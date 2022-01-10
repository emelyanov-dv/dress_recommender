from PIL import Image
import pandas as pd
import numpy as np
from torchvision import transforms
import torch 
import io
import base64
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

device = 'cpu'

def predict_bbox(model, img, image_size):
  '''
    run yolo
    return PIL image and bbox
  '''
  result = model(img, image_size)
  result.save('buffer')
  img_detected = Image.open('buffer/image0.jpg')
  return img_detected, result.pandas().xyxy[0]

def crop_dresses(img, result):
  '''
     takes source image and reult from predict_bbox function
  '''
  recognized_dresses = []
  for row_idx in result.index:
    bbox_coordinates = result.iloc[row_idx, :].values[:4]
    confidence = result.iloc[row_idx, :].values[4]
    cropped_img = img.crop(bbox_coordinates)
    #cropped_img.show()
    recognized_dresses.append(cropped_img)
  return recognized_dresses

def get_embeddings(images, model):
  '''
     images - PIL image list, model - resnet
     return - list of [512] embeddings
  '''
  # convert_tensor = transforms.ToTensor()
  trans = transforms.Compose([
    transforms.Resize((125, 225)), # mean sizes of detected bbox
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # imagenet
  ])
  embeddings = []
  for img in images:
    img = torch.unsqueeze(trans(img), 0).to(device)
    embeddings.append(model(img)) 
  return embeddings

def get_image_similarity(query_embedding, database_embeddings):
  '''
     query_embedding - embedding of the picture using in the search
     database_embeddings - embeddings in database from train dataset
     one embedding per run
  '''
  # get cosine similarity
  similarity = cosine_similarity(query_embedding.cpu().detach().numpy(), database_embeddings).reshape(-1)
  # similarity = pd.Series(similarity)

  # sort and get index to show
  similarity_result =  pd.Series(similarity)
  return similarity_result

def get_similar_images(query_embedding, num_to_show,  similarity_threshold, dbfloder, embeddings_total_cropped, recognized_dress):
  '''
     Find similar embeddings in local database, read similar images from 'database', response mosaic, built with mosaic_images
  '''
  recognized_dress = np.array(recognized_dress.resize((120, 120)))
  recognized_dress = np.expand_dims(recognized_dress,0)
  recognized_dress = np.transpose(recognized_dress, (0, 3, 1,2))
  images = [recognized_dress]

  similarity_result = get_image_similarity(query_embedding.reshape(1,-1), embeddings_total_cropped)
  # select only only above threshold
  index_to_show  = similarity_result[similarity_result > similarity_threshold].sort_values(ascending = False)[:num_to_show].index
  print(index_to_show)
  #nothing was found
  if len(index_to_show) == 0:
    return -1  
  
  # create mosaic of similar images
  for img_name in index_to_show:
    img = Image.open(f'{dbfloder}{img_name}.jpg')
    # to mosaic convert to (batch_size, channels, h, w)
    im_resized = img.resize((120, 120))
    np_img = np.array(im_resized)
    np_img = np.expand_dims(np_img,0)
    np_img = np.transpose(np_img, (0, 3, 1,2))
    images.append(np_img)
  mosaic = mosaic_images(np.concatenate(images, 0))
  mosaic = Image.fromarray(mosaic)
  return mosaic

def mosaic_images(images, max_size=1920, max_subplots=16):
    '''
       collects from numpy (batch_size, channels, h, w) single numpy image
    '''
    # Plot image grid with labels
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if np.max(images[0]) <= 1:
        images *= 255.0  # de-normalise (optional)
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Build Image
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i, im in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        im = im.transpose(1, 2, 0)
        mosaic[y:y + h, x:x + w, :] = im
        
    # Resize (optional)
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))
    return mosaic

def convert_to_bytes(img):
    '''
       convert image to base64 bytes
    '''
    buffered = io.BytesIO() 
    img = img.convert("RGB")
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str
  
# change the last layer in RESNET to get 512 flat embeddings
class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
