from torchvision import models
import torch
import pickle
import requests
import os
# functions to process images, get bbox, load similar images
from utilities import *

model_info = {
        "build": "0.1",
        "model": "yolov5s+resnet"
}


# load yolo trained on 320 picsize and set trained options
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolo320.pt', force_reload = False)
model.conf = 0.2  
model.iou = 0.2 
image_size = 320

# load resnet
device = 'cpu'
resnet = torch.load('resnet18.model') 
resnet.fc = Identity() # Identity - class to get the layer before last embeddings
resnet.to(device)

# load existing embeddings
with open('embeddings.pickle', 'rb') as f:
  embeddings_total = pickle.load(f)

similarity_threshold = 0.80
dbfloder = 'imagedb/'
similar_images_num = 16


def get_info():
    """
    This function responds to a request for /api/info
    with the sample model data

    :return:        model data
    """
    return str(model_info)


#used by post request
def run(sample):
    stream = io.BytesIO(sample)
    img = Image.open(stream)
    
    img_bbox, bbox = predict_bbox(model, img, image_size)    
    
    recognized_dresses = crop_dresses(img, bbox) 
    if len(recognized_dresses) == 0:
      return 'No dresses were found in the picture'
       
    # get embedding for every reckognized dress
    embeddings= []
    embeddings = get_embeddings(recognized_dresses,resnet)
    
    #construct mosaic of similar images, detected - first     	
    mosaic_list=[]
    for recognized_dress, emb in zip(recognized_dresses, embeddings):
        mosaic = get_similar_images(emb, similar_images_num,  similarity_threshold, dbfloder, embeddings_total, recognized_dress)
        if isinstance(mosaic, (Image.Image)):  #filter if nothing simialr was found     
           mosaic_list.append(mosaic)
    try:
       response = {'detected': convert_to_bytes(img_bbox).decode('utf-8')} # whole picture with bboxes
       for i, mosaic in enumerate(mosaic_list):
           response[f'{i}'] = convert_to_bytes(mosaic).decode('utf-8')
    except Exception as e: 
       return f'Nothing similar with {similarity_threshold} similarity threshold was found in the DB'
           
    return response
