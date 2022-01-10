# Dress recommender service

Demo project for dress recommendations, based on uploaded image.

There are several steps: 
1. Send POST request with an image using **client.py** 
2. **server.py** gets an image, runs YOLOv5, crop detected dresses
3. Pretrained **resnet-18** gets embeddings, searches for the most similar in local folder using cosine similarity
4. **server.py** responses with a mosaic image of similar dresses, found in local database
5. **client.py** creates folder, corresponding to a run number and saves responses images. 


The **.ipynb** notebook shows the prototyping process steps. 
