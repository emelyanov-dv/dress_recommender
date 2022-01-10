# Dress recommender service

Demo project for dress recommendations, similar to dresses found at an uploaded image.
Developed using: pytorch, YOLOv5, pretrained resnet-18.


Prototyping was done in [Colab](https://colab.research.google.com/drive/1esvpzlzLhfykj4c8VCd7bF01z8-QIUjr?usp=sharing).


Server part: 
* server.py – flask application;
*	yolo_evaluator.py – main part, model loading and inference;
*	utilities.py – image processing helpers;
*	swagger.yaml – API conf.

Client part:
* client.py - script for POST request sending;
* requirements.txt - requirments for client.py;
* valdataset - deepfashion dress validation part; 

Client arguments: 
* --file - send an image from local folder; 
* --url - download from url and send an image;
* --server - IP address of server app, default: localhost:9091

Examples:
* ~# python3 client.py --file valdataset/1.jpg
* ~# python3 client.py --url https://media.sezane.com/image/upload/c_fill,d_placeholder_dark.png,fl_progressive:semi,h_816,q_auto:best,w_582/esayegh8c9rtnb0rqtqm.jpg
* ~# python3 client.py --file valdataset/1.jpg --server localhost:9091



There are several steps: 
1. Send POST request with an image using **client.py** 
2. **server.py** gets an image, runs YOLOv5, crop detected dresses
3. Pretrained **resnet-18** gets embeddings, searches for the most similar in local folder using cosine similarity
4. **server.py** responses with a mosaic image of similar dresses, found in local database
5. **client.py** creates folder, corresponding to a run number and saves responses images. 
