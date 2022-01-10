import requests
import sys
import io
import base64
from PIL import Image
import os
import re
import argparse 

def create_run_dir():
  '''
     creates 'runs' dir where run_x subdirs contains results of the service runs
     returns created subdir relative path
  '''
  directory = 'runs'
  if not os.path.exists(directory):
      os.makedirs(directory)

  dirs = [dirname for dirname in os.listdir('runs')]

  if len(dirs)>0:
    run_numbers = list(map(lambda dir: int(re.findall('\d+', dir)[0]), dirs))
    run_dir = directory+f'/run_{max(run_numbers)+1}/'
  else:
    run_dir = directory+'/run_1/'
  os.makedirs(run_dir)
  return run_dir

def send_request():
   '''
      send request, recieve json with binary base 64 images strings, save to folder runs
   ''' 
   parser = argparse.ArgumentParser()
   parser.add_argument("--file", help="use filepath to eval")
   parser.add_argument("--url", help="use url to http get picture")
   parser.add_argument("--server", help="use server ipv4:port to post picture")
   args = parser.parse_args()
   
   if (args.url is None) & (args.file is None):
      parser.print_help()
      return -1
   
   if args.url:
      pload = requests.get(args.url, stream=True).raw
   elif args.file:
      pload = open(args.file, 'rb')
   
   
   #create dir to saveresults
   run_dir = create_run_dir()
   
   #send post
   if not args.server:
   	r = requests.post('http://localhost:9091/',data = pload, headers={'Content-Type': 'Content-type: image/png'})
   else:
   	r = requests.post('http://'+args.server+'/',data = pload, headers={'Content-Type': 'Content-type: image/png'})
   
   #check if response is json
   try:
      if isinstance(eval(r.text), dict):
        resp = eval(r.text)
      else:
        raise Exception('not an image') 
   except Exception as e:
        print(r.text)
        return -1
   
   #decode all response images and save to dir 
   try:
      for key, image in resp.items():      
         img_bin_str = base64.b64decode(image.encode('utf-8'))
         buffered = io.BytesIO()
         buffered.write(img_bin_str)
         img = Image.open(buffered)
         img.show()
         img.save(run_dir+key+'.jpg')
   except Exception as e:
         print('Response image is corrupted')
         return -1
         
   print (f'Results are saved to {run_dir} folder')
   if len(resp)==1:
      print('no similar dresses were found')
   return 0
   
send_request()
