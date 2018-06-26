#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 19:03:05 2018

@author: jai
"""

import tensorflow as tf
import os
import io
import re
import xml.etree.ElementTree as ET
from PIL import Image
from object_detection.utils import dataset_util

i=0

def create_tf_example(filename):
    
  print('working')  
  image_path='/home/jai/Downloads/Datasets/Taylor Swift/annotations/' + filename
  
  xml = re.sub('\.jpg$', '', filename)        
  
  label_path='/home/jai/Downloads/Datasets/Taylor Swift/labels/' + xml + '.xml' 
    
  img = Image.open(image_path)
  width, height = img.size
  img_bytes = io.BytesIO()
  img.save(img_bytes, format=img.format)

  height = height
  width = width
  encoded_image_data = img_bytes.getvalue()
  image_format = img.format.encode('utf-8')  
    
  # Read the label XML
  tree = ET.parse(label_path)
  root = tree.getroot()
  xmins = xmaxs = ymins = ymaxs = list()

  for coordinate in root.find('object').iter('bndbox'):
      xmins = [int(coordinate.find('xmin').text)]
      xmaxs = [int(coordinate.find('xmax').text)]
      ymins = [int(coordinate.find('ymin').text)]
      ymaxs = [int(coordinate.find('ymax').text)]

  classes_text = ['ts'.encode('utf-8')]
  classes = [1]
  
    

  
  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(encoded_image_data),
      'image/source_id': dataset_util.bytes_feature(encoded_image_data),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example




def main(_):
  writer = tf.python_io.TFRecordWriter('/home/jai/Downloads/training.record')

  for filename in os.listdir('/home/jai/Downloads/Datasets/Taylor Swift/annotations/'):
    tf_example = create_tf_example(filename)
    writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.app.run()
  
  



for example in tf.python_io.tf_record_iterator(''):
    result = tf.train.Example.FromString(example)