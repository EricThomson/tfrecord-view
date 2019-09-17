"""
converting from pascal voc to tf record.

Using some of main and setup from here:
    https://raw.githubusercontent.com/swirlingsand/deeper-traffic-lights/master/data_conversion_bosch.py
Directed from this not amazing site:
    https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-2-converting-dataset-to-tfrecord-47f24be9248d
    
The core conversion function is largely from here:
    https://github.com/AndrewCarterUK/tf-example-object-detection-api-race-cars/blob/master/create_pascal_tf_record.py
Which basically uses the code from the docs:
    https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py
    
Some of it is completely unecessary he pretty much mindlessly followed the docs. For something
that is more useful, going over what you actually need (e.g., not keys): 
    https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
    
For an actual helpful discussion of tf_records read up on it here:
    https://www.tensorflow.org/tutorials/load_data/tf_records
And some notes on working with bmp files:
    https://stackoverflow.com/questions/50871281/tensorflow-how-to-encode-and-read-bmp-images
    
usage
python vos_to_tfr_script2.py --output_path=/blah/blah


OK this is sort of ridiculous it stores all of teh data in a separate tfrecord file instead
 of just stoaring the locations? What teh fuck is this shit?
 
Ways to make this more efficient?
 https://stackoverflow.com/questions/47732186/tensorflow-tf-record-too-large-to-be-loaded-into-an-np-array-at-once
"""
import numpy as np
import tensorflow as tf
import os
import glob
from lxml import etree
from object_detection.utils import dataset_util

import logging 
logging.basicConfig(level=logging.DEBUG, filename = 'vos_to_tfr.log', filemode = 'a')  #INFO/Warning,/error/critical  filemode 'w/a' is for append




#%%
verbose = 1

def create_tf_example(data,
                       image_path,
                       label_map_dict,
                       ignore_difficult_instances=False):
  """
  From:
  https://raw.githubusercontent.com/AndrewCarterUK/tf-example-object-detection-api-race-cars/master/create_pascal_tf_record.py
  Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    image_path: Path to image described by the PASCAL XML file
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  with tf.gfile.GFile(image_path, 'rb') as fid:
      encoded_image = fid.read()

  if verbose: print(f"Encoding {image_path}")
  # For some reason after processing xml, it frequently returns width/height switched!
  width = int(data['size']['width'])
  height = int(data['size']['height'])


  try: 
    annotation_list = data['object']
    annotated = True
    num_bboxes = len(annotation_list)
  except KeyError:
    if verbose: print("No annotations for this one")
    annotated = False
    num_bboxes = 0
    
  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  difficult_obj = []
  
  if annotated: 
      for annotation in annotation_list:
        difficult = bool(int(annotation['difficult']))
        if ignore_difficult_instances and difficult:
          continue
    
        difficult_obj.append(int(difficult))
    
        x1 = annotation['bndbox']['xmin']
        y1 = annotation['bndbox']['ymin']
        x2 = annotation['bndbox']['xmax']
        y2 = annotation['bndbox']['ymax']
        xmin.append(float(x1) / width)
        xmax.append(float(x2) / width)
        ymin.append(float(y1) / height)
        ymax.append(float(y2) / height)
        classes_text.append(annotation['name'].encode('utf8'))
        classes.append(label_map_dict[annotation['name']])
    
  obj_features = {
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(data['filename'].encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_image),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }

  logging.info(f"    Features for {data['filename']}")
  logging.info(f"    Width/height: {width}, {height}")
  logging.info(f"    Num bboxes: {num_bboxes}" )

  tf_features = tf.train.Features(feature = obj_features)
  tf_example = tf.train.Example(features = tf_features)
            
  return tf_example




#%%
def main(_):
    class_labels =  {"dog" : 1, "cat": 2 }
    data_path = r"/home/eric/Pictures/cats_dogs/"
    output_path =  data_path + r'cats_dogs.record'
    
    
    
    filename_query = os.path.join(data_path, '*.png')  #can change to any format (bmp, png etc)
    image_paths = np.sort(glob.glob(filename_query))    
    writer = tf.python_io.TFRecordWriter(output_path)
    for idx, image_path in enumerate(image_paths):
        logging.info(f"\n\nStarting with test image {idx} stored at {image_path}")
        xml_path = os.path.splitext(image_path)[0] + '.xml'
      
        with tf.gfile.GFile(xml_path, 'rb') as fid:
            xml_str = fid.read()
    
        xml = etree.fromstring(xml_str)
        xml_data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        tf_example = create_tf_example(xml_data, image_path, class_labels)  #Leaving out  FLAGS.ignore_difficult_instances
        writer.write(tf_example.SerializeToString())
    
    writer.close()

    
