"""
voc_to_tfr.py
Create a TFRecord file from images and Pascal VOC encoded annotation xmls.

Part of tfrecord-view repo: https://github.com/EricThomson/tfrecord-view
"""
import numpy as np
import tensorflow as tf
import os
import glob
from lxml import etree


def create_tf_example(data,
                      image_path,
                      label_map_dict,
                      ignore_difficult_instances=False, 
                      verbose = 1):
    """
    Convert image/xml-derived annotation dict to tensorflow example file to be
    incorporated into a TFRecord. Adapted from:
            https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data, so they are between [0, 1].

    Inputs:
        data: dict holding PASCAL XML fields for a single image (obtained by
            running recursive_parse_xml_to_dict)
        image_path: Path to image
        label_map_dict: A map from string label names to integers ids.
        ignore_difficult_instances: Whether to skip difficult instances in the
            dataset    (default: False).
        verbose (default 1): 1 to show image info during encoding, 0 otherwise

    Returns:
        example: The converted tf.Example.

    """
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_image = fid.read()

    if verbose: print(f"Encoding {image_path}")
    # For some reason after processing xml, it frequently returns width/height switched!
    width = int(data['size']['width'])
    height = int(data['size']['height'])

    #If no data['object'] there are no bounding boxes
    if 'object' in data:
        annotation_list = data['object']
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        classes = []
        classes_text = []
        difficult_obj = []
    

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
                'image/height': int64_feature(height),
                'image/width': int64_feature(width),
                'image/filename': bytes_feature(data['filename'].encode('utf8')),
                'image/encoded': bytes_feature(encoded_image),
                'image/object/bbox/xmin': float_list_feature(xmin),
                'image/object/bbox/xmax': float_list_feature(xmax),
                'image/object/bbox/ymin': float_list_feature(ymin),
                'image/object/bbox/ymax': float_list_feature(ymax),
                'image/object/class/text': bytes_list_feature(classes_text),
                'image/object/class/label': int64_list_feature(classes),
                'image/annotated': int64_feature(0)
        }
    
        tf_features = tf.train.Features(feature = obj_features)
        tf_example = tf.train.Example(features = tf_features)
    
            
    else:
        if verbose: print("No annotations for this one")
        obj_features = {
                'image/height': int64_feature(height),
                'image/width': int64_feature(width),
                'image/filename': bytes_feature(data['filename'].encode('utf8')),
                'image/encoded': bytes_feature(encoded_image),
                'image/annotated': int64_feature(0)
        }
    
    tf_features = tf.train.Features(feature = obj_features)
    tf_example = tf.train.Example(features = tf_features)
    return tf_example


#Following feature encoders are from models/research/object_detection/dataset_util.py
def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

#Following is from models/research/object_detection/dataset_util.py
def recursive_parse_xml_to_dict(xml):
  """Recursively parses XML contents to python dict.

  We assume that `object` tags are the only ones that can appear
  multiple times at the same level of a tree.

  Args:
    xml: xml tree obtained by parsing XML file contents using lxml.etree

  Returns:
    Python dictionary holding XML contents.
  """
  if not xml:
    return {xml.tag: xml.text}
  result = {}
  for child in xml:
    child_result = recursive_parse_xml_to_dict(child)
    if child.tag != 'object':
      result[child.tag] = child_result[child.tag]
    else:
      if child.tag not in result:
        result[child.tag] = []
      result[child.tag].append(child_result[child.tag])
  return {xml.tag: result}

#%%
if __name__ == '__main__':
    # Repo
    class_labels =    {"dog" : 1, "cat": 2 }
    data_path = r"annotated_images/"
    output_path =    data_path + r'cats_dogs.record'
    
    
    verbose = 1
    filename_query = os.path.join(data_path, '*.png')    #can change to any format (bmp, png etc)
    image_paths = np.sort(glob.glob(filename_query))
    
    writer = tf.python_io.TFRecordWriter(output_path)
    for idx, image_path in enumerate(image_paths):
        xml_path = os.path.splitext(image_path)[0] + '.xml'

        with tf.gfile.GFile(xml_path, 'rb') as fid:
                xml_str = fid.read()

        xml = etree.fromstring(xml_str)
        xml_data = recursive_parse_xml_to_dict(xml)['annotation']
        tf_example = create_tf_example(xml_data, image_path, class_labels, verbose = verbose)  
        writer.write(tf_example.SerializeToString())

    writer.close()
    print("Done encoding data TFRecord file")
