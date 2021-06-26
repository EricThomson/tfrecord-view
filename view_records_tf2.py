"""
The tf2 version is based in https://github.com/jschw/tfrecord-view/blob/master/tfrecord_view_gui.py

view_records.py:
Consume and display data from a tfrecord file: pulls image and bounding boxes for display
so you can make sure things look reasonabloe, e.g., after augmentation.

Hit 'n' for 'next' image, or 'esc' to quit.

Part of tensorflow-view repo: https://github.com/EricThomson/tfrecord-view

"""

import cv2
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore', category = FutureWarning)  #tf 1.14 and np 1.17 are clashing: temporary solution

def cv_bbox(image, bbox, color = (255, 255, 255), line_width = 2):
    """
    use opencv to add bbox to an image
    assumes bbox is in standard form x1 y1 x2 y2
    """

    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, line_width)
    return


def parse_record(data_record):
    """
    parse the data record from a tfrecord file, typically pulled from an iterator,
    in this case a one_shot_iterator created from the dataset.
    """
    feature = {'image/encoded': tf.io.FixedLenFeature([], tf.string),
                'image/object/class/label': tf.io.VarLenFeature(tf.int64),
                'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
                'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
                'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
                'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
                'image/filename': tf.io.FixedLenFeature([], tf.string)
                }
    return tf.io.parse_single_example(data_record, feature)


def view_records(file_path, class_labels, stride = 1, verbose = 1):
    """
    peek at the data using opencv and tensorflow tools.
    Inputs:
        file_path: path to tfrecord file (usually has 'record' extension)
        class_labels: dictionary of labels with name:number pairs (start with 1)
        stride (default 1): how many records to jump (you might have thousands so skip a few)
        verbose (default 1): display text output if 1, display nothing except images otherwise.

    Usage:
    Within the image window, enter 'n' for next image, 'esc' to stop seeing images.
    """
    dataset = tf.data.TFRecordDataset([file_path])
    record_iterator = iter(dataset)
    num_records = dataset.reduce(np.int64(0), lambda x, _: x + 1).numpy()

    if verbose:
        print(f"\nGoing through {num_records} records with a stride of {stride}.")
        print("Enter 'n' to bring up next image in record.\n")
    for im_ind in range(num_records):

        #Parse and process example

        parsed_example = parse_record(record_iterator.get_next())
        if im_ind % stride != 0:
            continue

        fname = parsed_example['image/filename'].numpy()
        encoded_image = parsed_example['image/encoded']
        image_np = tf.image.decode_image(encoded_image, channels=3).numpy()

        labels =  tf.sparse.to_dense(parsed_example['image/object/class/label'], default_value=0).numpy()
        x1norm =  tf.sparse.to_dense( parsed_example['image/object/bbox/xmin'], default_value=0).numpy()
        x2norm =  tf.sparse.to_dense( parsed_example['image/object/bbox/xmax'], default_value=0).numpy()
        y1norm =  tf.sparse.to_dense( parsed_example['image/object/bbox/ymin'], default_value=0).numpy()
        y2norm =  tf.sparse.to_dense( parsed_example['image/object/bbox/ymax'], default_value=0).numpy()

        num_bboxes = len(labels)

        #% Process and display image
        height, width = image_np[:, :, 1].shape
        image_copy = image_np.copy()
        image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

        if num_bboxes > 0:
            x1 = np.int64(x1norm*width)
            x2 = np.int64(x2norm*width)
            y1 = np.int64(y1norm*height)
            y2 = np.int64(y2norm*height)
            for bbox_ind in range(num_bboxes):
                    bbox = (x1[bbox_ind], y1[bbox_ind], x2[bbox_ind], y2[bbox_ind])
                    label_name = list(class_labels.keys())[list(class_labels.values()).index(labels[bbox_ind])]
                    label_position = (bbox[0] + 5, bbox[1] + 20)
                    cv_bbox(image_rgb, bbox, color = (250, 250, 150), line_width = 2)
                    cv2.putText(image_rgb,
                                label_name,
                                label_position,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (10, 10, 255), 2); #scale, color, thickness

        if verbose:
            print(f"\nImage {im_ind}")
            print(f"    {fname}")
            print(f"    Height/width: {height, width}")
            print(f"    Num bboxes: {num_bboxes}")
        cv2.imshow("bb data", image_rgb)
        k = cv2.waitKey()
        if k == 27:
            break
        elif k == ord('n'):
            continue
    cv2.destroyAllWindows()
    if verbose:
        print("\n\ntfrecord-view: done going throug the data.")


#%%
if __name__ == '__main__':
    class_labels =  {"dog" : 1, "cat": 2 }
    #Make the following using voc_to_tfr.py
    data_path = r"annotated_images/cats_dogs.record"

    verbose = 1
    stride = 1
    view_records(data_path, class_labels, stride = stride, verbose = verbose)
