# tfrecord-view
How to consume data from TFRecord files, which are used in the Tensorflow [object detection api](https://github.com/tensorflow/models/tree/master/research/object_detection). I use it to double check my augmentation pipeline (built with [imgaug](https://github.com/aleju/imgaug)), and TFRecord encoding.

Currently tested in Linux. Not sure about behavior in Windows.

## Usage
### Creating a TFRecord file
If you need to create a TFRecord file, see `voc_to_tfr.py`. The images and annotation files are in `annotated_images/`.

### Consuming a TFRecord file
If you already have TFRecord file data, then use `view_records.py` to see how to consume it and show data. The function takes in the path to the TFRecord file, the dictionary of class labels, and a couple of optional keyword arguments like stride. It will then show the images with bounding boxes and labels for each object, if applicable.

### Installation
Prereqs: tensorflow, opencv, and numpy. Note the object detection API doesn't yet work with Tensorflow 2 (https://github.com/tensorflow/models/issues/6423), so we will be using version 1 of Tensorflow.

    git clone XXX
    conda create --name tfrecord-view
    conda activate --name tfrecord-view
    conda install python=3 opencv numpy
    conda install -c anaconda tensorflow-gpu>=1.12

 This also assumes you have installed the object detection api (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md), as we will use some utilities that come with it (in particular `utils.dataset_util`).

## For more info
Construction of TFRecord files:
- https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
- https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py

On consuming TFRrecord files, there aren't a lot of great resources out there. I used this (and probably 20 other sites I can't even remember: this is one aspect of the api that keeps changing, and will surely change again once it is ported to Tensorflow 2):
- https://stackoverflow.com/a/56932321/1886357

I recently found this repo which is similar to this one, and has some nice ideas:
- https://github.com/yinguobing/tfrecord_utility

## To do
- Functionalize encoder in voc_to_tfr.py.
- Look over tfrecord_utility repo maybe he found a way to simplify reading data?

#### Sources for images
The images are of cats and dogs, with one that has no label. The images were scraped from:
- https://huggablemuggs.com/8-tricks-to-help-your-cat-and-dog-to-get-along/
- https://2catsandablog.wordpress.com/2018/08/14/do-cats-and-dogs-really-fight-like-cats-and-dogs/
- http://www.waycooldogs.com/feeding-cats-dogs-together/
- https://phz8.petinsurance.com/ownership-adoption/pet-ownership/pet-behavior/7-tips-on-combining-multi-pet-household
- https://www.mercurynews.com/2019/04/15/whos-going-to-tell-mom-shes-feeding-her-dogs-the-wrong-food/
- https://www.meowingtons.com/blogs/lolcats/snuggly-cat-and-dog-best-friends-to-cheer-you-up
- https://www.thesprucepets.com/cute-aquarium-sea-critters-4146506
