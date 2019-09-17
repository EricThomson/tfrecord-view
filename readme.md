# tfrecord-view
Module to display images and bounding boxes from data encoded in tensorflow record file, for the tensorflow object detection api (https://github.com/tensorflow/models/tree/master/research/object_detection). Also includes script to illustrate how to create a tfrecord file from a bunch of images and annotations.

## To do
- create agitignore, include .record files and other whatever in that.
- post to git.
- make sure it works on another computer.

## Installation
Prereqs: tensorflow, opencv, and numpy. Note the object detection API doesn't yet work with tensorflow 2 (https://github.com/tensorflow/models/issues/6423), so we will be using version 1 of tensorflow.

    git clone XXX
    conda create --name tfrecord-view
    conda activate --name tfrecord-view
    conda install python=3 opencv numpy
    conda install -c anaconda tensorflow-gpu>=1.12

 This also assumes you have are using (and have installed) the object detection api (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md), as we will use some utilities that come with it (in particular `utils.dataset_util`).

## Usage
If you already have TFRecord file data and the associated vos-encoded annotations (xml files), then you can dive into `view_records.py`. Basically, you give the view_records() function the path to the tfrecord file, the dictionary of class labels, stride (in case you want to skip over some images in the record), and verbosity (set it to 1 to display some text info about each image, otherwise 0.

If you need to create a TFRecord file, I have included a script for that as well as some example images of cats/dogs for encoding, in  `vos_to_tfr.py`.

### For more info
Construction of TFRecord files:
- https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
- https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py

On consuming TFRrecord files, there aren't a lot of great resources out there. I used this (and probably 20 other sites I can't even remember):
- https://stackoverflow.com/a/56932321/1886357


#### Sources for images
- https://huggablemuggs.com/8-tricks-to-help-your-cat-and-dog-to-get-along/
- https://2catsandablog.wordpress.com/2018/08/14/do-cats-and-dogs-really-fight-like-cats-and-dogs/
- http://www.waycooldogs.com/feeding-cats-dogs-together/
- https://phz8.petinsurance.com/ownership-adoption/pet-ownership/pet-behavior/7-tips-on-combining-multi-pet-household
- https://www.mercurynews.com/2019/04/15/whos-going-to-tell-mom-shes-feeding-her-dogs-the-wrong-food/
- https://www.meowingtons.com/blogs/lolcats/snuggly-cat-and-dog-best-friends-to-cheer-you-up
- https://www.thesprucepets.com/cute-aquarium-sea-critters-4146506
