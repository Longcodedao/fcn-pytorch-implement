#!/bin/bash

DIR=~/fcn-pytorch-implement/datasets/VOC
mkdir -p $DIR
cd $DIR

# if [ ! -e benchmark_RELEASE ]; then
#     wget https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz
#     tar -xvf benchmark.tgz
# fi

if [ ! -e VOCdevkit/VOC2012 ]; then
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    tar -xvf VOCtrainval_11-May-2012.tar
fi
