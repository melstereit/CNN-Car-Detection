#!/usr/bin/env bash
# Get Cars-Dataset
wget http://imagenet.stanford.edu/internal/car196/car_ims.tgz
wget http://imagenet.stanford.edu/internal/car196/cars_annos.mat
wget http://imagenet.stanford.edu/internal/car196/cars_test_annos_withlabels.mat
tar -xzvf car_ims.tgz
rm car_ims.tgz
