#!/bin/bash
path_to_dataset='../bouncing_balls'

if [ -d $path_to_dataset ] 
then
  rm -r $path_to_dataset 
  echo $path_to_dataset 'exists'
fi
mkdir $path_to_dataset
mkdir $path_to_dataset'/1balls'
mkdir $path_to_dataset'/1balls/train'
mkdir $path_to_dataset'/1balls/test'
cp list* $path_to_dataset'/1balls'

mkdir $path_to_dataset'/2balls'
mkdir $path_to_dataset'/2balls/train'
mkdir $path_to_dataset'/2balls/test'
cp list* $path_to_dataset'/2balls'

mkdir $path_to_dataset'/3balls'
mkdir $path_to_dataset'/3balls/train'
mkdir $path_to_dataset'/3balls/test'
cp list* $path_to_dataset'/3balls'

python gen.py
