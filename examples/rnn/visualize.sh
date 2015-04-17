#!/bin/bash
# Scripts for training the pgpjerk model. 

visualization='visualization'

if [ -d $visualization ] 
then
  echo $visualization 'exists'
  mv $visualization $backup_root$visualization
fi
mkdir $visualization
mkdir $visualization'/train'
mkdir $visualization'/train/pred_of'
mkdir $visualization'/train/pred_ofx'
mkdir $visualization'/train/pred_ofy'
mkdir $visualization'/train/true_frames'
mkdir $visualization'/train/true_of'
mkdir $visualization'/train/true_ofx'
mkdir $visualization'/train/true_ofy'
mkdir $visualization'/test'
mkdir $visualization'/test/pred_of'
mkdir $visualization'/test/pred_ofx'
mkdir $visualization'/test/pred_ofy'
mkdir $visualization'/test/true_frames'
mkdir $visualization'/test/true_of'
mkdir $visualization'/test/true_ofx'
mkdir $visualization'/test/true_ofy'

python visualize.py
