#!/bin/bash
# Scripts for visualizing the predicted results
source /deep/u/kuanfang/optical-flow-pred/scripts/config.sh

if [ -d $visualization ] 
then
  rm -r $visualization
  echo $visualization ' removed'
fi

mkdir $visualization
mkdir $visualization'/wxf_left'
mkdir $visualization'/wxf_right'
mkdir $visualization'/train'
mkdir $visualization'/train/pred_frames'
mkdir $visualization'/train/pred_of'
mkdir $visualization'/train/pred_ofx'
mkdir $visualization'/train/pred_ofy'
mkdir $visualization'/train/true_frames'
mkdir $visualization'/train/true_of'
mkdir $visualization'/train/true_ofx'
mkdir $visualization'/train/true_ofy'
mkdir $visualization'/test'
mkdir $visualization'/test/pred_frames'
mkdir $visualization'/test/pred_of'
mkdir $visualization'/test/pred_ofx'
mkdir $visualization'/test/pred_ofy'
mkdir $visualization'/test/true_frames'
mkdir $visualization'/test/true_of'
mkdir $visualization'/test/true_ofx'
mkdir $visualization'/test/true_ofy'

python visualize.py
