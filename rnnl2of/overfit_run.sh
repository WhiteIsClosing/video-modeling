#!/bin/bash
# Scripts for training the pgpjerk model. 

logfile='LOG.txt'
models='models'
visualization='visualization'
prediction='prediction'
features='features'
backup='backup'
backup_root=$backup'/'

if [ -d $backup ] 
then
  rm -r $backup 
  echo $backup 'exists'
fi
mkdir $backup

if [ -f $logfile ] 
then
  echo $logfile 'exists'
  mv $logfile $backup_root$logfile 
fi

if [ -d $models ] 
then
  echo $models 'exists'
  mv $models $backup_root$models
fi
mkdir $models

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

if [ -d $prediction ] 
then
  echo $prediction 'exists'
  mv $prediction $backup_root$prediction
fi
mkdir $prediction
# mkdir $prediction'/train'
# mkdir $prediction'/test'

# if [ -d $features ] 
# then
#   echo $features 'exists'
#   mv $features $backup_root$features
# fi
# mkdir $features

echo 'Copied existed folders to backup/ and created new folders'

python overfit_train.py
