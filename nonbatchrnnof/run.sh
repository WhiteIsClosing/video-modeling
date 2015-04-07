#!/bin/bash
# Scripts for training the pgpjerk model. 

logfile='LOG.txt'
models='models'
predicted_frames='predicted_frames'
preds='preds'
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

if [ -d $predicted_frames ] 
then
  echo $predicted_frames 'exists'
  mv $predicted_frames $backup_root$predicted_frames
fi
mkdir $predicted_frames
mkdir $predicted_frames'/train'
mkdir $predicted_frames'/train/ofx'
mkdir $predicted_frames'/train/ofy'
mkdir $predicted_frames'/train/true_ofx'
mkdir $predicted_frames'/train/true_ofy'
mkdir $predicted_frames'/test'
mkdir $predicted_frames'/test/ofx'
mkdir $predicted_frames'/test/ofy'
mkdir $predicted_frames'/test/true_ofx'
mkdir $predicted_frames'/test/true_ofy'

if [ -d $preds ] 
then
  echo $preds 'exists'
  mv $preds $backup_root$preds
fi
mkdir $preds
mkdir $preds'/train'
mkdir $preds'/test'

if [ -d $features ] 
then
  echo $features 'exists'
  mv $features $backup_root$features
fi
mkdir $features

echo 'Copied existed folders to backup/ and created new folders'

python train.py
