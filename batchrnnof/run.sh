#!/bin/bash
# Scripts for training the pgpjerk model. 

logfile='LOG.txt'
models='models'
predicted_frames='predicted_frames'
predictions='predictions'
features='features'
backup='backup'
backup_root=$backup'/'

if [ -d $backup ] 
then
  rm -r $backup_root* 
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
mkdir $predicted_frames'/test'

if [ -d $predictions ] 
then
  echo $predictions 'exists'
  mv $predictions $backup_root$predictions
fi
mkdir $predictions
mkdir $predictions'/train'
mkdir $predictions'/test'

if [ -d $features ] 
then
  echo $features 'exists'
  mv $features $backup_root$features
fi
mkdir $features

echo 'Copied existed folders to backup/ and created new folders'

python train.py
