#!/bin/bash
# Make a backup folder without training new models

logfile='LOG.txt'
models='models'
predicted_frames='predicted_frames'
preds='preds'
features='features'
backup='backup_stored'
backup_root=$backup'/'

if [ -d $backup ] 
then
  echo $backup 'exists'
else
  mkdir $backup
fi

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

if [ -d $predicted_frames ] 
then
  echo $predicted_frames 'exists'
  mv $predicted_frames $backup_root$predicted_frames
fi

if [ -d $preds ] 
then
  echo $preds 'exists'
  mv $preds $backup_root$preds
fi

if [ -d $features ] 
then
  echo $features 'exists'
  mv $features $backup_root$features
fi

echo 'Copied existed folders to backup/ and created new folders'
echo 'Made a backup.'

