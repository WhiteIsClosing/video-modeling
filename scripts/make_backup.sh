#!/bin/bash
# Make a backup folder without training new models

logfile='LOG.txt'
models='models'
visualization='visualization'
prediction='prediction'
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

if [ -d $visualization ] 
then
  echo $visualization 'exists'
  mv $visualization $backup_root$visualization
fi

if [ -d $prediction ] 
then
  echo $prediction 'exists'
  mv $prediction $backup_root$prediction
fi

if [ -d $features ] 
then
  echo $features 'exists'
  mv $features $backup_root$features
fi

echo 'Copied existed folders to backup/ and created new folders'
echo 'Made a backup.'

