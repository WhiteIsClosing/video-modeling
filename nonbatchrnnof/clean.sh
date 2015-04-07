#!/bin/bash
# Clean the folder and keep only all source codes. 

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
fi

if [ -f $logfile ] 
then
  rm $logfile
fi

if [ -d $models ] 
then
  rm -r $models 
fi

if [ -d $predicted_frames ] 
then
  rm -r $predicted_frames 
fi

if [ -d $preds ] 
then
  rm -r $preds 
fi

if [ -d $features ] 
then
  rm -r $features 
fi

rm *.pyc
rm .*.swp

echo 'Removed all results.'

