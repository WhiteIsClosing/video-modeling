#!/bin/bash
# Scripts for training the model. 
source /deep/u/kuanfang/optical-flow-pred/scripts/config.sh

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
if [ -d $prediction ] 
then
  echo $prediction 'exists'
  mv $prediction $backup_root$prediction
fi
mkdir $prediction
# mkdir $prediction'/train'
# mkdir $prediction'/test'

echo 'Copied existed folders to backup/ and created new folders'

python train.py
