#!/bin/bash
# Clean the folder and keep only all source codes. 

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
fi

if [ -f $logfile ] 
then
  rm $logfile
fi

if [ -d $models ] 
then
  rm -r $models 
fi

if [ -d $visualization ] 
then
  rm -r $visualization 
fi

if [ -d $prediction ] 
then
  rm -r $prediction 
fi

if [ -d $features ] 
then
  rm -r $features 
fi

rm *.pyc
rm .*.swp

echo 'Removed all results.'

