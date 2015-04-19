#!/bin/bash
# Clean the folder and keep only all source codes. 
source /deep/u/kuanfang/optical-flow-pred/scripts/config.sh

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

rm *.pyc
rm .*.swp

echo 'Removed all results.'

