#!/bin/bash
# Add all files to git

logfile='LOG.txt'

git add *.py
git add *.sh

if [ -f $logfile ] 
then
  git add $logfile
else
  echo $logfile 'does not exit.'
fi

echo 'Added all files to git'
