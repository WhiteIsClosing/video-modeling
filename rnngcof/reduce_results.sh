#!/bin/bash
path='./'
models='models'
prediction='prediction'
idx=100

rm -r $path'/backup'

files="$(ls $path'models')"
for f in $files
do
  if [ $f == 'model_'$idx'.npy' ] 
  then
    echo 'save '$f
  elif [ $f == 'model.npy' ] 
  then
    echo 'save '$f
  else
    # echo 'rm '$f
    rm $path'models/'$f
  fi
done

files="$(ls $path'prediction')"
for f in $files 
do
  if [ $f == 'preds_train_'$idx'.npy' ] 
  then
    echo 'save '$f
  elif [ $f == 'preds_test_'$idx'.npy' ] 
  then
    echo 'save '$f
  else
    # echo 'rm '$f
    rm $path'prediction/'$f
  fi
done
