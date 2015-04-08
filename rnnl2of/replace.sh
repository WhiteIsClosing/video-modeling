#!/bin/bash

for f in *.py
do
  sed -i 's///g' $f

  sed -i 's/numtrain/numseqs_train/g' $f
  sed -i 's/numtest/numseqs_test/g' $f

  sed -i 's/frame_len/frame_dim/g' $f
  sed -i 's/seq_len/seq_dim/g' $f
  sed -i 's/numframes/seq_len/g' $f

  sed -i 's/trainframes/numframes_train/g' $f
  sed -i 's/testframes/numframes_test/g' $f

  sed -i 's/train_list/list_train/g' $f
  sed -i 's/train_features_numpy/features_train_numpy/g' $f
  sed -i 's/train_ofx/ofx_train/g' $f
  sed -i 's/train_ofy/ofy_train/g' $f

  sed -i 's/test_list/list_test/g' $f
  sed -i 's/test_features_numpy/features_test_numpy/g' $f
  sed -i 's/test_ofx/ofx_test/g' $f
  sed -i 's/test_ofy/ofy_test/g' $f

  sed -i 's/root/path/g' $f

  echo 'Done with '$f
done
