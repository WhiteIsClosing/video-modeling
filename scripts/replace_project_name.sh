#!/bin/bash
# find ./ -name hyper_params.py | while read line; do
#     f=$line
#     #sed -i 's/project_name/project_name=\'/deep/u/kuanfang/video-modeling\'/g' $f
#     sed -i 's/optical-flow-pred/video-modeling/g' $f
#     echo 'Done with '$f
# done

find ./ -name *.sh | while read line; do
    f=$line
    #sed -i 's/project_name/project_name=\'/deep/u/kuanfang/video-modeling\'/g' $f
    sed -i 's/optical-flow-pred/video-modeling/g' $f
    echo 'Done with '$f
done
