# Optical Flow prediction 
This project is used to predict optical flow using recurrent neural networks (RNNs) and other statistical models. 

Currently, we provide the following models:
* rnnl1: 1-hidden-layer RNN

## Setup
change the path_to_project_folder to the folder you store optical-flow-pred in 
* scripts/*.sh
* all the hyperParam.py in the examples you will run

## Data Generation
1. Bouncing Balls
```
cd data/bouncing_balls_generator
bash run.sh
```

## Training
```
cd examples/name_of_model
bash run.sh
```

## Visualize
After the training, you could visualize the prediction results by:
```
bash visualize.sh
```



