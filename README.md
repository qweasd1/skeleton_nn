# Fast Distributed Ensemble NN training for classification task

## dependencies
* pytorch
* mlxtend
* opencv

## file structure
* nn_mnist: generate training data for mnist, do training and evaluate its performance
* cifar: generate training data for CIFAR, do training and evaluate its performance. The file contains 100 is for 
CIFAR 100, otherwise CIFAR 10

The ```cnn_model.py``` file contains NN structure, the ```train_adaptive_binary_model.py``` contains the code to train FDE model, the ```train_model.py``` file contains the logic to train the all-in-one model. 
The ```valid_all_adaptive_binary_model.py``` file contains the validation logic of model performance.