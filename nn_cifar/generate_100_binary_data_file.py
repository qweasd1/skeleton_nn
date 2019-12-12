import torch
import pickle
import numpy as np
data_path = "/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/nn_cifar/data/cifar-100-python"


y_train = np.zeros((50000,),dtype="long")

with open("{0}/train".format(data_path), "rb") as file:
    _data = pickle.load(file, encoding='bytes')
    label = _data[b"fine_labels"]
    y_train = np.array(label).astype("long")

data_dir = "/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/nn_cifar/data/100"

def to_tensor(y,target):
    y = torch.from_numpy((y == target).astype('long'))
    torch.save(y,"{0}/{1}.pt".format(data_dir,"y_" + str(target)))

for i in range(100):
    to_tensor(y_train,i)


