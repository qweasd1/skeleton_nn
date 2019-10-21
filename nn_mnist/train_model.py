import os

from nn_mnist.load_train_data import MnistTrainData
from nn_mnist.cnn_model import CNNModel
import torch
from torch.optim.adamw import AdamW
import torch.nn.functional as F
import time

X_filepath = "/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/nn_mnist/data/X.pt"
y_filepath = "/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/nn_mnist/data/y.pt"
model_root = "/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/nn_mnist/all_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = MnistTrainData(X_filepath,y_filepath,device)


model = CNNModel(10)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

print("model parameters: {0}".format(get_n_params(model)))

model.to(device)
optimizer = AdamW(model.parameters(),lr=0.001)


start = time.time()
while True:
    iteration_start = time.time()
    for i, (X, y) in enumerate(train_data.batches()):
        optimizer.zero_grad()
        out = model(X)
        loss = F.nll_loss(out, y)
        loss.backward()
        optimizer.step()
    iteration_end = time.time()
    acc = model.calc_accuracy(train_data)
    if acc > 0.99:
        optimizer = AdamW(model.parameters(), lr=0.0001)

    if acc > 0.999:
        break
    message = "{0} (total {1}| current {2})".format(acc,iteration_end - start, iteration_end - iteration_start)
    print(message)

torch.save(model.state_dict(),os.path.join(model_root,"all_model"))






