from nn_cifar.load_train_data import TrainData
from nn_cifar.cnn_model import CNNModel
import torch
from torch.optim import Adam
import torch.nn.functional as F
import time

X_filepath = "/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/nn_cifar/data/X_edge.pt"
y_filepath = "/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/nn_cifar/data/y_0.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = TrainData(X_filepath,y_filepath,device)


model = CNNModel()

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
optimizer = Adam(model.parameters(),lr=0.001)


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
    # if acc > 0.999:
        # optimizer = Adam(model.parameters(), lr=0.0001)
    message = "{0} (total {1}| current {2})".format(acc,iteration_end - start, iteration_end - iteration_start)
    print(message)

    if acc > 0.999:
        break






