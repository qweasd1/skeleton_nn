import torch
import math

class TrainData:
    def __init__(self, X_filepath, y_filepath, device):

        self.X = torch.load(X_filepath).to(device)
        self.y = torch.load(y_filepath).to(device)
        self.size = len(self.X)


    def batches(self,batch_size=32):
        X = self.X
        y = self.y
        batch_count = math.ceil(self.size / batch_size)

        def iterator():
            for i in range(batch_count):
                start = i*batch_size
                end = (i+1)*batch_size
                yield X[start:end],y[start:end]
        return iterator()


class AdaptiveTrainData:
    def __init__(self, X_filepath, y_filepath, device,state_path=None):
        self.X = torch.load(X_filepath).to(device)
        self.y = torch.load(y_filepath).to(device)
        self.device = device
        self.init_state(state_path)


    def init_state(self,state_path=None):
        self.expand_size = 100
        self.end_size = 200

        self.models = []


        self.size = self.X.size()[0]
        self.next_train_indice = []

        self.is_target_table = torch.full((self.size,), True, dtype=torch.bool)
        self.is_target_table[self.y == 0] = False

        self.negative_samples_indice = (~self.is_target_table).nonzero().view(-1)
        self.negative_samples_X = self.X[self.negative_samples_indice]
        self.negative_samples_y = self.y[self.negative_samples_indice]
        if state_path is None:
            self.is_consumed_table = self.is_target_table.clone()
        else:
            self.is_consumed_table = torch.load(state_path)["is_consumed_table"]

        self.find_init_train_indice()


    def find_init_train_indice(self):

        print("left {0}".format(self.is_consumed_table.sum().item()))

        positive_sample_size = 500
        negative_sample_size = 100

        left_positive_indice = ((self.y == 1) & self.is_consumed_table).nonzero().view(-1)

        if len(left_positive_indice) == 0:
            return False

        if len(left_positive_indice) > self.end_size:
            positive_sample = left_positive_indice[:positive_sample_size]
        else:
            positive_sample = left_positive_indice[:self.end_size]

        negative_sample = self.y.eq(0).nonzero()[:negative_sample_size].view(-1)
        self.next_train_indice = torch.cat((positive_sample, negative_sample))

        self.is_consumed_table[self.next_train_indice] = False

        return True

    def batches(self, batch_size=32):
        self.current_X = X = self.X[self.next_train_indice]
        self.current_y = y = self.y[self.next_train_indice]
        self.current_size = current_size = len(X)
        batch_count = math.ceil(current_size / batch_size)

        def iterator():
            for i in range(batch_count):
                start = i * batch_size
                end = (i + 1) * batch_size
                yield X[start:end], y[start:end]

        return iterator()

    def new_model(self):
        # self.models.append(self.current_model)
        return self.find_init_train_indice()

    # def save_model(self, root, target):
    #     for i, model in enumerate(self.models):
    #         torch.save(model.state_dict(), "{2}/{0}_{1}".format(target, i, root))

    def find_to_expand(self, model):
        to_expand = torch.Tensor([]).long()
        find_error_size = self.expand_size * 5
        batch_count = math.ceil(len(self.negative_samples_indice) / find_error_size)
        left = self.expand_size
        for i in range(batch_count):
            start = i * find_error_size
            end = (i + 1) * find_error_size
            to_expand_segement = (model(self.negative_samples_X[start:end]).argmax(axis=1) != self.negative_samples_y[
                                                                                              start:end]).nonzero().view(
                -1)[:left] + start
            left -= len(to_expand_segement)
            to_expand = torch.cat((to_expand, self.negative_samples_indice[to_expand_segement]))
            if left == 0:
                break
        return to_expand

    def evaluate_and_expand(self, model):
        self.current_model = model
        y_p = model(self.current_X).argmax(axis=1)
        acc = (y_p == self.current_y).sum().item() / self.current_size
        if acc > 0.95:

            # old way to find to expand
            # y_p_all = model(self.X).argmax(axis=1)

            # all_negative_sample_indice = ((y_p_all != self.y) & ~self.is_target_table)
            # to_expand = all_negative_sample_indice.nonzero()[:self.expand_size].view(-1)

            # new way to find to expand
            to_expand = self.find_to_expand(model)

            self.next_train_indice = torch.cat((self.next_train_indice, to_expand))
            print("data_size: {0}".format(len(self.next_train_indice)))
            if len(to_expand) == 0:
                y_p_all = model(self.X).argmax(axis=1)
                all_positive_instance_indice = ((y_p_all == self.y) & self.is_target_table)
                self.is_consumed_table[all_positive_instance_indice] = False
                raise AddModelException()

        return acc

    def reset_current_model(self):
        print("reset model")
        self.is_consumed_table[self.next_train_indice] = True
        self.find_init_train_indice()

    def save(self,target):
        self.reset_current_model()
        torch.save({"is_consumed_table":self.is_consumed_table},"./state_{0}".format(target))






class AddModelException(Exception):
    pass

class FinishException(Exception):
    pass