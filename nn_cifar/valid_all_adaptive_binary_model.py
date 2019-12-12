import torch
import os

from nn_cifar.cnn_model import CNNModel

root = "/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/nn_cifar/better_models"
data_root = "/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/nn_cifar/data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BinaryModel:
    def __init__(self, device):
        self.models = []
        self.device = device

    def add_subsubmodel(self, path):
        model = CNNModel(2)
        model.load_state_dict(torch.load(path,map_location=device))
        model.to(self.device)
        model.eval()

        self.models.append(model)

    def predict(self, X):
        pred = torch.zeros((len(X),)).bool()
        prob = torch.zeros((len(X),)).fill_(-float("Inf"))

        for model in self.models:
            y_prob = model(X)
            y_p = y_prob.argmax(axis=1)
            y_target_prob = torch.gather(y_prob, 1, y_p.view(-1, 1)).view(-1)
            pred = (pred | y_p.bool())
            prob = torch.max(prob, y_target_prob)
        return pred, prob


class EnsembleModel:
    def __init__(self, device):
        self.models = {}
        self.models_ = {}
        self.device = device

    def load_binary_models(self, root):
        submodel_subsubmodels = [[int(num) for num in file.split("_")] for file in os.listdir(root) if
                                 not file.startswith(".")]
        submodel_subsubmodels.sort()
        for target, chunk in submodel_subsubmodels:
            if target not in self.models:
                self.models[target] = BinaryModel(device)
            subsubmodel_path = os.path.join(root, "{0}_{1}".format(target, chunk))
            self.models[target].add_subsubmodel(subsubmodel_path)

    def load_binary_models_(self, root):
        submodel_subsubmodels = [[int(num) for num in file.split("_")] for file in os.listdir(root) if
                                 not file.startswith(".")]
        submodel_subsubmodels.sort()

        for target, chunk in submodel_subsubmodels:
            if target not in self.models:
                self.models[target] = []
            subsubmodel_path = os.path.join(root, "{0}_{1}".format(target, chunk))
            self.models[target].append(subsubmodel_path)
    def predict_single(self,X,i):
        result = []

        for target, binary_models in self.models.items():
            for i, subsubmodel_path in enumerate(binary_models):
                print("loading {0}_{1}".format(target, i))
                subsubmodel = CNNModel(2)
                subsubmodel.load_state_dict(torch.load(subsubmodel_path, map_location=self.device))
                subsubmodel.eval()

                y_prob = subsubmodel(torch.Tensor([X[i]]))

                print("finished {0}_{1}".format(target, i))

                unit_prob = y_prob[:, 1]
                result.append((y_prob[0][1].item(),target))
                result.sort(reverse=True)

        return result

    def predict_(self, X):
        result_size = len(X)
        pred = torch.full((result_size,), -1).to(device)
        prob = torch.full((result_size,), -1).to(device)

        for target, binary_models in self.models.items():
            for i, subsubmodel_path in enumerate(binary_models):
                print("loading {0}_{1}".format(target, i))
                subsubmodel = CNNModel(2)
                subsubmodel.load_state_dict(torch.load(subsubmodel_path, map_location=self.device))
                subsubmodel.eval()

                y_prob = subsubmodel(X)

                print("finished {0}_{1}".format(target, i))

                unit_prob = y_prob[:,1]
                for i in range(result_size):
                    if unit_prob[i] > prob[i]:
                        pred[i] = target
                        prob[i] = unit_prob[i]


        return pred, prob


if __name__ == '__main__':
    # submodel = BinaryModel(device)
    # target = 0
    # chunk_count = 8
    # for i in range(chunk_count):
    #     submodel.add_subsubmodel("{0}/{1}_{2}".format(root, target, i))
    # _Xt = torch.load("{0}/Xt_att2.pt".format(data_root)).to(device)
    # _yt = torch.load("{0}/yt_{1}.pt".format(data_root, target)).to(device)
    # pred, _ = submodel.predict(_Xt)
    #
    # # False Positive ((pred == 1) & (_yt == 0)).sum()
    # # True Negative ((pred == 0) & (_yt == 1)).sum()
    # print("acc: {0}".format((pred == _yt).sum().item() / len(_yt)))

    # 0.95 training
    #
    # 0.923 for target '1' FP 204 TN 566


    # Xt = torch.load("{0}/Xt_att2.pt".format(data_root)).to(device)
    # yt = torch.load("{0}/yt.pt".format(data_root)).to(device)

    Xt = torch.load("{0}/Xt_att2.pt".format(data_root)).to(device)
    yt = torch.load("{0}/yt.pt".format(data_root)).to(device)

    # #
    # # # Ensemble model
    # model = EnsembleModel(device)
    # model.load_binary_models_(root)
    # #
    # y_p, candidates = model.predict_(Xt)
    # print("acc: {0}".format((y_p == yt).sum().item() / len(y_p)))

    # all in one model
    with torch.no_grad():
        model = CNNModel(10)
        model.load_state_dict(torch.load("/Users/zhendongwang/Documents/projects/phd/skeleton_nn/code/nn_cifar/all_model/all_model_251.88759517669678_0.99658",map_location=device))
        model.eval()

        y_p = model(Xt).argmax(axis=1)
        print("acc: {0}".format((y_p == yt).sum().item() / len(y_p)))
    # print("acc: {0}".format(((y_p == yt) & ( (yt == 0) | ( yt == 1) )).sum().item()))

# t = [18, 36, 62, 92, 115, 167, 214, 241, 247, 259, 290, 321,
#      340, 359, 417, 444, 445, 447, 449, 495, 551, 557, 582, 583,
#      610, 625, 659, 674, 684, 691, 716, 720, 740, 844, 846, 882,
#      883, 900, 924, 938, 947, 956, 962, 965, 1014, 1039, 1062, 1112,
#      1125, 1226, 1232, 1242, 1260, 1319, 1328, 1393, 1414, 1425, 1435, 1469,
#      1522, 1527, 1530, 1549, 1553, 1554, 1597, 1641, 1681, 1709, 1717, 1737,
#      1748, 1754, 1782, 1790, 1850, 1865, 1871, 1878, 1901, 2004, 2018, 2024,
#      2035, 2052, 2063, 2070, 2109, 2125, 2130, 2135, 2148, 2185, 2189, 2280,
#      2293, 2298, 2299, 2314, 2326, 2369, 2414, 2426, 2454, 2488, 2534, 2582,
#      2597, 2607, 2648, 2654, 2659, 2686, 2695, 2720, 2758, 2760, 2769, 2770,
#      2810, 2896, 2927, 2939, 2953, 2995, 3012, 3021, 3030, 3059, 3060, 3062,
#      3189, 3266, 3289, 3337, 3422, 3451, 3474, 3475, 3492, 3503, 3520, 3550,
#      3558, 3599, 3674, 3727, 3762, 3767, 3808, 3811, 3821, 3850, 3853, 3859,
#      3869, 3871, 3941, 3946, 3950, 3951, 3976, 3985, 4007, 4065, 4078, 4116,
#      4123, 4163, 4176, 4201, 4224, 4238, 4255, 4256, 4321, 4350, 4360, 4369,
#      4374, 4380, 4400, 4437, 4443, 4477, 4497, 4498, 4500, 4507, 4536, 4578,
#      4671, 4724, 4731, 4740, 4743, 4748, 4761, 4763, 4785, 4807, 4814, 4823,
#      4838, 4860, 4880, 4899, 4939, 4966, 4978, 4997, 5183, 5201, 5210, 5449,
#      5457, 5600, 5623, 5634, 5655, 5734, 5771, 5936, 5937, 5955, 5972, 5973,
#      5981, 5997, 6004, 6011, 6059, 6081, 6091, 6157, 6166, 6173, 6400, 6418,
#      6505, 6532, 6555, 6557, 6558, 6564, 6571, 6574, 6576, 6597, 6599, 6603,
#      6614, 6625, 6641, 6651, 6740, 6783, 6847, 7121, 7208, 7216, 7233, 7338,
#      7430, 7434, 7545, 7978, 8094, 8097, 8160, 8236, 8246, 8287, 8316, 8325,
#      8326, 8332, 8376, 8408, 8486, 8523, 8607, 9009, 9015, 9019, 9024, 9280,
#      9530, 9634, 9642, 9664, 9669, 9679, 9698, 9719, 9729, 9768, 9770, 9792,
#      9839, 9904, 9922]
#
# t2 = [241, 247, 290, 340, 445, 449, 610, 625, 720, 844, 846, 924,
#       947, 1014, 1112, 1125, 1226, 1260, 1393, 1425, 1527, 1530, 1549, 1901,
#       2024, 2035, 2130, 2135, 2299, 2369, 2534, 2597, 2607, 2654, 2659, 2686,
#       2695, 2720, 2810, 2896, 2953, 3422, 3558, 3674, 3762, 3941, 4123, 4176,
#       4201, 4238, 4255, 4321, 4369, 4374, 4477, 4536, 4578, 4740, 4748, 4763,
#       4785, 4838, 4860, 4880, 4899, 5183, 5449, 5771, 5937, 5955, 6004, 6418,
#       6557, 6564, 6599, 6614, 8236, 8326, 8332, 9642, 9669, 9719, 9729, 9770]
#
# t_1 = [18, 36, 62, 92, 115, 167, 214, 259, 321, 359, 417, 444,
#        447, 495, 551, 557, 582, 583, 659, 674, 684, 691, 716, 740,
#        882, 883, 900, 938, 956, 962, 965, 1039, 1062, 1232, 1242, 1319,
#        1328, 1414, 1435, 1469, 1522, 1553, 1554, 1597, 1641, 1681, 1709, 1717,
#        1737, 1748, 1754, 1782, 1790, 1850, 1865, 1871, 1878, 2004, 2018, 2052,
#        2063, 2070, 2109, 2125, 2148, 2185, 2189, 2280, 2293, 2298, 2314, 2326,
#        2414, 2426, 2454, 2488, 2582, 2648, 2758, 2760, 2769, 2770, 2927, 2939,
#        2995, 3012, 3021, 3030, 3059, 3060, 3062, 3189, 3266, 3289, 3337, 3451,
#        3474, 3475, 3492, 3503, 3520, 3550, 3599, 3727, 3767, 3808, 3811, 3821,
#        3850, 3853, 3859, 3869, 3871, 3946, 3950, 3951, 3976, 3985, 4007, 4065,
#        4078, 4116, 4163, 4224, 4256, 4350, 4360, 4380, 4400, 4437, 4443, 4497,
#        4498, 4500, 4507, 4671, 4724, 4731, 4743, 4761, 4807, 4814, 4823, 4939,
#        4966, 4978, 4997, 5201, 5210, 5457, 5600, 5623, 5634, 5655, 5734, 5936,
#        5972, 5973, 5981, 5997, 6011, 6059, 6081, 6091, 6157, 6166, 6173, 6400,
#        6505, 6532, 6555, 6558, 6571, 6574, 6576, 6597, 6603, 6625, 6641, 6651,
#        6740, 6783, 6847, 7121, 7208, 7216, 7233, 7338, 7430, 7434, 7545, 7978,
#        8094, 8097, 8160, 8246, 8287, 8316, 8325, 8376, 8408, 8486, 8523, 8607,
#        9009, 9015, 9019, 9024, 9280, 9530, 9634, 9664, 9679, 9698, 9768, 9792,
#        9839, 9904, 9922]
