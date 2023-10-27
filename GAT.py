# -*- coding:utf-8 -*-

import torch
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv
from sklearn.neighbors import NearestNeighbors
import pickle
import numpy as np
from torch_geometric.data import Data


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GATConv(num_node_features, 32)
        self.conv2 = GATConv(32, num_classes)
        self.norm = torch.nn.BatchNorm1d(32)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.norm(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x


def load_data(name):
    if name == 'NELL':
        print('./' + name + '/')
        dataset = NELL(root='./' + name + '/')
        # CUDA out of memory
        _device = torch.device('cpu')
    else:
        dataset = Planetoid(root='./' + name + '/', name=name)
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(_device)
    if name == 'NELL':
        data.x = data.x.to_dense()
        num_node_features = data.x.shape[1]
    else:
        num_node_features = dataset.num_node_features
    return data, num_node_features, dataset.num_classes

def data_load():
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = r'E:\projectstudy\pythonProject\data\patent\patent-text-image(cat=64+64).pkl'
    data_list = []
    with open(data_dir, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    # data = np.load(data_dir, allow_pickle=True)
    lbls = data['y']#.astype(np.long)
    fts = data['x']#[0].item()#.astype(np.float32)
    if lbls.min() == 1:
        lbls = lbls - 1
    idx_train = data['train']
    idx_test = data['test']

    # 转换为mask形式
    all_labels = np.unique(np.concatenate([idx_train, idx_test]))
    train_mask = np.zeros_like(all_labels, dtype=bool)
    train_mask[np.where(all_labels == idx_train[:, None])[1]] = True

    test_mask = np.zeros_like(all_labels, dtype=bool)
    test_mask[np.where(all_labels == idx_test[:, None])[1]] = True



# 到这一步就是把数据提取出来data然后送入下方的KNNGraph
#     data = np.array(data)###标记
    # node_edge, w = KNN_attr(data[i])  # node_edge, w=edge_index, edge_fea
     # 节点特征变为（10，512）相当于X即feature
    knn = NearestNeighbors(n_neighbors=4)
    knn.fit(fts)
    distances, indices = knn.kneighbors(fts, return_distance=True)
    edge_raw0 = []  # edge_raw0会生成50个数组[0,0...,1,1......9,9...]
    edge_raw1 = []
    for i in range(len(fts)):
        node_index = list(indices[i])
        local_index = np.zeros(4) + i

        edge_raw0 = np.hstack((edge_raw0, local_index))  # 按照水平方向堆叠数组
        edge_raw1 = np.hstack((edge_raw1, node_index))

    edge_index = [edge_raw0, edge_raw1]

    node_features = torch.tensor(fts, dtype=torch.float)  # 节点特征变为（10，512）相当于X即feature
    graph_label = torch.tensor(lbls, dtype=torch.long)  # 获得图标签 数量为10
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    train_mask = torch.tensor(train_mask)
    test_mask = torch.tensor(test_mask)

    graph = Data(x=node_features, y=graph_label, edge_index=edge_index,train_mask=train_mask, test_mask=test_mask).to(_device)


    return graph, fts.shape[1], 10
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def train(model, data, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    model.train()
    for epoch in range(200):
        out = model(data)
        # _,pred = out.max(dim=1)
        optimizer.zero_grad()
        loss = loss_function(out[data.train_mask], data.y[data.train_mask])
        acc_train = accuracy(out[data.train_mask], data.y[data.train_mask])

        # correct = int(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
        # train_acc = correct / int(data.train_mask.sum())
        # # correct = torch.eq(pred[data.train_mask], data.y[data.train_mask]).float().sum().item()
        # print('Train_Acc: {:.4f}'.format(train_acc))
        loss.backward()
        optimizer.step()

        print('Epoch {:03d} loss {:.4f}'.format(epoch, loss.item()),
              'acc_val: {:.4f}'.format(acc_train.item()))


def test(model, data,device):
    model.eval()
    # loss_function_test = torch.nn.CrossEntropyLoss().to(device)
    logits = model(data)
    # # loss = loss_function_test(logits[data.test_mask], data.y[data.test_mask])
    # loss = 2.0
    pred = logits.argmax(dim=1)
    correct = torch.eq(pred[data.test_mask], data.y[data.test_mask]).float().sum().item() # 这里好像有问题
    epoch_acc = correct/int(data.test_mask.sum())
    print(
          # "loss= {:.4f}".format(loss.item()),
          "accuracy= {}".format(epoch_acc))

    # _, pred = model(data).max(dim=1)
    # correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    # acc = correct / int(data.test_mask.sum())
    # print('GCN Accuracy: {:.4f}'.format(acc))

    loss_function_test = torch.nn.CrossEntropyLoss().to(device)
    output = model(data)
    # loss_test = loss_function_test(output[data.test_mask], data.y[data.test_mask])
    acc_test = accuracy(output[data.test_mask], data.y[data.test_mask])
    print("Test set results:",
          # "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

def main():
    # names = ['CiteSeer', 'Cora', 'PubMed', 'NELL']
    names = ['Cora']
    for name in names:
        # if name in names:
        #     print(name + '...')
        # else:
        print("Patent" + '...')
        # data, num_node_features, num_classes = load_data(name)
        data, num_node_features, num_classes = data_load()

        print(data, num_node_features, num_classes)
        _device = 'cpu' if name == 'NELL' else 'cuda'
        device = torch.device(_device)
        model = GCN(num_node_features, num_classes).to(device)
        train(model, data, device)
        test(model, data,device)


if __name__ == '__main__':
    main()
