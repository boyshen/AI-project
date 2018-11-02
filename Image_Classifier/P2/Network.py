#!／usr/bin/env python3
# 创建日期：2018.10.30
# 发布日期: 31/10/2018
# version: 1.0
# PURPOSE：创建任意隐藏层的前馈网络。
#
# 使用：
#   network = Network(<input_size>,<hidden>,<output_size>,<drop>)
#   <input_size> : 输入层节点数量
#   <hidden>     : 隐藏层阶段，使用list方式。如：[500,200,100]
#   <output_size>: 输出层节点数量
#   <drop>       : 随机化零个元素，输出按比例 1/（1-p）缩放
# PS: <>指示预期用户输入
#

#导入python 相关model
#conding=utf8
import torch
from torch import nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, input_size, hidden, output_size, drop_p=0.5):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden[0])])

        layer_sizes = zip(hidden[:-1], hidden[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden[-1], output_size)

        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):

        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)

        return F.log_softmax(x, dim=1)

if __name__ == '__main__':
    network = Network(1000,[500,200],100)
    print (network)
