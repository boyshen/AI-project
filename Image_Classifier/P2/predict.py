#!／usr/bin/env python3
# 创建日期：2018.10.30
# 发布日期: 2/11/2018
# version: 1.0
# PURPOSE：预测花的名称以及概率。
#
# 使用：
#   python train.py <input> <checkpoint>
# 返回K个类别概率：
#   python train.py <input> <checkpoint> --top_k 3
# 使用类别到真实名词的映射：
#   python train.py <input> <checkpoint> --category_names <filename.json>
# 使用GPU训练：
#   python train.py <input> <checkpoint> --gpu
# 
# <input>          --> 花的图片
# <checkpoint>     --> checkpoint.pth 文件的路径
# <filename.json>  --> 花的名称文件集，json 文件类型
#
# PS: <>指示预期用户输入
#
# Example: python train.py input checkpoint.pth --gpu

# 导入 python model
#conding=utf8
import argparse
import os
import torch

import json

import numpy as np

from torch import nn
from torch import optim
from PIL import Image
from Network import Network
from torchvision import models

# 定义网络的输入采用vgg模型，默认是25088
input_size = 25088

# 重要配置，定义网络输出
output_size = 102

# 输出颜色
red = '\033[1;31m'
green = '\033[1;32m'
yellow = '\033[1;33m'
blue = '\033[1;34m'
color = '\033[0m'

#均值和标准差标准化到网络期望的结果
# 均值
mean = [0.485,0.456,0.406]
# 方差
std = [0.229,0.224,0.225]

def Input_args():
    '''接收输入参数并进行解析
    input：无
    return：参数字典
    '''
    print (green + "1. Get input Parameters " + color)
    parser = argparse.ArgumentParser(description="Identifying flower names")

    parser.add_argument('input',metavar='picture file name and checkpoint',nargs=2,
                       help='picture file name of flower and checkpoint')

    parser.add_argument('--gpu',dest='run_mode',action='store_const',const='cuda',
                        help='cpu or cuda,By default,it detects whether the device supports CUDA,Otherwise run in the form of     CPU')

    parser.add_argument('--top_k',dest='top_k',action='store',type=int,default=3,
                       help='K class probability before returning. Default to return to the first 3')

    parser.add_argument('--category_names',dest='category_names',action='store',
                       help='Flower category JSON file name')

    result = parser.parse_args()

    dict_args = {}
    dict_args['picture_name'] = result.input[0]
    dict_args['checkpoint'] = result.input[1]
    dict_args['run_mode'] = result.run_mode
    dict_args['top_k'] = result.top_k
    dict_args['category_names'] = result.category_names

    return dict_args

def print_args(args):
    '''打印出配置参数
    input：args <参数字典>
    return: 无
    '''
    print (green + "2. Get Configuration Information:" + color)
    print ("=============== args config ===============")
    print ("Picture name    : {}".format(args['picture_name']))
    print ("run mode        : {}".format(args['run_mode']))
    print ("checkpoint file : {}".format(args['checkpoint']))
    print ("topk            : {}".format(args['top_k']))
    print ("Category names  : {}".format(args['category_names']))
    print ("===========================================")

def check_input(args):
    '''检查输入的参数。
        （1）判断图片文件、checkpoint文件是否存在，
        （2）文件可以读写
    input:args <输入参数字典>
    return：True / False 。验证通过返回True，否则返回False
    '''
    image = args['picture_name']
    checkpoint = args['checkpoint']
    category = args['category_names']

    if os.path.exists(image) == False:
        print (red + "picture not found ,{}".format(image) + color)
        return False
    elif os.access(image,os.R_OK) == False:
        print (red + "File not Read ,{}".format(image) + color)
        return False

    if os.path.exists(checkpoint) == False:
        print (red + "file not found ,{}".format(checkpoint) + color)
        return False
    elif os.access(checkpoint,os.R_OK) == False:
        print (red + "File not Read ,{}".format(checkpoint) + color)
        return False

    if category != None:
        if os.path.exists(category) == False:
            print (red + "File not found ,{}".format(category) + color)
            return False
        elif os.access(category,os.R_OK) == False:
            print (red + "File not Read, {}".format(category) + color)
            return False

    print (green + "3. check input parameters is ok !" + color)

    return True

def load_checkpoint(args):
    '''加载checkpoint文件。创建model、optimizer、class_to_idx
    input:args <参数字典>
    return: model、optimizer、class_to_idx
    '''
    print (green + "4. load checkpoint file " + color)

    checkpoint = torch.load(args['checkpoint'])
    hidden = [checkpoint['hidden']]
    output_size = checkpoint['output_size']
    lr = checkpoint['lr']

    # 初始化网络模型
    model = None
    arch = checkpoint['model']
    if arch == 'vgg13':
        model = models.vgg13()

    elif arch == 'vgg16':
        model = models.vgg16()

    elif arch == 'vgg19':
        model = models.vgg19()

    for params in model.parameters():
        params.requires_grad = False

    #加载 classifier，替换原来的 classifier
    classifier = Network(input_size,hidden,output_size)
    model.classifier = classifier

    #加载 model 的state_dict
    model.load_state_dict(checkpoint['state_dict'])
    print (blue + "---> load model is ok !" + color)

    #加载 optimizer 的 state_dict
    optimizer = optim.Adam(model.classifier.parameters(),lr=lr)
    optimizer.load_state_dict(checkpoint['optimizer'])

    print (blue + "---> load optimizer is ok !" + color)

    #加载 train 的 class_to_idx
    image_class_idx = checkpoint['image_class_idx']

    print (blue + "---> load image class idx is ok !" + color)

    return model,optimizer,image_class_idx

def choices_device(args):
    '''选择运行设备
    input：args <字典配置参数>
    return: str 运行设备
    '''
    device = None

    if args['run_mode'] == None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cuda'

    print (green + "4. choise run device is {} !".format(device) + color)

    return device

def Init_Picture(image):
    '''图片初始化，对图片进行标准化处理
    input：image <图片>
    return：image_ndarray <图片数据数组，ndarray对象>
    '''

    # 从图像的中心裁剪出 224x224
    img = Image.open(image)
    img_256 = img.resize(size=(256,256))
    img_224 = img_256.crop(box=(0,0,224,224))

    #转换颜色通道为浮点
    np_image = np.array(img_224)
    np_image = np_image.astype(float)

    #图像按照特定的方式标准化
    np_image = (np_image - mean)/std

    #对维度重新排
    np_image_T = np_image.transpose((2,0,1))

    return np_image_T

def predict(args,model,class_to_idx,device):
    '''类别预测
    input：args <参数字典>
           model
           class_to_idx
           device <运行设备>
    retrun: probs <预测概率>
            class <类别ID>
    '''
    print (green + "5. predict class " + color)
    image_path = args['picture_name']
    np_image = Init_Picture(image_path)

    image = torch.from_numpy(np_image).unsqueeze(0)
    image = image.type(torch.FloatTensor)

    if device == 'cuda':
        model.cuda()
        image = image.cuda()
    elif device == 'cpu':
        model.cpu()
        image = image.cpu()

    output = model.forward(image)

    k,idx = output.topk(args['top_k'])
    probs = list()
    for j in range(k.size()[1]):
        probs.append(k[0][j].item())

    # 颠倒字典。类别id为key，索引为value
    id_class = {}
    for key,value in class_to_idx.items():
        id_class[value] = key

    classes = list()
    for j in range(idx.size()[1]):
        key = idx[0][j].item()
        classes.append(id_class[key])

    print (blue + "---> probs : {} ".format(probs) + color)
    print (blue + "---> classes : {}".format(classes) + color)

    return probs,classes

def print_category_names(probs,class_idx,category_names):
    '''输出预测的图片名称
    '''
    print (green + "6. category class name" + color)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    names = [cat_to_name[idx] for idx in class_idx]
    print (blue + "---> probs   : {}".format(probs) + color)
    print (blue + "---> classes : {}".format(class_idx) + color)
    print (blue + "---> names   : {}".format(names) + color)

def main():
    # 获取输入的参数
    args = Input_args()

    # 输出配置参数
    print_args(args)

    if check_input(args) == False:
        print (red + "find Error !!!" + color)
        return False

    # 加载checkpoint
    model,optimizer,class_to_idx = load_checkpoint(args)

    # 选择运行设备
    device = choices_device(args)

    # 类别预测
    probs,class_idx = predict(args,model,class_to_idx,device)

    # 映射图片名称
    if args['category_names'] != None:
        print_category_names(probs,class_idx,args['category_names'])

    print (green + "over" + color)

if __name__ == '__main__':
    main()
