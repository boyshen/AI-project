#!／usr/bin/env python3
# 创建日期：2018.10.30
# 发布日期: 31/10/2018
# version: 1.0
# PURPOSE：训练神经网络。
#
# 使用：
#   python train.py <Image dir> 
# 设置保存检查点目录：
#   python train.py <Image dir> --save_dir <save directory>
# 选择架构：
#   python train.py <Image dir> --arch <architecture> 
# 设置参数：
#   python train.py <Image dir> --learnning_rate <learnning rate> --hidden_units <hidden units> --epochs <epochs>
# 使用GPU训练：
#   python train.py <Image dir> --gpu
# PS: <>指示预期用户输入
#
# Example: python train.py Cat_image_dir/ --gpu

# 导入 python model
#conding=utf8
import argparse
import os
import torch

import time

from torch import nn
from torch import optim
from torchvision import transforms,datasets,models
from Network import Network

# 定义网络的输入采用vgg模型，默认是25088
input_size = 25088

# 重要配置，定义网络输出
output_size = 102

# 数据集目录名，分别为train（训练）、valid（验证）、test（测试）
dir_list = ['train','valid','test']

# 网络模型
arch_list = ['vgg13','vgg16','vgg19']

# 输出颜色
red = '\033[1;31m'
green = '\033[1;32m'
yellow = '\033[1;33m'
blue = '\033[1;34m'
color = '\033[0m'

#均值和标准差标准化到网络期望的结果
# 均值
mean_size = [0.485,0.456,0.406]
# 方差
std_size = [0.229,0.224,0.225]

# 输出训练丢失频率
print_every = 30

def Input_args():
    '''接收输入参数并进行解析。
    input： 无
    return：参数字典
    '''
    print (green + "1. Get input parameters " + color)
    parser = argparse.ArgumentParser(description='deeping train network')

    parser.add_argument('data_directory',metavar='image dir',
                        help='Directory of pictures,Absolute path !')

    parser.add_argument('--gpu',dest='run_mode',action='store_const',const='cuda',
                        help='cpu or cuda,By default,it detects whether the device supports CUDA,Otherwise run in the form of CPU')

    parser.add_argument('--save_dir',dest='save_dir',action='store',
                       help='Directory of training data.')

    parser.add_argument('--arch',dest='arch',action='store',choices=arch_list,default='vgg19',
                       help='Selection architecture: [vgg13,vgg16,vgg19],default vgg19')

    parser.add_argument('--learnning_rate',dest='lr',action='store',type=float,default=0.001,
                       help='learnning rate ,default :0.001')

    parser.add_argument('--hidden_units',dest='hidden',action='store',type=int,default=6144,
                       help='hidden units ,default :6144')

    parser.add_argument('--epochs',dest='epochs',action='store',type=int,default=10,
                       help='epochs number ,default :10')

    result = parser.parse_args()

    dict_args = {}
    dict_args['data_directory'] = result.data_directory
    dict_args['run_mode'] = result.run_mode
    dict_args['save_dir'] = result.save_dir
    dict_args['arch'] = result.arch
    dict_args['lr'] = result.lr
    dict_args['hidden'] = result.hidden
    dict_args['epochs'] = result.epochs

    return dict_args

def print_args_conf(args):
    '''打印出相关配置参数
    input ： args <配置参数字典>
    return： 无
    '''
    print (green + "2. Get Configuration Information:" + color)
    print ("=============== args config ===============")
    print ("Image directory : {}".format(args['data_directory']))
    print ("run mode        : {}".format(args['run_mode']))
    print ("save directory  : {}".format(args['save_dir']))
    print ("architecture    : {}".format(args['arch']))
    print ("learnning rate  : {}".format(args['lr']))
    print ("hidden units    : {}".format(args['hidden']))
    print ("epochs          : {}".format(args['epochs']))
    print ("===========================================")

def check_args(args):
    '''检测函数,检测相关配置参数是否正确。
        （1）图片目录是否存在.
        （2）使用GPU运行，检测GPU是否运行，是否可用。
    input：args <配置参数字典>
    return：True / False，True：配在没有问题，可以运行。否则返回 False
    '''

    # 检测图片目录是否存在
    if os.path.exists(args['data_directory']) == False:
        print (red + "Error: data directory not found :",args['data_directory'] + color)
        return False

    for dir_name in dir_list:
        i_dir = os.path.join(args['data_directory'],dir_name)
        if os.path.exists(i_dir) == False:
            print (red + "Error: data directory not found :",i_dir)
            return False

    # 检测GPU是否运行
    if args['run_mode'] == 'cuda':
        if torch.cuda.is_available() == False:
            print (red + "Error: GPU not running or disable" + color)
            return False

    print (green + "3. check input parameters is ok !" + color)

    return True

def init_dataframes(args):
    '''初始化数据目录
        根据提供的图片目录,随机旋转、随机裁剪、随机翻转进行数据变换。
    input：args <参数配置字典>
    return：测试、验证、训练的 DataLoader,datasets  对象
    '''

    train_dir = os.path.join(args['data_directory'],dir_list[0])
    valid_dir = os.path.join(args['data_directory'],dir_list[1])
    test_dir = os.path.join(args['data_directory'],dir_list[2])

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=mean_size,std=std_size)])

    validation_transforms = transforms.Compose([transforms.Resize(225),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean_size,std=std_size)])

    test_transforms = transforms.Compose([transforms.Resize(225),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=mean_size,std=std_size)])

    train_datasets = datasets.ImageFolder(train_dir,transform=train_transforms)
    validation_datasets = datasets.ImageFolder(valid_dir,transform=train_transforms)
    test_datasets = datasets.ImageFolder(test_dir,transform=train_transforms)

    train_dataloader = torch.utils.data.DataLoader(train_datasets,batch_size=64,shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_datasets,batch_size=32)
    test_dataloader = torch.utils.data.DataLoader(test_datasets,batch_size=32)

    image_datasets = {'train':train_datasets,'test':test_datasets,'validation':validation_datasets}
    dataloaders = {'train':train_dataloader,'test':test_dataloader,'validation':validation_dataloader}

    print (green + "4. Data set Initialization is ok !" + color)

    return dataloaders,image_datasets

def create_train_network(args):
    '''构建训练网络
        根据制定的参数构建网络。
    input：args <参数配置字典>
    return： 网络模型model
    '''

    model = None

    if args['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)

    elif args['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)

    elif args['arch'] == 'vgg19':
        model = models.vgg19(pretrained=True)

    for params in model.parameters():
        params.requires_grad = False

    # 定义新的前馈网络作为分类器
    classifier = Network(input_size,[args['hidden']],output_size)

    model.classifier = classifier

    print (green + "5. create train network {} is ok!".format(args['arch']) + color)

    return model

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

    print (green + "6. choise run device is {} !".format(device) + color)

    return device

def Validation(model,data_loader,criterion,device='cpu'):
    '''验证函数
    input: model <网络模型>
           data_loader <验证数据集 DataLoader 对象>
           criterion <误差>
           device <运行设备>
    return: 数据丢失和验证精度
    '''

    loss = 0
    accuracy = 0

    for data in data_loader:
        images,labels = data

        images,labels = images.to(device),labels.to(device)

        output = model.forward(images)
        loss += criterion(output,labels)

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return loss,accuracy

def train_leanning(model,criterion,optimizer,data_loader,epochs=8,print_every=30,device='cpu'):
    '''训练网络函数,输入训练丢失、验证丢失、验证精度
    input: model <网络模型>
           criterion <误差>
           data_loader <数据集>
           epochs <训练次数>
           print_every <输出周期>
           device <运行设备>
    return: 无
    '''
    print (green + "7. start train network " + color)

    train_dataloader = data_loader['train']
    validation_dataloader = data_loader['validation']

    train_loss = 0
    steps = 0
    train_loss = 0

    for e in range(epochs):
        # 训练模式
        model.train()

        train_loss = 0
        model.to(device)

        for ii,(images,labels) in enumerate(train_dataloader):
            steps += 1

            images,labels = images.to(device),labels.to(device)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if steps % print_every == 0:
                # 验证模式
                model.eval()

                with torch.no_grad():
                    val_loss,accuracy = Validation(model,validation_dataloader,criterion,device)

                print(yellow + "device : {},".format(device),
                      "epochs : {}/{},".format(e+1,epochs),
                      "validation loss : {:.3f},".format(val_loss/len(validation_dataloader)),
                      "validation accuracy : {:.3f},".format(accuracy/len(validation_dataloader)),
                      "train loss : {:.4f}".format(train_loss/print_every) + color)

                train_loss = 0

                model.train()

def check_correct_on_test(model,dataloader,criterion,device='cpu'):
    '''对网络模型进行测试
    input: model <网络模型>
           dataloader <数据集>
           criterion <误差>
           device <运行设备>
    return: 无
    '''
    print (green + "9. start test network model " + color)

    test_dataloader = dataloader['test']

    correct = 0
    total = 0
    test_loss = 0

    model.to(device)
    with torch.no_grad():
        for i,(images,labels) in enumerate(test_dataloader):

            images,labels = images.to(device),labels.to(device)

            outputs = model(images)
            test_loss += criterion(outputs,labels).item()

            value,predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (labels == predicted).sum().item()

            print (yellow + "device : {},".format(device),
                   "test loss : {:.4f},".format(test_loss),
                   "correct : {:.1f}%".format(correct/total * 100) + color)

def save_checkpoint(model,args,datasets,optimizer):
    ''' 保存检查点
    input: model <网络模型>
           args <配置输入参数>
           datasets <数据集配置>
    return: True True:保存成功，否则产生异常
    '''
    print (green + "10. start save check_point " + color)

    check_point = {'lr':args['lr'],
                   'model':args['arch'],
                   'hidden':args['hidden'],
                   'output_size':output_size,
                   'state_dict':model.state_dict(),
                   'image_class_idx':datasets['train'].class_to_idx,
                   'optimizer':optimizer.state_dict()}
    save_path = None

    # 如果未设置保存目录，则保存在当前目录下
    if args['save_dir'] == None:
        save_path = os.getcwd()

    elif os.path.exists(args['save_dir']) == False:
        os.makedirs(args['save_dir'])
        save_path = args['save_dir']

    else:
        save_path = args['save_dir']

    os.chdir(save_path)
    torch.save(check_point,'checkpoint.pth')
    file_path = os.path.join(save_path,'checkpoint.pth')
    print (blue + " save succenss ! file : {}".format(file_path) + color)

    return True

def main():
    # 获取输入参数
    args = Input_args()

    # 输出配置参数
    print_args_conf(args)

    # 检查配置参数
    if check_args(args) == False:
        print (red + "find Error ! ! !" + color)
        return False

    # 初始化数据集
    dataloader,datasets = init_dataframes(args)

    # 构建训练网络
    model = create_train_network(args)
    #print (model)

    # 选择运行设备
    device = choices_device(args)

    # 误差 
    criterion = nn.NLLLoss()

    # 优化器
    optimizer = optim.Adam(model.classifier.parameters(),args['lr'])

    # 训练网络
    train_leanning(model,criterion,optimizer,dataloader,args['epochs'],print_every,device)

    # 测试网络
    check_correct_on_test(model,dataloader,criterion,device)

    time.sleep(10)

    # 保存 checkpoint
    save_checkpoint(model,args,datasets,optimizer)

    print (green + "11. over " + color)

if __name__ == '__main__':
    main()
