程序分为train.py 、predict.py 两个主要文件。train.py 用于训练神经网络，网络模型仅为 vgg13，vgg16，vgg19 三种，除该三种以外的模型可能会报错。

网络训练使用方法：
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

花卉图片识别使用方法：
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
# example：python predict.py data/test/1/image_06736.jpg checkpoint/checkpoint.pth

注意事项：
（1）网络中重新构建了classifier ，其中输出为102个节点，如果需要修改该输出，请修改train.py 文件中的 ：input_size 配置。

 (2) 设备训练。在没有使用 --gpu 命令配置时。默认情况下将检测GPU是否可用，如果GPU可用，则使用GPU,否则为cpu。

（3）checkpoint 保存目录。如果没有使用 --save_dir 指明保存路径，checkpoint 将保存在当前目录下。linux 下设置目录保存，请保证有权限执行。

 (4) train.py 文件中print_every 配置项。该配置项为训练输出数据的频率。默认为：30

（5） 使用python train.py <data_directory> 训练神经网络
	ps: <data_director> 该目录为图片目录的绝对路径。图片目录下需要包括“train”，“valid”，“test” 三个目录。及分别是训练、验证、测试数据目录。
如：
└── data
    ├── test
    │   ├── 1
    │   │   ├── image_06734.jpg
    │   │   ├── image_06735.jpg
    │   │   └── image_06736.jpg
    │   └── 10
    │       ├── image_07086.jpg
    │       ├── image_07087.jpg
    │       └── image_07088.jpg
    ├── train
    │   ├── 1
    │   │   ├── image_06734.jpg
    │   │   ├── image_06735.jpg
    │   │   └── image_06736.jpg
    │   └── 10
    │       ├── image_07086.jpg
    │       ├── image_07087.jpg
    │       └── image_07088.jpg
    └── valid
        ├── 1
        │   ├── image_06734.jpg
        │   ├── image_06735.jpg
        │   └── image_06736.jpg
        └── 10
            ├── image_07086.jpg
            ├── image_07087.jpg
            └── image_07088.jpg
