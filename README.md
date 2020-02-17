# ABD-Net Implement on Deep-Person-Reid
### 简介
在Deep-Person-Reid最新版本(2020.2.17 latest 1.0.9)上实现带有ABD分支的多分支ResNet50网络(ABD-Net)。
    
由于网络为多分支网络，因此与原Deep-Person-Reid的优化器、损失函数等不兼容，因此单独编写训练程序train.py，并修改了torchreid中的一些内容，使其能够兼容ABD-Net多输出网络的训练过程。

###训练

在args.py中指定训练参数。

首先使用Deep-Person-Reid中的数据集注册方法注册自己的数据集，在__init__函数中返回指定的query, gallery和train。注册数据集的具体方法[点我](https://github.com/KaiyangZhou/deep-person-reid/blob/master/docs/user_guide.rst)。

注意在训练时，需要设置args.py中的--evaluate为False，否则为只测试模式。

    --root 数据集根目录
    -s ['用于训练的数据集名称'， '如需交叉验证，添加其他数据集名称']
    -t ['用于测试的数据集名称'， '如需交叉验证，添加其他数据集名称']

### 测试

在args.py中，指定--evaluate参数为True，开启只测试模式。

### 参考与引用

[Deep-Person-Reid](https://github.com/KaiyangZhou/deep-person-reid)

[ABD-Net](https://github.com/TAMU-VITA/ABD-Net)

