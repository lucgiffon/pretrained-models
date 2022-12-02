# Pretrained Models
Repository of pretrained models weights with checksum and source python file

## Classification
| Task               | Dataset        | Architecture               | Layers                                     | Other informations   | Performance  | Source file                                                                                                                    |
| ------------------ | -------------- |:--------------------------:|:------------------------------------------:|:--------------------:| :----------: | :-----------------------------------------------------------------------------------------------------------------------------:|
| Classification     | Cifar10        | VGG19                      | Dense; Conv2D; Batchnorm; MaxPooling;      |  (4096x4096)         | 92.9%        |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/cifar10_vgg19.py)                 |
| Classification     | Cifar100       | VGG19                      | Dense; Conv2D; Batchnorm; MaxPooling;      |  (4096x4096)         | 69.6%        |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/cifar100_vgg19.py)                |
| Classification     | SVHN           | VGG19                      | Dense; Conv2D; Batchnorm; MaxPooling;      |  (4096x4096)         | 96.0%        |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/svhn_vgg19.py)                    |
| Classification     | Cifar10        | VGG19                      | Dense; Conv2D; Batchnorm; MaxPooling;      |  (2048x2048)         | 92.7%        |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/cifar10_vgg19.py)                 |
| Classification     | Cifar100       | VGG19                      | Dense; Conv2D; Batchnorm; MaxPooling;      |  (2048x2048)         | 66.5%        |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/cifar100_vgg19.py)                |
| Classification     | SVHN           | VGG19                      | Dense; Conv2D; Batchnorm; MaxPooling;      |  (2048x2048)         | 96.1%        |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/svhn_vgg19.py)                    |
| Classification     | MNIST          | Lenet                      | Dense; Conv2D;                             |                      | 99.3%        |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/mnist_lenet.py)                   |
| Classification     | Cifar10        | VGG19                      | Dense; Conv2D; Batchnorm; MaxPooling;      |  (2048x2048) seed:1  | 92.7%        |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/cifar10_vgg19.py)                 |
| Classification     | Cifar100       | VGG19                      | Dense; Conv2D; Batchnorm; MaxPooling;      |  (2048x2048) seed:1  | 66.4%        |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/cifar100_vgg19.py)                |
| Classification     | SVHN           | VGG19                      | Dense; Conv2D; Batchnorm; MaxPooling;      |  (2048x2048) seed:1  | 96.1%        |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/svhn_vgg19.py)                    |
| Classification     | MNIST          | Lenet                      | Dense; Conv2D;                             |              seed:1  | 99.3%        |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/mnist_lenet.py)                   |
| Classification     | Cifar10        | VGG19                      | Dense; Conv2D; Batchnorm; MaxPooling;      |  (2048x2048) seed:2  | 93.0%        |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/cifar10_vgg19.py)                 |
| Classification     | Cifar100       | VGG19                      | Dense; Conv2D; Batchnorm; MaxPooling;      |  (2048x2048) seed:2  | 68.8%        |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/cifar100_vgg19.py)                |
| Classification     | SVHN           | VGG19                      | Dense; Conv2D; Batchnorm; MaxPooling;      |  (2048x2048) seed:2  | 95.9%        |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/svhn_vgg19.py)                    |
| Classification     | MNIST          | Lenet                      | Dense; Conv2D;                             |              seed:2  | 99.3%        |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/mnist_lenet.py)                   |
| Classification     | Cifar10        | VGG19                      | Dense; Conv2D; Batchnorm; MaxPooling;      |  (2048x2048) seed:3  | 92.8%        |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/cifar10_vgg19.py)                 |
| Classification     | Cifar100       | VGG19                      | Dense; Conv2D; Batchnorm; MaxPooling;      |  (2048x2048) seed:3  | 65.2%        |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/cifar100_vgg19.py)                |
| Classification     | SVHN           | VGG19                      | Dense; Conv2D; Batchnorm; MaxPooling;      |  (2048x2048) seed:3  | 96.2%        |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/svhn_vgg19.py)                    |
| Classification     | MNIST          | Lenet                      | Dense; Conv2D;                             |              seed:3  | 99.3%        |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/mnist_lenet.py)                   |
| Classification     | MNIST          | Fully connected 500        | Dense; Conv2D; Batchnorm; MaxPooling;      |                      | 92.5%        |    No source                                                                                                                   |
| Classification     | Cifar100       | Resnet 20                  | Dense; Conv2D; Batchnorm;                  |    (deprecated)      | 66.2%        |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/cifar_resnet_old.py)              |
| Classification     | Cifar100       | Resnet 50                  | Dense; Conv2D; Batchnorm;                  |    (deprecated)      | 70.4%        |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/cifar_resnet_old.py)              |
| Classification     | Cifar100       | Resnet 20                  | Dense; Conv2D; Batchnorm;                  |                      | 73.2%        |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/cifar_resnet.py)                  |
| Classification     | Cifar100       | Resnet 50                  | Dense; Conv2D; Batchnorm;                  |                      | 76.0%        |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/cifar_resnet.py)                  |
| Classification     | Cifar100       | Resnet 50                  | Dense; Conv2D; Batchnorm;                  |      seed:1          | 76.0%        |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/cifar_resnet.py)                  |
| Classification     | Cifar100       | Resnet 50                  | Dense; Conv2D; Batchnorm;                  |      seed:2          | 76.0%        |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/cifar_resnet.py)                  |
| Classification     | Cifar100       | Resnet 50                  | Dense; Conv2D; Batchnorm;                  |      seed:3          | 76.0%        |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/cifar_resnet.py)                  |
| Classification     | Cifar100       | Resnet 20                  | Dense; Conv2D; Batchnorm;                  |      seed:1          | 73.9%        |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/cifar_resnet.py)                  |
| Classification     | Cifar100       | Resnet 20                  | Dense; Conv2D; Batchnorm;                  |      seed:2          | 72.9%        |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/cifar_resnet.py)                  |
| Classification     | Cifar100       | Resnet 20                  | Dense; Conv2D; Batchnorm;                  |      seed:3          | 73.4%        |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/cifar_resnet.py)                  |
| Classification     | Cifar10        | Tensor Train base          | Dense; Conv2D; Batchnorm; MaxPooling;      |                      | 89.3%        |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/cifar10_tensor_train_base.py)     |


__Note__: for architecture VGG19, (NumberxNumber) refers to the size of the 2 fully-connected layers on top of the convolution layers. In the source file the value `SIZE_DENSE` configure this number of hidden units. For example: VGG19 (4096x4096) refers to a VGG19 architecture with two fully connected layer of size 4096 hidden units between the output of convolutional layers and input of classification layer.

__Note__: for resnet architecture (deprecated), 20 layers have been obtained with a  `n_stack=3` and 50 layers have been obtained with `n_stack=8`. 
