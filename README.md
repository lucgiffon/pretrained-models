# Pretrained Models
Repository of pretrained models weights with checksum and source python file

## Classification
| Dataset        | Architecture             | Performance  | Weights                                                                                                       | Checksum                         | Source file                                                                                                       |
| -------------- |:------------------------:| :----------: | :-----------------------------------------------------------------------------------------------------------: | :------------------------------: | :----------------------------------------------------------------------------------------------------------------:|
| Cifar10        | VGG19 (4096x4096)        | 92.9%        | [Download](https://pageperso.lis-lab.fr/~luc.giffon/saved_models/cifar10_vgg19_4096x4096_1570693209.h5)       | a3ece534a8e02d17453dffc095048f65 |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/cifar10_vgg19.py)    |
| Cifar100       | VGG19 (4096x4096)        | 69.6%        | [Download](https://pageperso.lis-lab.fr/~luc.giffon/saved_models/cifar100_vgg19_4096x4096_1570789868.h5)      | cb1bd8558f385030c6c68808023918e0 |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/cifar100_vgg19.py)   |
| SVHN           | VGG19 (4096x4096)        | 96.0%        | [Download](https://pageperso.lis-lab.fr/~luc.giffon/saved_models/svhn_vgg19_4096x4096_1570786657.h5)          | 204e41afbc84d1806822a60a9558ea52 |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/svhn_vgg19.py)       |
| Cifar10        | VGG19 (2048x2048)        | 92.7%        | [Download](https://pageperso.lis-lab.fr/~luc.giffon/saved_models/cifar10_vgg19_2048x2048_1572303047.h5)       | 98cece5432051adc2330699a40940dfd |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/cifar10_vgg19.py)    |
| Cifar100       | VGG19 (2048x2048)        | 66.5%        | [Download](https://pageperso.lis-lab.fr/~luc.giffon/saved_models/cifar100_vgg19_2048x2048_1572278802.h5)      | 57d6bf6434428a81e702271367eac4d1 |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/cifar100_vgg19.py)   |
| SVHN           | VGG19 (2048x2048)        | 96.1%        | [Download](https://pageperso.lis-lab.fr/~luc.giffon/saved_models/svhn_vgg19_2048x2048_1572278831.h5)          | d5697042804bcc646bf9882a45dedd9e |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/svhn_vgg19.py)       |
| MNIST          | Lenet                    | 99.3%        | [Download](https://pageperso.lis-lab.fr/~luc.giffon/saved_models/mnist_lenet_1570207294.h5)                   | 26d44827c84d44a9fc8f4e021b7fe4d2 |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/mnist_lenet.py)      |
| MNIST          | Fully connected 500      | 92.5%        | [Download](https://pageperso.lis-lab.fr/~luc.giffon/saved_models/mnist_500.h5)                                | 1b023b05a01f24a99ac9a460488068f8 |    No source                                                                                                      |
| Cifar100       | Resnet 20                | 66.2%        | [Download](https://pageperso.lis-lab.fr/~luc.giffon/saved_models/resnet_20_cifar100.h5)                       | 4845ec6461c5923fc77f42a157b6d0c1 |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/cifar_resnet.py)     |
| Cifar100       | Resnet 50                | 70.4%        | [Download](https://pageperso.lis-lab.fr/~luc.giffon/saved_models/resnet_50_cifar100.h5)                       | d76774eb6f871b1192c144f0dc29340e |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/cifar_resnet.py)     |
| Cifar10        | Tensor Train base        | 89.3%        | [Download](https://pageperso.lis-lab.fr/~luc.giffon/saved_models/cifar10_tensor_train_base_1585409008.h5)     | e985fbe4ade6893b7fb92655be1b846f |    [Source](https://github.com/lucgiffon/pretrained-models/blob/master/models/classification/cifar10_tensor_train_base.py)     |


__Note__: for architecture VGG19, (NumberxNumber) refers to the size of the 2 fully-connected layers on top of the convolution layers. In the source file the value `SIZE_DENSE` configure this number of hidden units. For example: VGG19 (4096x4096) refers to a VGG19 architecture with two fully connected layer of size 4096 hidden units between the output of convolutional layers and input of classification layer.
