# Teacher-Assistant-Knowledge-Distilliation

Another version of code documentation is inside readme.pdf file.    
the paper: https://arxiv.org/pdf/1902.03393.pdf

----------------
----------------

# 0. Table of Contents
#### This manual has has two parts
### 1- Installation
Provides installation guide.
### 2- Running the code
Details on how to run the code.
### 3- Examples
Some examples to reproduce results in the paper.

----------------
# 1. Installation
##### 1.1 OS: Mac OS X or Linux(tested on Ubuntu and Debian)
##### 1.2 Pytorch(1.0.0) which can be installed with details provided here: https://pytorch.org/get-started/locally/
For most users, ```pip3 install torch torchvision``` should work.
If you are using Anaconda, ```conda install pytorch torchvision -c pytorch``` should work. 
##### 1.3 Microsoft NNI toolkit: Details of installation provided here: https://github.com/Microsoft/nni/
For most users, ```pip3 install nni``` should work.


# 2. Running The code
### 2.1 Important Files In codebase: 
#### 2.1.1 `model_factory.py` Creates neural networks(resnet and plane CNN models)  to be used by trainer.
#### 2.1.2 `data_loader.py` Loads datasets(CIFAR10/CIFAR100) to be used by trainer.
#### 2.1.3 `train.py` The main code. Given a training config which we will explain below, it will train models.
#### 2.1.4 `search_space.json` defines hyper parameter search space for the hyper parameter optimizer.
#### 2.1.5 `config.yml` config for the hyper parameter optimizer.

### 2.2 Configs
#### 2.2.1 Training Settings
These settings are needed duri
###### 2.2.1.1 epochs
Number of training epochs.
###### 2.2.1.2 dataset
Dataset name, can be 'cifar10' or 'cifar100'. Default is 'cifar100'.
###### 2.2.1.3 batch-size
mini batch size used in training. Default is 128.
###### 2.2.1.4 learning-rate
Initial learning rate for SGD optimizer. Depending on models in might be changed during training.
###### 2.2.1.5 momentum
momentum for SGD optimizer. Default is 0.9.
###### 2.2.1.6 weight-decay
Weight decay for SGD optimizer. Default is 0.0001.
###### 2.2.1.7 teacher
Teacher model name.
Can be `resnetX` for resnetwhere X can be any value in (8, 14, 20, 26, 32, 44, 56, 110)
or `PlaneY` for plane(vanilla CNN) networks where Y can be any value in (2, 4, 6, 8, 10). For details of network please refer to the paper.
###### 2.2.1.8 student
Student model name.
Values can be in forms of `resnetX` or `planeY` which explained before.
###### 2.2.1.9 teacher-checkpoint
Path for a file which has pretrained teacher. Defauls is empty which means we need to train the teacher also.
###### 2.2.1.10 cuda
wheather or not train on GPU. (must have GPU supportive pytorch installed). Values 1 and true can be used for training on GPU.
###### 2.2.1.11 dataset-dir
location of the dataset. default is './data/'

#### 2.2.2 Hyper Parameters 
NNI toolkit needs a search_space file (like `search_space.json`) consists `T` and `lambda` in equation 3 of the paper. Also, in order to get reliable results, there will be multiple `seeds` to avoid bad runs. For more details on search space file, please refer to the example sections or https://microsoft.github.io/nni/docs/SearchSpaceSpec.html. 

## 2.3 Running
You have to run the code using `nnictl create --config config.yml`. Then the hyper parameter optimizer will run experiments with different hyper parameters and the results will be available thorough a dashboard.

------

# 3. Examples
In all examples, you need to change the `command` line of `config.yml` and tell the nnictl runner to how to run an experiment.


### 3.1 'resnet 110' as teacher, 'resnet8' as student  on CIFAR100 (Baseline Knowledge Distillation)
You should change the command part of the config file like this:   
`command: python3 train.py --epochs 160 --teacher resnet110 --student resnet8 --cuda 1 --dataset cifar10`

### 3.2 'resnet 110' as teacher, 'resnet20' as TA, 'resnet8' as student  on CIFAR100(using GPU)
1. Train Teacher(Resnet110): This phase is not knowledge distillation. So there's no teacher and only a student trained alone.   
`command: python3 train.py --epochs 160 --student resnet110 --cuda 1 --dataset cifar100`

2. After first step, choose the weights which had best accuracy on valdiation data and train TA(Resnet20) with teacher (Resnet110) weights. Say the best resnet110 weights file was resnet110_XXXX_best.pth.tar   
`command: python3 train.py --epochs 160 --teacher resnet110 --teacher-checkpoint ./resnet110_XXXX_best.pth.tar --student resnet20 --cuda 1 --dataset cifar100`

3. Repeat like step two, distillate knowledge from TA to student (Teacher is resnet20, student is resnet8). Also, we assume the best weights from step two was resnet20_XXXX_best.pth.tar       
`command: python3 train.py --epochs 160 --teacher resnet20 --teacher-checkpoint ./resnet20_XXXX_best.pth.tar  --student resnet8 --cuda 1 --dataset cifar100`



### 3.3 'resnet 110' as teacher, 'resnet14' as TA, 'resnet8' as student  on CIFAR10
1. Train Teacher(Resnet110): This phase is not knowledge distillation. So there's no teacher and only a student trained alone.   
`command: python3 train.py --epochs 160 --student resnet110 --cuda 1 --dataset cifar10`

2. After first step, choose the weights which had best accuracy on valdiation data and train TA(Resnet14) with teacher (Resnet110) weights. Say the best resnet110 weights file was resnet110_XXXX_best.pth.tar
`command: python3 train.py --epochs 160 --teacher resnet110 --teacher-checkpoint ./resnet110_XXXX_best.pth.tar --student resnet14 --cuda 1 --dataset cifar10`

3. Repeat like step two, distillate knowledge from TA to student (Teacher is resnet14, student is resnet8). Also, we assume the best weights from step two was resnet14_XXXX_best.pth.tar     
`command: python3 train.py --epochs 160 --teacher resnet14 --teacher-checkpoint ./resnet14_XXXX_best.pth.tar --student resnet8 --cuda 1 --dataset cifar10`
