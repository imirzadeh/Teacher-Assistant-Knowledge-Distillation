from resnet_cifar import *
from plain_cnn_cifar import *


def is_resnet(name):
	"""
	Simply checks if name represents a resnet, by convention, all resnet names start with 'resnet'
	:param name:
	:return:
	"""
	name = name.lower()
	return name.startswith('resnet')


def create_cnn_model(name, dataset="cifar100", use_cuda=False):
	"""
	Create a student for training, given student name and dataset
	:param name: name of the student. e.g., resnet110, resnet32, plane2, plane10, ...
	:param dataset: the dataset which is used to determine last layer's output size. Options are cifar10 and cifar100.
	:return: a pytorch student for neural network
	"""
	num_classes = 100 if dataset == 'cifar100' else 10
	model = None
	if is_resnet(name):
		resnet_size = name[6:]
		resnet_model = resnet_book.get(resnet_size)(num_classes=num_classes)
		model = resnet_model
		
	else:
		plane_size = name[5:]
		model_spec = plane_cifar10_book.get(plane_size) if num_classes == 10 else plane_cifar100_book.get(plane_size)
		plane_model = ConvNetMaker(model_spec)
		model = plane_model

	# copy to cuda if activated
	if use_cuda:
		model = model.cuda()
		
	return model

# if __name__ == "__main__":
# 	dataset = 'cifar100'
# 	print('planes')
# 	for p in [2, 4, 6, 8, 10]:
# 		plane_name = "plane" + str(p)
# 		print(create_cnn_model(plane_name, dataset))
#
# 	print('-'*20)
# 	print("resnets")
# 	for r in [8, 14, 20, 26, 32, 44, 56, 110]:
# 		resnet_name = "resnet" + str(r)
# 		print(create_cnn_model(resnet_name, dataset))
