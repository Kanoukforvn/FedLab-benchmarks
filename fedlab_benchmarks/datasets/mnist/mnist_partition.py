import sys

from fedlab.utils.functional import save_dict
from fedlab.utils.dataset.slicing import noniid_slicing, random_slicing

import torchvision

trainset = torchvision.datasets.MNIST(root="./", train=True, download=False)

<<<<<<< HEAD
data_indices = noniid_slicing(trainset, num_clients=10, num_shards=2000)
save_dict(data_indices, "mnist_noniid.pkl")

data_indices = random_slicing(trainset, num_clients=10)
save_dict(data_indices, "mnist_iid.pkl")
=======
num_clients=10
num_shards=200

data_indices = noniid_slicing(trainset, num_clients=num_clients, num_shards=num_shards)
save_dict(data_indices, "mnist_noniid_{}_{}.pkl".format(num_shards, num_clients))

data_indices = random_slicing(trainset, num_clients=num_clients)
save_dict(data_indices, "mnist_iid_{}.pkl".format(num_clients))
>>>>>>> master
