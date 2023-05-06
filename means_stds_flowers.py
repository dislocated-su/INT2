# Find mean and std of each image as discibed https://www.geeksforgeeks.org/how-to-normalize-images-in-pytorch/
import torchvision.transforms as transforms # transform data
import torch # PyTorch package
import torchvision # load datasets
import numpy as np # for transformation
import torch.utils.data as U

transformToTensor = transforms.Compose([
    transforms.ToTensor()
])

trainsetNotNorm = torchvision.datasets.Flowers102("./data", split="train", download=True, transform=transformToTensor)
testsetNotNorm = torchvision.datasets.Flowers102("./data", split="test", download=True, transform=transformToTensor)
validsetNotNorm = torchvision.datasets.Flowers102("./data", split="val", download=True, transform=transformToTensor)
set = U.ConcatDataset([trainsetNotNorm, validsetNotNorm, testsetNotNorm])

means = [0, 0, 0]
stds = [0, 0, 0]
for img, _ in set:
  means = [sum(i) for i in zip(means, img.mean([1, 2]))]
  stds =  [sum(i) for i in zip(stds, img.std([1, 2]))]

means = np.array(means) / len(set)
stds = np.array(stds) / len(set)

print(means, stds)