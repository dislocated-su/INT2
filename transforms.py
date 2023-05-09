import torch # PyTorch package
import torch.utils.data as U
import torchvision.transforms as transforms
import numpy as np
import torchvision
import PIL

def crop_to_longest_side(image: PIL.Image.Image) -> PIL.Image.Image:
    """Crop the images so only a specific region of interest is shown to my PyTorch model"""
    w, h = image.size
    size = min(w, h)
    return transforms.functional.crop(image, left=(w//2) - (size//2), top=(h//2) - (size//2), width=size, height=size)
    #return transforms.CenterCrop(size=size)

#https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

means, stds = (0.436, 0.378, 0.288), (0.265, 0.212, 0.219) # means, stds over flowers-102 
unnormalize = transforms.Normalize((-np.array(means)/ np.array(stds)), (1.0 / np.array(stds)))

def rgb2hsvTransform(image: PIL.Image.Image) -> PIL.Image.Image:
    return image.convert('HSV')

def get_datasets(batch_size):
    # base transform
    transform_base = transforms.Compose([
        #transforms.Lambda(crop_to_longest_side),
        #transforms.Lambda(rgb2hsvTransform),
        transforms.Resize((256,256)), 
        transforms.ToTensor(), # to tensor object
        transforms.Normalize(means, stds)
        ]) # mean, std over rbg values

    # flip (p=1)
    transform_flip = transforms.Compose([
    transform_base,
    transforms.RandomHorizontalFlip(1.0)]) # definite horizontal flip

    # rotation
    transform_rotate = transforms.Compose([
        transform_base,
        transforms.RandomRotation(degrees = (-45, 45))]) # random rotation between -45 degrees and 45 degrees

    # multiple augmentations
    transform_flip_rotate = transforms.Compose([
        transforms.RandomRotation(degrees = (-45, 45)),
        transform_base,
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        #AddGaussianNoise(0, 0.4),
        transforms.GaussianBlur(kernel_size=9), 
        ])

    transform_perspective = transforms.Compose([
        transform_flip_rotate,
        transforms.RandomPerspective(p=1.0)
    ])

    transform_elastic = transforms.Compose([
        transform_flip_rotate,
        transforms.ElasticTransform(alpha=30.0,sigma=2.0)
    ])

    transform_test = transforms.Compose([
        transform_base
    ])



    # load train data
    print("Loading Datasets")
    set_no_aug = torchvision.datasets.Flowers102(root='./data', split="train",
                                            download=True, transform=transform_base)

    set_flip = torchvision.datasets.Flowers102(root='./data', split="train",
                                                    download = True, transform = transform_flip)

    set_rotate = torchvision.datasets.Flowers102(root='./data', split="train",
                                                    download = True, transform = transform_rotate)

    set_flip_rotate = torchvision.datasets.Flowers102(root='./data', split="train",
                                                    download = True, transform = transform_flip_rotate)

    set_perspective = torchvision.datasets.Flowers102(root='./data', split="train",
                                                    download = True, transform = transform_perspective)
    
    set_elastic = torchvision.datasets.Flowers102(root='./data', split="train",
                                                    download = True, transform = transform_elastic)

    #set_perspective, _ = U.random_split(set_perspective, [0.5, 0.5])
    set_elastic, _ = U.random_split(set_elastic, [0.50,0.50])
    set_no_aug, _ = U.random_split(set_no_aug, [0.2,0.8])

    trainset = U.ConcatDataset([set_flip_rotate, set_elastic, set_perspective])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # load test data
    testset = torchvision.datasets.Flowers102(root='./data', split="test",
                                            download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False)

    #load valid data
    validset = torchvision.datasets.Flowers102(root='./data', split="val",
                                            download=True, transform=transform_test)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size,
                                            shuffle=False)
    
    return trainloader, validloader, testloader