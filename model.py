import matplotlib.pyplot as plt # for plotting
import numpy as np # for transformation

import torch # PyTorch package
import torchvision # load datasets
import torchvision.transforms as transforms # transform data
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer
from tqdm import tqdm
import torch.utils.data as U
import json
import PIL

if (torch.cuda.is_available()):
    torch.cuda.empty_cache()
    print("GPU WORKING!")


with open("flower_classes.json", "r") as f:
    classes = json.load(f)
classes = list(classes.values())


# Hyperparameters

# Means and stds for flowers dataset (train,test,valid)
means, stds = (0.436, 0.378, 0.288), (0.265, 0.212, 0.219)

batch_size = 32
EPOCHS = 60

# for SGD
LEARNING_RATE =0.005
MOMENTUM=0.9
WEIGHT_DECAY=0.000001
# For lr Scheduler
SCHEDULER_PATIENCE=5
SCHEDULER_FACTOR= 0.2
MIN_LR = 0.000001

def imshow(img, title=None):
  ''' function to show image '''
  img = img / 2 + 0.5 # unnormalize
  npimg = img.numpy() # convert to numpy objects
  ax = plt.subplot() 
  ax.imshow(np.transpose(npimg, (1, 2, 0)))
  if (title is not None):
    ax.set_title(title)
  plt.show()

def crop_to_longest_side(image: PIL.Image.Image) -> PIL.Image.Image:
    """Crop the images so only a specific region of interest is shown to my PyTorch model"""
    w, h = image.size
    size = min(w, h)
    return transforms.functional.crop(image, left=(w//2) - (size//2), top=(h//2) - (size//2), width=size, height=size)
    #return transforms.CenterCrop(size=size)


# base transform
transform_base = transforms.Compose([
	transforms.Lambda(crop_to_longest_side),
	transforms.Resize((256,256)), 
	transforms.ToTensor(), # to tensor object
	transforms.Normalize(means, stds)]) # mean, std over rbg values

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
    transforms.GaussianBlur(kernel_size=9), 
	transform_base,
  	transforms.RandomHorizontalFlip(0.25),
	transforms.RandomVerticalFlip(0.25),
	])

transform_test = transforms.Compose([
	transform_base
    ]) # mean, std over rbg values



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



trainset = U.ConcatDataset([set_flip_rotate])
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

print("Displaying images")
images, labels = next(iter(trainloader))
y = np.array(classes)
# call function on our images
imshow(torchvision.utils.make_grid(images), np.apply_along_axis('   '.join, 0 ,y[labels]))



print("Defining network")
class Net(nn.Module):
    def __init__(self, num_channels, classes):
        super(Net, self).__init__() 
        self.features = nn.Sequential (
            nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=(3,3)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),

            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),

            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),

            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=512),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=512),
            
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=512 * 12 * 12, out_features=512),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=512),
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=classes),
            nn.ReLU(),
            nn.LogSoftmax(dim=1)    
        )

    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1) 
        output = self.classifier(x)
        return output


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train(model, iterator, optimizer, criterion, device):

    epoch_loss = 0
    epoch_acc = 0
    

    model.train()

    for (x, y) in tqdm(iterator, desc="TRAIN PROGRESS:", total=int(len(iterator.dataset) / batch_size), leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type='cuda'):
            y_pred = model.forward(x)
            loss = criterion(y_pred, y)
            acc = calculate_accuracy(y_pred, y)
        

        loss.backward()
        optimizer.step()

        epoch_loss += float(loss.cpu().item())
        epoch_acc += float(acc.cpu().item())

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def testing(model, iterator, criterion, device):
  total_loss = 0
  total_acc = 0
  model.eval()
  with torch.no_grad():
    for images, labels in iterator:
      images = images.to(device)
      labels = labels.to(device)

      pred_y= model.forward(images)

      loss = criterion(pred_y, labels)
      acc = calculate_accuracy(pred_y, labels)

      total_loss += float(loss.cpu())
      total_acc += float(acc.cpu())

    return total_loss / len(iterator), total_acc / len(iterator)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net(3, len(classes))
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)
print(f"The number of parameters: {count_parameters(model)}")

print("Defining Learning Rate Scheduler")
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=SCHEDULER_PATIENCE, factor=SCHEDULER_FACTOR, min_lr=MIN_LR,  verbose=True)

if (True):
    x = images.to(device)
    x = model.features(x)
    print(x.shape)

best_valid_loss = float('inf')
useful_info_dict = {
                "train_loss" : [],
                "train_acc" : [],
                "valid_loss" : [],
                "valid_acc" : []
                  }

print("Training...")
for epoch in tqdm(range(EPOCHS), desc="EPOCH PROGRESS:"):

    train_loss, train_acc = train(model, trainloader, optimizer, criterion, device)
    valid_loss, valid_acc = testing(model, validloader, criterion, device)

    scheduler.step(valid_loss)

    if valid_loss < best_valid_loss:
      best_valid_loss = valid_loss
      torch.save(model.state_dict(), 'best-model.pt')

    print(f'Epoch: {epoch+1:02}')
    
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    useful_info_dict["train_loss"].append(train_loss)
    useful_info_dict["train_acc"].append(train_acc)
    
    print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc*100:.2f}%')
    useful_info_dict["valid_loss"].append(valid_loss)
    useful_info_dict["valid_acc"].append(valid_acc)

model.load_state_dict(torch.load('best-model.pt'))

print("Testing...")
testing_loss, testing_acc = testing(model, testloader, criterion, device)

print(f"\tTesting Loss : {testing_loss}| Testing Acc: {testing_acc}")

def plot_loss_and_acc(useful_info):
    train_loss = useful_info["train_loss"]
    train_acc = useful_info["train_acc"]
    
    val_loss = useful_info["valid_loss"]
    val_acc = useful_info["valid_acc"]
    fig, axs = plt.subplots(2)

    (ax_loss, ax_acc) = axs

    train_loss_line = ax_loss.plot(train_loss, label="Train loss")
    train_acc_line = ax_acc.plot(train_acc, label="Train accuracy")
    val_loss_line = ax_loss.plot(val_loss, label="Valid loss")
    val_acc_line = ax_acc.plot(val_acc, label="Valid accuracy")
    
    fig.legend()
    
    plt.show()

plot_loss_and_acc(useful_info_dict)


print("Finished")


