import matplotlib.pyplot as plt # for plotting
import numpy as np
import sys, torch, json
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer


def progressbar(it, desc="", size=60, out=sys.stdout): # Python3.3+
    count = len(it)
    def show(j):
        x = int(size*j/count)
        print("{}[{}{}] {}/{}".format(desc, u"█"*x, "."*(size-x), j, count), 
                end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)

def imshow(img, title=None):
  ''' function to show image '''
  img = img / 2 + 0.5 # unnormalize
  npimg = img.numpy() # convert to numpy objects
  ax = plt.subplot() 
  ax.imshow(np.transpose(npimg, (1, 2, 0)))
  if (title is not None):
    ax.set_title(title)
  plt.show()
  
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

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
    
    return fig

def load_classes():   
    with open("flower_classes.json", "r") as f:
        classes = list(json.load(f).values())
    return classes


# https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
# Functions for tensorboard
def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels, classes, size):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    images = images.cpu()
    labels = labels.cpu()
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 12))
    for idx in np.arange(size):
        ax = fig.add_subplot(1, size, idx+1, xticks=[], yticks=[])
        image = np.transpose(images[idx], (1, 2, 0))
        ax.imshow(normalize_image(image))
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min = image_min, max = image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image    



def plot_images(images, labels, classes, normalize = True):

    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize = (15, 15))

    for i in range(rows*cols):

        ax = fig.add_subplot(rows, cols, i+1)
        
        image = images[i]

        if normalize:
            image = normalize_image(image)

        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        label = classes[labels[i]]
        ax.set_title(label)
        ax.axis('off')
    
    return fig

def display_matrix(pred_dict):
    fig = plt.figure(figsize = (15, 15))
    ax = fig.add_subplot()
    matrix = np.zeros((102, 102))
    
    for label  in pred_dict:
        for pred in pred_dict[label]:
            matrix[label][pred.item()] += 1
    ax.matshow(matrix)
    return fig