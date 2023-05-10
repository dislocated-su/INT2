import matplotlib.pyplot as plt # for plotting
import numpy as np
import sys, torch, torchvision, time
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer
from torch.cuda.amp import GradScaler, autocast
from torchsummary import summary
from model import VGG, VGG_config
import lib
from torch.utils.tensorboard import SummaryWriter

from transforms import get_datasets

if (torch.cuda.is_available()):
    torch.cuda.empty_cache()
    print("GPU WORKING!")

#Hyper parameters
batch_size = 16
gradient_accumulations = 4
EPOCHS = 150

# for SGD
LEARNING_RATE =0.01
MOMENTUM=0.95
WEIGHT_DECAY=0.005
# For lr Scheduler
SCHEDULER_PATIENCE=5
SCHEDULER_FACTOR= 0.2
MIN_LR = 0.0000001

scaler = GradScaler()

def train(model, iterator, optimizer, criterion, device, epoch):

    epoch_loss = 0
    epoch_acc = 0

    model.train()
    model.zero_grad()
        
    for i, (x, y) in enumerate(iterator, 0):
        
        x = x.to(device)
        y = y.to(device)

        with autocast():
            y_pred = model.forward(x)
            loss = criterion(y_pred, y)
            acc = lib.calculate_accuracy(y_pred, y)
        

        scaler.scale(loss / gradient_accumulations).backward()

        
        if (i + 1) % gradient_accumulations == 0:
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad()

        epoch_loss += float(loss.cpu().item())
        epoch_acc += float(acc.cpu().item())
            
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def test(model, iterator, criterion, device):
  total_loss = 0
  total_acc = 0
  model.eval()
  with torch.no_grad():
    for images, labels in iterator:
      images = images.to(device)
      labels = labels.to(device)

      pred_y= model.forward(images)

      loss = criterion(pred_y, labels)
      acc = lib.calculate_accuracy(pred_y, labels)

      total_loss += float(loss.cpu())
      total_acc += float(acc.cpu())
    
    #writer.add_figure('predictions vs. actuals', lib.plot_classes_preds(model, unnormalize(x), y, classes), global_step=epoch)
    return total_loss / len(iterator), total_acc / len(iterator)

def testing(model, testloader, criterion, device):
    print("Testing...")
    model.load_state_dict(torch.load('best-model.pt'))
    testing_loss, testing_acc = test(model, testloader, criterion, device)

    print(f"\tTesting Loss : {testing_loss}| Testing Acc: {testing_acc}")



if __name__ == '__main__':
    comment = input("Enter comment about current run: ")
    
    # TensorBoard
    writer = SummaryWriter(comment=comment)
    
    trainloader, validloader, testloader = get_datasets(batch_size)
    classes = lib.load_classes()

    print("Displaying images")
    train_images, train_labels = next(iter(trainloader))
    test_images, test_labels = next(iter(testloader))
    y = np.array(classes)
    # call function on our images
    # grid_train = torchvision.utils.make_grid(train_images)
    # grid_test = torchvision.utils.make_grid(test_images)

    # lib.imshow(grid_train, np.apply_along_axis('   '.join, 0 ,y[train_labels]))#
    
    
    # plot images in tensorboard
    
    train_images_plot, train_labels_plot = zip(*[(image, label) for image, label in
                       [trainloader.dataset[np.random.randint(0, len(trainloader.dataset))] for _ in range(25)]])
    
    test_images_plot, test_labels_plot = zip(*[(image, label) for image, label in
                       [testloader.dataset[np.random.randint(0, len(testloader.dataset))] for _ in range(25)]])
    
    writer.add_figure('images train', lib.plot_images(train_images_plot, train_labels_plot, classes))
    writer.add_figure('images test', lib.plot_images(test_images_plot, test_labels_plot, classes))
    
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Defining network")
    model = VGG(3, len(classes), VGG_config['VGG16'])
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    writer.add_graph(model, test_images)

    model = model.to(device)
    criterion = criterion.to(device)

    summary(model, (3,256,256))

    if (False):
        x = train_images.to(device)
        x = model.features(x)
        print(x.shape)

    print(f"The number of parameters: {lib.count_parameters(model)}")

    print("Defining Learning Rate Scheduler")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=SCHEDULER_PATIENCE, factor=SCHEDULER_FACTOR, min_lr=MIN_LR,  verbose=True)
    
    best_valid_loss = float('inf')
    log = {
                "train_loss" : [],
                "train_acc" : [],
                "valid_loss" : [],
                "valid_acc" : []
                  }

    start_time = time.time()
    try:
        print("Training...")
        for epoch in lib.progressbar(range(EPOCHS), desc="EPOCH PROGRESS:"):

            epoch_start_time = time.time()
            
            train_loss, train_acc = train(model, trainloader, optimizer, criterion, device, epoch)
            valid_loss, valid_acc = test(model, validloader, criterion, device)

            scheduler.step(valid_loss)
            
            epoch_end_time = time.time()
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'best-model.pt')
                

            print(f'\nEpoch: {epoch+1:02} : {epoch_end_time - epoch_start_time}s')
            
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            log["train_loss"].append(train_loss)
            log["train_acc"].append(train_acc)
            
            print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc*100:.2f}%')
            log["valid_loss"].append(valid_loss)
            log["valid_acc"].append(valid_acc)
            
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/valid', valid_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/valid', valid_acc, epoch)

        end_time = time.time()
        print(f"Training finished after {end_time - start_time}s")
    except KeyboardInterrupt:
        testing(model, testloader, criterion, device)
    else:
        testing(model, testloader, criterion, device)
    
    writer.add_figure("loss and acc", lib.plot_loss_and_acc(log))
    print("Finished")