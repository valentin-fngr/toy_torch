import torch 
import torch.nn as nn 
import sys 
from torch.utils.tensorboard import SummaryWriter
sys.path.insert(0,'/home/fontanger/toy_torch/toy_torch')
from models.resnet import ResNet_16_cifar
from data.utils import ImageNormalization
from torchsummary import summary
from utils.metrics import accuracy
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import configs.resnet_cifar as config


def get_model(): 
    model = ResNet_16_cifar(dropout=config.dropout, nb_classes=config.num_classes)
    model = model.to(config.device)
    print("device : ", next(model.parameters()).device)
    print(summary(model, (3, config.image_width, config.image_height)))
    return model 


def get_optimizer(model): 
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )
    return optimizer 

def get_scheduler(optimizer): 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 20, 25, 30], gamma=0.1)
    return scheduler


def get_criterion():
    
    criterion = nn.CrossEntropyLoss()
    return criterion 



def get_transforms(): 
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        ImageNormalization(),
        transforms.Pad(padding=4), 
        transforms.RandomCrop(config.image_width), 
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(), # 50 % 
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        ImageNormalization()
    ])



    return train_transform, test_transform

def get_data_loaders(train_transform, test_transform): 
    
    train_dataset =  CIFAR100(root="../../research/dataset/", download=True, train=True, transform=train_transform)
    test_dataset = CIFAR100(root="../../research/dataset/", download=True, train=False, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=4)

    return train_loader, test_loader




def train(model, optimizer, criterion, train_loader, epoch, writer=None, scheduler=None): 

    if not writer: 
        print("No writer selected, tensorboard will not be used")
    
    if not scheduler: 
        print("No learning rate scheduler") 

    
    total_loss = 0.0
    total_acc = 0.0
    for i, data in enumerate(train_loader): 
        
        model.zero_grad()
        imgs, labels = data 
        # fix use of global variable 
        imgs = imgs.to(config.device)
        labels = labels.to(config.device)

        predictions = model(imgs) 
        loss = criterion(predictions, labels)   
        total_loss += loss.item()
        total_acc += accuracy(predictions, labels)

        loss.backward()
        optimizer.step() 


        if i % 100 == 0 and i != 0: 
            # print(f"[EPOCH {epoch} | {i} / {len(train_loader)}] : cross entropy = {total_loss / 100}")
            grad_sum = 0.0 

            for name, p in model.named_parameters(): 
                norm = p.grad.data.norm(2)
                grad_sum += norm.item() ** 2

            grad_sum = grad_sum ** (1. / 2)
            if writer:
                writer.add_scalar("Train/loss", total_loss / 100, epoch*len(train_loader) + i + 1) 
                writer.add_scalar("Train/gradient_norm", grad_sum, epoch*len(train_loader) + i + 1)
                writer.add_scalar("Train/accuracy", total_acc / 100, epoch*len(train_loader) + i + 1)

            total_loss = 0.0
            total_acc = 0.0


    if scheduler is not None: 
        scheduler.step()




def test(model, criterion, scheduler, test_loader, epoch, writer=None): 

    validation_loss = 0.0 
    total_acc = 0.0
    lenght = len(test_loader)

    for i, data in enumerate(test_loader):

        imgs, labels = data 
        imgs = imgs.to(config.device)
        labels = labels.to(config.device)

        predictions = model(imgs) 
        loss = criterion(predictions, labels)   
        validation_loss += loss.item()
        total_acc += accuracy(predictions, labels)
    
    if writer is not None: 
        writer.add_scalar("Test/loss", validation_loss / lenght, epoch*lenght + i + 1)
        writer.add_scalar("Test/accuracy", total_acc / lenght, epoch*lenght + i + 1)
        if scheduler is not None: 
            writer.add_scalar("Train/Learning rate", scheduler.get_last_lr()[0], epoch)
        else: 
            writer.add_scalar("Train/Learning rate", config.lr, epoch)

    print(f"Validation at EPOCH={epoch} : ", validation_loss / lenght)
    print("Last learning rate : ", scheduler.get_last_lr()[0])



    
def training_loop(model, optimizer, criterion, train_loader, test_loader, writer=None, scheduler=None):
    
    print("--- Training ---")
    model.train()
    for epoch in range(config.epochs): 
        train(model, optimizer, criterion, train_loader, epoch, writer, scheduler)
        test(model, criterion, scheduler, test_loader, epoch, writer)
    print("--- Training : DONE ---")



def main(): 

    print("--- Loading model ---")
    model = get_model()
    print("--- Loading model : DONE ---") 

    print("--- Loading optimizer ---")
    optimizer = get_optimizer(model)
    print("--- Loading optimizer : DONE ---")

    print("--- Loading scheduler ---")
    scheduler = get_scheduler(optimizer)
    print("--- Loading scheduler : DONE ---")

    print("--- Loading criterion ---")
    criterion = get_criterion() 
    print("--- Load criterion : DONE ---")

    train_transform, test_transform = get_transforms()

    train_loader, test_loader = get_data_loaders(train_transform, test_transform)

    config_name = f"{type(model).__name__}_\
        lr={config.lr}_\
        bs={config.batch_size}\
        optimizer={type(optimizer).__name__}\
        scheduler={type(scheduler).__name__}\
        filter_size={(32, 32, 64, 128, 128)}\
        momentum={config.momentum}\
        cifar100\
        weight_decay={config.weight_decay}\
    "
    writer = SummaryWriter("runs/" + config_name)

    # train 
    training_loop(model, optimizer, criterion, train_loader, test_loader, writer, scheduler)






if __name__ == "__main__": 
    main()