import torch
import numpy as np
from torch.optim import lr_scheduler, optimizer
from torch.serialization import save
from torchvision.transforms.transforms import Scale
from model_genarate import *
import torchvision.datasets as dset
import torchvision.transforms as T
from torch.utils.data import DataLoader, sampler

def check_accuracy(loader, model, device=None, dtype=None):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
    return acc

def train(
        model=None, loader_train=None, 
        loader_val=None, scheduler=None,optimizer=None,wd=None, 
        epochs=1, device=None, dtype=None, check_point_dir=None, save_epochs=None, mode=None
):
    acc = 0
    model = model.to(device=device)
    accs = [0]
    train_acc = [0]
    losses = []
    record_dir_acc = check_point_dir + 'record_val_acc.npy'
    record_dir_train_acc = check_point_dir + 'record_train_acc.npy'
    record_dir_loss = check_point_dir + 'record_loss.npy'
    model_save_dir = check_point_dir + 'check_points.pth'
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, pre = scores.max(1)

            num_correct = (pre == y).sum()
            num_samples = pre.size(0)
            train_acc.append(float(num_correct) / num_samples)

            loss = torch.nn.functional.cross_entropy(scores, y)

            loss_value = np.array(loss.item())
            losses.append(loss_value)

            optimizer.zero_grad()
            loss.backward()
            for group in optimizer.param_groups:  # Adam-W
                for param in group['params']:
                    param.data = param.data.add(param.data, alpha=-wd * group['lr'])
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if t % 100 == 0:
                acc = check_accuracy(loader_val, model)
                accs.append(np.array(acc))
        print("Epoch:" + str(e) + ', Val acc = ' + str(acc))
        if (mode == 'run' and e % save_epochs == 0 and e != 0) or (mode == 'run' and e == epochs - 1):
            np.save(record_dir_acc, np.array(accs))
            np.save(record_dir_train_acc, np.array(train_acc))
            np.save(record_dir_loss, np.array(losses))
            torch.save(model.state_dict(), model_save_dir)
    return acc, accs, train_acc, losses

def run(
        mode='search', model = None,
        search_epoch=4, lr_range=[-5, -2], wd_range=[-4, -2],
        search_result_save_dir = None,
        run_epoch=30, lr=0, wd=0,  
        check_point_dir = None, save_epochs=None,
        T_mult=None, loader_train=None, loader_val=None, device=None, dtype=None
):
    if mode == 'search':
        num_iter = 10000000
        epochs = search_epoch
    else:
        num_iter = 1
        epochs = run_epoch
    if mode == 'search':
        print('Searching under lr: 10 ** (', lr_range[0], ',', lr_range[1],') , wd: 10 ** (', wd_range[0], ',', wd_range[1], '), every ', epochs, ' epoch')
    else:
        print(mode + 'ing under lr: ' + str(lr) + ' , wd: ' + str(wd) + ' for ', str(epochs), ' epochs.')
    for i in range(num_iter):
        model_ = model
        if mode == 'search':
            learning_rate = 10 ** np.random.uniform(lr_range[0], lr_range[1])
            weight_decay = 10 ** np.random.uniform(wd_range[0], wd_range[1])
        else:
            learning_rate = lr 
            weight_decay = wd
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                    betas=(0.9, 0.999), weight_decay=0)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=T_mult)
        args = {
            'model' : model_, 'loader_train' : loader_train, 'loader_val' : loader_val, 
            'scheduler' : lr_scheduler, 'optimizer' : optimizer, 'wd' : wd, 
            'epochs' : epochs, 'device' : device, 'dtype' : dtype,
            'check_point_dir' : check_point_dir, 'save_epochs' : save_epochs, 'mode' : mode
        }
        print('######################################     Training...     ######################################')
        val_acc, _, _, _ = train(**args)
        print('Training for ' + str(epochs) +' epochs, learning rate: ', learning_rate, ', weight decay: ', weight_decay, ', Val acc: ', val_acc)

        if mode == 'search':
            with open(search_result_save_dir + 'search_result.txt', "a") as f:
                f.write(str(epochs) + ' epochs, learning rate:'  + str(learning_rate) + ', weight decay: ' + str(weight_decay) + ', Val acc: ' + str(val_acc) + '\n')


if __name__ == '__main__':
    print()
    print('############################### Training test (cifar10 for example) ###############################')  
    print()
    dtype = torch.float32
    USE_GPU = True
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Device: ', device)
    print()
    print('############################### Dataset loading ... ###############################')
    NUM_TRAIN = 49000
    transform = T.Compose([
        T.Resize(224),
        T.RandomRotation(90),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    cifar_train = dset.CIFAR10('./dataset/', train=True, download=True, transform=transform)
    loader_train = DataLoader(cifar_train, batch_size=32, sampler=sampler.SubsetRandomSampler(range(0, NUM_TRAIN)))
    cifar_val = dset.CIFAR10('./dataset/', train=True, download=True, transform=transform)
    loader_val = DataLoader(cifar_val, batch_size=32, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))
    print('############################### Dataset loaded. ###############################')
    args = {
        'mode':'search', 'model' : mobile_former_294(10),
        'loader_train' : loader_train, 'loader_val' : loader_val, 
        'device' : device, 'dtype' : dtype, 'T_mult' : 2, 
        # If search:
        'search_epoch' : 4, 'lr_range' : [-5, -2], 'wd_range' : [-4, -2],
        'search_result_save_dir' : './search_result/',
        # If run
        'run_epoch' : 5, 'lr' : 0, 'wd' : 0,  
        'check_point_dir' : './check_point/', 'save_epochs' : 5,
    }
    run(**args)

