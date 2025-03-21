#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from unet import ResidualUNet, Denoiser
from utils import save_image, dotdict

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


#sigmas = 3*torch.pow(torch.ones(50)*0.9,torch.arange(50))
betat = torch.linspace(1e-4,0.02,1000)
alphat = 1-betat
alphabart = [1]
for alpha in alphat:
    alphabart.append(alphabart[-1]*alpha)

alphabart = torch.Tensor(alphabart[1:])


betat = betat[20:]
alphat = alphat[20:]
alphabart = alphabart[20:]
print(alphabart)


def forward(data, model, alphasb_batch = None):
    im = data

    # parameters of the langevin dynamics steps
    ind_randoms= torch.randint(0, alphat.shape[0], (data.shape[0],), device = data.device)
    
    noise_in = torch.randn_like(im)
    if alphasb_batch is None:
        alphasb_batch = alphabart[ind_randoms]

        im_input = (torch.sqrt(1-alphasb_batch)[:,None,None,None]*noise_in+torch.sqrt(alphasb_batch)[:,None,None,None]*im)
    else :
        im_input = im
    pred_eps = model(im_input, alphasb_batch)

    score = -pred_eps/torch.sqrt(1-alphasb_batch)[:,None,None,None]
    # corrected image 
    im_corrected = (im_input-torch.sqrt(1-alphasb_batch)[:,None,None,None]*pred_eps)/torch.sqrt(alphasb_batch)[:,None,None,None]

    dist_eps =torch.sum((pred_eps-noise_in)**2,(1,2,3)) # square norm of loss per image
    loss = dist_eps.sum()
    #loss = (dist_mean).sum()
    return loss, im_input, im_corrected, pred_eps, score


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
     
        optimizer.zero_grad()

        loss, im_input, im_corrected, pred_eps, score= forward(data, model)

        loss.backward()
        running_loss += loss.item()
        
        optimizer.step()

        if (batch_idx+1) % args.log_interval  == 0:
            running_loss = running_loss/(args.log_interval*args.batch_size)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), running_loss))
            running_loss = 0
            if args.dry_run:
                break

nid= "vnoise"
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    ctime = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            im = data
            loss, im_input, corr, pred_eps, score= forward(data, model)
            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set {:.4f} Average loss: {:.4f} \n'.format(time.time()-ctime,test_loss))
    orig = im
    noisy = im_input
    save_image(noisy[:10],"im/noisy.jpg")
    save_image(corr[:10],"im/corrected.jpg")
    save_image(orig[:10],"im/originals.jpg")
    gen_shape = list(im.shape)
    gen_shape[0] = 32
    gen_im_sde = sample_sde(model, device, gen_shape)
    save_image(gen_im_sde, "im/generatedsde.jpg")
    #gen_im_ddpm = sampleDDPM(model, device, gen_shape)
    #save_image(gen_im_ddpm, "im/generatedddpm.jpg")
    gen_im_ode = sample_ode(model, device, gen_shape)
    save_image(gen_im_ode, "im/generatedode.jpg")

def sample_ode(self, device, im_shape):
    with torch.no_grad():  # avoid backprop wrt model parameters
        x = torch.randn(im_shape, device=device)
        for t in range(alphabart.shape[0]-1,1,-1):
            alpha_t = alphat[t]
            beta_t = 1-alpha_t
            alphasb_batch = torch.ones((x.shape[0],), device=  device)*alphabart[t]
            loss, im_input, im_corrected, pred_eps, score = forward(x, model, alphasb_batch=alphasb_batch)
            x = (2-torch.sqrt(1-beta_t))*x+1/2*beta_t*score
            if t%100==0:
                print(t, beta_t, torch.std(x), torch.mean(x))
    return x

def sample_sde(self,device, im_shape):
    with torch.no_grad():  # avoid backprop wrt model parameters
        x = torch.randn(im_shape, device=device)
        for t in range(alphabart.shape[0]-1,1,-1):
            alpha_t = alphat[t]
            beta_t = 1-alpha_t
            alphasb_batch = torch.ones((x.shape[0],), device=  device)*alphabart[t]
            loss, im_input, im_corrected, pred_eps, score = forward(x, model, alphasb_batch=alphasb_batch)
            xp = (2-torch.sqrt(1-beta_t))*x+beta_t*score
            noise = torch.randn_like(x)
            x = xp + torch.sqrt(beta_t) * noise
    return x


# In[ ]:


TEST = False# set to true to load model from disk and only generate to test langevin

# Training settings
args_dict = {'batch_size' : 64, 'test_batch_size' :64, 'epochs' :1000, 'lr' : 0.0002, 'gamma' : 0.995, 'no_cuda' :False, 
             'dry_run':False, 'seed': 1, 'log_interval' : 200, 'save_model' :True, 'only_test':False, 'model_path':"models/denoiserceleb.pt", 
             'load_model_from_disk':False, 'dataset':"CELEBA", 'test':False}
args = dotdict(args_dict)
#parser = argparse.ArgumentParser(description="A simple argument parser example.")

# Add arguments
#parser.add_argument('--dataset', type=str, required=False, default = 'MNIST', help='Dataset can be one of MNIST, CIFAR, CELEBA')
#parser.add_argument('--test', type= str, required = False, help='wether to only test a model, requires path to the testing weights')

# Parse the arguments
#margs = parser.parse_args()
#args.dataset = margs.dataset
#if margs.test is not None:
#    print("TEST")
#    args.test = True
#    print(args.test)
#    args.model_path = margs.test

if args.test:
    print("TEST")
    args.load_model_from_disk = True
    args.only_test = True
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

train_kwargs = {'batch_size': args.batch_size}
test_kwargs = {'batch_size': args.test_batch_size}
if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                   'pin_memory': True,
                   'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

dataset = args.dataset
if dataset == "CIFAR":
    transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),#mean, std
    transforms.RandomHorizontalFlip(p=0.5)
    ])

    dataset1 = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
    dataset2 = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    model = Denoiser(3,3).to(device)
elif dataset == "CELEBA":
    transform = transforms.Compose([transforms.Resize((64,64)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5])])
    dataset1 = datasets.CelebA("../data/celeba", split = 'train',download=False, transform=transform)
    dataset2 = datasets.CelebA("../data/celeba", split = 'test', download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    model = Denoiser(3,3).to(device)

elif dataset=="MNIST":
    # loading dataset
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(2),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('./data/mnist', train=True, download=True,
    transform=transform)
    dataset2 = datasets.MNIST('./data/mnist', train=False,
    transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Denoiser(1,1).to(device)

if args.load_model_from_disk:
    model.load_state_dict(torch.load(args.model_path, weights_only= True))
optimizer = optim.Adam(model.parameters(), lr=args.lr)

scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

alphabart= alphabart.to(device)
for epoch in range(1, args.epochs + 1):
    if not args.only_test:
        train(args, model , device, train_loader, optimizer, epoch)
        scheduler.step()
        if args.save_model:
            torch.save(model.state_dict(), args.model_path)
    if epoch%3 == 0:
        test(model,  device, test_loader)

if args.save_model:
    torch.save(model.state_dict(), args.model_path)


# In[ ]:





# In[ ]:




