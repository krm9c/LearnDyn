import os, gc
from core.trainer import train_record
import torch
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt


class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    '''
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    '''
    """
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance
             by `params.dict['learning_rate']
        """
        return self.__dict__

parser = argparse.ArgumentParser()
parser.add_argument(
    '--save_dir',
    default='_sin_',
    help="Directory to save the data in")
parser.add_argument(
    '--json_file',
    default='sine.json',
    help="Directory containing params.json")
parser.add_argument(
    '--opt',
    default='DPMCL',
    help="The optimization")
parser.add_argument(
    '--kappa',
    default=None,
    help="kappa value")
parser.add_argument(
    '--zeta',
    default=None,
    help="zeta value")
parser.add_argument(
    '--eta',
    default=None,
    help="eta value")
parser.add_argument(
    '--total_runs',
    default=None,
    help="total number of runs value")
parser.add_argument(
    '--total_samples',
    default=None,
    help="total no. of tasks value")
parser.add_argument(
    '--batch_size',
    default=None,
    help="kappa value")

import time 

if __name__ == '__main__':
    start_time = time.time()
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join('params/', args.json_file)
    assert os.path.isfile(json_path),\
        "No json configuration file found at {}".format(
        json_path)
    params = Params(json_path).dict
    params['opt'] = args.opt
    params['save_file'] = args.save_dir
    if args.total_runs is not None:
        params['total_runs'] = int(args.total_runs)
    if args.total_samples is not None:
        params['total_samples'] = int(args.total_samples)
    if args.zeta is not None:
        params['zeta'] = int(args.zeta)
    if args.eta is not None:
        params['eta'] = int(args.eta)
    if args.kappa is not None:
        params['kappa'] = int(args.kappa)
    if params['problem'] == 'classification':
        params['criterion'] = torch.nn.CrossEntropyLoss()
    else: 
        params['criterion'] = torch.nn.MSELoss()
    print("The parameters are", params)
    RA = np.zeros([params['total_runs'], params['total_samples']])
    LA = np.zeros([params['total_runs'], params['total_samples']])
    TA = np.zeros([params['total_runs'], params['total_samples'],
                params['total_samples']])


    for i in range(params['total_runs']):
        Runner = train_record(params)
        RA[i, :], LA[i, :], TA[i, :, :] = Runner.main()
        # if params['task_wise']>0:
        #     np.savetxt(params['save_file']+str(i)+'TE.csv', TE, delimiter = ',')
        # Runner.show_gpu('after all stuff have been removed')
        # Runner.print_gpu_obj()

    CTE = LA
    CME = RA
    print(CTE.shape, CME.shape)
    # print(Runner.get_gpu_memory_map())
    # Runner.show_gpu(f'{0}: Before deleting objects')
    # Runner.show_gpu(f'{0}: After deleting objects') 
    # gc.collect()
    # Runner.show_gpu(f'{0}: After gc collect') 
    # Runner.show_gpu(f'{0}: After empty cache') 
    # Runner.show_gpu('after all stuff have been removed')
    # Runner.print_gpu_obj()
    np.savez(params['save_file']+'.npz', RA=RA, LA=LA, TA = TA)
    print("The time elapsed for on iterations", time.time()-start_time)
    
    ################################################
    ################################################
    ## Lets plot things and see how is the behavior
    Runner = None

    def cm2inch(value):
        return value/2.54

    small = 7
    med = 10
    large = 12
    plt.style.use('seaborn-white')
    COLOR = 'darkslategray'
    params1 = {'axes.titlesize': small,
              'legend.fontsize': small,
              'figure.figsize': (cm2inch(15),cm2inch(8)),
              'axes.labelsize': med,
              'axes.titlesize': small,
              'xtick.labelsize': small,
              'ytick.labelsize': med,
              'figure.titlesize': small, 
              'font.family': "sans-serif",
              'font.sans-serif': "Myriad Hebrew",
              'text.color' : COLOR,
              'axes.labelcolor' : COLOR,
              'axes.linewidth' : 0.3,
              'xtick.color' : COLOR,
              'ytick.color' : COLOR}

    plt.rcParams.update(params1)
    plt.rc('text', usetex = False)
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.rcParams['mathtext.fontset'] = 'cm'

    large = 24
    med = 8
    small = 7
    labels = ['Old', 'New', 'Original']
    titles = ['worst', 'median', 'best']
    # create plots with numpy array
    fig, a=plt.subplots(2, 1, sharex=False, dpi=600,
    gridspec_kw = {'wspace': 0.7, 'hspace': 0.7})

    #CME
    # Some Plot oriented settings 
    a[0].spines["top"].set_visible(False)
    a[0].spines["bottom"].set_visible(False)
    a[0].spines["right"].set_visible(False)
    a[0].spines["left"].set_visible(True)
    a[0].grid(linestyle=':', linewidth=0.5)
    a[0].get_xaxis().tick_bottom()
    a[0].get_yaxis().tick_left()
    # Some Plot oriented settings
    a[1].spines["top"].set_visible(False)
    a[1].spines["bottom"].set_visible(False)
    a[1].spines["right"].set_visible(False)
    a[1].spines["left"].set_visible(True)
    a[1].grid(linestyle=':', linewidth=0.5)
    a[1].get_xaxis().tick_bottom()
    a[1].get_yaxis().tick_left()
    t = np.arange(CME.shape[1])
    mean = np.mean(CME, axis=0)
    yerr = np.std(CME, axis=0)

    print(t.shape, mean.shape, yerr.shape)
    a[0].fill_between(t, (mean + yerr), (mean), alpha=0.4, color = color[3])
    # a[0].set_xlim([0, 500])
    #a[0].set_yscale('log')
    a[0].set_xlabel('Tasks')
    a[0].set_ylabel('CME')
    a[0].set_title('('+params["data_id"]+","+str(params["opt"])+')')
    # a[0].legend(bbox_to_anchor=(0.0008, -0.5, 0.3, 0.1), loc = 'upper left',ncol=3 )
    t = np.arange(CTE.shape[1])
    # The Final Plots with CME
    mean = np.mean(CTE, axis=0)
    yerr = np.std(CTE, axis=0)
    print(mean, yerr)
    a[1].fill_between(t, (mean + yerr), (mean), alpha=0.4, color = color[3])
    # a[1].set_yscale('log')
    a[1].set_xlabel('Tasks')
    a[1].set_ylabel('CTE')
    # a[1].legend(bbox_to_anchor=(0.0008, -0.5, 0.3, 0.1), loc = 'upper left',ncol=3 )
    plt.savefig( params["data_id"]+"_"+params['opt']+".png", dpi=600)






##########################################################
# import torch.nn as nn
# from torch.autograd import Variable, grad
# import torch.nn.functional as F
# import torch.optim as optim
# import torch.autograd.profiler as profiler
# from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
# import time, copy
# import gc
# import torch
# from core.dataloaders import *
# from sklearn.metrics import r2_score, mean_squared_error, accuracy_score


# # # The data 
# model_F = torch.nn.Sequential( 
# torch.nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
# torch.nn.MaxPool2d(kernel_size=2, stride=2),
# torch.nn.ReLU(),
# torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
# torch.nn.MaxPool2d(kernel_size=2, stride=2),
# torch.nn.ReLU(),
# )


# model_P = torch.nn.Sequential( 
# torch.nn.Linear(7 * 7 * 64, 100),
# torch.nn.ReLU(),
# torch.nn.Linear(100, 10)
# )



# model_F = torch.nn.Sequential( 
# torch.nn.Conv2d(3, 6, 5),
# torch.nn.ReLU(),
# torch.nn.MaxPool2d(kernel_size=2, stride=2),
# torch.nn.Conv2d(6, 16, 5),
# torch.nn.ReLU(),
# torch.nn.MaxPool2d(kernel_size=2, stride=2),
# torch.nn.Dropout()
# )


# model_P = torch.nn.Sequential( 
# torch.nn.Linear(256, 100),
# torch.nn.ReLU(),
# torch.nn.Linear(100,100),
# torch.nn.ReLU(),
# torch.nn.Linear(100, 10)
# )


# n_epochs = 3
# batch_size_train = 64
# batch_size_test = 1000
# learning_rate = 0.01
# momentum = 0.5
# log_interval = 10

# random_seed = 1
# torch.backends.cudnn.enabled = False
# torch.manual_seed(random_seed)


# train_loader = torch.utils.data.DataLoader(
#   torchvision.datasets.MNIST('../data', train=True, download=True,
#                              transform=torchvision.transforms.Compose([
#                                torchvision.transforms.ToTensor(),
#                                torchvision.transforms.Normalize(
#                                  (0.1307,), (0.3081,))
#                              ])),
#   batch_size=batch_size_train, shuffle=True)

# from core.dataloaders import *


# test_loader = torch.utils.data.DataLoader(
# torchvision.datasets.MNIST('../data', train=False, download=True,
#                             transform=torchvision.transforms.Compose([
#                             torchvision.transforms.ToTensor(),
#                             torchvision.transforms.Normalize(
#                                 (0.1307,), (0.3081,))
#                             ])),
# batch_size=batch_size_test, shuffle=True)

# self.model_F.train()
# self.model_P.train()
# self.optimizer = torch.optim.Adam(list(self.model_P.parameters()) + list(self.model_F.parameters()), lr = 0.0001 )
# # optimizer_curr = torch.optim.Adam(  list(model_P.parameters()) + list(model_F.parameters()), lr = 0.0001 )


# data = data_return(params)

# for s_n in range(10):
#     print("The sample number is", s_n)
#     datloader_curr, dataloader_exp = data.generate_dataset(task_id = s_n, batch_size= 64, phase = 'training')
#     test_loader_curr, test_loader = data.generate_dataset(task_id = s_n, batch_size= 64, phase = 'testing')
#     data.append_to_experience(task_id = s_n)
#     dataloader_curr, dataloader_exp = data.generate_dataset(task_id = s_n, batch_size= 64, phase = 'training')
    

#     for epoch in range(10):
#         for batch_idx, sample in enumerate(dataloader_curr):
#             dat = sample['x']
#             target = sample['y'].reshape([-1])
#             optimizer.zero_grad()
#             feature_out = model_F(dat)
#             y_pred = F.log_softmax(model_P(feature_out.reshape(feature_out.size(0), -1) ) )
#             loss   = F.nll_loss(y_pred, target)
#             # y_pred = model_P(feature_out.reshape(feature_out.size(0), -1) ) 
#             # loss   = torch.nn.CrossEntropyLoss()(y_pred, target)
#             loss.backward()
#             optimizer.step()

#     test_loss = 0.0
#     correct = 0.0
#     with torch.no_grad():
#         for sample in test_loader:

#             dat = sample['x']
#             target = sample['y'].reshape([-1])

#             feature_out = model_F(dat)
#             output = F.log_softmax(model_P(feature_out.reshape(feature_out.size(0), -1) ) )
#             test_loss += F.nll_loss(output, target, size_average=False).item()

#             pred = output.data.max(1, keepdim=True)[1]
#             correct += pred.eq(target.data.view_as(pred)).sum()


#         test_loss /= len(test_loader.dataset)
#         print('\n CME Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#             test_loss, correct, len(test_loader.dataset),
#             100. * correct / len(test_loader.dataset)))

        
#         test_loss = 0.0
#         correct = 0.0
#         for sample in test_loader_curr:
#             dat = sample['x']
#             target = sample['y'].reshape([-1])
#             feature_out = model_F(dat)
#             output = F.log_softmax(model_P(feature_out.reshape(feature_out.size(0), -1) ) )
            
#             test_loss += F.nll_loss(output, target, size_average=False).item()
            
#             pred = output.data.max(1, keepdim=True)[1]
#             correct += pred.eq(target.data.view_as(pred)).sum()

#         test_loss /= len(test_loader.dataset)
#         print('\n CTE Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#             test_loss, correct, len(test_loader_curr.dataset),
#             100. * correct / len(test_loader_curr.dataset)))


#     # data.append_to_experience(task_id = s_n)