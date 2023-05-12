import torch.nn as nn
from torch.autograd import Variable, grad
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd.profiler as profiler
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
import time, copy
import gc
import torch
from core.dataloaders import *
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from copy import deepcopy
################################################
# Sanity Check  and initialize the CPU/GPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

# The class for defining the network architecture and the optimizer
class Net(torch.nn.Module):
    def __init__(self, Config):
        super(Net, self).__init__()
        self.config = Config
        if self.config['network']== 'fcnn':
            # Model h
            self.model_F = torch.nn.Sequential(
                torch.nn.Linear(self.config['D_in'], self.config['H']),
                torch.nn.ReLU(),
                torch.nn.Linear(self.config['H'], self.config['H']),
                torch.nn.ReLU(),
                torch.nn.Linear(self.config['H'], self.config['D_in'])
            )

            # The g model and the buffer model are the same
            # Model g
            self.model_P = torch.nn.Sequential(
            torch.nn.Linear(self.config['D_in'], self.config['H']),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config['H'], self.config['H']),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config['H'], self.config['D_out']) 
            )


            # Model buffer
            self.model_F_buffer = torch.nn.Sequential(
            torch.nn.Linear(self.config['D_in'], self.config['H']),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config['H'], self.config['H']),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config['H'], self.config['D_in']) 
            )

            self.model_P_buffer = torch.nn.Sequential(
            torch.nn.Linear(self.config['D_in'], self.config['H']),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config['H'], self.config['H']),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config['H'], self.config['D_out']) 
            )
            
        elif self.config['network']== 'cnn':
            # # The data 
            self.model_F = torch.nn.Sequential( 
            torch.nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReLU(),
            )


            self.model_P = torch.nn.Sequential( 
            torch.nn.Linear(7 * 7 * 64, self.config['H']),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config['H'],  self.config['D_out'])
            )

            self.model_F_buffer = torch.nn.Sequential( 
            torch.nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReLU(),
            )

            self.model_P_buffer = torch.nn.Sequential( 
            torch.nn.Linear(7 * 7 * 64, self.config['H']),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config['H'],  self.config['D_out'])

            )

        # elif self.config['network']== 'cnn3':
        #    self.model_F = torch.nn.Sequential( 
        #     torch.nn.Conv2d(3, 6, 5),
        #     torch.nn.MaxPool2d(kernel_size=2, stride=2),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(6, 16, 5),
        #     torch.nn.MaxPool2d(kernel_size=2, stride=2),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout()
        #     )


        #    self.model_P = torch.nn.Sequential( 
        #     torch.nn.Linear(256, self.config['H']),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(self.config['H'], self.config['H']),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(self.config['H'], self.config['D_out'])
        #     )


        #    self.model_P_buffer = torch.nn.Sequential( 
        #     torch.nn.Linear(256, self.config['H']),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(self.config['H'], self.config['H']),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(self.config['H'], self.config['D_out'])
        #     )

        #    self.model_F_buffer = torch.nn.Sequential( 
        #     torch.nn.Conv2d(3, 6, 5),
        #     torch.nn.MaxPool2d(kernel_size=2, stride=2),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(6, 16, 5),
        #     torch.nn.MaxPool2d(kernel_size=2, stride=2),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout()
        #     )

        # if self.config['opt'] == 'ANML':
        #     if self.config['network']== 'fcnn':
        #         # Model h
        #         self.model_NLM = torch.nn.Sequential(
        #             torch.nn.Linear(self.config['D_in'], self.config['H']),
        #             torch.nn.ReLU(),
        #             torch.nn.Linear(self.config['H'], self.config['H']),
        #             torch.nn.ReLU(),
        #             torch.nn.Linear(self.config['H'], self.config['D_in'])
        #         )
        #     elif self.config['network']== 'cnn':
        #         self.model_NLM = torch.nn.Sequential( 
        #             torch.nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
        #             torch.nn.ReLU(),
        #             torch.nn.MaxPool2d(kernel_size=2, stride=2),
        #             torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
        #             torch.nn.ReLU(),
        #             torch.nn.MaxPool2d(kernel_size=2, stride=2),
        #             torch.nn.Dropout()
        #             )
        #     elif self.config['network']== 'cnn3':
        #         self.model_NLM = torch.nn.Sequential( 
        #             torch.nn.Conv2d(3, 6, 5),
        #             torch.nn.ReLU(),
        #             torch.nn.MaxPool2d(kernel_size=2, stride=2),
        #             torch.nn.Conv2d(6, 16, 5),
        #             torch.nn.ReLU(),
        #             torch.nn.MaxPool2d(kernel_size=2, stride=2),
        #             torch.nn.Dropout()     
        #         )
        #     self.optimizer_NLM = torch.optim.RMSprop( list(self.model_NLM.parameters()) +\
        #          list(self.model_P.parameters()), lr=self.config['learning_rate'] )

        # if self.config['opt'] == 'CML':
        #     self.opt_buffer = torch.optim.RMSprop(list(self.model_P.parameters()) \
        #     + list(self.model_buffer.parameters()),\
        #     lr= self.config['learning_rate'])

        self.optimizer  = torch.optim.Adam(list(self.model_P.parameters())
            +list(self.model_F.parameters()), lr = self.config['learning_rate'])
        
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=1.01)



    # Return the current score.
    def return_score(self, dataloader_eval_curr, dataloader_eval_exp):
        # print("I am in the model", self.evaluate_model(dataloader_eval_curr),\
        # self.evaluate_model(dataloader_eval_exp) )
        return self.evaluate_model(dataloader_eval_curr), self.evaluate_model(dataloader_eval_exp)

    def task_wise_Accuracy(self, data_obj, n_tasks):
        task_Wise_acc = []
        for i in range(n_tasks):
            (x, y), (dat_x, dat_y) = data_obj.retreive_data(i, phase= 'testing')
            data_loader = DataLoader(Continual_Dataset(self.config, \
                    data_x = x, data_y = y),\
                    batch_size= self.config['batch_size'], \
                    shuffle=True, num_workers=4)
            task_Wise_acc.append(self.evaluate_model(data_loader))
        return task_Wise_acc

    # The function to get the outputs
    def evaluate_model(self, test_loaders):
        self.model_P.eval()
        self.model_F.eval()
        test_loss = 0.0
        correct = 0.0
        # total = 0.0
        for sample in test_loaders:
            dat = sample['x'].float().to(device)

            if self.config['problem'] == 'classification': 
                feature_out = self.model_F(dat)
                output = self.model_P(feature_out.reshape(feature_out.size(0), -1) ) 
                pred = output.data.max(1, keepdim=True)[1]
                # print("The actual outputs", pred, sample['y'])
                # print("The actual correct value", pred.eq(sample['y'].long().reshape([-1]).to(device).data.view_as(pred)).float())
                correct += pred.eq(sample['y'].long().reshape([-1]).to(device).data.view_as(pred)).float().sum()
                # total += len(output)
                # print("The value that was calculated", correct, total)
            else:
                feature_out = self.model_F(dat)
                output = self.model_P(feature_out.reshape(feature_out.size(0), -1) ) 
                test_loss += torch.nn.MSELoss()(output, sample['y'].float().to(device)).item()
        
        if self.config['problem'] == 'classification':
            # print("correct is", correct, len(test_loaders.dataset), 100. * correct / len(test_loaders.dataset))
            # print(len(test_loaders.dataset), total)
            # xoo = input("Lets check how the evaluation happens")
            return (100 * correct / (len(test_loaders.dataset)+1))
        else:
            return ( test_loss / (len(test_loaders.dataset)+1) )


    def update_para(self, sample, optimizer):
        ################################
        dat = sample['x'].float().to(device)
        target = sample['y'].to(device)
        self.optimizer.zero_grad()
        if self.config['problem'] == 'classification':
            target = target.reshape([-1]).long()
            feature_out = self.model_F(dat)

            # print(feature_out.shape)
            y_pred = F.log_softmax(self.model_P(feature_out.reshape(feature_out.size(0), -1) ) )
            loss   = F.nll_loss(y_pred, target)
        else:
            y_pred = self.model_P( self.model_F( dat ) )
            loss   = torch.nn.MSELoss()(y_pred, target.float() )
        loss.backward(retain_graph = True)
        optimizer.step()
        return loss

###########################################################################
    def ER(self, dataloader_exp):
        exp_it = iter(dataloader_exp)
        for epoch in range(self.config['N']):
            try:
                sample = next(exp_it) 
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader 
                exp_it = iter(dataloader_exp)
                sample  = next(exp_it) 
            self.update_para(sample, self.optimizer)
        return self

###########################################################################
    def normalize_grad(self, input, p=2, dim=1, eps=1e-12):
        return input / input.norm(p, dim, True).clamp(min=eps).expand_as(input)

###########################################################################
    def learn_NASH(self, dataloader_curr, dataloader_exp, test_loader_ap,\
         samp_num, phase = None):
        exp_it  = iter(dataloader_exp)
        curr_it = iter(dataloader_curr)


        dat_x_loss=[]
        dat_theta_loss=[]
        dat_J=[]
        
        for epoch in range(self.config['kappa']):
            ## Generalization cost, J_N
            try:
                sample_c = next(curr_it) 
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # Reinitialize data loader 
                curr_it = iter(dataloader_curr)
                sample_c  = next(curr_it) 
            ################################
            dat    = sample_c['x'].float().to(device)
            target = sample_c['y'].to(device)
            self.optimizer.zero_grad()
            if self.config['problem'] == 'classification':
                target = target.reshape([-1]).long()
                feature_out = self.model_F(dat)
                # print(feature_out.shape)
                y_pred = F.log_softmax(self.model_P(feature_out.reshape(feature_out.size(0), -1) ) )
                J_N    = F.nll_loss(y_pred, target)
            else:
                y_pred = self.model_P( self.model_F( dat ) )
                J_N   = torch.nn.MSELoss()(y_pred, target.float() )
            ## Forgetting cost evaluation, J_P
            try:
                sample_e = next(exp_it) 
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader 
                exp_it = iter(dataloader_exp)
                sample_e  = next(exp_it)
            ################################
            dat = sample_e['x'].float().to(device)
            target = sample_e['y'].to(device)
            self.optimizer.zero_grad()
            if self.config['problem'] == 'classification':
                target = target.reshape([-1]).long()
                feature_out = self.model_F(dat)
                # print(feature_out.shape)
                y_pred = F.log_softmax(self.model_P(feature_out.reshape(feature_out.size(0), -1) ) )
                J_P   = F.nll_loss(y_pred, target)
            else:
                y_pred = self.model_P( self.model_F( dat ) )
                J_P   = torch.nn.MSELoss()(y_pred, target.float() )
            ## The loss due to the data changes.
            # Lets merge the sample arrays for forgetting and generalization    
            # The data oriented loss
            if self.config['problem'] == 'classification':
                x_PN = torch.cat((sample_e['x'], sample_c['x']), dim = 0).float().to(device)
                y_PN = torch.cat((sample_e['y'], sample_c['y']), dim = 0).to(device)
                target = y_PN.reshape([-1]).long()
                feature_out = self.model_F(x_PN)
                feature_out = feature_out.reshape(feature_out.size(0), -1)
                y_pred = F.log_softmax( self.model_P(feature_out) )
                loss_x_PN  = F.nll_loss(y_pred, target)
            else:
                x_PN = torch.cat((sample_e['x'], sample_c['x']), dim = 0).float().to(device)
                y_PN = torch.cat((sample_e['y'], sample_c['y']), dim = 0).to(device)
                feature_out = self.model_F( x_PN)
                y_pred = self.model_P(feature_out)
                loss_x_PN   = torch.nn.MSELoss()(y_pred, y_PN.float() )
            # Player 1 Strategies
            x_PN.requires_grad = True
            adv_grad = 0
            epsilon = 0.001
            #########################################################################################
            for j in range(self.config['zeta']+1): 
                if self.config['problem'] == 'classification':
                    x_PN = x_PN + epsilon*adv_grad 
                    feature_out = self.model_F(x_PN)
                    feature_out = feature_out.reshape(feature_out.size(0), -1)
                else:
                    x_PN = x_PN + epsilon*adv_grad 
                    feature_out = self.model_F( x_PN)
                # self.opt_buffer.zero_grad()
                if self.config['problem'] == 'classification':
                    y_pred = F.log_softmax( self.model_P(feature_out) )
                    loss_BUF  = F.nll_loss(y_pred, target)
                else:
                    y_pred = self.model_P(feature_out)
                    loss_BUF   = torch.nn.MSELoss()(y_pred, y_PN.float() )
                adv_grad = torch.autograd.grad(loss_BUF, x_PN)[0]
                # Normalize the gradient values.
                adv_grad = self.normalize_grad(adv_grad, p=2, dim=1, eps=1e-12)
            #########################################################################################    
            x_PN_next = x_PN
            # The data oriented loss
            if self.config['problem'] == 'classification':
                feature_out = self.model_F(x_PN_next)
                feature_out = feature_out.reshape(feature_out.size(0), -1)
                y_pred = F.log_softmax( self.model_P(feature_out) )
                loss_x_PN_1  = F.nll_loss(y_pred, target)
            else:
                feature_out = self.model_F( x_PN_next)
                y_pred = self.model_P(feature_out)
                loss_x_PN_1   = torch.nn.MSELoss()(y_pred, y_PN.float() )
            x_Loss = self.config['gamma']*(loss_x_PN_1 - loss_x_PN) 
            # Player 2 strategies, for changes in the parameters.
            # #####################################################
            self.model_P_buffer.load_state_dict(self.model_P.state_dict())
            self.model_F_buffer.load_state_dict(self.model_F.state_dict())
            self.opt_buffer = torch.optim.Adam( \
                list(self.model_P_buffer.parameters()) + list(self.model_P_buffer.parameters()) ,\
                lr = epsilon )
            #########################################################################################
            for j in range(self.config['zeta']): 
                self.opt_buffer.zero_grad()
                if self.config['problem'] == 'classification':
                    x_PN = torch.cat((sample_e['x'], sample_c['x']), dim = 0).float().to(device)
                    y_PN = torch.cat((sample_e['y'], sample_c['y']), dim = 0).to(device)
                    target = y_PN.reshape([-1]).long()
                    feature_out = self.model_F_buffer(x_PN)
                    feature_out = feature_out.reshape(feature_out.size(0), -1)
                    y_pred = F.log_softmax( self.model_P_buffer(feature_out) )
                    loss_BUF  = F.nll_loss(y_pred, target)
                else:
                    x_PN = torch.cat((sample_e['x'], sample_c['x']), dim = 0).float().to(device)
                    y_PN = torch.cat((sample_e['y'], sample_c['y']), dim = 0).to(device)
                    feature_out = self.model_F_buffer( x_PN)
                    y_pred = self.model_P_buffer(feature_out)
                    loss_BUF   = torch.nn.MSELoss()(y_pred, y_PN.float() )
                loss_BUF.backward()
                self.opt_buffer.step()
            #########################################################################################
            if self.config['problem'] == 'classification':
                x_PN = torch.cat((sample_e['x'], sample_c['x']), dim = 0).float().to(device)
                y_PN = torch.cat((sample_e['y'], sample_c['y']), dim = 0).to(device)
                target = y_PN.reshape([-1]).long()
                feature_out = self.model_F_buffer(x_PN)
                feature_out = feature_out.reshape(feature_out.size(0), -1)
                y_pred = F.log_softmax( self.model_P_buffer(feature_out) )
                loss_k_1  = F.nll_loss(y_pred, target)
            else:
                x_PN = torch.cat((sample_e['x'], sample_c['x']), dim = 0).float().to(device)
                y_PN = torch.cat((sample_e['y'], sample_c['y']), dim = 0).to(device)
                feature_out = self.model_F_buffer( x_PN)
                y_pred = self.model_P_buffer(feature_out)
                loss_k_1   = torch.nn.MSELoss()(y_pred, y_PN.float() )

            if self.config['problem'] == 'classification':
                x_PN = torch.cat((sample_e['x'], sample_c['x']), dim = 0).float().to(device)
                y_PN = torch.cat((sample_e['y'], sample_c['y']), dim = 0).to(device)
                target = y_PN.reshape([-1]).long()
                feature_out = self.model_F(x_PN)
                feature_out = feature_out.reshape(feature_out.size(0), -1)
                y_pred = F.log_softmax( self.model_P(feature_out) )
                loss_k = F.nll_loss(y_pred, target)
            else:
                x_PN = torch.cat((sample_e['x'], sample_c['x']), dim = 0).float().to(device)
                y_PN = torch.cat((sample_e['y'], sample_c['y']), dim = 0).to(device)
                feature_out = self.model_F( x_PN)
                y_pred = self.model_P(feature_out)
                loss_k  = torch.nn.MSELoss()(y_pred, y_PN.float() )
            #########################################################################################
            self.optimizer.zero_grad()
            
            
            Total_L_TH = self.config['gamma']*(loss_k_1 - loss_k) 
            ## Final calculation of the loss and the parameter updates
            Total_Loss =self.config['beta']*(J_P + J_N) + Total_L_TH + x_Loss
            dat_J.append((J_P+J_N).item() )
            dat_theta_loss.append((loss_k_1 - loss_k).item())
            dat_x_loss.append(x_Loss.item())
            
            # print("epoch", epoch, Total_Loss.item(), (J_P+J_N).item(), (loss_k_1 - loss_k).item(), x_Loss.item())

            Total_Loss.backward()
            self.optimizer.step()
        
        # x=input("This is the input")
        return self, dat_theta_loss, dat_x_loss, dat_J


###########################################################################
    def learn_MER(self, dataloader_curr, dataloader_exp, test_loader_ap, samp_num, phase = None):
        exp_it  = iter(dataloader_exp)
        curr_it = iter(dataloader_curr)
        before_P = deepcopy(self.model_P.state_dict())
        before_F = deepcopy(self.model_F.state_dict())
        for epoch in range(self.config['N_grad']):
            weights_before_P = deepcopy(self.model_P.state_dict())
            weights_before_F = deepcopy(self.model_F.state_dict())
            for epoch in range(self.config['N_meta']):
                ## Generalization to new task
                try:
                    sample_c = next(curr_it)
                except StopIteration:
                    # StopIteration is thrown if dataset ends
                    # reinitialize data loader
                    curr_it = iter(dataloader_curr)
                    sample_c  = next(curr_it)

                ## Compensate for forgetting.
                try:
                    sample_e = next(exp_it)
                except StopIteration:
                    # StopIteration is thrown if dataset ends
                    # reinitialize data loader
                    exp_it = iter(dataloader_exp)
                    sample_e  = next(exp_it)

                # Compensate for the third term
                #####################################################
                ### Compensate for the Third term
                if self.config['problem'] == 'classification':
                    x_PN = torch.cat((sample_e['x'], sample_c['x']), dim = 0).float().to(device)
                    y_PN = torch.cat((sample_e['y'], sample_c['y']), dim = 0).to(device)
                    target = y_PN.reshape([-1]).long()
                    feature_out = self.model_F(x_PN)
                    feature_out = feature_out.reshape(feature_out.size(0), -1)
                else:
                    x_PN = torch.cat((sample_e['x'], sample_c['x']), dim = 0).float().to(device)
                    y_PN = torch.cat((sample_e['y'], sample_c['y']), dim = 0).to(device)
                    feature_out = self.model_F( x_PN)

                self.optimizer.zero_grad()
                if self.config['problem'] == 'classification':
                    y_pred    = F.log_softmax( self.model_P(feature_out) )
                    loss_BUF  = F.nll_loss(y_pred, target)
                else:
                    y_pred = self.model_P(feature_out)
                    loss_BUF  = torch.nn.MSELoss()(y_pred, y_PN.float() )
                loss_BUF.backward(create_graph = True)
                self.optimizer.step()

            weights_after_P = deepcopy(self.model_P.state_dict())
            weights_after_F = deepcopy(self.model_F.state_dict())

            # Within batch Reptile meta-update:
            self.model_P.load_state_dict({name : weights_after_P[name] + ((weights_after_P[name]\
                    - weights_before_P[name]) * self.config['beta']) for name in weights_before_P})
            self.model_F.load_state_dict({name : weights_after_F[name] + ((weights_after_F[name]\
                    - weights_before_F[name]) * self.config['beta']) for name in weights_before_F})

        # print("I got through the first set of updates")
        after_P = deepcopy(self.model_P.state_dict())
        after_F = deepcopy(self.model_F.state_dict())

        # Across batch Reptile meta-update:
        self.model_P.load_state_dict({name : before_P[name] + ((after_P[name] -\
                before_P[name]) * self.config['gamma']) for name in before_P})
        self.model_F.load_state_dict({name : before_F[name] + ((after_F[name] -\
                before_F[name]) * self.config['gamma']) for name in before_F})
        return self



###########################################################################
    def backward(self, dataloader_curr, dataloader_exp, test_loader_ap, samp_num, phase = None):
        import copy
        if self.config['opt'] == 'NASH':
            return self.learn_NASH(dataloader_curr, dataloader_exp,\
                 test_loader_ap, samp_num, phase) 
        elif self.config['opt']=='MER':
            return self.learn_MER(dataloader_curr, dataloader_exp,\
                 test_loader_ap, samp_num, phase)
        elif self.config['opt'] == 'ER':
            return self.ER(dataloader_exp)
        elif self.config['opt'] == 'NAIVE':
            return self.ER(dataloader_curr)
        else:
            print("Optimizer not available")
