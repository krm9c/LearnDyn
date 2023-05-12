# python packages
from types import SimpleNamespace
import numpy as np
import torch
# The main code for the problem.
# from sklearn.metrics import r2_score, mean_squared_error
import subprocess
from core.model import *
from core.dataloaders import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

################################################
# Sanity Check  and initialize the CPU/GPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")  
    # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

################################################
# Main model running
class Run_Model():
    def __init__(self, Config):
        self.model = Net(Config).float()
        self.model.to(device)
        self.data = data_return(Config)
        self.config = Config

    ################################################
    def run_model_once(self):
        RA = np.zeros([self.config['total_samples']])
        LA = np.zeros([self.config['total_samples']])
        TA = np.zeros([self.config['total_samples'], self.config['total_samples']]) 

        for samp_n in range(self.config['total_samples']):
            self.model.model_F.train()
            self.model.model_P.train()

            # If ER, we will append the new task data 
            # to the experience array
            if self.config['opt'] == 'ER':

                # Create the datasets
                # Training set
                dataloader_curr, dataloader_exp = self.data.generate_dataset(
                    task_id =samp_n,
                    batch_size =self.config['batch_size'], 
                    phase ='training')   

                # Testing set
                test_loader_curr, test_loader =self.data.generate_dataset(
                    task_id =samp_n,
                    batch_size =self.config['batch_size'],
                    phase ='testing')


                print("I have to append")
                self.data.append_to_experience(task_id = samp_n)
                dataloader_curr, dataloader_exp = self.data.generate_dataset(
                    task_id = samp_n,
                    batch_size= 64,
                    phase = 'training')
            else:
                # Create the datasets
                # Training set
                dataloader_curr, dataloader_exp = self.data.generate_dataset(
                    task_id =samp_n,
                    batch_size =self.config['batch_size'], 
                    phase ='training')   

                # Testing set
                test_loader_curr, test_loader =self.data.generate_dataset(
                    task_id =samp_n,
                    batch_size =self.config['batch_size'],
                    phase ='testing')

            ## Main training routine
            self.model, dat_theta_loss, dat_x_loss, dat_J = self.model.backward(dataloader_curr,
             dataloader_exp, 
             test_loader, 
             samp_num = samp_n)
            
            self.model.scheduler.step()
            self.config['kappa'] = self.config['kappa']+int(self.config['kappa']*0.01)

            ## Main Evaluation Routines
            with torch.no_grad():

                # print("regular accuracies")
                LA[samp_n], RA[samp_n] = self.model.return_score(test_loader_curr, test_loader)

                # print("task_wise")
                TA[samp_n,:] = self.model.task_wise_Accuracy(self.data, self.config['total_samples'])
                
                # Printing data
                print('Sample_number {}/{}'.format(samp_n, self.config['total_samples']-1),
                "Retained Accuracy", RA[samp_n], " Learned Accuracy", LA[samp_n])

            if self.config['opt'] != 'ER': 
              self.data.append_to_experience(task_id= samp_n)

            print(np.array(dat_theta_loss).reshape([-1,1]).shape,\
                np.array(dat_x_loss).reshape([-1,1]).shape,\
                np.array(dat_J).reshape([-1,1]).shape)
            
            np.savetxt('/home/kraghavan/Projects/CL/NashMCL/Balance_test/theta_loss'+str(samp_n)+'.csv', np.concatenate(\
                                                        [np.array(dat_theta_loss).reshape([-1,1]),\
                                                        np.array(dat_x_loss).reshape([-1,1]),\
                                                        np.array(dat_J).reshape([-1,1])], axis=1 ), delimiter=',')    
        
        return LA, RA, TA



################################################
# Record the gpu memory things and evaluate
# 
class train_record():
    def __init__(self, Config):
        print("__initialized__")
        self.config = Config

    def print_gpu_obj(self):
        import gc
        count = 0
        for tracked_object in gc.get_objects():
            if torch.is_tensor(tracked_object):
                count+=1
                print("{} {} {}".format(
                    type(tracked_object).__name__,
                "GPU" if tracked_object.is_cuda else "" ,
                "pinned" if tracked_object.is_pinned() else "",
        ))

    def get_gpu_memory_map(self):
        """
        Get the current gpu usage.
        Returns
        -------
        usage: dict
            Keys are device ids as integers.
            Values are memory usage as integers in MB.
        """
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used',
                '--format=csv,nounits,noheader'
            ], encoding='utf-8')
        
        # Convert lines into a dictionary
        gpu_memory = [int(x) for x in result.strip().split('\n')]
        gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
        return gpu_memory_map


    def show_gpu(self, msg):
        """
        ref: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
        """
        def query(field):
            return(subprocess.check_output(
                ['nvidia-smi', f'--query-gpu={field}',
                    '--format=csv,nounits,noheader'], 
                encoding='utf-8'))
        def to_int(result):
            return int(result.strip().split('\n')[0])
        
        used = to_int(query('memory.used'))
        total = to_int(query('memory.total'))
        pct = used/total
        print('\n' + msg, f'{100*pct:2.1f}% ({used} out of {total})')    

    # The function comes here from py_run.
    def main(self):
        One_M = Run_Model(self.config)
        # with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        LA, RA, TA = One_M.run_model_once()

        # print(self.get_gpu_memory_map())
        # self.show_gpu(f'{0}: Before deleting objects')
        # self.show_gpu(f'{0}: After deleting objects')
        # gc.collect()
        # self.show_gpu(f'{0}: After gc collect')
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()
        # self.show_gpu(f'{0}: After empty cache')
        # self.show_gpu('after all stuff have been removed')
        # self.print_gpu_obj()
        return LA, RA, TA