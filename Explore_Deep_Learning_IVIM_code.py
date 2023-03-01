import time
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import nibabel as nib
from IPython import get_ipython
from pathlib import Path
get_ipython().run_line_magic('matplotlib', 'qt') #Inline,qt
import pickle
import random
import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#%% pytorch init
import torch

#CPU or GPU
dev = 'cpu'
if dev == 'gpu':
    device = torch.device("cuda:0")
    print('Device: {}'.format(torch.cuda.get_device_name(device.index)))
    torch.backends.cudnn.benchmark = True
elif dev == 'cpu':
    print('cpu')
    device = torch.device('cpu')
    torch.set_num_threads(1) # 1 cpu core

#%% hyperparameters
# Experiment parameters
snr = 200 # SNR is relative to unit-intensity signal, depends on normalization!
b = [  0,  10,  20,  40,  80, 110, 140, 170, 200, 300, 400, 500, 600, 700, 800, 900]
b_torch = torch.Tensor(b).to(device)[None,:,None,None]

# Network parameters
hidden_channels = 64 # depth of the neural network
out_channels = 4 # 4 IVIM parameters (S0, D, f, D*) output NN 

# Training parameters
n_epochs = 5000
batches_per_epoch = 500
learning_rate = 0.0001
criterion = torch.nn.MSELoss()
training_strategies = ['supervised', 'unsupervised']
bounds = np.array(([0, 0, 0, 3e-3], [1,3e-3, 0.5, 100e-3])) # Also scaling params
patience = 10 # Patience for Early stopping
Batch_size=128 # For training
Batch_size_val = 100000
patch = (1,1) # Patch size for uniform training is single voxel
network = 'MLP_3layer'
optmizer = 'Adam'

#%% Functions
# IVIM model
def IVIM_model(S0, Dt, Fp, Dp, bvalues):
    return (S0 * (Fp * torch.exp(-bvalues * Dp) + (1 - Fp) * torch.exp(-bvalues * Dt)))

def descale_params(paramnorm, Lower_bound, Upper_bound):
    a1 = 0
    b1 = 1
    return ((((paramnorm - a1) / (b1 - a1)) ) * (Upper_bound - Lower_bound)) + Lower_bound

def scaling(param, Lower_bound, Upper_bound):
    a1 = 0
    b1 = 1
    return (b1 - a1) * ((param - Lower_bound) / (Upper_bound - Lower_bound)) + a1

def Descale_params(params, bounds):
    S0_descaled = descale_params(params[:,0],bounds[0,0], bounds[1,0])
    D_descaled = descale_params(params[:,1],bounds[0,1], bounds[1,1])
    F_descaled = descale_params(params[:,2],bounds[0,2], bounds[1,2])
    Dp_descaled = descale_params(params[:,3],bounds[0,3], bounds[1,3])
    params_descaled = torch.cat((S0_descaled[:,None], D_descaled[:,None], F_descaled[:,None], Dp_descaled[:,None]), axis=1)
    return params_descaled

def Scale_params(params, bounds):
    S0_scaled = scaling(params[:,0],bounds[0,0], bounds[1,0])
    D_scaled = scaling(params[:,1],bounds[0,1], bounds[1,1])
    F_scaled = scaling(params[:,2],bounds[0,2], bounds[1,2])
    Dp_scaled = scaling(params[:,3],bounds[0,3], bounds[1,3])
    params_scaled = torch.cat((S0_scaled[:,None], D_scaled[:,None], F_scaled[:,None], Dp_scaled[:,None]), axis=1)
    return params_scaled

def signal_params(params, snr, b_torch):
    # Create signal with IVIM model
    signal = IVIM_model(params[:,0][:,None], params[:,1][:,None], params[:,2][:,None], params[:,3][:,None], b_torch)            
    # Add complex-valued noise
    signal_noise = signal + 1/snr * (torch.randn(signal.shape, device=device) + 1j*torch.randn(signal.shape, device=device))
    return signal_noise
        
def calculate_losses(criterion, out_network, params_scaled, signal, signal_pred):
    losses = {}
    losses['signals'] = criterion(signal, signal_pred)
    for index,name in zip([0,1,2,3], ['S0', 'D', 'F', 'D*']):
        losses[name] = criterion(out_network[:,index].float(), params_scaled[:,index].float())
    losses['parameters'] = criterion(out_network.float(), params_scaled.float())
    return losses

def generator_uniform(Batch_size, patch, bounds):
    while True:
        # Generate IVIM params
        S0 = bounds[0,0] + (torch.rand((Batch_size, 1) + patch, device=device) * (bounds[1,0] - bounds[0,0]))
        Dt = bounds[0,1] + (torch.rand((Batch_size, 1) + patch, device=device) * (bounds[1,1] - bounds[0,1]))
        Fp = bounds[0,2] + (torch.rand((Batch_size, 1) + patch, device=device) * (bounds[1,2] - bounds[0,2]))
        Dp = bounds[0,3] + (torch.rand((Batch_size, 1) + patch, device=device) * (bounds[1,3] - bounds[0,3]))
        
        params = torch.cat((S0, Dt, Fp, Dp), axis=1)
        params_scaled = Scale_params(params, bounds)
        signal_noise = signal_params(params, snr, b_torch)
        signal_noise = signal_noise.to(device=device, dtype=torch.float32)
        yield abs(signal_noise), params_scaled, params
        
def params_array_to_list(out_network_descaled):    
    IVIM_param = {}
    IVIM_param['S0'] = out_network_descaled[:,0].flatten().cpu().detach().numpy()
    IVIM_param['D'] = out_network_descaled[:,1].flatten().cpu().detach().numpy()
    IVIM_param['F'] = out_network_descaled[:,2].flatten().cpu().detach().numpy()*100 # F in %
    IVIM_param['D*'] = out_network_descaled[:,3].flatten().cpu().detach().numpy()
    return(IVIM_param)

def MLP_3layer(b, hidden_channels, out_channels, torch):
    if activation_function_choise == 'ELU':
        activation_function = torch.nn.ELU()
    elif activation_function_choise == 'RELU':
        # RELU
        activation_function = torch.nn.RELU()
    net = torch.nn.Sequential(torch.nn.Conv2d(len(b), hidden_channels, 1), activation_function,
                                    torch.nn.Conv2d(hidden_channels, hidden_channels, 1), activation_function,
                                    torch.nn.Conv2d(hidden_channels, hidden_channels, 1), activation_function,
                                    torch.nn.Conv2d(hidden_channels, out_channels, 1)).to(device)
    return net

def make_PredvsTrue_plots(ground_truth, estimate, plot_title):
    print('Making predicted vs truth plot ' + plot_title)
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3,figsize=(15,5))
    fig.suptitle(plot_title, fontsize = 25)
    for i in ['D', 'F', 'D*']:
        x_ = ground_truth[i]
        y_ = estimate[i]
        if i == 'D':
            bound_x = [0,0.003]; bound_y = [0,0.004]
            ax1.scatter(x_,y_, s=0.5,alpha = 1, c = ground_truth['S0'], cmap='inferno')
            ax1.plot(bound_y,bound_y, linestyle = 'dashed',  linewidth = 3, color='black')
            ax1.set_xlim(bound_x); ax1.set_ylim(bound_y)
            ax1.set_xlabel('truth ' + i, fontsize = 15); ax1.set_ylabel('pred ' + i, fontsize = 15)
            ax1.set_title(i, fontsize = 20)
        elif i == 'F':
            bound_x = [0,50]; bound_y = [0,60]
            ax2.scatter(x_,y_, s=1.5,alpha = 1, c = ground_truth['S0'], cmap='inferno')
            ax2.plot(bound_y,bound_y, linestyle = 'dashed',  linewidth = 3, color='black')
            ax2.set_xlim(bound_x); ax2.set_ylim(bound_y)
            ax2.set_xlabel('truth ' + i, fontsize = 15); ax2.set_ylabel('pred ' + i, fontsize = 15)
            ax2.set_title(i, fontsize = 20)
        elif i == 'D*':
            bound_x = [0,0.1]; bound_y = [0,0.14]
            ax3.scatter(x_,y_, s=1.5,alpha = 1, c = ground_truth['S0'], cmap='inferno')
            ax3.plot(bound_y,bound_y, linestyle = 'dashed',  linewidth = 3, color='black')
            ax3.set_xlim(bound_x); ax3.set_ylim(bound_y)
            ax3.set_xlabel('truth ' + i, fontsize = 15); ax3.set_ylabel('pred ' + i, fontsize = 15)
            ax3.set_title(i, fontsize = 20)
    fig.tight_layout()
    if plot_title == 'Min(MSE-D*)':
        # Change name for saving figure
        plot_title = 'Min_MSE_Dp'
    plt.savefig('runs/' + run + '/Plots and Figures/' + plot_title + '.png', dpi=600, bbox_inches="tight", pad_inches=0.8)
    plt.close()

#%% Generator init and test distribution:
# Init seed network
seed_orig = torch.seed()
torch.manual_seed(123)
random.seed(123)
np.random.seed(123)

# generator initialize and test set
generator_distribution = 'uniform'
if generator_distribution == 'uniform': 
    gen = generator_uniform(Batch_size, patch, bounds)
    signal_noise_val, params_scaled_val, params_val = next(generator_uniform(Batch_size_val, patch, bounds))
    print(generator_distribution + ' test set loaded')
    ground_truth = params_array_to_list(params_val)

# Options network
activation_function_choise = 'ELU'

training_NN = True
Making_loss_curves = True

#%% Training:
if training_NN:
    for training_strategy in training_strategies:
        # Initiolize network and other parameters
        if network == 'MLP_3layer':
            net = MLP_3layer(b, hidden_channels, out_channels, torch)
        # defining optimizer
        if optmizer == 'Adam':
            opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
        if optmizer == 'SGD':
            opt = torch.optim.SGD(net.parameters(), lr=learning_rate)
        if optmizer == 'RMSProp':
            opt = torch.optim.RMSProp(net.parameters(), lr=learning_rate)
        values_training = {}
        start_time_epoch = time.time()
        loss_curve = []
        best_val = 1e16
        best_val_Dp = 1e16
        num_bad_epochs = 0
        badepoch = 0 # Init for patience epoch in Early stopping
        run = training_strategy + '_' + generator_distribution # Naming training
        print(run)
        Path('runs').mkdir(parents=True, exist_ok=True)
        Path('runs/' + run).mkdir(parents=True, exist_ok=True)
        Path('runs/' + run + '/Saved_networks').mkdir(parents=True, exist_ok=True)
        Path('runs/' + run + '/Plots and Figures').mkdir(parents=True, exist_ok=True)
        loss_print = open('runs/' + run + '/loss_curve.txt',"w+")
        for epoch in range(n_epochs):
            for batch in range(batches_per_epoch):
                start_time_batch = time.time()
                # Generate new data
                signal_noise_train, params_scaled_train, params_train = next(gen)
                ## Training
                opt.zero_grad()
                net.train()
                out_network_train = net(signal_noise_train)        
                # Absolute constrain output network after descaling
                train_params_descaled = abs(Descale_params(out_network_train, bounds))
                # rescaling to 0-1 values
                out_network_train_rescaled = Scale_params(train_params_descaled, bounds)
                # Estimated signal from estimated parameters
                signal_pred_train = IVIM_model(train_params_descaled[:,0][:,None], train_params_descaled[:,1][:,None], train_params_descaled[:,2][:,None], train_params_descaled[:,3][:,None], b_torch)
                # Calculate losses and errors between simulated (ground truth) and estimates
                losses_train = calculate_losses(criterion, out_network_train_rescaled, params_scaled_train, signal_noise_train, signal_pred_train)
                # Define loss based on training strategy
                if training_strategy == 'supervised':
                    # Supervised learning is optimized on the parameters
                    loss_train = losses_train['parameters'].float()
                elif training_strategy == 'unsupervised': 
                    # Unsupervised learning is optimized on the signals
                    loss_train = losses_train['signals'].float()
                loss_train.backward() # Upgrade gradients
                opt.step() #Optimization step
                
            # Test set evaluation
            net.eval()
            with torch.no_grad():
                out_network_val = net(signal_noise_val)
                # Absolute constrain output network after descaling
                val_params_descaled = abs(Descale_params(out_network_val, bounds))
                # rescaling to 0-1 values
                out_network_val_rescaled = Scale_params(val_params_descaled, bounds)
                # Estimated signal from estimated parameters
                signal_pred_val = IVIM_model(val_params_descaled[:,0][:,None], val_params_descaled[:,1][:,None], val_params_descaled[:,2][:,None], val_params_descaled[:,3][:,None], b_torch)
                # Calculate losses and errors between simulated (ground truth) and estimates
                losses_val = calculate_losses(criterion, out_network_val_rescaled, params_scaled_val, signal_noise_val, signal_pred_val)
                # Define parameters in list
                for name in losses_val:
                    if losses_val[name] == 0:
                        values_training[name] = 0
                    else:
                        values_training[name] = losses_val[name].item()
                # Define loss based on training strategy on the test set
                if training_strategy == 'supervised':
                    # Supervised learning is optimized on the parameters
                    loss_val = losses_val['parameters'].float().item()
                elif training_strategy == 'unsupervised': 
                    # Unsupervised learning is optimized on the signals
                    loss_val = losses_val['signals'].float().item()
            
                # Spearman correlation between f and D*
                values_training['Spearman(fvD*)'], p_F_Dp_val = scipy.stats.spearmanr(val_params_descaled[:,2].cpu().detach().numpy().flatten(), val_params_descaled[:,3].cpu().detach().numpy().flatten().flatten())
                
            # Early stopping criterion
            if loss_val < best_val:
                loss_val < best_val
                best_val = loss_val
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1
                if num_bad_epochs == patience:
                    badepoch += 1
                    if badepoch == 1:
                        print('Early stopping patience {patience} at epoch {epoch}'.format(patience=patience, epoch=epoch))
                        values_training['epoch_early_stopping'] = epoch
                        torch.save(net.state_dict(), 'runs/' + run + '/Saved_networks/Early_stopping.txt')
                        output_network = params_array_to_list(val_params_descaled)
                        make_PredvsTrue_plots(ground_truth, output_network, plot_title = 'Early stopping')
                   
            # Min(MSE-D*) 
            if values_training['D*'] < best_val_Dp:
                best_val_Dp = values_training['D*']
                torch.save(net.state_dict(), 'runs/' + run + '/Saved_networks/Min_MSE_Dp.txt')
                output_network_Min_MSE_Dp = params_array_to_list(val_params_descaled)
                values_training['epoch_Dp'] = epoch
    
            # append losses
            loss_curve.append(values_training.copy())
    
            # Save network at certain epochs   
            if epoch % 100 == 0:
                print('training {} epoch {} network {} time {:.3f} loss_val {} Unsup {} S0 {} D {} F {} D* {} Sup {}\n'.format(training_strategy, str(epoch), str(network), time.time() - start_time_epoch, loss_train, values_training['signals'], values_training['S0'], values_training['D'], values_training['F'], values_training['D*'], values_training['parameters']))
            if epoch % 10 == 0:
                # Store values in txt
                loss_print.writelines('training {} epoch {} network {} epoch time {:.3f} loss_val {} Unsup {} S0 {} D {} F {} D* {} Sup {}\n'.format(training_strategy, str(epoch), str(network), time.time() - start_time_epoch, loss_train, values_training['signals'], values_training['S0'], values_training['D'], values_training['F'], values_training['D*'], values_training['parameters']))
        
        #%% Making plots when training is finished:        
        # Min(MSE-D*)
        print('Min(MSE-D*) at epoch {epoch}'.format(patience=patience, epoch=values_training['epoch_Dp']))
        make_PredvsTrue_plots(ground_truth, output_network_Min_MSE_Dp, plot_title = 'Min(MSE-D*)')
        
        # Epoch 50000
        output_network = params_array_to_list(val_params_descaled)
        make_PredvsTrue_plots(ground_truth, output_network, plot_title = 'Epoch-50000')
        
        # Save network for evaluation
        torch.save(net.state_dict(), 'runs/' + run + '/Saved_networks/network_lastepoch.txt'.format(epoch=epoch))
        with open('runs/' + run + '/Saved_networks/pickle_losscurve_lastepoch.txt'.format(epoch=epoch), 'wb') as f:  # Python 3: open(..., 'rb')
                                                                   pickle.dump([loss_curve,epoch],f)
        del net
        loss_print.close()
     
#%% Make loss/MSE curves:
if Making_loss_curves:
    print('Making MSE loss plots and Spearman(fvD*) plots')
    Path('runs/loss_curves').mkdir(parents=True, exist_ok=True)
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3,figsize=(25,6))
    fig.suptitle('Effect of training on MSE-signals (unsupervised loss), MSE-parameters (supervised loss) and Spearmans Rho', fontsize = 25)
    for MSE in ['signals', 'parameters', 'Spearman(fvD*)']: # 
        for training_strategy in training_strategies: 
            run = training_strategy + '_' + generator_distribution # Naming training
            file_training = 'runs/' + run + '/Saved_networks/pickle_losscurve_lastepoch.txt'
            with open(file_training, 'rb') as f:  
                loss_file_training, epoch = pickle.load(f)
                MSE_loss_file_training = np.array([x[MSE] for x in loss_file_training])
                epochs_loss_file_training = list(range(0,MSE_loss_file_training.shape[0]))
                if MSE == 'signals':
                    ax1.plot(epochs_loss_file_training, MSE_loss_file_training, label = run)
                    ax1.set_title(MSE, fontsize = 20); ax1.set_ylabel('MSE-' + MSE, fontsize = 15); ax1.set_xlabel('Epoch', fontsize = 15)
                    ax1.set_ylim([1e-5,0.7e-3])
                    ax1.legend()        
                elif MSE == 'parameters':
                    ax2.plot(epochs_loss_file_training, MSE_loss_file_training, label = run)
                    ax2.set_title(MSE, fontsize = 20); ax2.set_ylabel('MSE-' + MSE, fontsize = 15); ax2.set_xlabel('Epoch', fontsize = 15)
                    ax2.set_ylim([5e-3,3e-2])
                    ax2.legend()        
                elif MSE == 'Spearman(fvD*)':
                    ax3.plot(epochs_loss_file_training, MSE_loss_file_training, label = run)
                    ax3.set_title(MSE, fontsize = 20); ax3.set_ylabel('MSE-' + MSE, fontsize = 15); ax3.set_xlabel('Epoch', fontsize = 15)
                    ax3.legend()        
    fig.tight_layout()
    plt.savefig('runs/loss_curves/MSE_signals_parameters_Spearman.png', dpi=600, bbox_inches="tight", pad_inches=0.8)
    plt.close()
    
    print('Making normalized MSE per parameter plot')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(15,10))
    fig.suptitle('Normalized MSE per parameter', fontsize = 25)
    for MSE in ['S0', 'D', 'F', 'D*']:
        for training_strategy in training_strategies: 
            run = training_strategy + '_' + generator_distribution # Naming training
            file_training = 'runs/' + run + '/Saved_networks/pickle_losscurve_lastepoch.txt'
            with open(file_training, 'rb') as f:  
                loss_file_training, epoch = pickle.load(f)
                MSE_loss_file_training = np.array([x[MSE] for x in loss_file_training])
                epochs_loss_file_training = list(range(0,MSE_loss_file_training.shape[0]))
                if MSE == 'S0':
                    ax1.plot(epochs_loss_file_training, MSE_loss_file_training, label = run)
                    ax1.set_title(MSE, fontsize = 20); ax1.set_ylabel('MSE-' + MSE, fontsize = 15); ax1.set_xlabel('Epoch', fontsize = 15)
                    ax1.set_ylim([1e-5,4e-5])
                    ax1.legend()        
                elif MSE == 'D':
                    ax2.plot(epochs_loss_file_training, MSE_loss_file_training, label = run)
                    ax2.set_title(MSE, fontsize = 20); ax2.set_ylabel('MSE-' + MSE, fontsize = 15); ax2.set_xlabel('Epoch', fontsize = 15)
                    ax2.set_ylim([0.1e-3,1.5e-2])
                    ax2.legend()        
                elif MSE == 'F':
                    ax3.plot(epochs_loss_file_training, MSE_loss_file_training, label = run)
                    ax3.set_title(MSE, fontsize = 20); ax3.set_ylabel('MSE-' + MSE, fontsize = 15); ax3.set_xlabel('Epoch', fontsize = 15)
                    ax3.set_ylim([5e-3,4e-2])
                    ax3.legend()        
                elif MSE == 'D*':
                    ax4.plot(epochs_loss_file_training, MSE_loss_file_training, label = run)
                    ax4.set_title(MSE, fontsize = 20); ax4.set_ylabel('MSE-' + MSE, fontsize = 15); ax4.set_xlabel('Epoch', fontsize = 15)
                    ax4.set_ylim([0.1e-1,1.8e-1])
                    ax4.legend()        
    fig.tight_layout()
    plt.savefig('runs/loss_curves/normalized MSE per parameter.png', dpi=600, bbox_inches="tight", pad_inches=0.8)
plt.close('all')