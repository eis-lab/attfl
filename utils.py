import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from scipy.interpolate import make_interp_spline
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

def dataset_noniid(DATASETS, NUM_USERS, SHARDS):
    NUM_SHARDS, NUM_IMAGES = int(NUM_USERS * SHARDS), (len(DATASETS)//(NUM_USERS * SHARDS)) # 40, 1250
    IDX_SHARD              = [i for i in range(NUM_SHARDS)]
    DIC_USERS              = {i: np.array([]) for i in range(NUM_USERS)}
    IDXS                   = np.arange(NUM_SHARDS * NUM_IMAGES)
    LABELS                 = np.array(DATASETS.targets)
    
    IDXS_LABELS = np.vstack((IDXS, LABELS))
    IDXS_LABELS = IDXS_LABELS[:, IDXS_LABELS[1, :].argsort()]
    IDXS        = IDXS_LABELS[0, :]

    for i in range(NUM_USERS):
        rand_set  = set(np.random.choice(IDX_SHARD, SHARDS, replace=False))
        IDX_SHARD = list(set(IDX_SHARD) - rand_set)
        
        for rand in rand_set:
            DIC_USERS[i] = np.concatenate((DIC_USERS[i], IDXS[rand * NUM_IMAGES:(rand + 1) * NUM_IMAGES]), axis=0)
            
    return DIC_USERS

def get_dataset_mnist(NUM_USERS, DIR, SHARDS):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
    ])

    train_dataset = datasets.MNIST(DIR, train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(DIR, train=False, download=True, transform=transform)

    user_group = dataset_noniid(train_dataset, NUM_USERS, SHARDS)
    
    return train_dataset, test_dataset, user_group

def get_dataset_cifar10(NUM_USERS, DIR, SHARDS):
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = datasets.CIFAR10(root=DIR, train=True, download=True, transform=transform)
    test_dataset  = datasets.CIFAR10(root=DIR, train=False, download=True, transform=transform)
    user_group    = dataset_noniid(train_dataset, NUM_USERS, SHARDS)
    return train_dataset, test_dataset, user_group

class DatasetProvider(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs    = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        data, label = self.dataset[self.idxs[item]]
        return torch.tensor(data), torch.tensor(label)

def model_size_calculator(model):
    param_size = 0
    for key in model.state_dict().keys():
        if any(sub in key for sub in ['Local', 'SubGlobal', 'Global', 'conn']):
            if 'weight' in key:
                a = np.shape(model.state_dict()[key])
                param_size += np.prod(a)
                
    param_size *= 4
    ours = param_size / 1024**2
    
    param_size = 0
    for key in model.state_dict().keys():
        if 'weight' in key:
            a = np.shape(model.state_dict()[key])
            param_size += np.prod(a)
    
    param_size *= 4
    origin = param_size / 1024**2
    
    print('\n*** The size of params in the baseline DNN model (i.e., communication cost): {:.3f}MB'.format(origin-ours))
    print('*** The size of params in our modules (i.e., communication cost)           : {:.3f}MB'.format(ours))
    print('*** Communication efficiency {:.2f} times better'.format(origin/ours))
    
def TB_CLI_logger(local_acc_test_avg, local_acc_test_avg_global, local_losses_avg, local_train_T_avg, PRINT_LOG_INTERVAL, TB_WRITER, MODEL_NAME, round_idx, TB_log_flag):
    
    if TB_log_flag:
        TB_WRITER.add_scalar(MODEL_NAME + ': Local Accuracy Test' , local_acc_test_avg, round_idx+1)
        TB_WRITER.add_scalar(MODEL_NAME + ': Local Accuracy Test Global' , local_acc_test_avg_global, round_idx+1)
        TB_WRITER.add_scalar(MODEL_NAME + ': Local Loss'  , local_losses_avg, round_idx+1)
        TB_WRITER.add_scalar(MODEL_NAME + ': Local Train T ' , local_train_T_avg, round_idx+1)
        TB_WRITER.close()
    
            
    ### Print log each 'N'-epoch
    if (round_idx+1) % PRINT_LOG_INTERVAL == 0:
        print('\n--| Avg Training Stats after ', round_idx,'global rounds:  |--')
        print('Local Training Loss                 : {:.6f}'.format(local_losses_avg))
        print('Local Training Accuracy Test        : {:.2f}%'.format(100*local_acc_test_avg))
        print('Local Training Time                 : {:.8f}\n'.format(local_train_T_avg))
        
def result_plot(train_loss, train_acc_test_local, train_acc_test_global):
    spline = make_interp_spline(np.arange(0, 50, 1), train_loss)
    X_ = np.linspace(0, 50, 200)
    train_loss_sp = spline(X_)
    
    plt.rcParams["figure.figsize"] = (13, 5)
    plt.rcParams["axes.titlesize"] = 10
    plt.subplot(1, 2, 1)
    plt.title('Loss', fontsize=25)
    plt.xlabel('Round', fontsize=25)
    plt.ylabel('Value', fontsize=25)
    plt.xticks(np.arange(0, 51, 5), fontsize=20)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=20)
    plt.xlim(0, 50)
    plt.ylim(0, 1)
    plt.plot(train_loss_sp, label='Loss')
    plt.legend(loc='upper right', fontsize=15)

    spline = make_interp_spline(np.arange(0, 50, 1), train_acc_test_local)
    X_ = np.linspace(0, 50, 100)
    train_acc_test_local_sp = spline(X_)
    
    spline = make_interp_spline(np.arange(0, 50, 1), train_acc_test_global)
    X_ = np.linspace(0, 50, 100)
    train_acc_test_global_sp = spline(X_)
    plt.subplot(1, 2, 2)
    plt.title('Local Accuracy', fontsize=25)
    plt.xlabel('Round', fontsize=25)
    plt.ylabel('Accuracy (%)', fontsize=25)
    plt.xticks(np.arange(0, 51, 5), fontsize=15)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=15)
    plt.xlim(0, 50)
    plt.ylim(0, 1)
    plt.plot(train_acc_test_global_sp, label='Global')
    plt.plot(train_acc_test_local_sp, label='Local')
    plt.legend(loc='upper right', fontsize=15)
    plt.show()
    