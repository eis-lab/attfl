import time
import numpy as np
from utils import *
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

class LocalUpdate(object):
    def __init__(self, train_dataset, test_dataset, idxs_train, tb_writer, batch_size, device):
        
        self.tb_writer  = tb_writer
        self.device     = device
        self.criterion  = nn.NLLLoss().to(self.device)
        self.batch_size = batch_size
        
        self.local_train_loader, self.local_test_loader, self.global_test_loader = self.local_train_val_test(train_dataset, test_dataset, list(idxs_train))
        
    def local_train_val_test(self, train_dataset, test_dataset, idxs_train):
        idxs_train_label                        = np.array(train_dataset.targets)[np.array(idxs_train, dtype=int)]
        idxs_local_train, idxs_local_test, _, _ = train_test_split(idxs_train, idxs_train_label, test_size=0.2, stratify=idxs_train_label)
        
        local_train_loader = DataLoader(DatasetProvider(train_dataset, idxs_local_train), batch_size=self.batch_size, shuffle=True, drop_last=True)
        local_test_loader  = DataLoader(DatasetProvider(train_dataset, idxs_local_test), batch_size=self.batch_size, shuffle=False, drop_last=True)
        global_test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        return local_train_loader, local_test_loader, global_test_loader
    
    def update_weight(self, local_model, global_epoch, optimizer, lr, l2, local_epochs, print_log):
        local_model.train()
        epoch_loss = []
        epoch_acc_test, epoch_acc_test_global = [], []
        
        local_saf_list, subglobal_saf_list, global_saf_list = [], [], []
        local_saf, subglobal_saf, global_saf = None, None, None
        scaler = MinMaxScaler()
        
        training_T = []
        
        if optimizer == 'sgd':
            optimizer = optim.SGD(local_model.parameters(), lr=lr, momentum=0.5, weight_decay=l2)
        elif optimizer == 'adam':
            optimizer = optim.Adam(local_model.parameters(), lr=lr, weight_decay=l2)
        
        for local_epoch in range(local_epochs):
            local_model.train()
            
            batch_loss = []
            
            st_training_time = time.time()
            for batch_idx, (data, label) in enumerate(self.local_train_loader):
                local_model.zero_grad()
                
                data, label                                  = data.to(self.device), label.to(self.device)
                output, local_saf, subglobal_saf, global_saf = local_model(data)
                loss                                         = self.criterion(output, label)
                
                batch_loss.append(loss.item())
                loss.backward()
                optimizer.step()
                
                local_saf_list.append(local_saf.cpu().detach().numpy())
                subglobal_saf_list.append(subglobal_saf.cpu().detach().numpy())
                global_saf_list.append(global_saf.cpu().detach().numpy())
                
                if print_log and (batch_idx % 1 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_epoch, local_epoch, batch_idx * len(data),
                        len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader), 
                        loss.item()))
                    
            ed_training_time = time.time()
            training_T.append(ed_training_time - st_training_time)
            
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
                    
            acc = self.local_inference(local_model, 'test')
            epoch_acc_test.append(acc)
            acc = self.local_inference(local_model, 'test_global')
            epoch_acc_test_global.append(acc)
            
            return local_model.state_dict(), sum(epoch_loss)     / len(epoch_loss), \
                                              sum(epoch_acc_test) / len(epoch_acc_test), \
                                              sum(epoch_acc_test_global) / len(epoch_acc_test_global), \
                                              sum(training_T)            / len(training_T), \
                                              scaler.fit_transform(np.average(local_saf_list, axis=0)), \
                                              scaler.fit_transform(np.average(subglobal_saf_list, axis=0)), \
                                              scaler.fit_transform(np.average(global_saf_list, axis=0))
        
    
    def local_inference(self, local_model, infer_type):
        local_model.eval()
        
        infer_loader = None
        if infer_type == "test":
            infer_loader = self.local_test_loader
        elif infer_type == "test_global":
            infer_loader = self.global_test_loader
        
        total_loss, total_len, total_correct_sum, accuracy = 0.0, 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(infer_loader):
                data, label = data.to(self.device), label.to(self.device)

                output, _, _, _ = local_model(data)
                loss            = self.criterion(output, label)
                total_loss      += loss.item()

                _, pred = torch.max(output, 1)
                pred    = pred.view(-1)

                total_correct_sum += torch.sum(torch.eq(pred, label)).item()
                total_len         += len(label)

            accuracy = total_correct_sum / total_len
        return accuracy
