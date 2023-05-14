import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM(nn.Module):
    def __init__(self, time_step, input_size, hidden_unit, batch_size, dname, lstm_num_layer=1, num_classes=10, init_weights=True):
        super().__init__()
        self.batch_size = batch_size
        self.dname      = dname
        self.time_step  = time_step
        self.input_size = input_size
        
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        '''                         Baseline DNN model                             '''
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        self.fwlstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_unit, num_layers=lstm_num_layer, batch_first=True)
        self.bwlstm2 = nn.LSTM(input_size=input_size, hidden_size=hidden_unit, num_layers=lstm_num_layer, batch_first=True)
        self.fc1   = nn.Linear(time_step*hidden_unit*4, 1000)
        self.fc2   = nn.Linear(1000, 500)
        self.fc3   = nn.Linear(500, num_classes)
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        
        
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        '''                      SA mechanism-based modules                        '''
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        self.Local_SA_Module = nn.Sequential(
            nn.Conv1d(in_channels=hidden_unit, out_channels=input_size, kernel_size=1, stride=1, padding=0),
            nn.Conv1d(in_channels=time_step,   out_channels=input_size, kernel_size=1, stride=1, padding=0),
            nn.Conv1d(in_channels=time_step,   out_channels=input_size, kernel_size=1, stride=1, padding=0)
        )
        
        self.SubGlobal_SA_Module = nn.Sequential(
            nn.Conv1d(in_channels=hidden_unit, out_channels=input_size, kernel_size=1, stride=1, padding=0),
            nn.Conv1d(in_channels=time_step,   out_channels=input_size, kernel_size=1, stride=1, padding=0),
            nn.Conv1d(in_channels=time_step,   out_channels=input_size, kernel_size=1, stride=1, padding=0)
        )
        
        self.Global_SA_Module = nn.Sequential(
            nn.Conv1d(in_channels=hidden_unit, out_channels=input_size, kernel_size=1, stride=1, padding=0),
            nn.Conv1d(in_channels=time_step,   out_channels=input_size, kernel_size=1, stride=1, padding=0),
            nn.Conv1d(in_channels=time_step,   out_channels=input_size, kernel_size=1, stride=1, padding=0)
        )
        
        self.conn_local     = nn.Conv1d(in_channels=input_size*2, out_channels=input_size, kernel_size=1, stride=1, padding=0)
        self.conn_subglobal = nn.Conv1d(in_channels=input_size, out_channels=hidden_unit, kernel_size=1, stride=1, padding=0)
        self.conn_global    = nn.Conv1d(in_channels=input_size, out_channels=hidden_unit, kernel_size=1, stride=1, padding=0)
        
    def adaptive_SA_Module(self, Q, K, V, conv_layer):
        Q = conv_layer[0](Q.permute(0, 2, 1)).permute(0, 2, 1)
        K = conv_layer[1](K)
        V = conv_layer[2](V)
        
        QK                = torch.bmm(Q, K) * (1.0 / np.sqrt(K.size()[1]))
        QK_size           = QK.size()
        Soft_SA_Weight_QK = F.softmax(QK, dim=1)
        
        SA_FeatureMap = torch.bmm(Soft_SA_Weight_QK, V)
        SA_output     = SA_FeatureMap + Q
        
        return SA_output, F.softmax(SA_FeatureMap.squeeze(1), dim=1)
        
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        
    def forward(self, x):
        if self.dname == 'MNIST':
            x = x.squeeze(1)
        else:
            x = x.view(x.size(0), self.time_step, self.input_size)
        
        lstm1_out, _        = self.fwlstm1(x)
        local_out, local_FM = self.adaptive_SA_Module(lstm1_out, x, x, self.Local_SA_Module)
        
        inv_local_out = torch.flip(local_out, [0, 1])
        inv_x         = torch.flip(x, [0, 1])
        inv_local_out = torch.cat([inv_x, inv_local_out], 2)
        inv_local_out = self.conn_local(inv_local_out.permute(0, 2, 1)).permute(0, 2, 1).squeeze(1)
        
        lstm2_out, _                = self.bwlstm2(inv_local_out)
        subglobal_out, subglobal_FM = self.adaptive_SA_Module(lstm2_out, local_out, local_out, self.SubGlobal_SA_Module)
        subglobal_out               = self.conn_subglobal(subglobal_out.permute(0, 2, 1)).permute(0, 2, 1)
        
        global_out, global_FM = self.adaptive_SA_Module(lstm2_out, x, x, self.Global_SA_Module)
        global_out            = self.conn_global(global_out.permute(0, 2, 1)).permute(0, 2, 1)
        
        bilstm_out = torch.cat([lstm1_out, lstm2_out, subglobal_out, global_out], dim=1)
        bilstm_out = bilstm_out.view(bilstm_out.size(0), bilstm_out.size(1) * bilstm_out.size(2))
        
        fc1_o = self.fc1(bilstm_out)
        fc2_o = self.fc2(fc1_o)
        fc3_o = self.fc3(fc2_o)
        
        return F.log_softmax(fc3_o, dim=1), local_FM.view(self.batch_size,-1), \
                subglobal_FM.view(self.batch_size, -1), global_FM.view(self.batch_size, -1)