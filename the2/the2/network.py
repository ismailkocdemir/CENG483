import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from copy import deepcopy
import math


class FCNetwork(nn.Module):
    def __init__(self, input_size=512, output_size=1, hidden_sizes=[512]):
        super(FCNetwork, self).__init__()
        self.layers = nn.ModuleList()
        all_layers = [input_size] + hidden_sizes
        for i in range(len(all_layers)-1):
            self.layers.append(nn.Linear(all_layers[i], all_layers[i+1]) )
            self.layers.append(nn.ReLU())
        
        self.layers.append(nn.Linear(all_layers[len(all_layers)-1], output_size) )
    
    def forward(self,x):
        out = x.clone()
        for layer in self.layers:
            out = layer(out)
        return out

    
class Trainer():
    def __init__(self, train_path="train.npy", val_path="valid.npy",\
                 train_gt_path="train_gt.npy", val_gt_path="valid_gt.npy"):
        
        self.train_data = torch.utils.data.TensorDataset(torch.Tensor(np.load(train_path)),\
                                         torch.Tensor(np.load(train_gt_path)))
        self.val_data = torch.utils.data.TensorDataset( torch.Tensor( np.load(val_path)), \
                                        torch.Tensor(np.load(val_gt_path)))
       
    def train(self, model, batch_size=32, num_epochs=300, verbose=True,\
              lr=0.000001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0 ):
    
        # Some initializations
        train_loss_history = []
        val_loss_history = []
        best_model = None
        patience_acc = 0.0
        patience_loss = math.inf
        early_stop_patience = 30
        early_stop_improve = False
        last_epoch = num_epochs
        
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01,\
                                             alpha=0.99, eps=1e-08, weight_decay=0, momentum=0)  
        loss_function = nn.MSELoss()
        
        
        train_loader = DataLoader( self.train_data, batch_size=batch_size, shuffle=True)
        train_loader_history = DataLoader( self.train_data, batch_size=len(self.train_data), shuffle=True)
        val_loader = DataLoader( self.val_data, batch_size=batch_size, shuffle=True)
        
        # Start training.
        for epoch in range(num_epochs):
            for i, (features, labels) in enumerate(train_loader):  
                
                features = Variable(features)
                labels = Variable(labels.type(torch.FloatTensor)).view(-1)
                
                
                optimizer.zero_grad()  
                
                # One forward pass.
                outputs = model(features)
                train_loss = loss_function(outputs, labels)
                #train_loss_history.append(train_loss.data[0])
                
                # One backward pass and parameter updates.
                train_loss.backward()
                optimizer.step()
                
                # Calculate loss on val_data.
                val_loss = 0.0
                _size = 0
                for j, (val_features, val_labels) in enumerate(val_loader):

                    val_features = Variable(val_features)
                    val_labels = Variable(val_labels.type(torch.FloatTensor)).view(-1)

                    val_out =  model(val_features)
                    val_loss += loss_function(val_out, val_labels).data[0]
                    _size += 1 
                    if j*batch_size > 500:
                        break
                
                val_loss /= _size
                #val_loss_history.append(val_loss)

                if i == len(self.train_data)//batch_size and verbose==True:
                    print ('Epoch [%d/%d], Loss: %.4f, Val Loss: %.4f' 
                           %(epoch+1, num_epochs, \
                             train_loss.data[0], val_loss ))
            
            for j, (_features, _labels) in enumerate(train_loader_history):
                _features = Variable(_features)
                _labels = Variable(_labels.type(torch.FloatTensor)).view(-1)

                _out = model(_features)
                _loss = loss_function(_out, _labels)
                train_loss_history.append(_loss.data[0])

            
            val_acc, val_loss_mean, _ = self.evaluate(model)
            val_loss_history.append(val_loss_mean)
            if val_acc > patience_acc or val_loss_mean < patience_loss:
                if patience_acc < val_acc:
                    if verbose:
                        print("improved accuracy")
                    patience_acc = val_acc
                    best_model = deepcopy(model)
                if patience_loss > val_loss_mean:
                    if verbose:
                        print("improved val loss")
                    patience_loss = val_loss_mean
                early_stop_patience = 30
            
            else:
                early_stop_patience -= 1
            
            if early_stop_patience == 0:
                print("Early stopping at epoch:", epoch+1)
                last_epoch = epoch+1
                break
            
        print("Training finished. Best val acc:", patience_acc)
        '''
        if not long_history:
            step = int(len(train_loss_history)/ min(num_epochs,last_epoch) )
            train_loss_history = train_loss_history[::step]
            val_loss_history = val_loss_history[::step]
        '''
        return best_model, patience_acc, train_loss_history, val_loss_history
        
    def evaluate(self, model, test_path=None):
        data = None
        
        if test_path:
            test_data = np.load(test_path)
            data = torch.utils.data.TensorDataset( torch.Tensor(test_data),\
                                        torch.Tensor(np.ones(shape=test_data.shape))) 
        else:
            data = self.val_data
        
        size = len(data)
        loader = DataLoader( data, batch_size=size, shuffle=False)
        
        if test_path:
            for j, (_features, _labels) in enumerate(loader):
                _features = Variable(_features)
                _out = model(_features)
            return _out.data.numpy()
        
        val_loss = 0.0
        loss_function =  nn.L1Loss(reduce=False)
        loss_function2 = nn.MSELoss()
        for j, (val_features, val_labels) in enumerate(loader):
            
            val_features = Variable(val_features)
            val_labels = Variable(val_labels.type(torch.FloatTensor)).view(-1)
            
            val_out = model(val_features)
            _loss = loss_function(val_out, val_labels)
            mse_loss = loss_function2(val_out, val_labels)
            for row in _loss:
                if row.data[0] > 10.0:
                    val_loss += 1
        val_loss /= size
        return 1.0-val_loss, mse_loss.data[0] ,val_out.data.numpy()
        
if __name__ == "__main__":
    model = FCNetwork()
    trainer = Trainer()
    best_model, best_val_err, train_hist, val_hist = trainer.train(model)
    
