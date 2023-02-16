import torch
from torch import nn, optim
from tqdm import tqdm


class TrainUtils:
    
    def __init__(self, model, loss_fn: str, optimizer: str, learning_rate: float = 1e-3, device: str = None):
        
        super(TrainUtils, self).__init__()
        
        # Set model
        self.model = model
        
        # Set loss function
        if loss_fn not in ['CrossEntropyLoss']:
            raise ValueError('Loss function is not supported!')
        else:
            if loss_fn == 'CrossEntropyLoss':
                self.loss_fn = nn.CrossEntropyLoss()
        
        # Set optimizer
        if optimizer not in ['Adam', 'SGD']:
            raise ValueError('Optimizer is not supported!')
        else:
            if optimizer == 'Adam':
                self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            elif optimizer == 'SGD':
                self.optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        
        # Set device
        if device is not None:
            self.device = torch.device(device)
            print('Using GPU!')
            print('Device name:', torch.cuda.get_device_name(self.device), '\n')
        else:
            self.device = torch.device('cpu')
            print('Using CPU!\n')

    def train(self, dataloader, print_log: bool = False):
        
        model = self.model
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        device = self.device
        
        loss_history = []
        
        for batch, (X, y) in enumerate(tqdm(dataloader)):
            
            # Switch to train mode
            model.train()
            
            # Send tensors to the device
            X, y, model = X.to(device), y.to(device), model.to(device)
            
            # Make predictions
            pred = model(X)
            
            # Compute loss (error)
            loss = loss_fn(pred, y)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Append batch loss history
            if batch % 100 == 0:
                loss_history.append([batch, loss])
                
        # Print loss history
        if print_log == True:
            print('Loss over batches:')
            print(' Batch\tLoss')
            for item in loss_history:
                print(f' {item[0]}\t{item[1]:>7f}')
    
        # Return loss history
        return (loss_history)

    def test(self, dataloader, print_log: bool = False):
        
        model = self.model
        loss_fn = self.loss_fn
        device = self.device
        
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        
        # Switch to eval mode
        model.eval()
        
        test_loss, correct = 0, 0
        
        with torch.no_grad():
            
            for X, y in tqdm(dataloader):
                
                # Send tensors to the device
                X, y, model = X.to(device), y.to(device), model.to(device)
                
                # Make predictions
                pred = model(X)
            
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
        test_loss /= num_batches
        correct /= size
        
        # Print test accuracy and test lost
        if print_log == True:
            print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}')
        
        # Return test accuracy
        return (correct)
