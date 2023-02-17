import torch


class Inference:
    
    def __init__(self, device: str = None):
        
        super(Inference, self).__init__()
        
        # Set device
        if device is not None and str(device)[0:3].lower() in 'cuda':            
            self.device = torch.device(device)
            print('Using GPU!')
            print('Device name:', torch.cuda.get_device_name(self.device))
        else:
            self.device = torch.device('cpu')
            print('Using CPU!')
    
    def get_device(self):
        
        return self.device
        
    def set_device(self, device: str):
        
        self.device = torch.device(device)
        
        if device == 'cpu':
            print('Using CPU!')
        else:
            print('Using GPU!')
            print('Device name:', torch.cuda.get_device_name(self.device))
        
    def make_prediction(self, model, X: torch.Tensor):
        
        model = model
        device = self.device
        
        # Switch to eval mode
        model.eval()
        
        with torch.no_grad():
            
            # Send tensors to the device
            X, model = X.to(device), model.to(device)
            
            # Make prediction
            pred = model(X)
            
            # Return prediction
            return pred.argmax(1)
            