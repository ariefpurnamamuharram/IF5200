import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, classification_report
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
                self.optimizer = optim.Adam(
                    model.parameters(), lr=learning_rate)
            elif optimizer == 'SGD':
                self.optimizer = optim.SGD(
                    model.parameters(), lr=learning_rate)

        # Set device
        if device is not None:
            self.device = torch.device(device)
            print('Using GPU!')
            print('Device name:', torch.cuda.get_device_name(self.device))
        else:
            self.device = torch.device('cpu')
            print('Using CPU!')

    def get_model(self):
        # Return the model object
        return self.model

    def train(self, dataloader, print_log: bool = False):
        model = self.model
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        device = self.device

        loss_history = []

        for batch, (_, image, label) in enumerate(tqdm(dataloader)):

            # Switch to train mode
            model.train()

            # Send tensors to the device
            image, label, model = image.to(device), label.to(device), model.to(device)

            # Make predictions
            preds = model(image)

            # Compute loss (error)
            loss = loss_fn(preds, label)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Append batch loss history
            if batch % 100 == 0:
                loss_history.append([batch, loss])

        # Print loss history
        if print_log:
            print('Loss over batches:')
            print(' Batch\tLoss')
            for item in loss_history:
                print(f' {item[0]}\t{item[1]:>7f}')

        # Return loss history
        return loss_history

    def test(self, dataloader, print_log: bool = False, print_clf_result: bool = False,
             print_conf_matrix: bool = False, print_clf_report: bool = False, export_clf_result: bool = False,
             filename_clf_result: str = 'clf_result.csv'):
        model = self.model
        loss_fn = self.loss_fn
        device = self.device

        size = len(dataloader.dataset)
        num_batches = len(dataloader)

        # Switch to eval mode
        model.eval()

        test_loss, correct = 0, 0
        global_result, global_filename, global_preds, global_y = [], [], [], []

        with torch.no_grad():
            for filename, image, label in tqdm(dataloader):
                # Send tensors to the device
                image, label, model = image.to(
                    device), label.to(device), model.to(device)

                # Make predictions
                pred = model(image)

                global_filename += filename
                global_preds += pred.argmax(1).tolist()
                global_y += label.tolist()

                test_loss += loss_fn(pred, label).item()
                correct += (pred.argmax(1) ==
                            label).type(torch.float).sum().item()

        for idx, name in enumerate(global_filename):
            global_result.append([name, global_y[idx], global_preds[idx]])

        global_result = pd.DataFrame(global_result, columns=['filename', 'label', 'pred'])

        # Print classification result
        if print_clf_result:
            print('\nItem result:')
            print(global_result)

        # Print confusion matrix:
        if print_conf_matrix:
            print('\nConfusion matrix:')
            print(confusion_matrix(global_y, global_preds))

        # Print classification report
        if print_clf_report:
            print('\nClassification report:')
            print(classification_report(global_y, global_preds))

        # Export classification result
        if export_clf_result:
            global_result.to_csv(filename_clf_result, index=False)

        test_loss /= num_batches
        correct /= size

        # Print test accuracy and test lost
        if print_log:
            print(f'Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}')

        # Return test accuracy
        return correct
