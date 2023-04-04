import torch


class InferenceUtils:

    def __init__(self, model, device: torch.device = torch.device('cpu')):
        super(InferenceUtils, self).__init__()
        self.model = model
        self.device = device

    def make_prediction(self, x: torch.Tensor):
        """
        Args:
            x: Image tensor

        Returns:
            Prediction result
        """

        model = self.model
        device = self.device
        print(f'Use device {device}!')
        if not device == torch.device('cpu'):
            print(f'Device name: {torch.cuda.get_device_name(device)}')

        # Switch to eval mode
        model.eval()

        with torch.no_grad():
            # Send tensors to the device
            x, model = x.to(device), model.to(device)

            # Make prediction
            pred = model(x)

            # Return prediction
            return pred.argmax(1)
