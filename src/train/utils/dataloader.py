from torch.utils.data import DataLoader

from src.train.utils.dataset import CustomDataset


class CustomDataLoader:

    def __init__(self, batch_size: int, image_path: str):
        super(CustomDataLoader, self).__init__()

        self.batch_size = batch_size
        self.image_path = image_path

    def get_batch_size(self):
        return self.batch_size

    def get_image_path(self):
        return self.image_path

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_image_path(self, image_path):
        self.image_path = image_path

    def create_dataloader(self, data):
        dataset = CustomDataset(data, self.image_path)
        return DataLoader(dataset, batch_size=self.batch_size)
