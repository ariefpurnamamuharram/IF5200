from torch.utils.data import Dataset

from engine.utils.image import read_image, ToTensorTransform


class CustomDataset(Dataset):

    def __init__(self, data, image_path):

        super(CustomDataset, self).__init__()

        self.data = data
        self.image_path = image_path
        self.transformer = ToTensorTransform()

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        data = self.data.iloc[:, 0:len(self.data)].iloc[idx]

        # Image preprocessing
        filename = data['filename']
        image = read_image(f'{self.image_path}/{filename}')
        image = self.transformer.transform(image)

        label = data['label']

        return (filename, image, label)
