from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class CustomDataset(Dataset):
    
    def __init__(self, data, image_path):
        
        super(CustomDataset, self).__init__()
        
        self.data = data
        self.image_path = image_path
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, idx):
        
        data = self.data.iloc[:, 0:len(self.data)].iloc[idx]
        
        # Image preprocessing
        filename = data['filename']
        image = Image.open(f'{self.image_path}/{filename}').convert('RGB')
        image = self.transform(image)
        
        label = data['label']
        
        return (image, label)