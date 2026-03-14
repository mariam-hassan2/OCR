import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from .utils import preprocess_image
from transformers import TrOCRProcessor

class LicenseDataset(Dataset):
    def __init__(self, folder_path, processor, augment=False):
        self.df = build_df(folder_path)  
        self.processor = processor
        self.augment = augment
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        image = preprocess_image(image)  
        
        pixels = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        tokens = self.processor.tokenizer(row['label'], padding="max_length", max_length=20).input_ids
        labels = torch.tensor([t if t != self.processor.tokenizer.pad_token_id else -100 for t in tokens])
        return {"pixel_values": pixels, "labels": labels}
