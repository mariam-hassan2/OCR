import torch
from PIL import Image, ImageEnhance, ImageFilter
from transformers import TrOCRProcessor

def clean_plate(text):
    return ''.join(c for c in text.upper() if c.isalnum())[:16]

def preprocess_image(image):
    image = ImageEnhance.Contrast(image).enhance(1.4)
    image = ImageEnhance.Sharpness(image).enhance(1.3)
    image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=40))
    return image

def load_model(repo_id="mtarek123456/ocr-eu-car-plates-fp16"):
    processor = TrOCRProcessor.from_pretrained(repo_id)
    model = VisionEncoderDecoderModel.from_pretrained(repo_id, torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, processor, device
