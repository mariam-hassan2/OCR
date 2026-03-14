from transformers import VisionEncoderDecoderModel, TrOCRProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
from .dataset import LicenseDataset
from .utils import clean_plate
import torch

def train_model(train_dir, val_dir, num_epochs=25, output_dir="./trocr-plates"):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    
    train_ds = LicenseDataset(train_dir, processor, augment=True)
    val_ds = LicenseDataset(val_dir, processor)
    
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=8,
        weight_decay=1e-4,
        fp16=True,
        logging_steps=50,
        save_steps=500,
        eval_strategy="epoch"
    )
    
    trainer = Seq2SeqTrainer(model, args, train_dataset=train_ds, eval_dataset=val_ds)
    trainer.train()
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    return model, processor
