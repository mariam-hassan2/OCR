import os, torch
from src.dataset import build_df, LicenseDataset
from src.model import load_model, apply_pruning, remove_pruning
from src.eval import eval_model
from src.utils import count_model_params
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator

DATASET_DIR = "/kaggle/input/datasets/abdelhamidzakaria/european-license-plates-dataset/dataset_final/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # Data
    train_df = build_df(os.path.join(DATASET_DIR, "train"))
    val_df   = build_df(os.path.join(DATASET_DIR, "val"))
    test_df  = build_df(os.path.join(DATASET_DIR, "test"))

    model, processor = load_model(DEVICE)

    train_ds = LicenseDataset(train_df, processor, augment=True)
    val_ds   = LicenseDataset(val_df, processor)
    test_ds  = LicenseDataset(test_df, processor)

    # Train
    args = Seq2SeqTrainingArguments(
        output_dir="/tmp/trocr", num_train_epochs=5, learning_rate=2e-5,
        per_device_train_batch_size=8, gradient_accumulation_steps=2,
        fp16=True, weight_decay=1e-4, eval_strategy="epoch",
        save_strategy="epoch", load_best_model_at_end=True,
        predict_with_generate=False, report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=default_data_collator,
    )
    trainer.train()

    # Eval + Prune
    count_model_params(model)
    acc, _, _ = eval_model(model, processor, test_ds)
    print(f"Final acc: {acc:.4f}")

if __name__ == "__main__":
    main()
