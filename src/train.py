from src.model import train_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="./data")
args = parser.parse_args()

model, processor = train_model(
    f"{args.data_dir}/train",
    f"{args.data_dir}/val",
    "./model"
)
