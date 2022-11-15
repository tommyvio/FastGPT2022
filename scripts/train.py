import yaml
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from utils.data_loader import load_text_data

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def train_model():
    texts = load_text_data("data/raw/text_data.txt")
    tokenizer = GPT2Tokenizer.from_pretrained(config['model']['name'])
    model = GPT2LMHeadModel.from_pretrained(config['model']['name'])
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    inputs = tokenizer(texts, return_tensors="pt", max_length=512, truncation=True, padding=True)
    training_args = TrainingArguments(
        output_dir=config['model']['save_dir'],
        num_train_epochs=config['training']['epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        save_steps=config['training']['save_steps'],
        save_total_limit=2,
        fp16=config['training']['fp16']
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=inputs['input_ids'])
    trainer.train()
    model.save_pretrained(config['model']['save_dir'])
    tokenizer.save_pretrained(config['model']['save_dir'])

if __name__ == "__main__":
    train_model()