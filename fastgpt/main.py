from fastapi import FastAPI
from models.fastgpt_model import FastGPT
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

app = FastAPI()
model = FastGPT(model_name=config['model']['name'], model_path=config['model']['save_dir'])

@app.post("/generate")
def generate(prompt: str, max_length: int = config['model']['max_length']):
    generated_text = model.generate_text(prompt, max_length)
    return {"generated_text": generated_text}