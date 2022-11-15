from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class FastGPT:
    def __init__(self, model_name="gpt2", model_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_path or model_name).to(self.device)

    def generate_text(self, prompt, max_length=50):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)