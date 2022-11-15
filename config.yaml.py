yaml
model:
  name: gpt2
  max_length: 100
  save_dir: "models/v1.0"
training:
  epochs: 10
  batch_size: 10
  learning_rate: 3e-5
  save_steps: 100000
  fp16: true