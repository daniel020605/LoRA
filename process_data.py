def format_example(example):
  prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"
  return {"text": prompt + example["output"]}

dataset = load_dataset("json", data_files="your_data.json")["train"]
dataset = dataset.map(format_example)