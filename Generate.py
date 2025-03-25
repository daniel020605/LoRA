inputs = tokenizer("### Instruction:\n如何做西红柿炒鸡蛋?\n\n### Response:\n", 
                  return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))