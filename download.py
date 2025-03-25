from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-r1-7b")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-r1-7b")

# 保存模型和分词器到本地目录
model.save_pretrained("./local_model")
tokenizer.save_pretrained("./local_model")