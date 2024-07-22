from peft import PeftModel
from utils import load_llama3, pipe_llama



model, tokenizer = load_llama3()
ft_model = PeftModel.from_pretrained(model, "checkpoint-llama")
pipe_llama(tokenizer, model)

