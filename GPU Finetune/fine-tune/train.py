from utils import load_data_splits, train, generate_train_val, generate_test
from utils import load_llama_instruct, lora_llama, train
from datasets import load_dataset, DatasetDict
import pandas as pd

model, tokenizer = load_llama_instruct()
lora_config, lora_model = lora_llama(model)

# # To split dataset
# test,val,train = load_data_splits()
# dataset_train=generate_train_val(train,tokenizer)
# dataset_val=generate_train_val(val,tokenizer)
# dataset_test=generate_test(test,tokenizer)

# #To load dataset that has been split
# dataset=load_dataset('csv', data_files={'train':'train.csv', 'val':'val.csv'})
# results=train(tokenizer,lora_model, lora_config, dataset["train"], dataset["val"])

#To preprocess dataset that has been split
dataset=load_dataset('csv', data_files={'train':'train.csv', 'val':'val.csv'})

train(tokenizer,model, lora_config, dataset['train'], dataset['val'])
