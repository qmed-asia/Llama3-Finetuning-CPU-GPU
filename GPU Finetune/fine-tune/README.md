# GPU Fine-Tuning Code for Language Model

This repository contains the code for fine-tuning a large language model (LLM) using GPU.

## Environment Setup

Set your HuggingFace token as an environment variable:
```python
os.environ["HF_TOKEN"] = 'your_huggingface_token'
```


## Launch Jobs

1) Add train.csv and val.csv to Llama3-Finetuning-CPU-GPU\GPU Finetune\fine-tune.

2) Setup and run Docker

Step 1: Build the Docker Image:

docker build -t fine-tuning-image .

Step 2: RUn the Docker Image
docker run --rm -it -v Llama3-Finetuning-CPU-GPU/GPU\ Finetune/fine-tune/:/workspace fine-tuning-image

## Functions

### `load_llama_instruct()`

This function loads the LLM with 4-bit quantization using the specified model ID or path. It returns the model and tokenizer.

### `load_data_splits()`

This function loads and prepares the dataset from a CSV file. It formats the data, saves it as a JSON file, and then splits it into training, validation, and test datasets.

### `generate_train_format(data_point)`

Generates input text for training based on a given prompt, task instruction, and answer.

### `generate_train_val(train, tokenizer)`

Adds a "training_prompt" column to the training dataset and tokenizes the prompts.

### `generate_test_format(data_point)`

Generates input text for testing based on a given prompt, task instruction, and answer.

### `generate_test(test, tokenizer)`

Adds "test_instructions" and "test_summary" columns to the test dataset and tokenizes the prompts.

### `load_llama3()`

Loads the tokenizer for the LLM. This function is similar to `load_llama_instruct()` but does not load the model.

### `lora_llama(model)`

Configures and applies LoRA (Low-Rank Adaptation) to the model for parameter-efficient fine-tuning.

### `train(tokenizer, model, lora_config, llm_train_data, llm_val_data)`

Trains the model using the SFT (Supervised Fine-Tuning) Trainer from the `trl` library. The training parameters such as batch size, learning rate, and number of epochs are specified here.

## Notes

- Ensure the paths to the dataset and model are correctly specified.
- Adjust the prompts and configuration settings as needed for your specific use case.
- The code assumes Secure Boot is disabled on your machine.

## Example

Here's a minimal example to run the fine-tuning process:

```python
from utils import load_data_splits, train, generate_train_val, generate_test
from utils import load_llama3, lora_llama, train
from datasets import load_dataset, DatasetDict
import pandas as pd

model, tokenizer = load_llama3()
lora_config, lora_model = lora_llama(model)

#To preprocess dataset that has been split
dataset=load_dataset('csv', data_files={'train':'train.csv', 'val':'val.csv'})

train(tokenizer,model, lora_config, dataset['train'], dataset['val'])
```

# Extra Remarks
Prompts are needed for finetuning. Please prepare your own prompts as needed. Refer to the code file.
