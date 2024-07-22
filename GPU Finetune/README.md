### README

# GPU Fine-Tuning Code for Language Model

This repository contains the code for fine-tuning a large language model (LLM) using GPU. The code utilizes various libraries such as `transformers`, `bitsandbytes`, `peft`, and `trl` to handle model loading, dataset preparation, and training. Below is a brief explanation of the code:

## Dependencies

- `pandas`
- `json`
- `os`
- `torch`
- `transformers`
- `datasets`
- `peft`
- `bitsandbytes`
- `trl`
- `evaluate`
- `numpy`

Make sure to install these packages using `pip` before running the code.

## Environment Setup

Set your HuggingFace token as an environment variable:
```python
os.environ["HF_TOKEN"] = 'your_huggingface_token'
```

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

## Usage

1. Load the model and tokenizer:
    ```python
    model, tokenizer = load_llama_instruct()
    ```

2. Prepare the dataset:
    ```python
    test_data, val_data, train_data = load_data_splits()
    ```

3. Format the training and validation data:
    ```python
    train_dataset = generate_train_val(train_data, tokenizer)
    ```

4. Configure LoRA and prepare the model:
    ```python
    lora_config, model = lora_llama(model)
    ```

5. Train the model:
    ```python
    train(tokenizer, model, lora_config, train_dataset, val_data)
    ```

6. Format the test data (optional):
    ```python
    test_dataset = generate_test(test_data, tokenizer)
    ```

## Notes

- Ensure the paths to the dataset and model are correctly specified.
- Adjust the prompts and configuration settings as needed for your specific use case.
- The code assumes Secure Boot is disabled on your machine.

## Example

Here's a minimal example to run the fine-tuning process:

```python
from your_module import load_llama_instruct, load_data_splits, generate_train_val, lora_llama, train

# Load model and tokenizer
model, tokenizer = load_llama_instruct()

# Load and prepare data
test_data, val_data, train_data = load_data_splits()

# Generate training format
train_dataset = generate_train_val(train_data, tokenizer)

# Configure and prepare model with LoRA
lora_config, model = lora_llama(model)

# Train the model
train(tokenizer, model, lora_config, train_dataset, val_data)
```

Feel free to customize the code according to your needs and share your improvements with the community!
