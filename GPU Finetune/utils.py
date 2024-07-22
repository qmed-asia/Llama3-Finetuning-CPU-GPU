import pandas as pd
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import bitsandbytes as bnb
import transformers
from trl import SFTTrainer
from datasets import load_dataset
import datasets
import evaluate
import numpy as np

os.environ["HF_TOKEN"] = 'xxxxxxxxxxxx' # insert your own HuggingFace token

def load_llama_instruct():
    base_model_id = "xxxxxxxxx" # specify the path of the LLM that u wanna use here (can be local path on your PC or model ID from HuggingFace such as "meta-llama/Meta-Llama-3-8B-Instruct")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",#sym_int4 for CPU
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id,add_eos_token=True,add_bos_token=True)

    return model, tokenizer



def load_data_splits():
    dataset_path = r"xxxxxx.csv"
    file_path = r"xxxxxx.json"
    df = pd.read_csv(dataset_path)
    extracted_data = df[['Input Query', 'Output Response']]
    formatted_prompts = [
        {
            "Prompt": "xxxxxxxxx", # insert your own prompt here
            "Input": row['Input Query'], # name of the column which is the input that u wanna feed for finetune
            "Summary": row['Output Response'] # 
        }
        for index, row in extracted_data.iterrows()
    ]
    with open(file_path, 'w') as file:
        json.dump(formatted_prompts, file, indent=4)

    # Load dataset
    
    dataset = load_dataset("json", data_files=file_path, split='train')  # Specify split here if your dataset only contains one split

    # Shuffle dataset
    dataset = dataset.shuffle(seed=1234)

    # First, split off the test set (10% of the initial dataset)
    split_datasets = dataset.train_test_split(test_size=0.1)

    train_val_datasets = split_datasets['train'].train_test_split(test_size=1/9)

    test_data= split_datasets['test']
    val_data=train_val_datasets['test']
    train_data=train_val_datasets['train']
    return test_data, val_data, train_data

def generate_train_format(data_point):
        """Generate input text based on a prompt, task instruction, (context info.), and answer

        :param data_point: dict: Data point
        :return: str: Tokenized prompt
        """

        # Generate prompt (insert your own prompt in 'prefic_text')
        prefix_text = "xxxxxxxxx"
        # Samples with additional context info.
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{prefix_text}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{data_point["Input"]}<|eot_id|><|start_header_id|>system<|end_header_id|>\n\n{data_point["Summary"]}<|eot_id|>"""


def generate_train_val(train, tokenizer):
    # Add the "prompt" column in the dataset
    text_column = [generate_train_format(data_point) for data_point in train]
    train_dataset = train.add_column("training_prompt", text_column)

    # # Map to tokenize the prompt
    # train_dataset = train_dataset.map(lambda samples: tokenizer(samples["training_prompt"]), batched=True)

    return train_dataset

def generate_test_format(data_point):
        """Generate input text based on a prompt, task instruction, (context info.), and answer

        :param data_point: dict: Data point
        :return: str: Tokenized prompt
        """
        # Generate prompt (insert your own prompt in 'prefic_text')
        prefix_text = "xxxxxxxxx"
        # Samples with additional context info.
        test_instruction = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{prefix_text}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{data_point["Input"]}<|eot_id|><|start_header_id|>system<|end_header_id|>\n\n{data_point["Summary"]}<|eot_id|>"""
        test_summary= f"""{data_point["Summary"]}"""
        return test_instruction, test_summary
def generate_test(test,tokenizer):
    # Add the "prompt" column in the dataset
    test_instructions, test_summaries = zip(*[generate_test_format(data_point) for data_point in test])
    print(test_instructions)
    test_dataset = test.add_column("test_instructions", test_instructions)
    test_dataset = test_dataset.add_column("test_summary", test_summaries)

    # Map to tokenize the prompt
    test_dataset = test_dataset.map(lambda samples: tokenizer(samples["test_instructions"]), batched=True)

    return test_dataset

def load_llama3():
    base_model_id = "xxxxxxxxx" # specify the path of the LLM that u wanna use here (can be local path on your PC or model ID from HuggingFace such as "meta-llama/Meta-Llama-3-8B-Instruct")
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",#sym_int4 for CPU
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    # model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id,add_eos_token=True,add_bos_token=True)

    # return model, tokenizer
    return tokenizer

def lora_llama(model):

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    ###RUN1 
    # lora_config = LoraConfig(
    #     r=16,
    #     lora_alpha=32,
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    #     target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
    # )
    ###RUN 2
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['q_proj', 'v_proj']
    )

    model = get_peft_model(model, lora_config)

    return lora_config, model

def train(tokenizer,model, lora_config, llm_train_data, llm_val_data):
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side='right'
    torch.cuda.empty_cache()

    trainer = SFTTrainer(
        model=model,
        train_dataset=llm_train_data,
        eval_dataset=llm_val_data,
        dataset_text_field="prompt",
        peft_config=lora_config,
        max_seq_length=2500,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            warmup_steps=0.03,  # This might still need to be specified in steps, not epochs
            num_train_epochs=9,  # Specify the number of epochs directly
            learning_rate=2e-4,
            logging_steps=10,  # Logs every 10 steps, you can also use `logging_strategy='Epoch'`
            output_dir="outputs",
            optim="paged_adamw_8bit",
            evaluation_strategy="epoch",  # Evaluates at the end of each epoch
            save_strategy="epoch",  # Saves at the end of each epoch
            load_best_model_at_end=True,  # Optional: load the best model at the end of training
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()
    print('-----Training Done-------')


