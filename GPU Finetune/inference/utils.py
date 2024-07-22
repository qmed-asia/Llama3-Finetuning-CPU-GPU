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

os.environ["HF_TOKEN"] = 'xxxx'

def load_llama3():
    base_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",#sym_int4 for CPU
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id,add_eos_token=True,add_bos_token=True)

    return model, tokenizer


def pipe_llama(llama_tokenizer, model):
    model.eval()

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=llama_tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16}
    )

    df = pd.read_csv('test.csv')

    input_cases = df['Input Query'].tolist()
    original_summaries = df['Output Response'].tolist()

    # Initialize an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=['Generated Summary', 'GPT Summary'])

    for patient_case, original_summary in zip(input_cases, original_summaries):
        messages = [
            {"role": "system", "content": "xxxx"}, # Replace with prompt
            {"role": "user", "content": f'{patient_case}' },
        ]

        prompt = pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
        )

        terminators = [
            llama_tokenizer.eos_token_id,
            llama_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
        )
        print(outputs[0]["generated_text"][len(prompt):])
        # Extract the generated text
        generated_text = outputs[0]["generated_text"][len(prompt):]

        current_df = pd.DataFrame({'Generated Summary': [generated_text], 'GPT Summary': [original_summary]})

        # Append the current DataFrame to the results DataFrame using concat
        results_df = pd.concat([results_df, current_df], ignore_index=True)
        # results_df = results_df.append({'Generated Summary': generated_text, 'Original Summary': original_summary}, ignore_index=True)
    # Save the DataFrame to a CSV file
    results_df.to_csv('outputs/test-inference.csv', index=False)
