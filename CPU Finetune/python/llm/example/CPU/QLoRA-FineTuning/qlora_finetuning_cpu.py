#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import os
import wandb
import transformers
from transformers import LlamaTokenizer, AutoTokenizer
from transformers import BitsAndBytesConfig

##For ipex-llm runs
from ipex_llm.transformers.qlora import get_peft_model, prepare_model_for_kbit_training, LoraConfig
from ipex_llm.transformers import AutoModelForCausalLM
# ## For raw transformers run
# from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
# from transformers import AutoModelForCausalLM


from datasets import load_dataset
import argparse
from ipex_llm.utils.isa_checker import ISAChecker
from trl import SFTTrainer

current_dir = os.path.dirname(os.path.realpath(__file__))
common_util_path = os.path.join(current_dir, '..', '..', 'GPU', 'LLM-Finetuning')
import sys
sys.path.append(common_util_path)
from common.utils import Prompter, get_train_val_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama2 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="xxxxxx",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--dataset', type=str, default="xxxxxx")

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    dataset_path = args.dataset
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    dataset=load_dataset('csv', data_files={'train':'train.csv', 'val':'val.csv'})

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="int4",  # nf4 not supported on cpu yet
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(model_path,quantization_config=bnb_config, device_map='cpu')


    # below is also supported
    # model = AutoModelForCausalLM.from_pretrained(model_path,
    #                                              # nf4 not supported on cpu yet
    #                                              load_in_low_bit="sym_int4",
    #                                              optimize_model=False,
    #                                              torch_dtype=torch.bfloat16,
    #                                              modules_to_not_convert=["lm_head"], )

    model = model.to('cpu')
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.enable_input_require_grads()
    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules= "all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    
    # To avoid only one core is used on client CPU
    isa_checker = ISAChecker()
    bf16_flag = isa_checker.check_avx512()
    from transformers import TrainerCallback, TrainingArguments
    import time

    class EpochTimeCallback(TrainerCallback):
        def __init__(self):
            self.epoch_start_time = None
            self.epoch_times = []

        def on_epoch_begin(self, args, state, control, **kwargs):
            self.epoch_start_time = time.time()

        def on_epoch_end(self, args, state, control, **kwargs):
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - self.epoch_start_time
            self.epoch_times.append(epoch_duration)
            print(f"Epoch {state.epoch} took {epoch_duration:.2f} seconds")

        def get_epoch_times(self):
            return self.epoch_times
        
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side='right'
    epoch_time_callback = EpochTimeCallback()

    ## Wandb
    wandb.init(project="xxxxxx", entity="xxxxxx")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset= dataset["val"],
        dataset_text_field="llama_prompt",
        peft_config=config,
        max_seq_length=2500,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            warmup_steps=0.03,  # This might still need to be specified in steps, not epochs
            num_train_epochs=8,  # Specify the number of epochs directly
            learning_rate=2e-4,
            bf16=bf16_flag,
            logging_steps=10,  # Logs every 10 steps, you can also use `logging_strategy='Epoch'`
            output_dir="outputs",
            optim="adamw_hf",
            evaluation_strategy="epoch",  # Evaluates at the end of each epoch
            save_strategy="epoch",  # Saves at the end of each epoch
            load_best_model_at_end=True,  # Optional: load the best model at the end of training
            report_to="wandb", # Log to wandb
            # gradient_checkpointing=True, # can further reduce memory but slower
        ),
        # Inputs are dynamically padded to the maximum length of a batch
        # data_collator=transformers.DataCollatorForSeq2Seq(
        #     tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        # ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[epoch_time_callback]  # Add the custom callback here
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    result = trainer.train()
    print(result)
