##  Parameter Efficient Fine-tuning on Local CPUs

# Setup 

Step 1: In _CPU Finetune/docker/llm/finetune/qlora/cpu/docker/start-qlora-finetuning-on-cpu.sh_, add your Weights and Biases and HuggingFace API Key .

Step 2: Configure AMX and tcmalloc setup
There are 3 main configurations for CPU training.  Make the follwing adjustments in the shell file in _CPU Finetune/docker/llm/finetune/qlora/cpu/docker/start-qlora-finetuning-on-cpu.sh_

1. ipex-llm + tcmalloc + AMX

export ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX

source ipex-llm-init -t

2. ipex-llm + tcmalloc wout amx

ONEDNN_MAX_CPU_ISA=AVX512_CORE_VNNI

source ipex-llm-init -t

3. ipex-llm 

ONEDNN_MAX_CPU_ISA=AVX512_CORE_VNNI

# Hardware Used
This code was tested on 4th Gen Intel® Xeon® Scalable Processors (Formerly Sapphire Rapids) [https://www.intel.com/content/www/us/en/developer/articles/technical/fourth-generation-xeon-scalable-family-overview.html repeat] to provide the necessary computational power for training and fine-tuning large language models on CPUs. Memory management was facilitated by 512GB DDR4 RAM and 2TB NVMe SSDs for storage. 

# Extra Remarks
Prompts are needed for finetuning. Please prepare your own prompts as needed. Refer to the code file. 
