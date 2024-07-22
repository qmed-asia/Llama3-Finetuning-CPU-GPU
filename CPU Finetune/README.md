# Parameter Efficient Fine-tuning on Local CPUs

This repo contains code for the paper {attach link}

There are 3 main configurations for CPU training.  

1. ipex-llm + tcmalloc + AMX

export ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX

source ipex-llm-init -t

2. ipex-llm + tcmalloc wout amx

ONEDNN_MAX_CPU_ISA=AVX512_CORE_VNNI

source ipex-llm-init -t

3. ipex-llm 

ONEDNN_MAX_CPU_ISA=AVX512_CORE_VNNI

Add code for GPU training
