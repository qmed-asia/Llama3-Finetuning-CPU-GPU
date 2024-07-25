# Llama3-Finetuning-CPU-GPU

## Contributors
- Shamus Sim Zi Yang
- Goh Man Fye
- Diong Zi Yu
- Yap Wei Chung

This repo contains implementation to the paper, "Enhancing Medical Summarisation with Parameter Efficient Fine Tuning on Local CPUs". [Paper Link]

## What is included inside this repository
- Evaluation metrics: conformance test and Deepeval.
- GPU fine-tuning.
- CPU fine-tuning (with link to the repo).

**Note: These are organized into three different folders. You can choose any folder based on your requirements.

## CPU Fine-tuning
This section covers the CPU fine-tuning process using tools like `tcmalloc` and `AMX`. The impact of using these tools includes improved memory management and enhanced computational efficiency, enabling effective local fine-tuning on CPUs.

## GPU Fine-tuning
The GPU fine-tuning process utilizes `NVIDIA A100 GPUs` from Microsoft Azure, which provided a `GPU instance with 220 GiB of memory, and 64 GiB of temporary disk space`. Storage was managed by 960 GB NVMe SSDs, which ensures high-speed data loading and checkpoint saving capabilities with 80 GiB of GPU memory.

## Hardware Specifications and Achievements
1. **CPU Setup**: 4th Gen Intel Xeon Scalable Processors (Sapphire Rapids), 512GB DDR4 RAM, 2TB NVMe SSDs.
2. **GPU Setup**: NVIDIA A100 GPU with 220 GiB memory, 64 GiB temporary disk space, 960 GB NVMe SSDs.
   
The following is our results:

![image](https://github.com/user-attachments/assets/d6a6d718-4a20-4d05-8e5c-7d274707ac0b)


## Custom Summarisation Dataset Structure
The training and validation datasets are loaded as CSV files: `train.csv` and `val.csv`.

A standard preprocessing pipeline is applied to it.
1. Ensure that the instruction fine-tuning dataset follows this template:
    ```
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{Insert your instruction prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{Insert your input}<|eot_id|><|start_header_id|>system<|end_header_id|>\n\n{Summary}<|eot_id|>
    ```
2. Rename this section as `_llama_prompt_` as one of the columns in `train.csv` and `val.csv`.

## License
Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

For any questions, please contact [shamus@qmed.asia], [manfye@qmed.asia], [dion.zy627@gmail.com], [yapweichung@gmail.com].

