# Llama3-Finetuning-CPU-GPU

- Date of Conference: 30-31 October 2024

- IEEE Link: https://ieeexplore.ieee.org/document/10823619

- DOI: 10.1109/ICECCE63537.2024.10823619

- Date Added to IEEE Xplore: 10 January 2025

## Contributors
- Shamus Sim Zi Yang [@Shamus](https://github.com/shamussim-ai)
- Manfye [@manfye](https://github.com/manfye)
- Diong Zi Yu [@Ziyu](https://github.com/diongzy)
- Yap Wei Chung [@Rain](https://github.com/YapWeiChung)

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

|     Param      | Trans (A100 GPU) (BF16) |     Ipex (BF16)    | Ipex-llm + tcmal (BF16) | Ipex + tcmal + AMX (BF16) |
|----------------|-------------------------|--------------------|-------------------------|---------------------------|
|Time/epoch      |            7.5          |         77.6       |           74.8          |            11.6           |
|Train Epoch     |             8           |          8         |            8            |             8             |
|Total Train Time|           60.0          |        621.0       |          598.6          |            93.0           |
|CPU Usage (%)   |            N/A          |     Fluctuates     |           ~25           |            ~25            |
|RAM Usage (MB)  |            N/A          |     Fluctuates     |          135000         |           167000          |

#### Remarks:
1. The unit used for all the time variables is "minutes (mins)."
2. Time/epoch = Time per epoch (mins)
3. Train Epoch = Training Epoch
4. Total Train Time = Total Training Time (mins)
5. Param = Parameters
6. Trans = Transformers
7. Ipex = Ipex-llm
8. tcmal = tcmalloc


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

