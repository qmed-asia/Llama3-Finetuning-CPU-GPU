# GPU Inferencing Code for Language Model

This repository contains the code for inferencing a large language model (LLM) using GPU.

## Environment Setup

Set your HuggingFace token as an environment variable:
```python
os.environ["HF_TOKEN"] = 'your_huggingface_token'
```

## Launch Jobs

1) Add test.csv to Llama3-Finetuning-CPU-GPU\GPU Finetune\inference.

2) Add checkpoint to Llama3-Finetuning-CPU-GPU\GPU Finetune\inference.

3) Setup and run Docker

Step 1: Build the Docker Image:

docker build -t inference-image .

Step 2: Run the Docker Image
docker run --rm -it inferencing-image


## Notes

- Ensure the paths to the dataset and model are correctly specified.
- Adjust the prompts and configuration settings as needed for your specific use case.

