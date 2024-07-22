#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu118-py38-torch201:biweekly.202404.1

USER root

RUN apt-get update && apt-get -y upgrade
RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir
RUN python -m nltk.downloader punkt
RUN MAX_JOBS=4 pip install flash-attn==2.5.5 --no-build-isolation

# Set the working directory
WORKDIR /workspace

# Copy the inferencing scripts and any other necessary files to the container
COPY Llama3-Finetuning-CPU-GPU/GPU\ Finetune/inference/inference.py /workspace/inference.py
COPY Llama3-Finetuning-CPU-GPU/GPU\ Finetune/inference/utils.py /workspace/utils.py
COPY Llama3-Finetuning-CPU-GPU/GPU\ Finetune/inference/test.csv /workspace/test.csv

# Command to run the inferencing script
CMD ["python", "/workspace/inference.py"]