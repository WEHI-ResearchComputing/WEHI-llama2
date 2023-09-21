#!/bin/bash

#SBATCH --job-name=inf-13b
#SBATCH --time 2:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=10
#SBATCH  --mem 400G
#SBATCH --nodes=1
# SBATCH --gpus-per-task=1
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A30:2
#SBATCH --output=slurm_13b.out
#SBATCH -q bonus


source /stornext/Home/data/allstaff/i/iskander.j/micromamba/etc/profile.d/micromamba.sh
micromamba activate llm
### the command to run
srun torchrun  --nproc-per-node 2  \
        example_chat_completion.py --ckpt_dir /vast/projects/RCP/WEHI-llama2/llama-2-13b-chat/  \
        --tokenizer_path /vast/projects/RCP/WEHI-llama2/tokenizer.model \
        --max_seq_len 512 --max_batch_size 6


srun torchrun --nproc-per-node  2 \
        example_text_completion.py \
        --max_seq_len 128 --max_batch_size 4 --ckpt_dir /vast/projects/RCP/WEHI-llama2/llama-2-13b/  \
        --tokenizer_path /vast/projects/RCP/WEHI-llama2/tokenizer.model