#!/bin/bash

#SBATCH --job-name=inf-13b-2nodes
#SBATCH --time 2:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=10
#SBATCH  --mem 400G
#SBATCH --nodes=2
# SBATCH --gpus-per-task=1
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A30:4 
#SBATCH --output=slurm.out
#SBATCH -q bonus


#master_hostname=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
#master_ip=$(ifconfig bond1.1330 | grep 'inet ' | cut -d ' ' -f 10)
#export MASTER_ADDR=${master_ip}
#export MASTER_ADDR=10.11.3.102 #$master_addr
#echo "MASTER_ADDR="$MASTER_ADDR
#export NCCL_DEBUG=INFO
#echo master_ip=$master_ip

export MASTER_ADDR=$(hostname -I | grep -o '10.11.\w*.\w*') MASTER_PORT=29500

echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

export NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=bond1.1330 NCCL_IB_DISABLE=1

source /stornext/Home/data/allstaff/i/iskander.j/micromamba/etc/profile.d/micromamba.sh
micromamba activate llm
### the command to run
srun torchrun --rdzv_id $RANDOM --rdzv_backend c10d \
        --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --nproc-per-node 4 --nnodes 2 \
        example_chat_completion.py --ckpt_dir /vast/projects/RCP/WEHI-llama2/70B/  \
        --tokenizer_path /vast/projects/RCP/WEHI-llama2/tokenizer.model \
        --max_seq_len 512 --max_batch_size 6