#!/bin/bash
#SBATCH --job-name=tuhz-no-attn
#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=2-00:00:00
#SBATCH -c 30
#SBATCH --mem=60G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-user=aidan.li@mail.utoronto.ca
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

# prepare your environment here
module load python/3.10.12
module load cuda-12.4
source /h/liaidan/TGMamba/tgmamba_venv/bin/activate

# put your command here
python tuhz_search_no_attn.py
