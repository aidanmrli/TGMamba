#!/bin/bash
#SBATCH --job-name=tuhz-2attn
#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=4-00:00:00
#SBATCH -c 30
#SBATCH --mem=60G
#SBATCH --output=/h/liaidan/TGMamba/slurm/outputs/slurm-%j.out
#SBATCH --error=/h/liaidan/TGMamba/slurm/outputs/slurm-%j.err
#SBATCH --mail-user=aidan.li@mail.utoronto.ca
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

# prepare your environment here
module load python/3.10.12
module load cuda-12.4
source /h/liaidan/TGMamba/tgmamba_venv/bin/activate

# put your command here
python /h/liaidan/TGMamba/hyperparam_search_attn.py --dataset 'tuhz'
