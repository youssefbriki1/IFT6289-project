#!/bin/bash
#SBATCH --job-name=fine-tuning/1.5b
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=4
#SBATCH -p compute_full_node
#SBATCH --output=/home/m/mehrad/brikiyou/scratch/ift6289/IFT6289-project/stock_predictor_sentiment_analysis/scripts/out/slurm-%j.out
#SBATCH --error=/home/m/mehrad/brikiyou/scratch/ift6289/IFT6289-project/stock_predictor_sentiment_analysis/scripts/out/slurm-%j.err
#SBATCH --mail-user=youssef.briki@umontreal.ca
#SBATCH --mail-type=FAIL 



module load BalamEnv
module load python/3.11.5
module load pytorch/2.1.2
module load gcc/12.3.0
module load cuda/12.3.1
cd /home/m/mehrad/brikiyou/scratch/ift6289/IFT6289-project/
source project_venv_2/bin/activate
cd stock_predictor_sentiment_analysis/dora_fine-tuning/
python3 training.py > output_1.5b.log