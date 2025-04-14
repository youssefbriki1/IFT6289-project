#!/bin/bash
#SBATCH --job-name=zero_shot_deepseek32
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=/home/m/mehrad/brikiyou/scratch/ift6289/IFT6289-project/stock_predictor_sentiment_analysis/scripts/out/slurm-%j.out
#SBATCH --error=/home/m/mehrad/brikiyou/scratch/ift6289/IFT6289-project/stock_predictor_sentiment_analysis/scripts/out/slurm-%j.err
#SBATCH --mail-user=youssef.briki@umontreal.ca
#SBATCH --mail-type=FAIL 

module load BalamEnv
source /home/m/mehrad/brikiyou/scratch/to_run.sh
ollama serve > /home/m/mehrad/brikiyou/scratch/ollama.log 2>&1 &
source /home/m/mehrad/brikiyou/scratch/ift6289/IFT6289-project/project_venv/bin/activate
cd /home/m/mehrad/brikiyou/scratch/ift6289/IFT6289-project/stock_predictor_sentiment_analysis/dora_fine-tuning/
python3 deepseek_zero_shot.py