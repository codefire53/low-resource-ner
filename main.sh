#!/bin/bash
#SBATCH --job-name=ner_mixed-all_wikiann-jv-test_glot500 # Job name
#SBATCH --error=./logs/%j%x.err # error file
#SBATCH --output=./logs/%j%x.out # output log file
#SBATCH --time=24:00:00 # 10 hours of wall time
#SBATCH --nodes=1  # 1 GPU node
#SBATCH --mem=16000 # 16 GB of RAM
#SBATCH --nodelist=ws-l2-003


python main.py --config-name glot500_jv
