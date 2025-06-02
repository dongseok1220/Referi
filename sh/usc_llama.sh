#!/bin/bash


mkdir -p logs


for model in 'llama'; do
    for task in 'math500' 'mmlu_pro' 'gpqa' 'drop' 'hotpotqa' 'musr_efficiently' 'musr_location'; do

        for shot in 'few'; do 
            sbatch --job-name="usc_${model}_${task}" \
                --output="logs/usc_${model}_${task}.txt" \
                --time=72:00:00 \
                --gres=gpu:1 \
                --partition=suma_a6000 \
                --wrap="source ~/.bashrc; conda activate proj2; python -u USC/usc_with_fewshot_llama.py --model ${model} --task '${task}' --shot_type ${shot} --output_dir 'baselines/usc_with_fewshot' --num_examples -1"
        done
    done
done
