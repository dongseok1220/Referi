#!/bin/bash


mkdir -p logs


for model in 'llama'; do
    for task in 'math500' 'mmlu_pro' 'gpqa' 'drop' 'hotpotqa' 'musr_efficiently' 'musr_location'; do

        for shot in 'few'; do 
            sbatch --job-name="llm_prompting_${model}_${task}" \
                --output="logs/llm_prompting_${model}_${task}.txt" \
                --time=72:00:00 \
                --gres=gpu:1 \
                --partition=suma_a6000 \
                --wrap="source ~/.bashrc; conda activate proj2; python -u LLM-as-judge/llm_as_judge_llama.py --model ${model} --task '${task}' --shot_type ${shot} --output_dir 'LLM-as-judge/llm_as_judge' --num_examples -1"
        done
    done
done
