#!/bin/bash

mkdir -p logs

tasks=("math500" "mmlu_pro" "gpqa" "drop" "gpqa" "musr_efficiently" "musr_location")
subjects=('business' 'law' 'psychology' 'biology' 'chemistry' 'history' 'other' 'health' 'economics' 'math' 'physics' 'computer science' 'philosophy' 'engineering')

MODELS=('llama')

MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
INPUT_DIR="result"
OUTPUT_DIR="baselines/baseline"
SHOT_TYPES=('few' 'zero')


for model in "${MODELS[@]}"; do
  for task in "${tasks[@]}"; do

    if [[ "$task" == "mmlu_pro" ]]; then

      for subject in "${subjects[@]}"; do
        for shot_type in "${SHOT_TYPES[@]}"; do
          sbatch --job-name="${model}_${task}_${subject}_${shot_type}" \
                 --output="logs/${model}_${task}_${subject}_${shot_type}.txt" \
                 --time=72:00:00 \
                 --gres=gpu:1 \
                 --partition=suma_a6000 \
                 --wrap="source ~/.bashrc; conda activate proj2; \
                         python -u baselines/response_likelihood_llama_mmlu.py \
                           --model ${model} \
                           --model_name ${MODEL_NAME} \
                           --task ${task} \
                           --subject '${subject}' \
                           --input_dir ${INPUT_DIR} \
                           --output_dir ${OUTPUT_DIR} \
                           --shot_type ${shot_type}"
        done
      done

    else
      for shot_type in "${SHOT_TYPES[@]}"; do
        sbatch --job-name="${model}_${task}_${shot_type}" \
               --output="logs/${model}_${task}_${shot_type}.txt" \
               --time=72:00:00 \
               --gres=gpu:1 \
               --partition=suma_a6000 \
               --wrap="source ~/.bashrc; conda activate proj2; \
                       python -u baselines/response_likelihood_llama.py \
                         --model ${model} \
                         --model_name ${MODEL_NAME} \
                         --task ${task} \
                         --input_dir ${INPUT_DIR} \
                         --output_dir ${OUTPUT_DIR} \
                         --shot_type ${shot_type}"
      done

    fi

  done
done
