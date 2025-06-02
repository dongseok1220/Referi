#!/bin/bash

mkdir -p logs


tasks=("math500" "mmlu_pro" "gpqa" "drop" "gpqa" "musr_efficiently" "musr_location")
subjects=('business' 'law' 'psychology' 'biology' 'chemistry' 'history' 'other' 'health' 'economics' 'math' 'physics' 'computer science' 'philosophy' 'engineering')


MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
INPUT_DIR="result"
OUTPUT_DIR="likelihood"
MODELS=('llama')

for task in "${tasks[@]}"; do

  if [[ "$task" == "mmlu_pro" ]]; then
    for subject in "${subjects[@]}"; do
      for model in "${MODELS[@]}"; do
        sbatch --job-name="${task}_${subject}_${model}" \
               --output="logs/${task}_${subject}_${model}.txt" \
               --time=72:00:00 \
               --gres=gpu:1 \
               --partition=suma_a6000 \
               --wrap="source ~/.bashrc; conda activate proj2; \
                        python -u ${task}/${task}_likelihood_llama.py \
                        --model ${model} \
                        --model_name ${MODEL_NAME} \
                        --task ${task} \
                        --subject '${subject}' \
                        --input_dir ${INPUT_DIR} \
                        --output_dir ${OUTPUT_DIR}"
      done
    done


  elif [[ "$task" == musr_* ]]; then
    for model in "${MODELS[@]}"; do
      sbatch --job-name="${task}_${model}" \
             --output="logs/${task}_${model}.txt" \
             --time=72:00:00 \
             --gres=gpu:1 \
             --partition=suma_a6000 \
             --wrap="source ~/.bashrc; conda activate proj2; \
                      python -u musr/musr_likelihood_llama.py \
                      --model ${model} \
                      --model_name ${MODEL_NAME} \
                      --task ${task} \
                      --input_dir ${INPUT_DIR} \
                      --output_dir ${OUTPUT_DIR}"
    done


  else
    for model in "${MODELS[@]}"; do
      sbatch --job-name="${task}_${model}" \
             --output="logs/${task}_${model}.txt" \
             --time=72:00:00 \
             --gres=gpu:1 \
             --partition=suma_a6000 \
             --wrap="source ~/.bashrc; conda activate proj2; \
                      python -u ${task}/${task}_likelihood_llama.py \
                      --model ${model} \
                      --model_name ${MODEL_NAME} \
                      --task ${task} \
                      --input_dir ${INPUT_DIR} \
                      --output_dir ${OUTPUT_DIR}"
    done
  fi
done
