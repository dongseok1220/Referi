#!/bin/bash

mkdir -p logs

tasks=("math500" "mmlu_pro" "gpqa" "drop" "gpqa" "musr_efficiently" "musr_location")
subjects=('business' 'law' 'psychology' 'biology' 'chemistry' 'history' 'other' 'health' 'economics' 'math' 'physics' 'computer science' 'philosophy' 'engineering')


MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
INPUT_DIR="result"
OUTPUT_DIR="likelihood"

for task in "${tasks[@]}"; do

  if [[ "$task" == "mmlu_pro" ]]; then
    for subject in "${subjects[@]}"; do
      for gpt in 'gpt-4o-mini' 'gpt-4o'; do
        sbatch --job-name="${task}_${subject}_${gpt}" \
               --output="logs/${task}_${subject}_${gpt}.txt" \
               --time=72:00:00 \
               --gres=gpu:1 \
               --partition=suma_a6000 \
               --wrap="source ~/.bashrc; conda activate proj2; \
                        python -u ${task}/${task}_likelihood_gpt.py \
                        --model ${gpt} \
                        --model_name ${MODEL_NAME} \
                        --task ${task} \
                        --subject ${subject} \
                        --input_dir ${INPUT_DIR} \
                        --output_dir ${OUTPUT_DIR}"
      done
    done


  elif [[ "$task" == musr_* ]]; then
    for gpt in 'gpt-4o-mini' 'gpt-4o'; do
      sbatch --job-name="${task}_${gpt}" \
             --output="logs/${task}_${gpt}.txt" \
             --time=72:00:00 \
             --gres=gpu:1 \
             --partition=suma_a6000 \
             --wrap="source ~/.bashrc; conda activate proj2; \
                      python -u musr/musr_likelihood_gpt.py \
                      --model ${gpt} \
                      --model_name ${MODEL_NAME} \
                      --task ${task} \
                      --input_dir ${INPUT_DIR} \
                      --output_dir ${OUTPUT_DIR}"
    done


  else
    for gpt in 'gpt-4o-mini' 'gpt-4o'; do
      sbatch --job-name="${task}_${gpt}" \
             --output="logs/${task}_${gpt}_${seed}.txt" \
             --time=72:00:00 \
             --gres=gpu:1 \
             --partition=suma_a6000 \
             --wrap="source ~/.bashrc; conda activate proj2; \
                      python -u ${task}/${task}_likelihood_gpt.py \
                      --model ${gpt} \
                      --model_name ${MODEL_NAME} \
                      --task ${task} \
                      --input_dir ${INPUT_DIR} \
                      --output_dir ${OUTPUT_DIR}"
    done
  fi
done
