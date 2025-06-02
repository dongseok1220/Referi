#!/usr/bin/env bash

source ~/.bashrc
conda activate proj


MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
INPUT_DIR="result_gpt"
OUTPUT_DIR="likelihood_qwen"


MODELS=("gpt-4o-mini" "gpt-4o")
TASKS=("math500" "mmlu_pro" "gpqa" "drop" "hotpotqa" "musr_efficiently" "musr_location")
SUBJECTS=('business' 'law' 'psychology' 'biology' 'chemistry' 'history' 'other' 'health' 'economics' 'math' 'physics' 'computer science' 'philosophy' 'engineering')



declare -a JOBS=()
for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do

    if [[ "$task" == "mmlu_pro" ]]; then
      for subject in "${SUBJECTS[@]}"; do
        JOBS+=(
          "python mmlu_pro/mmlu_pro_likelihood_gpt.py \
            --task $task \
            --subject $subject \
            --model $model \
            --model_name $MODEL_NAME \
            --input_dir $INPUT_DIR \
            --output_dir $OUTPUT_DIR"
        )
      done

    elif [[ "$task" == musr_* ]]; then
      JOBS+=(
        "python musr/musr_likelihood_gpt.py \
          --task $task \
          --model $model \
          --model_name $MODEL_NAME \
          --input_dir $INPUT_DIR \
          --output_dir $OUTPUT_DIR"
      )

    else
      JOBS+=(
        "python $task/${task}_likelihood_gpt.py \
          --task $task \
          --model $model \
          --model_name $MODEL_NAME \
          --input_dir $INPUT_DIR \
          --output_dir $OUTPUT_DIR"
      )

    fi
  done
done


EXTRA_MODELS=("gpt-4o" "gpt-4o-mini")
EXTRA_TASKS=("math500" "mmlu_pro" "gpqa" "drop" "hotpotqa" "musr_efficiently" "musr_location")
EXTRA_INPUT_DIRS=("result")

for model in "${EXTRA_MODELS[@]}"; do
  for task in "${EXTRA_TASKS[@]}"; do
    for input_dir in "${EXTRA_INPUT_DIRS[@]}"; do
      output_dir="${input_dir}_baseline"
      for shot_type in "few" "zero"; do
        JOBS+=(
          "python -u baselines/response_likelihood_gpt.py \
            --model $model \
            --task $task \
            --input_dir $input_dir \
            --output_dir $output_dir \
            --shot_type $shot_type"
        )
      done
    done
  done
done



mkdir -p logs



GPU_IDS=($(nvidia-smi --query-gpu=index --format=csv,noheader))
NUM_GPUS=${#GPU_IDS[@]} || { echo "No GPU detected"; exit 1; }

declare -A PID_OF
is_free() {
  local gid=$1
  local pid=${PID_OF[$gid]}
  [[ -z $pid || ! -e /proc/$pid ]]
}
launch() {
  local gid=$1; shift
  local job_id=$1; shift
  local cmd="$*"
  export CUDA_VISIBLE_DEVICES=$gid

  local logfile="logs/job_${job_id}_gpu${gid}.log"
  echo "[`date '+%F %T'`] ▶️  Job $job_id on GPU $gid: $cmd" | tee -a "$logfile"
  bash -c "$cmd" &>> "$logfile" &
  PID_OF[$gid]=$!
}



next_job=0
total_jobs=${#JOBS[@]}

while (( next_job < total_jobs )); do
  launched=false
  for gid in "${GPU_IDS[@]}"; do
    if is_free $gid; then
      launch $gid $next_job "${JOBS[$next_job]}"
      (( next_job++ ))
      launched=true
      (( next_job == total_jobs )) && break
    fi
  done
  $launched || sleep 2
done

for pid in "${PID_OF[@]}"; do
  wait $pid
done

echo "✅  All $total_jobs jobs finished"
