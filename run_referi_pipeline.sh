#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL="${MODEL:-llama}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
STYLE="${STYLE:-auto}"

RESULT_DIR="${RESULT_DIR:-result}"
FORWARD_DIR="${FORWARD_DIR:-forward}"
BACKWARD_DIR="${BACKWARD_DIR:-backward}"
REFERI_OUT="${REFERI_OUT:-referi.txt}"

TASKS_CSV="${TASKS_CSV:-math500,mmlu_pro,gpqa,drop,hotpotqa,musr_location,musr_efficiently}"
SHOTS_CSV="${SHOTS_CSV:-few}"
SUBJECTS_CSV="${SUBJECTS_CSV:-business,law,psychology,biology,chemistry,history,other,health,economics,math,physics,computer science,philosophy,engineering}"

NUM_EXAMPLES="${NUM_EXAMPLES:--1}"
TEMPERATURE="${TEMPERATURE:-1}"
N="${N:-5}"

RUN_GENERATE=1
RUN_EVALUATE=1
RUN_FORWARD=1
RUN_BACKWARD=1
RUN_REFERI=1

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --model LABEL
  --model-name NAME
  --style {auto|gpt|llama}
  --result-dir DIR
  --forward-dir DIR
  --backward-dir DIR
  --referi-out FILE
  --tasks CSV
  --shots CSV
  --subjects CSV
  --num-examples N
  --temperature T
  --n N
  --skip-generate
  --skip-evaluate
  --skip-forward
  --skip-backward
  --skip-referi
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --style)
      STYLE="$2"
      shift 2
      ;;
    --result-dir)
      RESULT_DIR="$2"
      shift 2
      ;;
    --forward-dir)
      FORWARD_DIR="$2"
      shift 2
      ;;
    --backward-dir)
      BACKWARD_DIR="$2"
      shift 2
      ;;
    --referi-out)
      REFERI_OUT="$2"
      shift 2
      ;;
    --tasks)
      TASKS_CSV="$2"
      shift 2
      ;;
    --shots)
      SHOTS_CSV="$2"
      shift 2
      ;;
    --subjects)
      SUBJECTS_CSV="$2"
      shift 2
      ;;
    --num-examples)
      NUM_EXAMPLES="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --n)
      N="$2"
      shift 2
      ;;
    --skip-generate)
      RUN_GENERATE=0
      shift
      ;;
    --skip-evaluate)
      RUN_EVALUATE=0
      shift
      ;;
    --skip-forward)
      RUN_FORWARD=0
      shift
      ;;
    --skip-backward)
      RUN_BACKWARD=0
      shift
      ;;
    --skip-referi)
      RUN_REFERI=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

IFS=',' read -r -a TASKS <<< "$TASKS_CSV"
IFS=',' read -r -a SHOTS <<< "$SHOTS_CSV"
IFS=',' read -r -a SUBJECTS <<< "$SUBJECTS_CSV"

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

for i in "${!TASKS[@]}"; do
  TASKS[$i]="$(trim "${TASKS[$i]}")"
done
for i in "${!SHOTS[@]}"; do
  SHOTS[$i]="$(trim "${SHOTS[$i]}")"
done
for i in "${!SUBJECTS[@]}"; do
  SUBJECTS[$i]="$(trim "${SUBJECTS[$i]}")"
done

TASKS_CSV="$(IFS=,; echo "${TASKS[*]}")"
SHOTS_CSV="$(IFS=,; echo "${SHOTS[*]}")"

resolve_style() {
  if [[ "$STYLE" != "auto" ]]; then
    printf '%s\n' "$STYLE"
    return
  fi
  shopt -s nocasematch
  if [[ "$MODEL" == *llama* ]]; then
    printf 'llama\n'
  else
    printf 'gpt\n'
  fi
  shopt -u nocasematch
}

STYLE_RESOLVED="$(resolve_style)"

run_generate() {
  echo "== Generate =="
  if [[ "$STYLE_RESOLVED" == "llama" ]]; then
    echo "Skipping generation for llama. Prepare ${RESULT_DIR} first with lm-eval-harness."
    return
  fi

  "${PYTHON_BIN}" -u generate_gpt.py \
    --model "$MODEL" \
    --tasks "$TASKS_CSV" \
    --shots "$SHOTS_CSV" \
    --output-dir "$RESULT_DIR" \
    --num-examples "$NUM_EXAMPLES" \
    --temperature "$TEMPERATURE" \
    --n "$N"
}

run_evaluate() {
  echo "== Evaluate =="
  for task in "${TASKS[@]}"; do
    "${PYTHON_BIN}" -u scripts/evaluate.py \
      --task "$task" \
      --models "$MODEL" \
      --shot_types "$SHOTS_CSV" \
      --base_dir "$RESULT_DIR"
  done
}

run_forward() {
  echo "== Forward =="
  for task in "${TASKS[@]}"; do
    for shot in "${SHOTS[@]}"; do
      "${PYTHON_BIN}" -u scripts/forward.py \
        --model "$MODEL" \
        --model_name "$MODEL_NAME" \
        --task "$task" \
        --input_dir "$RESULT_DIR" \
        --output_dir "$FORWARD_DIR" \
        --shot_type "$shot"
    done
  done
}

run_backward_one() {
  local task="$1"
  shift
  "${PYTHON_BIN}" -u scripts/backward.py \
    --model "$MODEL" \
    --model_name "$MODEL_NAME" \
    --task "$task" \
    --style "$STYLE_RESOLVED" \
    --input_dir "$RESULT_DIR" \
    --output_dir "$BACKWARD_DIR" \
    "$@"
}

run_backward() {
  echo "== Backward =="
  for task in "${TASKS[@]}"; do
    if [[ "$task" == "mmlu_pro" ]]; then
      for subject in "${SUBJECTS[@]}"; do
        run_backward_one "$task" --subject "$subject"
      done
    else
      run_backward_one "$task"
    fi
  done
}

run_referi() {
  echo "== Referi =="
  "${PYTHON_BIN}" -u scripts/check_referi.py \
    --models "$MODEL" \
    --tasks "${TASKS[@]}" \
    --forward_dir "$FORWARD_DIR" \
    --backward_dir "$BACKWARD_DIR" \
    --result_dir "$RESULT_DIR" | tee "$REFERI_OUT"
}

cd "$ROOT_DIR"

if [[ "$RUN_GENERATE" == "1" ]]; then
  run_generate
fi

if [[ "$RUN_EVALUATE" == "1" ]]; then
  run_evaluate
fi

if [[ "$RUN_FORWARD" == "1" ]]; then
  run_forward
fi

if [[ "$RUN_BACKWARD" == "1" ]]; then
  run_backward
fi

if [[ "$RUN_REFERI" == "1" ]]; then
  run_referi
fi
