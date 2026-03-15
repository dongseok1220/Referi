#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL="${MODEL:-gpt-4o-mini}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-1B-Instruct}"
TASKS_CSV="${TASKS_CSV:-math500,mmlu_pro,gpqa,drop,hotpotqa,musr_location,musr_efficiently}"
SUBJECTS_CSV="${SUBJECTS_CSV:-business,law,psychology,biology,chemistry,history,other,health,economics,math,physics,computer science,philosophy,engineering}"
GENERATE_SHOTS="${GENERATE_SHOTS:-few,zero}"
SHOTS="${SHOTS:-few}"
NUM_EXAMPLES="${NUM_EXAMPLES:-1}"
TEMPERATURE="${TEMPERATURE:-1}"
N="${N:-5}"

TEST_ROOT="${TEST_ROOT:-${ROOT_DIR}/test}"
RESULT_DIR="${RESULT_DIR:-${TEST_ROOT}/result}"
FORWARD_DIR="${FORWARD_DIR:-${TEST_ROOT}/forward}"
BACKWARD_DIR="${BACKWARD_DIR:-${TEST_ROOT}/backward}"
BASELINE_DIR="${BASELINE_DIR:-${TEST_ROOT}/baselines}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required for test.sh because generation uses generate_gpt.py." >&2
  exit 1
fi

IFS=',' read -r -a TASKS <<< "$TASKS_CSV"
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
for i in "${!SUBJECTS[@]}"; do
  SUBJECTS[$i]="$(trim "${SUBJECTS[$i]}")"
done

for task in "${TASKS[@]}"; do
  if [[ "$task" == "mmlu_pro" ]]; then
    for subject in "${SUBJECTS[@]}"; do
      topk_path="${ROOT_DIR}/embedding_oneshot/sim/mmlu_pro/${subject}_k.jsonl"
      if [[ ! -f "$topk_path" ]]; then
        echo "Missing top-k file: $topk_path" >&2
        exit 1
      fi
    done
  else
    topk_path="${ROOT_DIR}/embedding_oneshot/sim/${task}/${task}_k.jsonl"
    if [[ ! -f "$topk_path" ]]; then
      echo "Missing top-k file: $topk_path" >&2
      exit 1
    fi
  fi
done

rm -rf "$TEST_ROOT"
mkdir -p "$TEST_ROOT"

cd "$ROOT_DIR"

PYTHON_BIN="$PYTHON_BIN" RESULT_DIR="$RESULT_DIR" TASKS_CSV="$TASKS_CSV" MODEL_LABELS_CSV="$MODEL" SHOTS_CSV="$GENERATE_SHOTS" SUBJECTS_CSV="$SUBJECTS_CSV" NUM_EXAMPLES="$NUM_EXAMPLES" TEMPERATURE="$TEMPERATURE" N="$N" \
  bash "${ROOT_DIR}/sh/local/run_generate.sh"

PYTHON_BIN="$PYTHON_BIN" RESULT_DIR="$RESULT_DIR" OUTPUT_ROOT="$BASELINE_DIR" TASKS_CSV="$TASKS_CSV" MODEL_LABELS_CSV="$MODEL" SHOTS_CSV="$SHOTS" SUBJECTS_CSV="$SUBJECTS_CSV" \
  bash "${ROOT_DIR}/sh/local/run_usc.sh"

PYTHON_BIN="$PYTHON_BIN" RESULT_DIR="$RESULT_DIR" OUTPUT_ROOT="$BASELINE_DIR" TASKS_CSV="$TASKS_CSV" MODEL_LABELS_CSV="$MODEL" SHOTS_CSV="$SHOTS" SUBJECTS_CSV="$SUBJECTS_CSV" \
  bash "${ROOT_DIR}/sh/local/run_llm_as_judge.sh"

PYTHON_BIN="$PYTHON_BIN" RESULT_DIR="$RESULT_DIR" OUTPUT_ROOT="$RESULT_DIR" TASKS_CSV="$TASKS_CSV" MODEL_LABELS_CSV="$MODEL" SHOTS_CSV="$SHOTS" SUBJECTS_CSV="$SUBJECTS_CSV" \
  CONF_MODEL_NAME="$MODEL_NAME" \
  bash "${ROOT_DIR}/sh/local/run_self_certainty.sh"

PYTHON_BIN="$PYTHON_BIN" RESULT_DIR="$RESULT_DIR" FORWARD_DIR="$FORWARD_DIR" TASKS_CSV="$TASKS_CSV" MODEL_LABELS_CSV="$MODEL" SHOTS_CSV="$SHOTS" SUBJECTS_CSV="$SUBJECTS_CSV" MODEL_NAME="$MODEL_NAME" NUM_EXAMPLES="$NUM_EXAMPLES" \
  bash "${ROOT_DIR}/sh/local/run_forward.sh"

PYTHON_BIN="$PYTHON_BIN" RESULT_DIR="$RESULT_DIR" BACKWARD_DIR="$BACKWARD_DIR" TASKS_CSV="$TASKS_CSV" MODEL_LABELS_CSV="$MODEL" SUBJECTS_CSV="$SUBJECTS_CSV" MODEL_NAME="$MODEL_NAME" STYLE=auto \
  bash "${ROOT_DIR}/sh/local/run_backward.sh"

echo
echo "Smoke test artifacts are ready under: $TEST_ROOT"
echo "Evaluate with:"
echo "MODELS_CSV='$MODEL' RESULT_DIR='$RESULT_DIR' FORWARD_DIR='$FORWARD_DIR' BACKWARD_DIR='$BACKWARD_DIR' BASELINE_DIR='$BASELINE_DIR' OUTPUT_ROOT='${TEST_ROOT}/evaluation' COT_TOKENIZER='$MODEL_NAME' bash sh/run_evaluate.sh"
