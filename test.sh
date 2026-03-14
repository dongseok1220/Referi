#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL="${MODEL:-gpt-4o-mini}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-1B-Instruct}"
TASKS_CSV="${TASKS_CSV:-math500,mmlu_pro,gpqa,drop,hotpotqa,musr_location,musr_efficiently}"
SUBJECTS_CSV="${SUBJECTS_CSV:-business,law,psychology,biology,chemistry,history,other,health,economics,math,physics,computer science,philosophy,engineering}"
SHOTS="${SHOTS:-few}"
NUM_EXAMPLES="${NUM_EXAMPLES:-1}"
TEMPERATURE="${TEMPERATURE:-1}"
N="${N:-1}"

TEST_ROOT="${TEST_ROOT:-${ROOT_DIR}/test_artifacts/referi_smoke}"
RESULT_DIR="${RESULT_DIR:-${TEST_ROOT}/result}"
FORWARD_DIR="${FORWARD_DIR:-${TEST_ROOT}/forward}"
BACKWARD_DIR="${BACKWARD_DIR:-${TEST_ROOT}/backward}"
REFERI_OUT="${REFERI_OUT:-${TEST_ROOT}/referi.txt}"

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

PYTHON_BIN="$PYTHON_BIN" bash run_referi_pipeline.sh \
  --model "$MODEL" \
  --model-name "$MODEL_NAME" \
  --style auto \
  --tasks "$TASKS_CSV" \
  --shots "$SHOTS" \
  --subjects "$SUBJECTS_CSV" \
  --num-examples "$NUM_EXAMPLES" \
  --temperature "$TEMPERATURE" \
  --n "$N" \
  --result-dir "$RESULT_DIR" \
  --forward-dir "$FORWARD_DIR" \
  --backward-dir "$BACKWARD_DIR" \
  --referi-out "$REFERI_OUT"

echo
echo "Smoke test output:"
cat "$REFERI_OUT"
