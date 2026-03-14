# Training-free LLM Verification via Recycling Few-shot Examples
This repository provides the code for the following paper [Training-free LLM Verification via Recycling Few-shot Examples](https://arxiv.org/abs/2506.17251), and provides the responses and data used in the experiment to reproduce the experiment.


## Environment Setup

1. **Create a new conda environment with Python 3.10**
   ```bash
   conda create -n my_env python=3.10
   ```

2. **Activate the environment and install dependencies**
   ```bash
   conda activate my_env
   pip install -r requirements.txt
   ```

3. **Install LaTeX-to-SymPy converter**
   ```bash
   cd tasks/math500/latex2sympy2
   pip install -e .
   ```

## Workflow
### Main pipeline
The default experiment tasks are:
- `math500`
- `mmlu_pro`
- `gpqa`
- `drop`
- `hotpotqa`
- `musr_location`
- `musr_efficiently`

For a single-machine run:

```bash
bash run_referi_pipeline.sh --model gpt-4o-mini --style gpt
```

For LLaMA generation, keep using [`lm-eval-harness`](https://github.com/EleutherAI/lm-evaluation-harness/tree/main) to prepare `result/...`, then skip generation here:

```bash
bash run_referi_pipeline.sh --model llama --style llama --skip-generate
```

Current directories:
- `result/`: generated model outputs and evaluation files
- `forward/`: forward scores
- `backward/`: backward likelihood files
- `embedding_oneshot/sim/`: top-k retrieval files used by ReFeri

### Manual steps
1. Generate outputs

```bash
python generate_gpt.py \
  --model gpt-4o-mini \
  --tasks math500,mmlu_pro,gpqa,drop,hotpotqa,musr_location,musr_efficiently \
  --shots few,zero \
  --output-dir result \
  --n 5
```

2. Evaluate generated outputs

```bash
python scripts/evaluate.py \
  --task math500 \
  --models gpt-4o-mini \
  --shot_types few,zero \
  --base_dir result
```

3. Compute forward scores

```bash
python scripts/forward.py \
  --task math500 \
  --model gpt-4o-mini \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --input_dir result \
  --output_dir forward \
  --shot_type few
```

4. Compute backward scores

```bash
python scripts/backward.py \
  --task math500 \
  --model gpt-4o-mini \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --style gpt \
  --input_dir result \
  --output_dir backward
```

5. Print the final ReFeri table

```bash
python scripts/check_referi.py \
  --models gpt-4o-mini \
  --tasks math500 mmlu_pro gpqa drop hotpotqa musr_location musr_efficiently \
  --forward_dir forward \
  --backward_dir backward \
  --result_dir result
```

### Smoke test
`test.sh` runs a one-sample smoke test over all main tasks and all `mmlu_pro` subjects. It checks the full path:

`generate -> evaluate -> forward -> backward -> referi`

Requirements:
- `OPENAI_API_KEY` must be set
- top-k files under `embedding_oneshot/sim/...` must already exist

Run:

```bash
bash test.sh
```

## Acknowledgements
We adapted the original implementations from the **reference repositories** of each benchmark as listed below.

| Benchmark | Reference repository |
|-----------|---------------------|
| **MATH500** | <https://github.com/QwenLM/Qwen2.5-Math> |
| **MMLU-Pro** | <https://github.com/TIGER-AI-Lab/MMLU-Pro> |
| **GPQA** | <https://github.com/idavidrein/gpqa> |
| **HotpotQA** | <https://github.com/bbuing9/ICLR24_SuRe> |
| **DROP** | <https://github.com/allenai/allennlp-reading-comprehension> |
| **MuSR** | <https://github.com/Zayne-sprague/MuSR> <br> <https://github.com/Zayne-sprague/To-CoT-or-not-to-CoT> |


## Citation 
If you find this work useful for your research, please cite our papers:
```
@article{lee2025training,
  title={Training-free LLM Verification via Recycling Few-shot Examples},
  author={Lee, Dongseok and Hong, Jimyung and Kim, Dongyoung and Kim, Jaehyung},
  journal={arXiv preprint arXiv:2506.17251},
  year={2025}
}
```
