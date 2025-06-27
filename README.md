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
   cd math500/latex2sympy2
   pip install -e .
   ```

## Step-by-Step Workflow
### Notes
Most shell scripts inside `sh/` (e.g. `likelihood_all_gpt.sh`, `response_all_gpt.sh`) include `#SBATCH` directives and are meant to be submitted to a Slurm scheduler.

If you're running on a single node machine, Docker, or a cloud provider without Slurm (e.g. Vast.ai), please refer to the example script without Slurm in the same folder and write a new one.

— for example:
```
bash sh/vast_likelihood_all_gpt.sh
```

We basically ran our experiments using the A6000 GPU.


### Step 1: Generate Model Outputs
- Open and run `generate.ipynb`.
- For LLaMA models, leverage the [`lm-eval-harness`](https://github.com/EleutherAI/lm-evaluation-harness/tree/main).
- `parquet.ipynb`: This file creates prompts for use with LLaMA and saves them in Parquet format.
- The accuracy can be checked by running it on the `.ipynb` file in each task folder.

### Step 2: Compute Likelihoods
- Run the likelihood script:
  ```bash
  ./sh/likelihood_all_gpt.sh
  ```
- Ensure you have set the correct `model_name`, `input_dir`, and `output_dir` variables at the the script.
- After running this file, it will create an `all_likelihoods.json` file in `output_dir`.
- This file is used to calculate the `backward consistency` score.

### Step 3: Compute Baselines
- Execute the response-based baseline script:
  ```bash
  ./sh/response_all_gpt.sh
  ```
- Verify the same configuration variables (`model_name`, `input_dir`, `output_dir`) in this script as well.
- After running this file, it will create an `{task}_few_few.jsonl` and `{task}_few_zero.jsonl`  file in `output_dir`.
- This `{task}_few_few.jsonl` file is used to calculate the `forward confidence` score, and the `{task}_few_zero.jsonl` file is used to calculate the `direct` score.

### Step 4: Update Correctness Annotations
- After likelihood computation, update the `"is_correct"` key in the `all_likelihoods.json` file.
- Helper notebooks are available in each task folder (e.g., `math500/math500.ipynb`) to guide this update.
- You should be able to see the `update_predictions_with_is_correct` function.

### Step 5: Apply Custom Evaluation Methods
- Use `check.ipynb` to run our method on updated likelihood data.
- You can check `direct_score`, `forward_score`, `backward_score` and `referi` which represent our final score. 
- Also you can see the `no_replace` related metrics, see appendix B of our paper.

### Step 6: Evaluate Baseline Strategies
- `cot_wp.ipynb`: This is an implementation of the paper [Chain-of-Thought Reasoning Without Prompting](https://arxiv.org/abs/2402.10200).
  * Requires the `{task}_few_few.jsonl` file generated in **Step 3**.
- `USC`: This baseline is based on [Universal Self-Consistency for Large Language Model Generation](https://arxiv.org/abs/2311.17311).
- `LEAP`: This baseline is based on [In-Context Principle Learning from Mistakes](https://arxiv.org/abs/2402.05403).

## Baselines Directory
- The `baselines/` folder contains below..
  - `response_likelihood_*.py`: File that calculates the forward, direct score required by our metric.
---

## Acknowledgements
We adapted the original implementations from the **reference repositories** of each benchmark as listed below.

| Benchmark | Reference repository |
|-----------|---------------------|
| **MATH500** | <https://github.com/QwenLM/Qwen2.5-Math> |
| **MMLU-Pro** | <https://github.com/TIGER-AI-Lab/MMLU-Pro> |
| **GPQA** | <https://github.com/idavidrein/gpqa> |
| **HotpotQA** | <https://github.com/bbuing9/ICLR24_SuRe> |
| **DROP** | <https://github.com/allenai/allennlp-reading-comprehension> <br>  `allennlp_rc/eval/drop_eval.py` |
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