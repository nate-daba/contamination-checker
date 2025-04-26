# Benchmark Contamination Testing with LLMSanitize (Modified)

This repository builds upon the [LLMSanitize](https://github.com/ntunlp/LLMSanitize) framework for detecting benchmark contamination in large language models (LLMs). It adapts, extends, and evaluates contamination detection techniques on recent open-source reasoning models and math-focused datasets, such as AIME-2024 and subsets of MMLU.

> **NOTE:** This is a forked and modified version of LLMSanitize for targeted experiments. For the original implementation and full methodology, please refer to:
>
> ```
> @article{ravaut2024much,
>   title={How Much are LLMs Contaminated? A Comprehensive Survey and the LLMSanitize Library},
>   author={Ravaut, Mathieu and Ding, Bosheng and Jiao, Fangkai and Chen, Hailin and Li, Xingxuan and Zhao, Ruochen and Qin, Chengwei and Xiong, Caiming and Joty, Shafiq},
>   journal={arXiv preprint arXiv:2404.00699},
>   year={2024}
> }
> ```

---

## Overview

This project explores contamination in LLMs by testing how well models can regenerate benchmark data under memorization-style prompting. It specifically builds on two methods from LLMSanitize:

- **TS-Guessing (Testset Slot Guessing)** (introduced [here](https://arxiv.org/abs/2311.09783))
- **Guided Prompting** (introduced [here](https://arxiv.org/abs/2308.08493))

Modifications in this fork include:

- Adapting instruction templates for math-specific benchmarks (AIME-2024)
- Creating stricter variations of guided prompts for more faithful verbatim regeneration
- Modifying splitting functions for MMLU and AIME to match realistic first-part/second-part decomposition
- Automating analysis using bootstrap hypothesis testing to determine significance of contamination-like behavior
- Logging and evaluation of models such as `DeepSeek-R1-Distill-Qwen-7B` and `Qwen2.5-Math-7B-Instruct`

---

## Setup Instructions

### 1. Clone this repository
```bash
git clone https://github.com/nate-daba/detect-benchmark-contamination.git
cd detect-benchmark-contamination
```

### 2. Create and activate a virtual environment (optional but recommended)
```bash
conda create -n llmsanitize python=3.10 -y
conda activate llmsanitize
```

Or using `venv`:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

To run the `Guided Prompting` contamination detection method on the AIME-2024 dataset with a specific model, use the following command:
```bash
sh tests/closed_data/guided-prompting/test_aime.sh --model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

To run the `TS-Guessing` contamination detection method on the MMLU dataset (100 questions in the “high-school mathematics” category) with a specific model, use the following command:
```bash
sh tests/closed_data/ts-guessing-question-based/test_mmlu.sh --model=Qwen/Qwen2.5-Math-7B-Instruct
```

Logs will be saved with prompts, responses, and bootstrap statistics in the `output/` folder.

## License
This repository inherits the license from LLMSanitize (Apache 2.0). See `LICENSE` for details.

## Acknowledgments
Special thanks to the LLMSanitize authors and maintainers for their comprehensive framework and open-source release.

