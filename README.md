# LLMSanitize
An open-source library for contamination detection in NLP datasets and Large Language Models (LLMs).  

## Installation
The library has been designed and tested with **Python 3.9** and **CUDA 11.8**.  

First make sure you have CUDA 11.8 installed, and create a conda environment with Python 3.9: 
```bash
conda create -name llmsanitize python=3.9
```

Next activate the environment:
```bash
conda activate llmsanitize
```

Then install all the dependencies for LLMSanitize:
```bash
pip install -r requirements.txt
```

Alternatively, you can combine the three steps above by just running:  
```bash
sh scripts/install.sh
```

Notably, we use the following important libraries:
- datasets 2.17.1
- einops 0.7.0
- huggingface-hub 0.20.3
- openai 0.27.8
- torch 2.1.2
- transformers 4.38.0
- vllm 0.3.3

## Supported Methods
So far we support the following contamination detection methods:

| **Method** | **Use Case** | **Short description** | **White-box access?** | **Reference** |  
|-----------------------------------|---|---|---|---|
| gpt-2 | data contamination | String matching | _ | [paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) |
| gpt-3 | data contamination | String matching | _ | [paper](https://arxiv.org/abs/2005.14165) |
| exact | data contamination | String matching | _ | [paper](https://arxiv.org/abs/2104.08758) |
| palm | data contamination | String matching | _ | [paper](https://arxiv.org/abs/2204.02311) |
| gpt-4 | data contamination | String matching | _ | [paper](https://arxiv.org/abs/2303.08774) |
| platypus | data contamination | Embeddings similarity | _ | [paper](https://arxiv.org/abs/2308.07317) |
| guided-prompting | model contamination | Likelihood | no | [paper](https://arxiv.org/abs/2308.08493) |
| sharded-likelihood | model contamination | Likelihood | yes | [paper](https://arxiv.org/abs/2310.17623) |
| min-prob | model contamination | LLM-based method | yes | [paper](https://arxiv.org/abs/2310.16789) |
| cdd | model contamination | Likelihood | no | [paper](https://arxiv.org/abs/2402.15938) |
| ts-guessing-question-based | model contamination | Completion | no | [paper](https://arxiv.org/abs/2311.09783) |
| ts-guessing-question-multichoice | model contamination | Completion | no | [paper](https://arxiv.org/abs/2311.09783) |

## vLLM
The following methods require to launch a vLLM instance which will handle model inference:

| **Method** | 
|---|
| guided-prompting |
| min-prob |
| cdd |
| ts-guessing-question-based |
| ts-guessing-question-multichoice |

To launch the instance, first run the following command in a terminal: 
```bash
sh scripts/vllm_hosting.sh
```
You are required to specify a **port number** and **model name** in this shell script. 

## Run Contamination Detection
To run contamination detection, follow the multiple test scripts in scripts/tests/ folder.  

For instance, to run sharded-likelihood on Hellaswag with Llama-2-7B:
```bash
sh scripts/tests/model/sharded-likelihood/test_hellaswag.sh -m <path_to_your_llama-2-7b_folder> 
```

To run a method using vLLM like guided-prompting for instance, the only difference is to pass the port number as argument:
```bash
sh scripts/tests/model/guided-prompting/test_hellaswag.sh -m <path_to_your_llama-2-7b_folder> -p <port_number_from_your_vllm_instance>
```


## Citation

If you find our paper or this project helps your research, please kindly consider citing our paper in your publication.


```
@article{ravaut2024much,
  title={How Much are LLMs Contaminated? A Comprehensive Survey and the LLMSanitize Library},
  author={Ravaut, Mathieu and Ding, Bosheng and Jiao, Fangkai and Chen, Hailin and Li, Xingxuan and Zhao, Ruochen and Qin, Chengwei and Xiong, Caiming and Joty, Shafiq},
  journal={arXiv preprint arXiv:2404.00699},
  year={2024}
}
```
