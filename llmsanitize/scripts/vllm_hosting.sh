# needed for the following methods:
# 1/ guided-prompting
# 2/ min-prob
# Run this script in one tab first, then run the script calling the method in another tab
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export RAY_memory_monitor_refresh_ms=0;
export CUDA_VISIBLE_DEVICES=2,3;

export HF_HOME=/workspace/ndaba/hf_cache
export TRANSFORMERS_CACHE=/workspace/ndaba/hf_cache
export HF_DATASETS_CACHE=/workspace/ndaba/hf_cache

export CLASSPATH="/workspace/ndaba/research/code/LLMSanitize/postagger/stanford-postagger-full-2020-11-17/stanford-postagger.jar"
export STANFORD_MODELS="/workspace/ndaba/research/code/LLMSanitize/postagger/stanford-postagger-full-2020-11-17/models"

server_type=vllm.entrypoints.openai.api_server

python3 -m $server_type \
    --model meta-llama/Llama-2-7b-chat-hf \
    --tokenizer meta-llama/Llama-2-7b-chat-hf \
    --gpu-memory-utilization=0.9 \
    --max-num-seqs=200 \
    --disable-log-requests \
    --host 127.0.0.1 --port 6001 --tensor-parallel-size 1 \
    --download-dir /workspace/ndaba/vllm_cache
