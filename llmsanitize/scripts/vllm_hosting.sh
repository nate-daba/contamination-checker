#!/bin/bash

# needed for the following methods:
# 1/ guided-prompting
# 2/ min-prob
# Run this script in one tab first, then run the script calling the method in another tab

# Set default model name
DEFAULT_MODEL="meta-llama/Llama-2-7b-chat-hf"
MODEL_NAME=$DEFAULT_MODEL

# Parse command line arguments
while [ $# -gt 0 ]; do
  case $1 in
    --model=*)
      MODEL_NAME="${1#*=}"
      shift
      ;;
    --model)
      if [ $# -gt 1 ]; then
        MODEL_NAME="$2"
        shift 2
      else
        echo "Error: Missing argument for --model"
        exit 1
      fi
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--model MODEL_NAME]"
      exit 1
      ;;
  esac
done

# Export environment variables
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export RAY_memory_monitor_refresh_ms=0
export CUDA_VISIBLE_DEVICES=2,3

export HF_HOME=/workspace/ndaba/hf_cache
export TRANSFORMERS_CACHE=/workspace/ndaba/hf_cache
export HF_DATASETS_CACHE=/workspace/ndaba/hf_cache

export CLASSPATH="/workspace/ndaba/research/code/LLMSanitize/postagger/stanford-postagger-full-2020-11-17/stanford-postagger.jar"
export STANFORD_MODELS="/workspace/ndaba/research/code/LLMSanitize/postagger/stanford-postagger-full-2020-11-17/models"

# Create directories if they don't exist
mkdir -p /workspace/ndaba/hf_cache
mkdir -p /workspace/ndaba/vllm_cache

server_type=vllm.entrypoints.openai.api_server

echo "Starting server with model: $MODEL_NAME"

python3 -m $server_type \
    --model "$MODEL_NAME" \
    --tokenizer "$MODEL_NAME" \
    --gpu-memory-utilization=0.9 \
    --max-num-seqs=200 \
    --disable-log-requests \
    --host 127.0.0.1 --port 6001 \
    --tensor-parallel-size 1 \
    --download-dir /workspace/ndaba/vllm_cache