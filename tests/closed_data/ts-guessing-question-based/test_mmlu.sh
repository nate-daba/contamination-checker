#!/bin/bash
# Set default values
port=6001
model_name="meta-llama/Llama-2-7b-chat-hf"
# Parse command line arguments
while [ $# -gt 0 ]; do
  case $1 in
    --model=*)
      model_name="${1#*=}"
      shift
      ;;
    --model)
      if [ $# -gt 1 ]; then
        model_name="$2"
        shift 2
      else
        echo "Error: Missing argument for --model"
        exit 1
      fi
      ;;
    --port=*)
      port="${1#*=}"
      shift
      ;;
    --port)
      if [ $# -gt 1 ]; then
        port="$2"
        shift 2
      else
        echo "Error: Missing argument for --port"
        exit 1
      fi
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--model MODEL_NAME] [--port PORT]"
      exit 1
      ;;
  esac
done
echo "Model name: $model_name"
echo "Local port: $port"
export HF_HOME=/workspace/ndaba/hf_cache
export TRANSFORMERS_CACHE=/workspace/ndaba/hf_cache
export HF_DATASETS_CACHE=/workspace/ndaba/hf_cache
# Set environment variables for Stanford POS tagger
export CLASSPATH="/workspace/ndaba/research/code/LLMSanitize/postagger/stanford-postagger-full-2020-11-17/stanford-postagger.jar"
export STANFORD_MODELS="/workspace/ndaba/research/code/LLMSanitize/postagger/stanford-postagger-full-2020-11-17/models"
# Wait until the vLLM server is ready
echo "Waiting for vLLM server at port $port..."
until curl -s http://127.0.0.1:$port/v1/completions -o /dev/null; do
    sleep 1
done
echo "vLLM server is ready!"
# test guided prompting closed_data contamination method
python main.py \
--eval_data_name cais/mmlu \
--eval_data_config_name high_school_mathematics \
--eval_set_key test \
--text_key question \
--label_key answer_text \
--n_eval_data_points 100 \
--num_proc 16 \
--method ts-guessing-question-based \
--local_port "$port" \
--model_name "$model_name" \
--max_output_tokens 2048
#--ts_guessing_type_hint \
#--ts_guessing_category_hint \
#--ts_guessing_url_hint \