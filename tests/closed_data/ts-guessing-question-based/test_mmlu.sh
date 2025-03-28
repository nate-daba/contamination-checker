# # Get the options
# while getopts ":p:m:" option; do
#    case $option in
#       p) # port number
#          port=$OPTARG;;
#       m) # Enter closed_data name
#          model_name=$OPTARG;;
#    esac
# done

# echo "model name ", $model_name
# echo "local port: ", $port

port=6001
model_name=meta-llama/Llama-2-7b-chat-hf

export HF_HOME=/workspace/ndaba/hf_cache
export TRANSFORMERS_CACHE=/workspace/ndaba/hf_cache
export HF_DATASETS_CACHE=/workspace/ndaba/hf_cache

# Set environment variables for Stanford POS tagger
export CLASSPATH="/workspace/ndaba/research/code/LLMSanitize/postagger/stanford-postagger-full-2020-11-17/stanford-postagger.jar"
export STANFORD_MODELS="/workspace/ndaba/research/code/LLMSanitize/postagger/stanford-postagger-full-2020-11-17/models"

# Wait until the vLLM server is ready
echo "Waiting for vLLM server at port 6001..."
until curl -s http://127.0.0.1:6001/v1/completions -o /dev/null; do
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
--num_proc 8 \
--method ts-guessing-question-based \
--local_port $port \
--model_name $model_name \
#--ts_guessing_type_hint \
#--ts_guessing_category_hint \
#--ts_guessing_url_hint \
