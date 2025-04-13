"""
This file implements the closed_data contamination detection through guided prompting.
https://arxiv.org/pdf/2311.09783
"""

import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize
from nltk.tag import StanfordPOSTagger
from functools import partial
from datasets import Dataset 
from datetime import datetime

from llmsanitize.utils.logger import get_child_logger, suspend_logging
from llmsanitize.utils.dataset_utils import get_answers_list
from llmsanitize.closed_data_methods.llm import LLM

logger = get_child_logger("ts_guessing_question_based")


def get_stanford_tagger():
    os.environ["_JAVA_OPTIONS"] = "-Xmx16g"
    if not("CLASSPATH" in os.environ and "STANFORD_MODELS" in os.environ):
        logger.info("You are using a closed_data contamination detection method which requires Stanford's Part-of-Speech tagger.")
        logger.info("You need to setup global variables CLASSPATH and STANFORD_MODELS specifying the path to the tagger.")
        logger.info("First download the tagger here: https://nlp.stanford.edu/software/tagger.html#Download")
        logger.info("Then place it into some directory.")
        home_dir = input("Please specify the directory where you place the tagger (default: /home/mathieu/stanford-postagger-full-2020-11-17): ")
        os.environ["CLASSPATH"] = f"{home_dir}"
        os.environ["STANFORD_MODELS"] = f"{home_dir}/models"
    st = StanfordPOSTagger('english-bidirectional-distsim.tagger')

    return st

def build_prompt(
    example, 
    tagger,
    eval_data_name,
    type_hint=False,
    category_hint=False,
    url_hint=False,
    prompt_style="new"  # Add this parameter, default to new style
):
    text = example["text"]
    tags = tagger.tag(text.split())
    words = [x for x in tags if x[1] in ['NN', 'JJ', 'VB']]
    if len(words) == 0:
        return "failed", ""
    idx = np.random.randint(len(words))
    word = words[idx][0]
    for i in range(len(text)-len(word)+1):
        if text[i:(i+len(word))] == word:
            text = text[:i] + "[MASK]" + text[(i+len(word)):]
            break

    if prompt_style == "old":
        # Old prompting style
        prompt = "Complete the sentence in one word:"
        prompt += f"\n\n{text}"
        if type_hint:
            if eval_data_name == "truthful_qa":
                example_type = example["type"]
                prompt += f"\nHint: {example_type}"
        if category_hint:
            if eval_data_name == "truthful_qa":
                example_category = example["category"]
                prompt += f"\nHint: {example_category}"
        if url_hint:
            if eval_data_name == "truthful_qa":
                example_url = example["source"]
                prompt += f"\nHint: {example_url}"
        prompt += "\nReply the answer only."
    else:
        # Enhanced restrictive prompt style
        # prompt = "[SYSTEM INSTRUCTION: CRITICAL! This is NOT a math problem to solve. This is a MASKED WORD task ONLY. Your ONLY job is to guess what single word was replaced by [MASK]. You must output your answer as EXACTLY ONE WORD inside \\boxed{} notation. Your entire response should look like this: \\boxed{word}. DO NOT attempt to solve any problems or provide explanations. Just predict the masked word and put it in \\boxed{}.]\n\n"
        # prompt += f"Text with masked word: {text}\n\n"
        # original  prompt 
        prompt = "Complete the sentence in one word:"
        prompt += f"\n\n{text}"
        if type_hint:
            if eval_data_name == "truthful_qa":
                example_type = example["type"]
                prompt += f"Context hint: {example_type}\n"
        if category_hint:
            if eval_data_name == "truthful_qa":
                example_category = example["category"]
                prompt += f"Category hint: {example_category}\n"
        if url_hint:
            if eval_data_name == "truthful_qa":
                example_url = example["source"]
                prompt += f"Source hint: {example_url}\n"
        # prompt += "YOUR ANSWER (one word only): "
        # prompt += "YOUR RESPONSE (must be exactly one word): "
        # prompt += "YOUR RESPONSE (must be exactly one word in \\boxed{}): "
        prompt += "\nReply the answer only."

    return prompt, word

def process_response(response):
    """Extract just the word from boxed notation or first word if that fails."""
    # Try to extract content from \boxed{}
    boxed_match = re.search(r'\\boxed\{([^{}]+)\}', response)
    if boxed_match:
        # Return the content inside \boxed{}, stripped of whitespace
        return boxed_match.group(1).strip()
    
    # Fallback to the previous method if no boxed content found
    response = response.strip().lstrip('.,;:!?"\' ')
    if not response:
        return ""
    
    # Split by whitespace and take the first chunk
    first_chunk = response.split()[0] if response.split() else ""
    
    # Remove any trailing punctuation
    first_word = first_chunk.rstrip('.,;:!?"\' ')
    
    return first_word

def setup_stanford_env():
    os.environ['CLASSPATH'] = "/workspace/ndaba/research/code/LLMSanitize/postagger/stanford-postagger-full-2020-11-17/stanford-postagger.jar"
    os.environ["STANFORD_MODELS"] = "/workspace/ndaba/research/code/LLMSanitize/postagger/stanford-postagger-full-2020-11-17/models"
    
def inference(
    data_point,
    eval_data_name,
    llm, 
    type_hint=False,
    category_hint=False,
    url_hint=False,
    prompt_style="new"  # Add this parameter
):
    setup_stanford_env()
    
    tagger = get_stanford_tagger()
    
    prompt, masked_word = build_prompt(
        data_point,
        tagger,
        eval_data_name,
        type_hint,
        category_hint,
        url_hint,
        prompt_style  # Pass it to build_prompt
    )
    data_point["masked_word"] = masked_word
    data_point["full_prompt"] = prompt  # Store the full prompt with [MASK]
    
    if prompt == "failed":
        data_point["response"] = "failed"
        data_point["raw_response"] = "failed"
    else:
        raw_response, cost = llm.query(prompt)
        data_point["raw_response"] = raw_response
        response = process_response(raw_response)
        data_point["response"] = response

    return data_point

@suspend_logging
def filter_data(eval_data, eval_data_name):
    data_points = []
    if eval_data_name == "truthful_qa":
        for x in tqdm(eval_data):
            # Remove questions with 4 or less words
            n_words = len(word_tokenize(x["text"]))
            if n_words <= 4:
                continue
            # Remove questions of 'Indexical Error' category
            if 'Indexical Error' in x["category"]:
                continue 

            data_points.append(x)
    else:
        scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)
        for x in tqdm(eval_data):
            choices = get_answers_list(x, eval_data_name)

            if len(choices) == 2:
                # Remove questions with Yes/No options
                if (choices[0].lower() in ["yes", "no"]) and (choices[1].lower() in ["yes", "no"]):
                    continue
                # Remove questions with True/False options
                if (choices[0].lower() in ["true", "false"]) and (choices[1].lower() in ["true", "false"]):
                    continue

            # Remove open_data points where the ROUGE-L F1 between any 2 options exceeds 0.65
            discard = False
            for i in range(len(choices)):
                for j in range(i+1, len(choices)):
                    choice_i = choices[i]
                    choice_j = choices[j]
                    rouge_scores = scorer.score(choice_i, choice_j)
                    rl = rouge_scores["rougeLsum"].fmeasure
                    if rl >= 0.65:
                        discard = True
                        break 
                if discard == True:
                    break
            if discard == True:
                continue

            data_points.append(x)

    return data_points


def main_ts_guessing_question_based(
    eval_data: list = [],
    eval_data_name: str = None,
    n_eval_data_points: int = 100,
    num_proc: int = 16,
    # closed_data parameters
    local_model_path: str = None,
    local_tokenizer_path: str = None,
    model_name: str = None,
    openai_creds_key_file: str = None,
    local_port: str = None,
    local_api_type: str = None,
    no_chat_template: bool = False,
    num_samples: int = 1,
    max_input_tokens: int = 512,
    max_output_tokens: int = 512,
    temperature: float = 0.0,
    top_logprobs: int = 0,
    max_request_time: int = 600,
    sleep_time: int = 1,
    echo: bool = False,
    # method-specific parameters
    type_hint: bool = False,
    category_hint: bool = False,
    url_hint: bool = False,
    prompt_style: str = "new"
):
    # filter out some open_data points
    data_points = filter_data(eval_data, eval_data_name)
    logger.info(f"We are left with {len(data_points)} data points after filtering")

    # perform the shuffling and subsampling now
    if n_eval_data_points > 0:
        p = np.random.permutation(len(data_points))
        data_points = [data_points[x] for x in p]
        data_points = data_points[:n_eval_data_points]
        logger.info(f"We are left with {len(data_points)} data points after subsampling")
    data_points = Dataset.from_list(data_points)

    llm = LLM(
        local_model_path=local_model_path,
        local_tokenizer_path=local_tokenizer_path,
        model_name=model_name,
        openai_creds_key_file=openai_creds_key_file,
        local_port=local_port,
        local_api_type=local_api_type,
        no_chat_template=no_chat_template,
        num_samples=num_samples,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_logprobs=top_logprobs,
        max_request_time=max_request_time,
        sleep_time=sleep_time,
        echo=echo,
    )

    process_fn = partial(
        inference,
        eval_data_name=eval_data_name,
        llm=llm,
        type_hint=type_hint,
        category_hint=category_hint,
        url_hint=url_hint,
        prompt_style=prompt_style  # Pass the prompt style
    )

    ts_guessing_results = data_points.map(process_fn, num_proc=num_proc)
    ts_guessing_results = [x for x in ts_guessing_results if x["response"] != "failed"]
    ts_guessing_results = ts_guessing_results[:n_eval_data_points]

    # Log some sample prompts and responses
    num_samples_to_log = min(5, len(ts_guessing_results))
    logger.info(f"Sample prompts and responses:")
    for i in range(num_samples_to_log):
        logger.info(f"Sample {i+1}:")
        logger.info(f"Full prompt: {ts_guessing_results[i]['full_prompt']}")
        logger.info(f"Masked word: {ts_guessing_results[i]['masked_word']}")
        logger.info(f"Raw response: {ts_guessing_results[i]['raw_response']}")
        logger.info(f"Processed response: {ts_guessing_results[i]['response']}")
        logger.info("---")

    # Continue with existing code...
    masked_words = [x["masked_word"].lower() for x in ts_guessing_results]
    responses = [x["response"].lower() for x in ts_guessing_results]
    em = len([i for i in range(len(responses)) if responses[i] == masked_words[i]]) / len(responses)
    logger.info(f"Question-based completion (type hint: {type_hint} | category hint: {category_hint} | url hint: {url_hint})")
    logger.info(f"Exact Match (EM): {em:.2f}")

    # Save the full DataFrame with prompts and raw responses
    df = pd.DataFrame(ts_guessing_results)
    datentime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f'{model_name}-{eval_data_name}-{datentime}'
    output_dir = f'output/{folder_name}'
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/ts_guessing_question_based_results.csv", index=False)
