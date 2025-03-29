"""
This file implements the closed_data contamination detection through guided prompting.
https://arxiv.org/pdf/2311.09783
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize, sent_tokenize
from functools import partial
from datasets import Dataset
from datetime import datetime

from llmsanitize.utils.logger import get_child_logger, suspend_logging
from llmsanitize.utils.dataset_utils import get_answers_list, get_answer_index
from llmsanitize.closed_data_methods.llm import LLM
from llmsanitize.closed_data_methods.ts_guessing_question_based import filter_data

logger = get_child_logger("ts_guessing_question_multichoice")

def setup_stanford_env():
    os.environ['CLASSPATH'] = "/workspace/ndaba/research/code/LLMSanitize/postagger/stanford-postagger-full-2020-11-17/stanford-postagger.jar"
    os.environ["STANFORD_MODELS"] = "/workspace/ndaba/research/code/LLMSanitize/postagger/stanford-postagger-full-2020-11-17/models"

def build_prompt(
    example,
    eval_data_name,
    prompt_style="new"  # Add prompt style parameter
):
    text = example["text"]
    choices = get_answers_list(example, eval_data_name)
    answer_index = get_answer_index(example, eval_data_name)
    answer = choices[answer_index]
    
    wrong_choices_indices = [i for i in range(len(choices)) if i != answer_index]
    index = np.random.randint(len(wrong_choices_indices))
    wrong_choice_index = wrong_choices_indices[index]
    wrong_choice_content = choices[wrong_choice_index]  # Save the content that will be masked
    
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    wrong_letter = alphabet[wrong_choice_index]
    
    if prompt_style == "old":
        # Old prompting style
        prompt = f"Please fill in the [] in option {wrong_letter} based on your benchmark knowledge."
        prompt += "\n\nThe crucial rule is that you should provide different answer in other options below."
        prompt += f"\n\nQuestion: {text}"
        prompt += "\nOptions:"
        for i in range(len(choices)):
            letter = alphabet[i]
            if i == wrong_choice_index:
                prompt += f"\n{letter}: [MASK]"
            else:
                choice = choices[i]
                prompt += f"\n{letter}: [{choice}]"
        prompt += "\n\nReply with answer only."
    else:
        # New more restrictive prompting style with crucial rule included
        # prompt = f"[INSTRUCTION: This is a multiple-choice question task. You are required to fill in ONLY the missing option {wrong_letter}. The CRUCIAL RULE is that your answer must be DIFFERENT from all other options shown. Respond with EXACTLY the content that should replace [MASK] in option {wrong_letter}. No explanations, no prefixes, no other text - JUST THE ANSWER CONTENT. Violation of this format will result in task failure.]\n\n"
        prompt = f"[INSTRUCTION: This is a multiple-choice question task. You are required to fill in ONLY the missing option {wrong_letter}. Respond with EXACTLY the content that should replace [MASK] in option {wrong_letter}. No explanations, no prefixes, no other text - JUST THE ANSWER CONTENT. Violation of this format will result in task failure.]\n\n"
        prompt += f"Question: {text}\n\n"
        prompt += "Options:\n"
        for i in range(len(choices)):
            letter = alphabet[i]
            if i == wrong_choice_index:
                prompt += f"{letter}: [MASK]\n"
            else:
                choice = choices[i]
                prompt += f"{letter}: [{choice}]\n"
        prompt += f"\nYOUR ANSWER FOR OPTION {wrong_letter} (replace [MASK]): "

    return prompt, answer, wrong_letter, wrong_choice_content


def process_response(response, wrong_letter):
    """Extract the answer from the response more robustly."""
    # First try the original method
    symbol = wrong_letter + ":"
    if symbol in response:
        response = response.split(symbol)[1]
        sents = sent_tokenize(response)
        if len(sents) > 0:
            response = sents[0]
    
    # Clean up the response
    response = response.strip()
    
    # Remove common prefixes models might add
    prefixes_to_remove = [
        "The answer is ", "My answer is ", "Answer: ", 
        "[MASK] = ", "[MASK] is ", "Option " + wrong_letter + ": ",
        wrong_letter + ": ", "The missing text is ", "The content is "
    ]
    
    for prefix in prefixes_to_remove:
        if response.lower().startswith(prefix.lower()):
            response = response[len(prefix):]
    
    # Remove brackets if the model added them
    if response.startswith("[") and response.endswith("]"):
        response = response[1:-1]
        
    return response.strip()


def inference(
    data_point, 
    eval_data_name, 
    llm,
    prompt_style="new"
):
    setup_stanford_env()
    
    prompt, answer, wrong_letter, wrong_choice_content = build_prompt(
        data_point,
        eval_data_name,
        prompt_style
    )
    
    # Store all the relevant information
    data_point["full_prompt"] = prompt
    data_point["answer_text"] = answer
    data_point["wrong_letter"] = wrong_letter
    data_point["masked_content"] = wrong_choice_content  # Store the actual content that was masked
    
    raw_response, cost = llm.query(prompt)
    data_point["raw_response"] = raw_response
    response = process_response(raw_response, wrong_letter)
    data_point["response"] = response

    return data_point

def main_ts_guessing_question_multichoice(
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
    max_output_tokens: int = 128,
    temperature: float = 0.0,
    top_logprobs: int = 0,
    max_request_time: int = 600,
    sleep_time: int = 1,
    echo: bool = False,
    # Added parameter for prompt style
    prompt_style: str = "old"
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
        prompt_style=prompt_style  # Pass prompt style
    )

    # After processing all examples
    ts_guessing_results = data_points.map(process_fn, num_proc=num_proc)
    
    # Create a list from the dataset to modify it
    results_list = list(ts_guessing_results)
    
    # Add columns for masked choice and match result
    for result in results_list:
        # Add the masked choice (letter)
        result["masked_choice"] = result["wrong_letter"]
        
        # Add match status
        result["match"] = result["response"].lower() == result["masked_content"].lower()
    
    # Convert back to a dataset for further processing
    ts_guessing_results = Dataset.from_list(results_list)
    
    # Now log the samples
    num_samples_to_log = min(5, len(ts_guessing_results))
    logger.info(f"Sample prompts and responses:")
    for i in range(num_samples_to_log):
        logger.info(f"Sample {i+1}:")
        logger.info(f"Full prompt: {ts_guessing_results[i]['full_prompt']}")
        logger.info(f"Answer: {ts_guessing_results[i]['answer_text']}")
        # In the logging section
        logger.info(f"Masked choice: {ts_guessing_results[i]['wrong_letter']}: [{ts_guessing_results[i]['masked_content']}]")
        logger.info(f"Raw response: {ts_guessing_results[i]['raw_response']}")
        logger.info(f"Processed response: {ts_guessing_results[i]['response']}")
        logger.info(f"Match: {ts_guessing_results[i]['match']}")
        logger.info("---")
    
    # Calculate metrics by comparing responses with masked_content instead of answer_text
    masked_contents = [x["masked_content"].lower() for x in ts_guessing_results]
    responses = [x["response"].lower() for x in ts_guessing_results]
    em = len([i for i in range(len(responses)) if responses[i] == masked_contents[i]]) / len(responses)
    scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)
    rl = np.mean(np.array([scorer.score(responses[i], masked_contents[i])["rougeLsum"].fmeasure for i in range(len(responses))]))

    logger.info(f"Question-based guessing (prompt style: {prompt_style})")
    logger.info(f"Exact Match (EM): {em:.2f}, ROUGE-L F1: {rl:.2f}")
    
    # Save to CSV
    df = pd.DataFrame(ts_guessing_results)
    datentime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f'{model_name}-{eval_data_name}-{datentime}'
    output_dir = f'output/{folder_name}'
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/ts_guessing_question_multichoice_results.csv", index=False)