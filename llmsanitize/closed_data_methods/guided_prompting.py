"""
This file implements the closed_data contamination detection through guided prompting.
https://arxiv.org/pdf/2308.08493.pdf
"""
import os
import sys
import pandas as pd
from datetime import datetime
import nltk
import random
import numpy as np
from rouge_score import rouge_scorer
from datasets import Value
from functools import partial
from llmsanitize.utils.utils import seed_everything, fill_template
from llmsanitize.utils.logger import get_child_logger, suspend_logging
from llmsanitize.closed_data_methods.llm import LLM
import llmsanitize.prompts.guided_prompting.general_instructions as gi_prompts
import llmsanitize.prompts.guided_prompting.guided_instructions as gui_prompts
from scipy.stats import bootstrap

logger = get_child_logger("guided_prompting")


def guided_prompt_split_fn(
    example,
    idx,
    dataset_name,
    text_key
):
    ''' split content per example to part 1 and part 2
        For AGnews: split ['text'] into 2 parts
            ARC: split ['question']+['choices']
    '''
    seed_everything(idx)
    splits = {'guided_prompt_part_1': '', 'guided_prompt_part_2': ''}
    # split the input field to two parts
    if dataset_name in ['ag_news', 'gsm8k', 'cais/mmlu']:
        text = example[text_key]
        sentences = nltk.sent_tokenize(text)
        if len(sentences) < 2:
            return splits
        first_part_length = random.randint(1, len(sentences) - 1)
        splits['guided_prompt_part_1'] = ''.join(sentences[:first_part_length])
        splits['guided_prompt_part_2'] = ''.join(sentences[first_part_length:])
    
    # split to question + choices[0], choices[1:]
    elif dataset_name in ['allenai/ai2_arc']:
        choices = example['choices']
        choices = [_label+'.'+_text for _text, _label in zip(choices['text'], choices['label'])]
        splits['guided_prompt_part_1'] = example[text_key] + '\n' + choices[0]
        splits['guided_prompt_part_2'] = '\n'.join(choices[1:])
    
    # NLI tasks
    elif dataset_name in ['Rowan/hellaswag']:
        splits['guided_prompt_part_1'] = example[text_key]
        splits['guided_prompt_part_2'] = example['endings'][int(example['label'])] 
    elif dataset_name in ['truthful_qa']:
        splits['guided_prompt_part_1'] = example[text_key]
        splits['guided_prompt_part_2'] = example['best_answer']
    elif dataset_name == "winogrande":
        sents = example[text_key].split('_')
        splits['guided_prompt_part_1'] = sents[0]
        splits['guided_prompt_part_2'] = sents[1]
    
    # Math reasoning tasks        
    elif dataset_name == 'HuggingFaceH4/aime_2024':
        # ONLY use the problem field, not text or anything else
        if 'problem' in example:
            problem_text = example['problem']
        else:
            problem_text = example[text_key]
        
        # Make sure we're only working with the problem, not solution
        if 'Solution' in problem_text:
            problem_text = problem_text.split('Solution', 1)[0].strip()
        
        # Also check for "Solution (beginning):" format
        if 'Solution (beginning)' in problem_text:
            problem_text = problem_text.split('Solution (beginning)', 1)[0].strip()
        
        # Remove any Problem: prefix if it exists
        if problem_text.startswith('Problem:'):
            problem_text = problem_text[len('Problem:'):].strip()
        
        sentences = nltk.sent_tokenize(problem_text)
        
        if len(sentences) < 2:
            return splits
        
        first_part_length = random.randint(1, len(sentences) - 1)
        
        splits['guided_prompt_part_1'] = ' '.join(sentences[:first_part_length])
        splits['guided_prompt_part_2'] = ' '.join(sentences[first_part_length:])
    else:
        raise(f"Error! guided_prompt_split_fn not found processing for dataset_name: {dataset_name}")

    return splits

def guided_prompt_process_label(example, dataset_name):
    new_example = example.copy()

    if dataset_name == 'cais/mmlu':
        new_example['answer_text'] = new_example['choices'][int(new_example['answer'])]
    elif dataset_name == 'winogrande':
        new_example['answer_token'] = new_example['option1'] + '/' + new_example['option2']
    elif dataset_name == 'HuggingFaceH4/aime_2024':
        print("\n[DEBUG in label_fn]")
        print("Original problem:", new_example.get("problem"))
        print("Original solution:", new_example.get("solution"))
        print("Original answer:", new_example.get("answer"))
        new_example['answer_text'] = str(new_example['answer'])

    return new_example

def bootstrap_test(data):
    ''' bootstrap test (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
        to check if to reject the H0 hypothesis that there's no difference between guided-prompt and general-prompt
    Args:
        data: a sequence of score difference (s_guided - s_general)
    Return:
        p-value of diff <= 0
    '''
    res = bootstrap((data,), np.mean, n_resamples=10000)

    return (res.bootstrap_distribution <= 0.).sum() / 10000.

def process_deepseek_response(response):
    """
    Process responses from DeepSeek models by removing the thinking process
    and keeping only the final answer after the </think> tag.
    """
    # Check if the response contains the </think> tag
    if '</think>' in response:
        # Split by the </think> tag and get the part after it
        parts = response.split('</think>')
        if len(parts) > 1:
            # Return everything after the last </think> tag, stripped of whitespace
            return parts[-1].strip()
    
    # If no </think> tag is found, return the original response
    return response

@suspend_logging
def guided_prompt_process_fn(
    example,
    idx,
    llm,
    split_name,
    dataset_name,
    label_key,
    text_key,
    general_template,
    guided_template
):
    seed_everything(idx)
    
    # Debug what fields are available
    print(f"\n[DEBUG process_fn] Keys in example: {example.keys()}")
    print(f"[DEBUG process_fn] text_key: {text_key}")
    print(f"[DEBUG process_fn] Original text: {example.get(text_key, 'Not found')[:100]}")
    print(f"[DEBUG process_fn] Problem field: {example.get('problem', 'Not found')[:100]}")
    print(f"[DEBUG process_fn] Solution field: {example.get('solution', 'Not found')[:100]}")
    label = str(example[label_key])
    first_part = example['guided_prompt_part_1']
    second_part = example['guided_prompt_part_2']
    print("\n=== Guided Prompt first and second part (in guided_prompt_process_fn)  ===")
    print("First part", first_part)
    print("Second part", second_part)
    print("===================================\n")
    # sys.exit()
    # query llm
    vars_map = {"split_name": split_name, "dataset_name": dataset_name, "first_piece": first_part, "label": label}
    general_prompt = fill_template(general_template, vars_map)
    guided_prompt = fill_template(guided_template, vars_map)
    
    print("\n=== Guided Prompt Sent to Model ===")
    print(guided_prompt)
    print("===================================\n")
    sys.exit()
    general_response_raw, cost = llm.query(general_prompt)
    guided_response_raw, cost_ = llm.query(guided_prompt)
    
    # Process responses to remove thinking part
    general_response = process_deepseek_response(general_response_raw)
    guided_response = process_deepseek_response(guided_response_raw)

    # get scores
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    general_score = scorer.score(second_part, general_response)['rougeL'].fmeasure
    guided_score = scorer.score(second_part, guided_response)['rougeL'].fmeasure

    # return
    example['general_score'] = general_score
    example['guided_score'] = guided_score
    example['general_response_raw'] = general_response_raw  # Store the raw response too
    example['guided_response_raw'] = guided_response_raw    # Store the raw response too
    example['general_response'] = general_response          # Processed response
    example['guided_response'] = guided_response            # Processed response
    example['first_part'] = first_part
    example['second_part'] = second_part

    return example

def main_guided_prompting(
    eval_data: list = [],
    eval_data_name: str = None,
    eval_set_key: str = "test",
    text_key: str = "problem",
    label_key: str = "label",
    num_proc: int = 16,
    n_eval_data_points: int = 100,
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
    # method-specific parameters
    guided_prompting_task_type: str = None,
):
    # Add at the beginning of main_guided_prompting
    print("=== ORIGINAL DATA SAMPLE ===")
    print("First example keys:", eval_data[0].keys())
    print("First example problem field:", eval_data[0].get("problem", "Not found"))
    print("First example solution field:", eval_data[0].get("solution", "Not found"))
    print("First example text field:", eval_data[0].get("text", "Not found"))
    print("=========================")
    # based on task type, choose prompt template
    type_str = guided_prompting_task_type
    guided_template = getattr(gui_prompts, f"GUI_{type_str}")
    general_template = getattr(gi_prompts, f"GI_{type_str}")

    # process selected examples parallely
    num_examples_to_test = n_eval_data_points
    split_fn = partial(guided_prompt_split_fn, dataset_name=eval_data_name, text_key=text_key)
    label_fn = partial(guided_prompt_process_label, dataset_name=eval_data_name)
    
    logger.info(f"Starting guided prompting evaluation on {eval_data_name}")
    eval_data = eval_data.map(split_fn, num_proc=num_proc, load_from_cache_file=False, with_indices=True)\
        .filter(lambda example: len(example['guided_prompt_part_1']) > 0 and len(example['guided_prompt_part_2']) > 0)\
        .map(label_fn, num_proc=num_proc)
    print("\n=== AFTER label_fn ===")
    print("Eval keys:", eval_data[0].keys())
    print("First piece:\n", eval_data[0]['guided_prompt_part_1'])
    print("Second piece:\n", eval_data[0]['guided_prompt_part_2'])
    print("Problem field:\n", eval_data[0].get("problem"))
    print("Solution field:\n", eval_data[0].get("solution"))
    print("Answer field:\n", eval_data[0].get("answer"))
    print("Answer text field:\n", eval_data[0].get("answer_text"))
    print("======================\n")
    # sys.exit()
    logger.info(f"After filtering, {len(eval_data)} examples remaining")
    random_examples = eval_data.shuffle(seed=42).filter(lambda _, idx: idx < num_examples_to_test, with_indices=True)
    logger.info(f"Selected {len(random_examples)} examples for testing")
    
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
        guided_prompt_process_fn,
        llm=llm,
        split_name=eval_set_key,
        dataset_name=eval_data_name,
        label_key=label_key,
        text_key=text_key,
        general_template=general_template,
        guided_template=guided_template
    )

    # somehow I need to do this to avoid datasets bug (https://github.com/huggingface/datasets/issues/6020#issuecomment-1632803184)
    features = eval_data.features
    features['general_score'] = Value(dtype='float64', id=None)
    features['guided_score'] = Value(dtype='float64', id=None)
    features["general_response"] = Value(dtype='string', id=None)
    features["guided_response"] = Value(dtype='string', id=None)
    features["first_part"] = Value(dtype='string', id=None)
    features["second_part"] = Value(dtype='string', id=None)
    features["general_prompt"] = Value(dtype='string', id=None)
    features["guided_prompt"] = Value(dtype='string', id=None)
    features["general_response_raw"] = Value(dtype='string', id=None)
    features["guided_response_raw"] = Value(dtype='string', id=None)

    # Add the prompts to the process_fn to save them
    def process_fn_with_prompts(example, idx):
        print("=== BEFORE TEMPLATE FILL ===")
        print("guided_prompt_part_1:", example['guided_prompt_part_1'])
        print("Does it contain 'Solution'?", 'Solution' in example['guided_prompt_part_1'])
        
        vars_map = {"split_name": eval_set_key, "dataset_name": eval_data_name, 
                "first_piece": example['guided_prompt_part_1'], "label": str(example[label_key])}
        
        # Check if any templates add Problem/Solution formatting
        print("General template:", general_template[:200])
        print("Guided template:", guided_template[:200])
        
        example["general_prompt"] = fill_template(general_template, vars_map)
        example["guided_prompt"] = fill_template(guided_template, vars_map)
        
        print("=== AFTER TEMPLATE FILL ===")
        print("Does general_prompt include 'Solution'?", 'Solution' in example["general_prompt"])
        print("Does guided_prompt include 'Solution'?", 'Solution' in example["guided_prompt"])
        
        return process_fn(example, idx)

    processed_examples = random_examples.map(
        process_fn_with_prompts,
        with_indices=True,
        num_proc=num_proc,
        features=features,
        load_from_cache_file=False
    )
    
    # Convert to list for easier processing
    processed_examples_list = [example for example in processed_examples if (len(example['general_response']) > 0) and (len(example['guided_response']) > 0)]
    logger.info(f"Successfully processed {len(processed_examples_list)} examples")

    # Log sample prompts and responses
    num_samples_to_log = min(10, len(processed_examples_list))
    logger.info(f"Sample prompts and responses:")
    for i in range(num_samples_to_log):
        logger.info(f"Sample {i+1}:")
        # logger.info(f"First part: {processed_examples_list[i]['first_part']}")
        logger.info(f"Second part (ground truth): {processed_examples_list[i]['second_part']}")
        # logger.info(f"General prompt: {processed_examples_list[i]['general_prompt']}")
        # logger.info(f"General response: {processed_examples_list[i]['general_response']}")
        # logger.info(f"General score: {processed_examples_list[i]['general_score']:.4f}")
        logger.info(f"Guided prompt: {processed_examples_list[i]['guided_prompt']}")
        logger.info(f"Guided response: {processed_examples_list[i]['guided_response']}")
        logger.info(f"Guided score: {processed_examples_list[i]['guided_score']:.4f}")
        logger.info(f"Score difference (guided - general): {processed_examples_list[i]['guided_score'] - processed_examples_list[i]['general_score']:.4f}")
        logger.info("---")

    # Calculate and log overall metrics
    scores_diff = [example['guided_score'] - example['general_score'] for example in processed_examples_list]
    logger.info(f"Tested {len(processed_examples_list)} examples with guided-prompting for closed_data {model_name}")
    
    # Conduct bootstrap test for significance
    p_value = bootstrap_test(scores_diff)
    logger.info(f"dataset: {eval_data_name}, guided_score - general_score (RougeL)")
    logger.info(f"mean: {np.mean(scores_diff):.3f}, std: {np.std(scores_diff):.3f}, p-value of diff <= 0: {p_value:.3f}")
    
    # Save results to CSV
    df = pd.DataFrame(processed_examples_list)
    datentime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f'{model_name}-{eval_data_name}-{datentime}'
    output_dir = f'output/{folder_name}'
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/guided_prompting_results.csv", index=False)
    
    # Also save a summary file with the overall metrics
    summary_df = pd.DataFrame({
        'model_name': [model_name],
        'dataset_name': [eval_data_name],
        'num_examples': [len(processed_examples_list)],
        'mean_score_diff': [np.mean(scores_diff)],
        'std_score_diff': [np.std(scores_diff)],
        'p_value': [p_value],
        'timestamp': [datentime]
    })
    summary_df.to_csv(f"{output_dir}/guided_prompting_summary.csv", index=False)
    
    return scores_diff, p_value