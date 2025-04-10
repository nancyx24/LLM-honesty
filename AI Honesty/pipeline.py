import re
import math
from anthropic import Anthropic
import pandas as pd
import numpy as np
import asyncio
from datasets import load_dataset
from scipy import stats

async def llm_query(
    client: Anthropic,
    model: str,
    system_prompt: str,
    messages: list,
    max_tokens=4000,
    temperature=0.7,
) -> str:
    """
    Query a Claude model.
    Args:
        client (Anthropic): The Anthropic client.
        model (str): The model to use.
        system_prompt (str): The system prompt to use.
        messages (list): The messages to send to the model.
        max_tokens (int): The maximum number of tokens to generate.
        temperature (float): The temperature to use for sampling.
    Returns:
        str: The model's response.
    """
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=messages,
        system=system_prompt
    )
    
    return response.content[0].text

async def run_experiment(client: Anthropic,
                   system_prompt: str,
                   model: str,
                   questions: list
                   ) -> list:
    """
    Experiment with how a Claude model responds to a set of questions given different system prompts.
    Args:
        system_prompt (str): The system prompt to use.
        model (str): The model to use.
        questions (list): The questions to ask.
    Returns:
        list: The model's responses to the questions.
    """
    messages = [
        [{"role": "user", "content": questions[idx]}] for idx in range(len(questions))
    ]

    responses = [llm_query(client, model, system_prompt=system_prompt, messages=message) for message in messages]
    responses = await asyncio.gather(*responses)
    return responses

def _response_processing(response: str) -> tuple:
    """
    Processes responses from the Claude model.
    Args:
        response (str): The response from the model.
    Returns:
        tuple: The processed response, answer, and confidence.
    """
    # process answer
    if '<answer>' not in response or '</answer>' not in response:
        answer = None
    else:
        answer = response.split('<answer>')[1].split('</answer>')[0]
        answer = answer.split("=")[-1].strip() # some show equation
        answer = answer.split(".")[0] # some have decimals -- from response_processing
        answer = re.sub(r'[\$\€,}]', '', answer) # get rid of symbols and commas
        answer = answer if answer == 'N/A' else re.sub(r'\b[a-zA-Z]+\b', '', answer).strip() # some is 'N/A' or have words
        try:
            answer = int(answer)
        except:
            answer = answer
    # process confidence
    if '<confidence>' not in response or '</confidence>' not in response:
        confidence = None
    else:
        confidence = int(response.split('<confidence>')[1].split('</confidence>')[0])
    return response, answer, confidence

def all_responses_processing(responses: str, 
                             data: pd.DataFrame) -> list:
    """
    Processes all LLM responses using _response_processing and appends the ground truth.
    Args:
        responses (str): The responses from the model.
        data (pd.DataFrame): The original dataframe.
    Returns:
        list: A list of dictionaries containing the processed responses, answers, confidence, truth, and ID.
    """
    processed_responses = []

    for idx in range(len(responses)):
        response, answer, confidence = _response_processing(responses[idx])
        try:
            truth = int(data['Answer'].iloc[idx])
        except:
            truth = str(data['Answer'].iloc[idx])
        try:
            id = int(data['ID'].iloc[idx])
        except:
            id = str(data['ID'].iloc[idx])
        output = {'response': response, 'answer': answer, 'confidence': confidence, 'truth': truth, 'id': id}
        # only include outputs with answer and confidence defined
        if answer and confidence:
            processed_responses.append(output)
    
    return processed_responses

def _get_incorrect_responses(processed_responses: list) -> list:
    """
    Get incorrect responses from the processed responses.
    Args:
        responses (list): The processed responses.
    Returns:
        list: A list of incorrect responses.
    """
    incorrect_responses = []
    for output in processed_responses:
        if type(output['answer']) == str or type(output['truth']) == str:
            if output['answer'] != output['truth']:
                incorrect_responses.append(output)
        else:
            # also consider correct if off by power of 10
            check = abs(output['answer'] / output['truth'])
            if 10**round(math.log10(check)) != check:
                incorrect_responses.append(output)
    return incorrect_responses

def _get_incorrect_responses(processed_responses: list) -> list:
    """
    Get incorrect responses from the processed responses.
    Args:
        responses (list): The processed responses.
    Returns:
        list: A list of incorrect responses.
    """
    incorrect_responses = []
    for output in processed_responses:
        if type(output['answer']) == str or type(output['truth']) == str:
            if output['answer'] != output['truth']:
                incorrect_responses.append(output)
        else:
            # also consider correct if off by power of 10
            check = abs(output['answer'] / output['truth'])
            if 10**round(math.log10(check)) != check:
                incorrect_responses.append(output)
    return incorrect_responses

def analysis(baseline_processed: list,
             exp_processed: list) -> dict:
    """
    Analyze the results of the experiment.
    Args:
        baseline_processed (list): The processed responses from the baseline experiment.
        exp_processed (list): The processed responses from the experimental run.
    Returns:
        dict: A dictionary containing summary statistics.
    """
    # get number of valid responses
    baseline_valid = len(baseline_processed)
    exp_valid = len(exp_processed)
    # get incorrect responses
    baseline_incorrect = _get_incorrect_responses(baseline_processed)
    exp_incorrect = _get_incorrect_responses(exp_processed)
    # get accuracy
    baseline_accuracy = len(baseline_processed) - len(baseline_incorrect)
    exp_accuracy = len(exp_processed) - len(exp_incorrect)
    # get accuracy rate
    baseline_accuracy_rate = baseline_accuracy / len(baseline_processed)
    exp_accuracy_rate = exp_accuracy / len(exp_processed)
    # get confidence
    baseline_confidence = np.mean([output['confidence'] for output in baseline_processed])
    exp_confidence = np.mean([output['confidence'] for output in exp_processed])
    # get response lengths
    baseline_response_lengths = [len(output['response']) for output in baseline_processed]
    exp_response_lengths = [len(output['response']) for output in exp_processed]
    # get average response length
    baseline_response_length = np.mean(baseline_response_lengths)
    exp_response_length = np.mean(exp_response_lengths)
    length_t_statistics, length_p_value = stats.ttest_ind(baseline_response_lengths, exp_response_lengths)
    # get confidence rate when both incorrect
    wrong_ids = [output['id'] for output in baseline_incorrect if output['id'] in [output['id'] for output in exp_incorrect]]
    baseline_confidences = [output['confidence'] for output in baseline_incorrect if output['id'] in wrong_ids]
    experiment_confidences = [output['confidence'] for output in exp_incorrect if output['id'] in wrong_ids]
    # get average confidence rate when both incorrect
    baseline_confidence_rate = np.mean(baseline_confidences)
    experiment_confidence_rate = np.mean(experiment_confidences)
    t_statistics, p_value = stats.ttest_ind(baseline_confidences, experiment_confidences)

    # return output
    output = {'baseline': {
        'valid': baseline_valid,
        'accuracy': baseline_accuracy,
        'accuracy_rate': baseline_accuracy_rate,
        'confidence': baseline_confidence,
        },
        'experiment': {
            'valid': exp_valid,
            'accuracy': exp_accuracy,
            'accuracy_rate': exp_accuracy_rate,
            'confidence': exp_confidence,
        },
        'length': {
            'baseline_length': baseline_response_length,
            'experiment_length': exp_response_length,
            't_statistics': length_t_statistics,
            'p_value': length_p_value
        },
        'both_incorrect': {
            'baseline_confidence_rate': baseline_confidence_rate,
            'experiment_confidence_rate': experiment_confidence_rate,
            't_statistics': t_statistics,
            'p_value': p_value
        }
    }
    return output