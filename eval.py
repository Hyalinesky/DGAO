import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
import re
import os

def _calculate_single_reward(completion, answer, task):
    """
    Calculate reward for a single completion based on task type.
    
    Args:
        completion (str): Generated completion text
        answer (str): Ground truth answer
        task (str): Task type identifier
        
    Returns:
        float: Reward value (1.0 for correct, 0.0 for incorrect)
    """
    import re
    
    if task in ['squad', 'searchqa']:
        # For SQuAD and SearchQA: check if answer appears in completion (case-insensitive)
        completion_lower = completion.lower().strip()
        answer_lower = answer.lower().strip()
        
        if answer_lower in completion_lower:
            return 1.0
        else:
            return 0.0
            
    elif task in ['CM17k', 'gsm8k']:
        # For CM17k and GSM8k: extract and compare values after '#### '
        def extract_final_answer(text):
            match = re.search(r'####\s*(.+)', text)
            if match:
                answer_text = match.group(1).strip()
                answer_text = answer_text.replace(' ', '').replace('.', '')
                return answer_text
            return None
        
        completion_answer = extract_final_answer(completion)
        ground_truth_answer = extract_final_answer(answer)
        
        if completion_answer is not None and ground_truth_answer is not None:
            if completion_answer == ground_truth_answer:
                return 1.0
        
        return 0.0
    
    elif task == 'sst2':
        # For SST2: extract first occurrence of 'positive' or 'negative'
        completion_lower = completion.lower()
        
        positive_match = re.search(r'\bpositive\b', completion_lower)
        negative_match = re.search(r'\bnegative\b', completion_lower)
        
        completion_sentiment = None
        if positive_match and negative_match:
            if positive_match.start() < negative_match.start():
                completion_sentiment = 'positive'
            else:
                completion_sentiment = 'negative'
        elif positive_match:
            completion_sentiment = 'positive'
        elif negative_match:
            completion_sentiment = 'negative'
        
        if completion_sentiment and completion_sentiment == answer.lower().strip():
            return 1.0
        else:
            return 0.0
    
    else:
        # Default behavior for unknown tasks
        import re
        
        completion_lower = completion.lower().strip()
        answer_lower = answer.lower().strip()
        
        if answer_lower in completion_lower:
            return 1.0
        
        # Additional matching logic for numbers
        completion_numbers = re.findall(r'-?\d+\.?\d*', completion)
        answer_numbers = re.findall(r'-?\d+\.?\d*', answer)
        
        if completion_numbers and answer_numbers:
            try:
                if float(completion_numbers[0]) == float(answer_numbers[0]):
                    return 1.0
            except:
                pass
        
        return 0.0

def normalize_response(response, task='squad'):
    """
    Normalize response using the same logic as _calculate_single_reward for consistency comparison
    """
    if task in ['squad', 'searchqa']:
        return response.lower().strip()
    elif task in ['CM17k', 'gsm8k']:
        def extract_final_answer(text):
            match = re.search(r'####\s*(.+)', text)
            if match:
                answer_text = match.group(1).strip()
                answer_text = answer_text.replace(' ', '').replace('.', '')
                return answer_text
            return text.lower().strip()
        return extract_final_answer(response)
    elif task == 'sst2':
        response_lower = response.lower()
        positive_match = re.search(r'\bpositive\b', response_lower)
        negative_match = re.search(r'\bnegative\b', response_lower)
        
        if positive_match and negative_match:
            if positive_match.start() < negative_match.start():
                return 'positive'
            else:
                return 'negative'
        elif positive_match:
            return 'positive'
        elif negative_match:
            return 'negative'
        else:
            return response.lower().strip()
    else:
        return response.lower().strip()

def calculate_metrics(output_path="eval/llama0/searchqa-90.json", task="searchqa"):
    """
    Calculate three metrics: average accuracy, consistency rate, and overconfidence rate.
    
    Args:
        output_path (str): Path to the results JSON file
        task (str): Task type identifier
    """
    print("\nStarting metric calculation...")

    # Read results file
    with open(output_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # Metric 1: Average accuracy
    total_correct = 0
    total_count = 0

    for group in results:
        for item in group:
            score = _calculate_single_reward(item['response'], item['answer'], task)
            total_correct += score
            total_count += 1

    average_accuracy = total_correct / total_count
    print(f"1. Average Accuracy: {average_accuracy:.4f} ({total_correct}/{total_count})")

    # Metric 2: Consistency Rate
    consistency_scores = []

    for group in results:
        normalized_responses = [normalize_response(item['response'], task) for item in group]
        response_counts = Counter(normalized_responses)
        max_count = max(response_counts.values())
        consistency = max_count / len(group)
        consistency_scores.append(consistency)

    average_consistency = sum(consistency_scores) / len(consistency_scores)
    print(f"2. Consistency Rate: {average_consistency:.4f}")

    # Metric 3: Overconfidence Rate
    overconfidence_scores = []

    for group in results:
        # Get correctness and normalized responses for each item
        correctness = []
        normalized_responses = []
        
        for item in group:
            score = _calculate_single_reward(item['response'], item['answer'], task)
            correctness.append(score)
            normalized_responses.append(normalize_response(item['response'], task))
        
        # Find the most common response and its count
        response_counts = Counter(normalized_responses)
        most_common_response, most_common_count = response_counts.most_common(1)[0]
        
        # Check if the most common response is correct
        most_common_correctness = None
        for i, norm_resp in enumerate(normalized_responses):
            if norm_resp == most_common_response:
                most_common_correctness = correctness[i]
                break
        
        # If most common response is wrong, record its ratio in current group
        if most_common_correctness == 0:  # Wrong answer
            overconfidence_ratio = most_common_count / len(group)
            overconfidence_scores.append(overconfidence_ratio)
        else:  # Correct answer
            overconfidence_scores.append(0.0)

    average_overconfidence = sum(overconfidence_scores) / len(overconfidence_scores)
    print(f"3. Overconfidence Rate: {average_overconfidence:.4f}")

    print("\nAll metrics calculated!")
    
    return {
        "average_accuracy": average_accuracy,
        "consistency_rate": average_consistency,
        "overconfidence_rate": average_overconfidence
    }

if __name__ == "__main__":
    # Default parameters
    task = "searchqa"
    output_path = f"eval/llama/searchqa.json"
    
    # Calculate metrics
    metrics = calculate_metrics(output_path, task)