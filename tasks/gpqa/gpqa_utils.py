import re 
import random
import csv
from collections import namedtuple
from typing import Dict, List, Tuple, Union, Any
import json

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

Example = namedtuple('Example', ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index', 'explanation'])


def load_examples(path: str, seed: int) -> List[Example]:
    """Load questions from csv file and return a list of Example namedtuples."""
    random.seed(seed)

    def shuffle_choices_and_create_example(row) -> Example:
        if hasattr(row, "__getitem__"):
            question = row["Question"]
            incorrect1 = row["Incorrect Answer 1"]
            incorrect2 = row["Incorrect Answer 2"]
            incorrect3 = row["Incorrect Answer 3"]
            correct = row["Correct Answer"]
            explanation = row["Explanation"]
        else:
            question = row.Question
            incorrect1 = row["Incorrect Answer 1"]
            incorrect2 = row["Incorrect Answer 2"]
            incorrect3 = row["Incorrect Answer 3"]
            correct = row["Correct Answer"]
            explanation = row["Explanation"]

        list_choices = [incorrect1, incorrect2, incorrect3, correct]
        random.shuffle(list_choices)
        example = Example(
            question,
            list_choices[0],
            list_choices[1],
            list_choices[2],
            list_choices[3],
            list_choices.index(correct),
            explanation,
        )
        return example

    if pd is not None:
        question_df = pd.read_csv(path)
        return [shuffle_choices_and_create_example(row) for _, row in question_df.iterrows()]

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [shuffle_choices_and_create_example(row) for row in reader]


LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

def parse_sampled_answer(answer):
        patterns = [r'answer is \((.)\)', r'Answer: \((.)\)', r'answer: \((.)\)', r'answer \((.)\)', r'\((.)\)']
        for pattern in patterns:
            match = re.search(pattern, answer)
            if match and match.group(1) in LETTER_TO_INDEX:
                return match.group(1)
        return None


def generate_prompt_from_examples(json_data, with_explanations=True):
    output = ""
    for q in json_data["questions"]:
        output += f'Question: {q["question"]}\nChoices:\n'
        for choice, value in q["choices"].items():
            output += f'({choice}) {value}\n'
        
        if with_explanations:
            output += f"Let's think step by step: \n{q['explanation']}\n"
        
        output += f'The correct answer is ({q["correct_answer"]})\n'
    
    return output

def chain_of_thought_prompt(json_data, example: Example) -> str:
    """Creates a chain-of-thought prompt given a single example."""
    prompt = f"Here are some example questions from experts. An explanation is given before the final answer. Answer the final question yourself, giving your reasoning beforehand.\n"
    prompt += generate_prompt_from_examples(json_data, with_explanations=True)
    prompt += f"Question: {example.question}"
    prompt += f"\nChoices:\n(A) {example.choice1}\n(B) {example.choice2}\n(C) {example.choice3}\n(D) {example.choice4}"
    prompt += "\nGive step by step reasoning before you answer, and when you're ready to answer, please use the format \"The correct answer is (insert answer here)\":\n"
    return prompt


def zero_shot_chain_of_thought_prompt(example: Example) -> str:
    prompt = f"What is the correct answer to this question: {example.question}"
    prompt += f"\n\nChoices:\n(A) {example.choice1}\n(B) {example.choice2}\n(C) {example.choice3}\n(D) {example.choice4}"
    prompt += "\nGive step by step reasoning before you answer, and when you're ready to answer, please use the format \"The correct answer is (insert answer here)\":\n"
    return prompt
