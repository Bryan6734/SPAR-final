"""
Script to generate and evaluate compressed versions of GPQA questions using Claude.
Each question will be compressed to exact word counts (75%, 50%, 25%) while preserving core intent.
"""

import pandas as pd
import anthropic
import os
from pathlib import Path
import time
import re
import logging
from datetime import datetime
import math
from tqdm import tqdm
import numpy as np

# Set up logging
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"compression_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Read API key from environment variable
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not CLAUDE_API_KEY:
    raise ValueError("Please set ANTHROPIC_API_KEY environment variable")

client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

def load_compression_prompt():
    """Load the compression prompt template"""
    prompt_path = Path(__file__).parent.parent / "prompts" / "compress_question.txt"
    with open(prompt_path) as f:
        return f.read()

def get_word_count(text):
    """Get word count of text, counting hyphenated words and numbers as single words"""
    if not isinstance(text, str):
        logging.warning(f"Non-string input to get_word_count: {type(text)}")
        return 0
    
    # Split on whitespace and filter out empty strings
    words = [w for w in text.split() if w.strip()]
    return len(words)

def calculate_target_words(original_count: int) -> dict:
    """Calculate target word counts for each compression level"""
    return {
        75: max(1, round(original_count * 0.75)),
        50: max(1, round(original_count * 0.50)),
        25: max(1, round(original_count * 0.25))
    }

def extract_compressions(response_text):
    """Extract all compressions from Claude's response"""
    if not isinstance(response_text, str):
        logging.error(f"Unexpected response type: {type(response_text)}")
        return {}
        
    compressions = {}
    patterns = {
        75: r"(\d+) words:\n(.*?)(?=\n\n|\Z)",
        50: r"(\d+) words:\n(.*?)(?=\n\n|\Z)",
        25: r"(\d+) words:\n(.*?)(?=\n\n|\Z)"
    }
    
    for percent, pattern in patterns.items():
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            compressions[percent] = match.group(2).strip()
        else:
            logging.warning(f"Could not find {percent}% compression in response")
    
    return compressions

def verify_word_count(text: str, target_count: int, tolerance: int = 0) -> bool:
    """Verify that a compression has the target number of words (within tolerance)"""
    actual_count = get_word_count(text)
    return abs(actual_count - target_count) <= tolerance

def evaluate_answer(question: str, correct_answer: str) -> tuple:
    """
    Use Claude to evaluate if the answer to a question is correct.
    Returns (predicted_answer, is_correct)
    """
    try:
        system_prompt = """You are an expert at answering academic questions accurately and evaluating answers.
        Your task is to:
        1. Answer the given question
        2. Compare your answer to the provided correct answer
        3. Return a tuple of (your_answer, True/False) where True means the answers match
        
        Be strict in your evaluation - answers must match in meaning, even if the wording is different."""
        
        eval_prompt = f"""Question: {question}
        Correct Answer: {correct_answer}
        
        Provide your answer and evaluation in this format:
        (your_answer, is_correct)"""
        
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": eval_prompt}]
        )
        
        # Extract answer and evaluation from response
        response_text = response.content[0].text
        match = re.search(r"\((.*?), (True|False)\)", response_text, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            is_correct = match.group(2) == "True"
            return answer, is_correct
        else:
            logging.warning("Could not parse evaluation response")
            return response_text, False
            
    except Exception as e:
        logging.error(f"Error evaluating answer: {str(e)}")
        return str(e), False

def compress_and_evaluate_question(question: str, correct_answer: str, prompt_template: str):
    """
    Generate compressions for a question and evaluate their performance
    """
    if not isinstance(question, str):
        logging.error(f"Skipping non-string question: {type(question)}")
        return None, None
    
    original_count = get_word_count(question)
    target_words = calculate_target_words(original_count)
    
    prompt = prompt_template.format(
        question=question,
        original_word_count=original_count,
        target_75=target_words[75],
        target_50=target_words[50],
        target_25=target_words[25]
    )
    
    try:
        # Get compressions
        logging.info(f"Original ({original_count} words): {question}")
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=2000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        compressions = extract_compressions(response.content[0].text)
        
        # Verify word counts and evaluate each compression
        results = {}
        for percent, text in compressions.items():
            target = target_words[percent]
            if verify_word_count(text, target):
                # Evaluate compressed question
                predicted_answer, is_correct = evaluate_answer(text, correct_answer)
                
                results[percent] = {
                    'compression': text,
                    'target_words': target,
                    'actual_words': get_word_count(text),
                    'predicted_answer': predicted_answer,
                    'is_correct': is_correct
                }
                
                logging.info(f"{percent}% compression ({get_word_count(text)} words):")
                logging.info(f"Q: {text}")
                logging.info(f"A: {predicted_answer}")
                logging.info(f"Correct: {is_correct}\n")
            else:
                actual = get_word_count(text)
                logging.warning(f"Compression {percent}% has {actual} words, expected {target}")
        
        return compressions, results
    
    except Exception as e:
        logging.error(f"Error compressing question: {str(e)}")
        return None, None

def save_progress(df, output_path, idx):
    """Save progress to CSV and backup file"""
    # Save to main output file
    df.iloc[:idx+1].to_csv(output_path, index=False)
    
    # Also save to a backup file with timestamp
    backup_dir = Path(output_path).parent / "backups"
    backup_dir.mkdir(exist_ok=True)
    backup_file = backup_dir / f"gpqa_diamond_compressed_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.iloc[:idx+1].to_csv(backup_file, index=False)
    
    logging.info(f"Progress saved through question {idx}")

def main():
    logging.info("Starting compression generation and evaluation")
    
    # Load data
    input_path = Path(__file__).parent.parent / "datasets" / "gpqa_diamond.csv"
    output_path = Path(__file__).parent.parent / "datasets" / "gpqa_diamond_compressed.csv"
    results_path = Path(__file__).parent.parent / "results" / "compression_results.csv"
    
    logging.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    prompt_template = load_compression_prompt()
    
    # Add original metrics
    logging.info("Calculating original metrics...")
    df["original_word_count"] = df["Question"].apply(get_word_count)
    
    # Initialize compression columns
    compression_levels = [75, 50, 25]
    for target in compression_levels:
        col_name = f"compression_{target}"
        df[col_name] = None
        df[f"{col_name}_word_count"] = None
        df[f"{col_name}_target_words"] = None
        df[f"{col_name}_predicted_answer"] = None
        df[f"{col_name}_is_correct"] = None
    
    # Process each question
    logging.info(f"Starting compression generation for {len(df)} questions...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        logging.info(f"\nProcessing question {idx}/{len(df)}...")
        
        compressions, results = compress_and_evaluate_question(
            row["Question"], 
            row["Correct Answer"], 
            prompt_template
        )
        
        if compressions and results:
            for target, result in results.items():
                col_name = f"compression_{target}"
                df.at[idx, col_name] = result['compression']
                df.at[idx, f"{col_name}_word_count"] = result['actual_words']
                df.at[idx, f"{col_name}_target_words"] = result['target_words']
                df.at[idx, f"{col_name}_predicted_answer"] = result['predicted_answer']
                df.at[idx, f"{col_name}_is_correct"] = result['is_correct']
        else:
            logging.error(f"Failed to generate compressions for question {idx}")
        
        # Save progress every 10 questions
        if idx % 10 == 0:
            save_progress(df, output_path, idx)
        
        # Sleep to avoid rate limits
        time.sleep(1)
    
    # Final save
    logging.info(f"Saving final results to {output_path}")
    df.to_csv(output_path, index=False)
    
    # Calculate and save summary statistics
    summary = pd.DataFrame({
        'compression_level': ['original'] + [f"{level}%" for level in compression_levels],
        'accuracy': [1.0] + [df[f"compression_{level}_is_correct"].mean() for level in compression_levels],
        'relative_performance': [1.0] + [df[f"compression_{level}_is_correct"].mean() for level in compression_levels]
    })
    
    summary.to_csv(results_path, index=False)
    print("\nCompression Results Summary:")
    print("---------------------------")
    print(summary)
    
    logging.info("Compression generation and evaluation complete!")

if __name__ == "__main__":
    main()