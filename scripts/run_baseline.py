"""
Evaluate model performance on original and compressed questions from GPQA dataset.
"""

import pandas as pd
import numpy as np
import logging
import os
import time
import re
from datetime import datetime
from pathlib import Path
import anthropic
from tqdm import tqdm

# Set up logging
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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

def parse_response(response_text: str) -> tuple:
    """Parse Claude's response to extract answer and correctness"""
    try:
        # Try to find a tuple pattern (answer, True/False)
        match = re.search(r"\((.*?), (True|False)\)", response_text, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            is_correct = match.group(2) == "True"
            return answer, is_correct
        
        # If no tuple found, try to find just the answer
        answer_match = re.search(r"answer: (.*?)(?=\n|$)", response_text, re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).strip()
            return answer, False
            
        # If still no match, return the first line as answer
        first_line = response_text.split('\n')[0].strip()
        return first_line, False
        
    except Exception as e:
        logging.error(f"Error parsing response: {str(e)}")
        return str(e), False

def evaluate_answer(question: str, correct_answer: str) -> tuple:
    """
    Use Claude to evaluate if the answer to a question is correct.
    Returns (predicted_answer, is_correct, response_text)
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
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": eval_prompt}]
        )
        
        response_text = response.content[0].text
        answer, is_correct = parse_response(response_text)
        return answer, is_correct, response_text
            
    except Exception as e:
        logging.error(f"Error evaluating answer: {str(e)}")
        return str(e), False, str(e)

def save_progress(df, output_path, idx):
    """Save progress to CSV and backup file"""
    # Save to main output file
    df.iloc[:idx+1].to_csv(output_path, index=False)
    
    # Also save to a backup file with timestamp
    backup_dir = Path(output_path).parent / "backups"
    backup_dir.mkdir(exist_ok=True)
    backup_file = backup_dir / f"gpqa_diamond_compressed_2_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.iloc[:idx+1].to_csv(backup_file, index=False)
    
    logging.info(f"Progress saved through question {idx}")

def main():
    logging.info("Starting evaluation of original and compressed questions")
    
    # Load compressed data
    input_path = Path(__file__).parent.parent / "datasets" / "gpqa_diamond_compressed_1.csv"
    output_path = Path(__file__).parent.parent / "datasets" / "gpqa_diamond_compressed_2.csv"
    
    logging.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    # Add evaluation columns for original and compressed questions
    df['original_predicted_answer'] = None
    df['original_is_correct'] = None
    df['original_response'] = None
    
    for level in [75, 50, 25]:
        df[f'compression_{level}_predicted_answer'] = None
        df[f'compression_{level}_is_correct'] = None
        df[f'compression_{level}_response'] = None
    
    # Process each question
    logging.info(f"Starting evaluation for {len(df)} questions...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        logging.info(f"\nProcessing question {idx}/{len(df)}...")
        
        # Evaluate original question
        logging.info(f"Original ({row['original_word_count']} words): {row['Question']}")
        predicted, is_correct, response = evaluate_answer(row['Question'], row['Correct Answer'])
        df.at[idx, 'original_predicted_answer'] = predicted
        df.at[idx, 'original_is_correct'] = is_correct
        df.at[idx, 'original_response'] = response
        logging.info(f"Original Answer: {predicted}")
        logging.info(f"Original Correct: {is_correct}")
        
        # Evaluate compressed versions
        for level in [75, 50, 25]:
            compressed_q = row[f'compression_{level}']
            if pd.isna(compressed_q):
                continue
                
            logging.info(f"\n{level}% compression ({row[f'compression_{level}_word_count']} words): {compressed_q}")
            predicted, is_correct, response = evaluate_answer(compressed_q, row['Correct Answer'])
            df.at[idx, f'compression_{level}_predicted_answer'] = predicted
            df.at[idx, f'compression_{level}_is_correct'] = is_correct
            df.at[idx, f'compression_{level}_response'] = response
            logging.info(f"{level}% Answer: {predicted}")
            logging.info(f"{level}% Correct: {is_correct}")
        
        # Save progress every 10 questions
        if idx % 10 == 0:
            save_progress(df, output_path, idx)
            
        # Calculate and log running accuracy
        accuracies = {
            'original': df['original_is_correct'].mean(),
            '75%': df['compression_75_is_correct'].mean(),
            '50%': df['compression_50_is_correct'].mean(),
            '25%': df['compression_25_is_correct'].mean()
        }
        logging.info("\nRunning Accuracies:")
        for version, acc in accuracies.items():
            if not pd.isna(acc):
                logging.info(f"{version}: {acc:.2%}")
        
        # Sleep to avoid rate limits (same as generate_compressions.py)
        time.sleep(1)
    
    # Final save
    logging.info(f"Saving final results to {output_path}")
    df.to_csv(output_path, index=False)
    
    # Print final statistics
    print("\nFinal Accuracy Results:")
    print("-----------------------")
    for version, acc in accuracies.items():
        if not pd.isna(acc):
            print(f"{version}: {acc:.2%}")
    
    logging.info("Evaluation complete!")

if __name__ == "__main__":
    main()