"""
Our goal: Generate multiple compressed versions of each question. We'll use tokens.
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

# get claude
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

def load_compression_prompt():
    """Load the compression prompt template text file"""
    prompt_path = Path(__file__).parent.parent / "prompts" / "compress_question.txt"
    with open(prompt_path) as f:
        return f.read()

def get_token_count(text):
    """ Estimate the number of tokens in a string (around 4 characters / token). This is a rough estimate, but it's accurate enough for our purposes based on playing around w/ character vs tokenizer."""

    total_characters = len(text)
    total_tokens = math.ceil(total_characters / 4)
    return total_tokens

def calculate_target_tokens(total_tokens: int) -> dict:
    """Calculate target token counts for each compression level"""
    return {
        75: max(1, round(total_tokens * 0.75)),
        50: max(1, round(total_tokens * 0.50)),
        25: max(1, round(total_tokens * 0.25))
    }

def extract_compressions(response_text, target_words):
    """Extract all compressions from Claude's response"""
    if not isinstance(response_text, str):
        logging.error(f"Unexpected response type: {type(response_text)}")
        return {}
        
    compressions = {}
    patterns = {
        75: rf"{target_words[75]} words:\n(.*?)(?=\n\n|\n{target_words[50]}|\Z)",
        50: rf"{target_words[50]} words:\n(.*?)(?=\n\n|\n{target_words[25]}|\Z)",
        25: rf"{target_words[25]} words:\n(.*?)(?=\n\n|\Z)"
    }
    
    for percent, pattern in patterns.items():
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            compressions[percent] = match.group(1).strip()
        else:
            logging.warning(f"Could not find {percent}% compression in response")
            logging.debug(f"Response text: {response_text}")
    
    return compressions

def compress_question(question: str, prompt_template: str):
    """Generate compressions for a question
    """
    total_tokens = get_token_count(question)
    target_tokens = calculate_target_tokens(total_tokens)
    

    compressions = {}
    results = {}
    for percent, target in target_tokens.items():
        prompt = prompt_template.format(question=question, target_words=target)
        response = client.completion(
            prompt=prompt,
            model="claude-3-5-sonnet-20241022",
            temperature=0.3,
            max_tokens=target
        )
        compressions[percent] = response["completion"]
        results[percent] = {
            "compression": compressions[percent],
            "actual_words": get_token_count(compressions[percent]),
            "target_words": target
        }
    return compressions, results

    

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
    logging.info("Starting compression generation")
    
    # Load data
    input_path = Path(__file__).parent.parent / "datasets" / "gpqa_diamond.csv"
    output_path = Path(__file__).parent.parent / "datasets" / "gpqa_diamond_compressed_1.csv"
    
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
    
    # Process each question
    logging.info(f"Starting compression generation for {len(df)} questions...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        logging.info(f"\nProcessing question {idx}/{len(df)}...")
        
        compressions, results = compress_question(row["Question"], prompt_template)
        
        if compressions and results:
            for target, result in results.items():
                col_name = f"compression_{target}"
                df.at[idx, col_name] = result['compression']
                df.at[idx, f"{col_name}_word_count"] = result['actual_words']
                df.at[idx, f"{col_name}_target_words"] = result['target_words']
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
    
    logging.info("Compression generation complete!")

if __name__ == "__main__":
    main()


    