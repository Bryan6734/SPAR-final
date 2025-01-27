"""
Evaluate model performance on original and compressed GPQA questions using Claude-3.
Uses multiple-choice answer parsing and evaluates all compression levels.
"""

import os
import re
import logging
import pickle
import pandas as pd
import anthropic
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import time
import json
from dotenv import load_dotenv
from utils import chain_of_thought_prompt, call_model_with_retries, Example
from typing import Dict, Any

# Load environment variables from .env file
load_dotenv()

# Constants
MODEL_NAME = "claude-3-sonnet-20240229"
COMPRESSION_LEVELS = [75, 50, 25]

# Load CoT examples
with open(Path(__file__).parent.parent / "prompts" / "chain_of_thought_examples.json", 'r') as f:
    COT_EXAMPLES = json.load(f)["questions"]

class ColorFormatter(logging.Formatter):
    """Custom formatter with colors"""
    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    green = "\x1b[32;20m"
    bold_white = "\x1b[1;37m"
    reset = "\x1b[0m"

    def format(self, record):
        if record.levelno == logging.INFO:
            if '[CORRECT]' in record.msg:
                record.msg = record.msg.replace('[CORRECT]', f'{self.green}[CORRECT]{self.reset}')
            if '[WRONG]' in record.msg:
                record.msg = record.msg.replace('[WRONG]', f'{self.red}[WRONG]{self.reset}')
            if record.msg.startswith('Q:'):
                record.msg = f"{self.bold_white}{record.msg}{self.reset}"
            if 'Correct answer:' in record.msg:
                record.msg = f"{self.green}{record.msg}{self.reset}"
        return record.msg

class ExperimentRunner:
    LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    
    def __init__(self, input_path: str, cache_path: str = None):
        """Initialize the experiment runner.
        
        Args:
            input_path: Path to input CSV with original and compressed questions
            cache_path: Optional path to cache file for model responses
        """
        self.input_path = Path(input_path)
        self.cache_path = Path(cache_path) if cache_path else Path("response_cache.pkl")
        
        # Load API key
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Please set ANTHROPIC_API_KEY environment variable")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Initialize cache
        self.cache = self._load_cache()
        
        # Setup logging
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = Path('logs')
        self.log_dir.mkdir(exist_ok=True)
        
        # Main log file
        log_file = self.log_dir / f'experiment_{timestamp}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        
        # Model I/O log file
        self.model_io_file = self.log_dir / f'model_io_{timestamp}.txt'
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            handlers=[
                logging.StreamHandler(),
                file_handler
            ]
        )

    def _load_cache(self):
        """Load or create response cache."""
        if self.cache_path.exists():
            with open(self.cache_path, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def _save_cache(self):
        """Save response cache to disk."""
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.cache, f)
            
    def log_model_io(self, prompt: str, response: str, question_idx: int):
        """Log model inputs and outputs to a separate file"""
        with open(self.model_io_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Question {question_idx}\n")
            f.write(f"{'='*80}\n")
            f.write("MODEL INPUT:\n")
            f.write(f"{prompt}\n")
            f.write(f"\nMODEL OUTPUT:\n")
            f.write(f"{response}\n")
            f.write(f"{'='*80}\n")

    def evaluate_question(self, row: pd.Series, idx: int) -> Dict[str, Any]:
        """Evaluate a single question."""
        try:
            # Create Example from the row data
            example = Example(
                question=row['Question'],
                choice1=row['Correct Answer'],
                choice2=row['Incorrect Answer 1'],
                choice3=row['Incorrect Answer 2'],
                choice4=row['Incorrect Answer 3'],
                correct_index=0  # Since correct answer is always first
            )

            # Log the question being processed
            logging.info("="*80)
            logging.info(f"\nQuestion: {example.question}")
            logging.info(f"Choices:\n(A) {example.choice1}\n(B) {example.choice2}\n(C) {example.choice3}\n(D) {example.choice4}\n")

            # Generate chain of thought prompt using utils
            prompt = chain_of_thought_prompt(example)
            
            # Get model response using utils
            logging.info("[Generating Chain of Thought Response]")
            response = call_model_with_retries(
                prompt=prompt,
                model_name=MODEL_NAME,
                call_type='sample',  
                temperature=0.0
            )
            
            # Log model I/O
            self.log_model_io(prompt, response, idx)
            
            if not response:
                return None

            # Extract the final answer (looking for "(A)", "(B)", "(C)", or "(D)")
            import re
            final_answer = None
            matches = re.findall(r'\([A-D]\)', response)
            if matches:
                final_answer = matches[-1][1]  # Get the letter from the last match

            if not final_answer:
                logging.error("Could not extract final answer from response")
                return None

            # Check if answer is correct (A is always correct since we put correct answer first)
            is_correct = final_answer.upper() == 'A'

            return {
                'response': response,
                'final_answer': final_answer,
                'is_correct': is_correct
            }

        except Exception as e:
            logging.error(f"Error evaluating question: {str(e)}")
            return None

    def save_results(self, df, results, output_path):
        """Save evaluation results to CSV."""
        # Add result columns
        for version in ['original'] + [f'compression_{l}' for l in COMPRESSION_LEVELS]:
            df[f'{version}_predicted'] = None
            df[f'{version}_correct'] = None
            df[f'{version}_response'] = None
            
            # Update results for each row
            for idx, row in df.iterrows():
                if str(idx) in results and version in results[str(idx)]:
                    result = results[str(idx)][version]
                    df.at[idx, f'{version}_predicted'] = result['final_answer']
                    df.at[idx, f'{version}_correct'] = result['is_correct']
                    df.at[idx, f'{version}_response'] = result['response']
            
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        # Also save backup
        backup_dir = Path(output_path).parent / "backups"
        backup_dir.mkdir(exist_ok=True)
        backup_path = backup_dir / f"results_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(backup_path, index=False)

    def run(self):
        """Run the experiment."""
        # Configure colored logging
        handler = logging.StreamHandler()
        handler.setFormatter(ColorFormatter())
        logging.getLogger().handlers = [handler]
        logging.getLogger().setLevel(logging.INFO)
        
        # Disable HTTP request logging
        logging.getLogger("httpx").setLevel(logging.WARNING)
        
        # Load questions
        df = pd.read_csv(self.input_path)
        results = {}
        
        print(f"\n{MODEL_NAME} Evaluation ({len(df)} questions)\n")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), ncols=80):
            print(f"\n{'='*80}")
            logging.info(f"Q: {row['Question']}\n")
            logging.info(f"Correct answer: (A) {row['Correct Answer'].strip()}\n")
            
            # Evaluate original question
            results[str(idx)] = {'original': self.evaluate_question(row, idx)}
            original_result = results[str(idx)]['original']
            
            # Track results for this question
            compression_results = {
                '100%': (original_result['is_correct'], original_result['final_answer'], original_result['response'])
            }
            
            # Print original result
            status = '[CORRECT]' if original_result['is_correct'] else '[WRONG]'
            answer = original_result['response'].split("The correct answer is")[-1].strip()[:50]
            logging.info(f"[100%] {status} {answer}...")
            
            # Evaluate compressed versions
            for level in COMPRESSION_LEVELS:
                compressed_q = row.get(f'compression_{level}')
                if pd.isna(compressed_q):
                    logging.info(f"[{level}%] No compression available")
                    continue
                
                example = Example(
                    question=compressed_q,
                    choice1=row['Correct Answer'],
                    choice2=row['Incorrect Answer 1'],
                    choice3=row['Incorrect Answer 2'],
                    choice4=row['Incorrect Answer 3'],
                    correct_index=0
                )
                
                prompt = chain_of_thought_prompt(example)
                response = call_model_with_retries(
                    prompt=prompt,
                    model_name=MODEL_NAME,
                    call_type='sample',
                    temperature=0.0
                )
                
                if not response:
                    logging.info(f"[{level}%] Failed to get response")
                    continue

                # Extract the final answer
                import re
                final_answer = None
                matches = re.findall(r'\([A-D]\)', response)
                if matches:
                    final_answer = matches[-1][1]
                    
                if not final_answer:
                    logging.info(f"[{level}%] Could not extract answer")
                    continue

                # Check if answer is correct and print result
                is_correct = final_answer.upper() == 'A'
                status = '[CORRECT]' if is_correct else '[WRONG]'
                answer = response.split("The correct answer is")[-1].strip()[:50]
                logging.info(f"[{level}%] {status} {answer}...")
                
                results[str(idx)][f'compression_{level}'] = {
                    'response': response,
                    'final_answer': final_answer,
                    'is_correct': is_correct
                }
            
            # Calculate and log running accuracy
            accuracy = sum(1 for r in results.values() if r['original'] and r['original']['is_correct']) / len(results)
            logging.info(f"\nRunning Accuracy: {accuracy:6.2%}")
            
            # Sleep to avoid rate limits
            time.sleep(1)
        
        # Save final results
        self.save_results(df, results, self.input_path.parent / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        # Print final results
        logging.info("\nFinal Results:")
        logging.info(f"Accuracy: {accuracy:6.2%}")
        
        logging.info("\nEvaluation complete! ")

def main():
    """Main entry point."""
    # Set up input path
    input_path = Path(__file__).parent.parent / "datasets" / "gpqa_diamond_compressed_1.csv"
    
    # Initialize and run experiment
    runner = ExperimentRunner(input_path)
    runner.run()

if __name__ == "__main__":
    main()
