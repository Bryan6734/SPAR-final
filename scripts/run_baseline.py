import json
import logging
import os
import random
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import anthropic
import time
import multiprocessing as mp
from functools import partial
import re

class AnswerPredictor:
    LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    
    def __init__(self, data_filename: str, examples_path: str = "prompts/chain_of_thought_examples.json", enable_raw_logging=False):
        self.model_name = "claude-3-5-sonnet-20241022"
        self.data_filename = data_filename
        self.last_request_time = 0
        self.min_request_interval = 60/16  # 16 requests per minute
        self.enable_raw_logging = enable_raw_logging
        self.examples_path = examples_path  # Store path instead of loading
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Set random seed for reproducibility
        random.seed(42)

    def initialize_process(self):
        """Initialize process-specific resources"""
        # Load examples
        with open(self.examples_path) as f:
            self.examples = json.load(f)["questions"]
            
        # Create new Anthropic client for this process
        self.client = anthropic.Anthropic()
        
        # Prepare cached system message and example
        example = self.examples[0]
        few_shot_example = f"\nQuestion: {example['question']}\n"
        few_shot_example += "Choices:\n"
        for letter in sorted(example['choices'].keys()):
            few_shot_example += f"({letter}) {example['choices'][letter]}\n"
        few_shot_example += f"\nLet me think through this step by step:\n{example['explanation']}\n"
        few_shot_example += f"Therefore, the answer is ({example['correct_answer']})\n\n---\n"
        
        # Cache the constant parts of our prompt
        self.system_content = [
            {
                "type": "text",
                "text": "You are an expert at answering academic questions. For each question:\n"
                        "1. Think through the problem step by step\n"
                        "2. Show your reasoning\n"
                        "3. Conclude with 'Therefore, the answer is (X)' where X is A, B, C, or D\n\n"
                        "Here is an example:\n",
                "cache_control": {"type": "ephemeral"}
            },
            {
                "type": "text",
                "text": few_shot_example,
                "cache_control": {"type": "ephemeral"}
            }
        ]

    @staticmethod
    def parse_sampled_answer(answer):
        """Parse Claude's response to extract the letter answer"""
        patterns = [
            r'answer is \((.)\)', 
            r'Answer: \((.)\)', 
            r'answer: \((.)\)', 
            r'answer \((.)\)', 
            r'\((.)\)'
        ]
        for pattern in patterns:
            match = re.search(pattern, answer)
            if match and match.group(1) in AnswerPredictor.LETTER_TO_INDEX:
                return match.group(1)
        return None

    def evaluate_question(self, question: str, choices: dict):
        """Evaluate a single question using one-shot chain of thought with prompt caching"""
        # Rate limiting
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)

        # Format the test question
        test_prompt = f"Question: {question}\nChoices:\n"
        for letter in sorted(choices.keys()):
            test_prompt += f"({letter}) {choices[letter]}\n"
        test_prompt += "\nLet me think through this step by step:"
        
        # Get model response using cached system content
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1000,
            temperature=0,
            system=self.system_content,
            messages=[{"role": "user", "content": test_prompt}]
        )
        self.last_request_time = time.time()
        
        response_text = response.content[0].text
        predicted_answer = self.parse_sampled_answer(response_text)
        
        # For logging purposes, reconstruct the full prompt
        full_prompt = "SYSTEM:\n" + "".join(item["text"] for item in self.system_content)
        full_prompt += "\nUSER:\n" + test_prompt
        
        return predicted_answer, response_text, full_prompt

    def evaluate_chunk(self, df_chunk, compressed_questions, results_path, process_id):
        """Process a chunk of questions in a single process"""
        # Initialize process-specific resources
        self.initialize_process()
        
        results = []
        completed_pairs = set()
        
        try:
            for idx, row in tqdm(df_chunk.iterrows(), total=len(df_chunk), 
                               desc=f'Process {process_id}', position=process_id):
                record_id = row['Record ID']
                original_question = row['Question']
                original_char_count = len(original_question)
                
                # Get all versions to evaluate
                questions_to_evaluate = [original_question]
                if record_id in compressed_questions:
                    questions_to_evaluate.extend(compressed_questions[record_id])
                
                # Create list of all answers
                answers = [
                    row['Incorrect Answer 1'],
                    row['Incorrect Answer 2'],
                    row['Incorrect Answer 3'],
                    row['Correct Answer']
                ]
                
                # Randomly shuffle answers
                answer_letters = ['A', 'B', 'C', 'D']
                random.shuffle(answer_letters)
                choices = dict(zip(answer_letters, answers))
                
                # Find correct letter
                correct_letter = None
                for letter, answer in choices.items():
                    if answer == row['Correct Answer']:
                        correct_letter = letter
                        break
                
                chunk_correct = 0
                chunk_total = 0
                
                for compression_idx, question in enumerate(questions_to_evaluate):
                    if (record_id, compression_idx) in completed_pairs:
                        continue
                        
                    compressed_char_count = len(question)
                    percent_of_original = (compressed_char_count / original_char_count) * 100
                    
                    predicted_letter, response, prompt = self.evaluate_question(question, choices)
                    
                    is_correct = predicted_letter == correct_letter
                    chunk_correct += int(is_correct)
                    chunk_total += 1
                    
                    # Console logging
                    mark = '✓' if is_correct else '✗'
                    compression_str = "original" if compression_idx == 0 else f"compressed_{compression_idx}"
                    print(f"Process {process_id} - Q{idx} ({compression_str}): {mark} pred={predicted_letter}, true={correct_letter}")
                    
                    result = {
                        'record_id': record_id,
                        'compression_index': compression_idx,
                        'question_id': idx,
                        'question': question,
                        'predicted_letter': predicted_letter,
                        'predicted_answer': choices[predicted_letter] if predicted_letter else 'INVALID',
                        'correct_letter': correct_letter,
                        'correct_answer': choices[correct_letter],
                        'is_correct': is_correct,
                        'original_char_count': original_char_count,
                        'compressed_char_count': compressed_char_count,
                        'percent_of_original': percent_of_original
                    }
                    results.append(result)
                    completed_pairs.add((record_id, compression_idx))
                
                # Save results periodically
                if len(results) % 5 == 0:
                    pd.DataFrame(results).to_csv(f"{results_path}_process_{process_id}.csv", index=False)
                    # Print chunk accuracy
                    if chunk_total > 0:
                        chunk_accuracy = chunk_correct / chunk_total
                        print(f"Process {process_id} - Current Accuracy: {chunk_accuracy:.2%} ({chunk_correct}/{chunk_total})")
                    
        finally:
            # Save final results
            if results:
                pd.DataFrame(results).to_csv(f"{results_path}_process_{process_id}.csv", index=False)
        
        return results

    def main(self, num_processes=4):
        """Run parallel evaluation on all questions"""
        print(f"\nStarting evaluation with {num_processes} processes")
        print(f"Rate limit: {60/self.min_request_interval:.1f} requests/minute per process")
        print(f"Total system throughput: {60/self.min_request_interval * num_processes:.1f} requests/minute\n")
        
        # Load data
        df = pd.read_csv(self.data_filename)
        with open('compressions_20250130_175049.json', 'r') as f:
            compressed_questions = json.load(f)
        
        # Create timestamp for files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f'results_{timestamp}'
        
        # Calculate chunk sizes for even distribution
        total_questions = len(df)
        base_chunk_size = total_questions // num_processes
        remainder = total_questions % num_processes
        
        # Create chunks with even distribution
        chunks = []
        start_idx = 0
        for i in range(num_processes):
            # Add one extra item to some chunks if we have a remainder
            chunk_size = base_chunk_size + (1 if i < remainder else 0)
            end_idx = start_idx + chunk_size
            chunks.append(df[start_idx:end_idx])
            start_idx = end_idx
        
        print(f"Distributing {total_questions} questions across {num_processes} processes:")
        for i, chunk in enumerate(chunks):
            print(f"Process {i}: {len(chunk)} questions ({len(chunk) * 6} total evaluations)")
        print()
        
        # Create pool and run parallel processing
        with mp.Pool(num_processes) as pool:
            # Create list of arguments for each process
            process_args = [
                (chunk, compressed_questions, results_path, i) 
                for i, chunk in enumerate(chunks)
            ]
            
            # Run processes
            all_results = pool.starmap(self.evaluate_chunk, process_args)
        
        # Combine results from all processes
        combined_results = []
        for process_results in all_results:
            combined_results.extend(process_results)
        
        # Save final combined results
        final_df = pd.DataFrame(combined_results)
        final_df.to_csv(f"{results_path}_final.csv", index=False)
        
        # Print final accuracy
        accuracy = sum(r['is_correct'] for r in combined_results) / len(combined_results)
        print(f"\nFinal Accuracy: {accuracy:.2%}")
        print(f"\nResults saved to: {results_path}_final.csv")

if __name__ == "__main__":
    predictor = AnswerPredictor('datasets/gpqa_diamond.csv', enable_raw_logging=False)
    predictor.main(num_processes=4)