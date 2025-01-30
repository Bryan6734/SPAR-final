"""
Evaluate model performance on original and compressed questions using Claude-3 with chain of thought reasoning.
"""

import os
import re
import logging
import pickle
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from utils import chain_of_thought_prompt, call_model_with_retries, call_models_in_parallel, Example, load_examples
from experiment_logger import ExperimentLogger
import asyncio

class AnswerPredictor:
    LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    INDEX_TO_LETTER = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    COMPRESSION_LEVELS = ['100', '75', '50', '25']

    def __init__(self, data_filename: str, verbose: bool = False, overwrite_cache: bool = False):
        self.model_name = "claude-3-5-sonnet-20241022"
        self.data_filename = data_filename
        self.verbose = verbose
        self.cache_filename = f"cache_{self.model_name}.pkl"
        self.overwrite_cache = overwrite_cache
        if os.path.exists(self.cache_filename):
            with open(self.cache_filename, 'rb') as file:
                self.cache = pickle.load(file)
        else:
            self.cache = {}
        
        # Initialize logger
        self.logger = ExperimentLogger("chain_of_thought")
        
        # Load and store full dataset
        self.df = pd.read_csv(self.data_filename)
        self.results = []
    
    def save_cache(self):
        with open(self.cache_filename, 'wb') as file:
            pickle.dump(self.cache, file)

    def get_response_from_cache_or_model(self, prompt):
        key = (self.model_name, prompt)
        if key in self.cache and not self.overwrite_cache:
            return self.cache[key]
        
        resp = call_model_with_retries(prompt, self.model_name, call_type='sample')
        self.cache[key] = resp
        self.save_cache()
        return resp

    @staticmethod
    def parse_sampled_answer(answer):
        pattern = r'answer is \((.)\)|Answer: \((.)\)|answer: \((.)\)|answer \((.)\)|\((.)\)'
        matches = re.findall(pattern, answer)
        valid_answers = []
        for match in matches:
            letter = next((group for group in match if group), None)
            if letter in AnswerPredictor.LETTER_TO_INDEX:
                valid_answers.append(letter)
        return valid_answers[-1] if valid_answers else None
    
    def get_answer(self, prompt):
        resp = self.get_response_from_cache_or_model(prompt)
        # resp is now a string, not an Anthropic response object
        
        if self.verbose:
            logging.info(f"====================PROMPT====================\n{prompt}\n====================PROMPT====================\n")
            logging.info(f'====================ANSWER====================\n{resp}\n====================ANSWER====================')
        
        return self.parse_sampled_answer(resp), resp
    
    def save_results(self):
        """Save current results to CSV"""
        results_df = pd.DataFrame(self.results)
        output_file = 'evals.csv'
        
        # If file exists, read existing results and append new ones
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            # Append only new results
            results_df = pd.concat([existing_df, results_df], ignore_index=True).drop_duplicates()
        
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        return output_file

    def main(self):
        """Run the experiment"""
        print("\nStarting evaluation...\n")
        
        # Process each row (original question) in the dataset
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            print(f"\n{'='*80}\nQuestion {idx+1}/{len(self.df)}:")
            
            # Shuffle answer choices once for all versions of this question
            choices = [
                row['Correct Answer'],
                row['Incorrect Answer 1'],
                row['Incorrect Answer 2'],
                row['Incorrect Answer 3']
            ]
            import random
            random.seed(42 + idx)  # Ensure reproducibility but different for each question
            random.shuffle(choices)
            correct_index = choices.index(row['Correct Answer'])
            
            # Store results for all versions of this question
            question_results = {
                'Question': row['Question'],
                'Correct Answer': row['Correct Answer'],
                'Incorrect Answer 1': row['Incorrect Answer 1'],
                'Incorrect Answer 2': row['Incorrect Answer 2'],
                'Incorrect Answer 3': row['Incorrect Answer 3'],
                'Shuffled Choice 1': choices[0],
                'Shuffled Choice 2': choices[1],
                'Shuffled Choice 3': choices[2],
                'Shuffled Choice 4': choices[3],
                'Correct Index': correct_index
            }
            
            # Create examples for all versions
            examples = []
            prompts = []
            for level in self.COMPRESSION_LEVELS:
                question_text = row[f'compression_{level}'] if level != '100' else row['Question']
                example = Example(
                    question_text,
                    choices[0],
                    choices[1],
                    choices[2],
                    choices[3],
                    correct_index
                )
                examples.append(example)
                prompts.append(chain_of_thought_prompt(example))
            
            # Get model responses for all versions in parallel
            responses = asyncio.run(call_models_in_parallel(
                prompts=prompts,
                model_name=self.model_name,
                call_type='sample',
                temperature=0.3,
                stop=None,
                max_tokens=1000
            ))
            
            # Process responses and store results
            versions_log = []
            for level, response in zip(self.COMPRESSION_LEVELS, responses):
                predicted_letter = self.parse_sampled_answer(response)
                if predicted_letter is None:
                    predicted_index = -1
                else:
                    predicted_index = self.LETTER_TO_INDEX[predicted_letter]
                
                # Store version results
                versions_log.append({
                    'version': level,
                    'predicted_letter': predicted_letter
                })
                
                # Store in CSV results
                question_results[f'compression_{level}_predicted'] = predicted_letter
                question_results[f'compression_{level}_correct'] = (predicted_index == correct_index)
            
            # Log all versions of this question
            self.logger.log_interaction(
                question_number=idx + 1,
                original_question=dict(row),
                choices=choices,
                versions=versions_log
            )
            
            # Store results
            self.results.append(question_results)
            
            # Save results every 5 questions
            if (idx + 1) % 5 == 0:
                output_file = self.save_results()
                print(f"Progress save: {output_file}")
        
        # Final save and statistics
        output_file = self.save_results()
        results_df = pd.DataFrame(self.results)
        
        print("\nFinal Results:")
        print("="*80)
        for level in self.COMPRESSION_LEVELS:
            accuracy = results_df[f'compression_{level}_correct'].mean()
            print(f"{level}% compression accuracy: {accuracy:.2%}")
        
        print(f"\nResults saved to:")
        print(f"1. {output_file} (summary)")
        print(f"2. {self.logger.get_log_file()} (raw model interactions)")

if __name__ == '__main__':
    predictor = AnswerPredictor('datasets/compressions.csv', verbose=True)
    predictor.main()