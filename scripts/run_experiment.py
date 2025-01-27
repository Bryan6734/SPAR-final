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
from utils import chain_of_thought_prompt, call_model_with_retries, Example, load_examples
from experiment_logger import ExperimentLogger

class AnswerPredictor:
    LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    INDEX_TO_LETTER = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

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

    def main(self):
        """Run the experiment"""
        results = []
        examples = load_examples(self.data_filename)  # This will shuffle the answers
        
        print("\nStarting evaluation...\n")
        for idx, example in tqdm(enumerate(examples), total=len(examples)):
            print(f"\n{'='*80}\nQuestion {idx+1}/{len(examples)}:")
            print(f"Q: {example.question}")
            
            print("\nPossible Answers:")
            choices = [example.choice1, example.choice2, example.choice3, example.choice4]
            for i, choice in enumerate(choices):
                print(f"{self.INDEX_TO_LETTER[i]}) {choice}")
            print()
            
            prompt = chain_of_thought_prompt(example)
            predicted_letter, full_response = self.get_answer(prompt)
            
            # Log the raw interaction
            self.logger.log_interaction(
                question_number=idx + 1,
                question=example.question,
                choices=choices,
                prompt=prompt,
                response=full_response
            )
            
            if predicted_letter is None:
                predicted_index = -1
                print(" Model failed to provide a valid answer")
            else:
                predicted_index = self.LETTER_TO_INDEX[predicted_letter]
                is_correct = predicted_index == example.correct_index
                print(f"Model's answer: ({predicted_letter})")
                print(" Correct!" if is_correct else " Incorrect!")
            
            print(f"\nModel's reasoning:\n{full_response}\n")
            
            results.append({
                'question': example.question,
                'predicted_index': predicted_index,
                'correct_index': example.correct_index,
                'correct': predicted_index == example.correct_index,
                'full_response': full_response
            })
            
        results_df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df.to_csv(f'results_{timestamp}.csv', index=False)
        
        accuracy = results_df['correct'].mean()
        print(f"\n{'='*80}")
        print(f"Final Accuracy: {accuracy:.2%}")
        print(f"Results saved to:")
        print(f"1. results_{timestamp}.csv (summary)")
        print(f"2. {self.logger.get_log_file()} (raw model interactions)")

if __name__ == '__main__':
    predictor = AnswerPredictor('datasets/compressions.csv', verbose=True)
    predictor.main()