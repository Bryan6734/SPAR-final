import json
import logging
import os
import re
from datetime import datetime
import anthropic
from tqdm import tqdm
import random

class AnswerPredictor:
    LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    
    def __init__(self, data_filename: str, examples_path: str = "prompts/chain_of_thought_examples.json"):
        self.model_name ="claude-3-5-sonnet-20241022"
        self.data_filename = data_filename
        
        # Load few-shot examples
        with open(examples_path) as f:
            self.examples = json.load(f)["questions"]
            
        # Initialize Claude client
        self.client = anthropic.Anthropic()
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Set random seed for reproducibility
        random.seed(42)

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
        """Evaluate a single question using few-shot chain of thought"""
        # Format examples for few-shot prompt
        few_shot_examples = ""
        for ex in self.examples:
            few_shot_examples += f"\nQuestion: {ex['question']}\n"
            few_shot_examples += "Choices:\n"
            # Present example choices in alphabetical order
            for letter in sorted(ex['choices'].keys()):
                few_shot_examples += f"({letter}) {ex['choices'][letter]}\n"
            few_shot_examples += f"\nLet me think through this step by step:\n{ex['explanation']}\n"
            few_shot_examples += f"Therefore, the answer is ({ex['correct_answer']})\n\n---\n"

        # Format the test question
        test_prompt = f"Question: {question}\nChoices:\n"
        # Present test choices in alphabetical order
        for letter in sorted(choices.keys()):
            test_prompt += f"({letter}) {choices[letter]}\n"
        test_prompt += "\nLet me think through this step by step:"

        # Combine into final prompt
        prompt = f"You are an expert at answering academic questions. For each question:\n"
        prompt += "1. Think through the problem step by step\n"
        prompt += "2. Show your reasoning\n"
        prompt += "3. Conclude with 'Therefore, the answer is (X)' where X is A, B, C, or D\n\n"
        prompt += "Here are some examples:\n"
        prompt += few_shot_examples
        prompt += "\nNow solve this question:\n"
        prompt += test_prompt
        
        # Get model response
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text
        predicted_answer = self.parse_sampled_answer(response_text)
        
        return predicted_answer, response_text, prompt  # Return prompt as well

    def test_single(self, question: str, choices: dict):
        """Test a single question and show exact model input/output"""
        # Format examples for few-shot prompt
        few_shot_examples = ""
        for ex in self.examples:
            few_shot_examples += f"\nQuestion: {ex['question']}\n"
            few_shot_examples += "Choices:\n"
            for letter, choice in ex['choices'].items():
                few_shot_examples += f"({letter}) {choice}\n"
            few_shot_examples += f"\nLet me think through this step by step:\n{ex['explanation']}\n"
            few_shot_examples += f"Therefore, the answer is ({ex['correct_answer']})\n\n---\n"

        # Format the test question
        test_prompt = f"Question: {question}\nChoices:\n"
        for letter, choice in choices.items():
            test_prompt += f"({letter}) {choice}\n"
        test_prompt += "\nLet me think through this step by step:"

        # Combine into final prompt
        prompt = f"You are an expert at answering academic questions. For each question:\n"
        prompt += "1. Think through the problem step by step\n"
        prompt += "2. Show your reasoning\n"
        prompt += "3. Conclude with 'Therefore, the answer is (X)' where X is A, B, C, or D\n\n"
        prompt += "Here are some examples:\n"
        prompt += few_shot_examples
        prompt += "\nNow solve this question:\n"
        prompt += test_prompt

        print("\n=== EXACT MODEL INPUT ===")
        print(prompt)
        
        # Get model response
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text
        predicted_answer = self.parse_sampled_answer(response_text)
        
        print("\n=== EXACT MODEL OUTPUT ===")
        print(response_text)
        print("\n=== PARSED ANSWER ===")
        print(f"Model predicted: ({predicted_answer})")
        
        return predicted_answer, response_text

    def main(self):
        """Run evaluation on all questions"""
        # Load questions from CSV
        import pandas as pd
        df = pd.read_csv(self.data_filename)
        results = []
        
        # Create timestamp for files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Open raw log file
        raw_log_path = f'raw_interactions_{timestamp}.txt'
        raw_log = open(raw_log_path, 'w', encoding='utf-8')
        
        try:
            # Process each question
            for idx, row in tqdm(df.iterrows(), total=len(df)):
                # Create list of all answers
                answers = [
                    row['Incorrect Answer 1'],
                    row['Incorrect Answer 2'],
                    row['Incorrect Answer 3'],
                    row['Correct Answer']
                ]
                
                # Randomly shuffle answers
                answer_letters = ['A', 'B', 'C', 'D']
                random.shuffle(answer_letters)  # Shuffle the letters
                choices = dict(zip(answer_letters, answers))  # Map shuffled letters to answers
                
                # Find which letter has the correct answer
                correct_letter = None
                for letter, answer in choices.items():
                    if answer == row['Correct Answer']:
                        correct_letter = letter
                        break
                
                # Evaluate question
                predicted_letter, response, prompt = self.evaluate_question(row['Question'], choices)
                
                # Log raw input/output to file
                raw_log.write(f"\n{'='*100}\n")
                raw_log.write(f"QUESTION {idx}\n")
                raw_log.write(f"{'='*100}\n\n")
                raw_log.write("=== MODEL INPUT ===\n")
                raw_log.write(f"{prompt}\n\n")
                raw_log.write("=== MODEL OUTPUT ===\n")
                raw_log.write(f"{response}\n\n")
                raw_log.write("=== PARSED RESULT ===\n")
                raw_log.write(f"Predicted: {predicted_letter} ({choices[predicted_letter] if predicted_letter else 'INVALID'})\n")
                raw_log.write(f"Correct: {correct_letter} ({choices[correct_letter]})\n")
                raw_log.write(f"{'='*100}\n\n")
                raw_log.flush()  # Ensure writing to disk
                
                # Check correctness
                is_correct = predicted_letter == correct_letter
                
                # Store results
                results.append({
                    'question_id': idx,
                    'question': row['Question'],
                    'predicted_answer': predicted_letter,
                    'correct_answer': correct_letter,
                    'is_correct': is_correct,
                    'model_response': response
                })
                
                # Log progress
                logging.info(f"Question {idx}: {'✓' if is_correct else '✗'} (predicted {predicted_letter}, correct {correct_letter})")
                
            # Save results to CSV
            results_df = pd.DataFrame(results)
            results_df.to_csv(f'results_{timestamp}.csv', index=False)
            
            # Print final accuracy
            accuracy = sum(r['is_correct'] for r in results) / len(results)
            print(f"\nFinal Accuracy: {accuracy:.2%}")
            
        finally:
            # Always close the raw log file
            raw_log.close()
            print(f"\nRaw interactions saved to: {raw_log_path}")

if __name__ == "__main__":
    predictor = AnswerPredictor('datasets/gpqa_diamond.csv')  # Using compressed dataset
    predictor.main()