import json
import logging
import os
import re
from datetime import datetime
import anthropic
from tqdm import tqdm

class AnswerPredictor:
    LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    
    def __init__(self, data_filename: str, examples_path: str = "prompts/chain_of_thought_examples.json"):
        self.model_name = "claude-3-sonnet-20240229"
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

        # Get model response
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text
        predicted_answer = self.parse_sampled_answer(response_text)
        
        return predicted_answer, response_text

    def main(self):
        """Run evaluation on all questions"""
        # Load questions from CSV
        import pandas as pd
        df = pd.read_csv(self.data_filename)
        results = []
        
        # Process each question
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # Format choices
            choices = {
                'A': row['Incorrect Answer 1'],
                'B': row['Incorrect Answer 2'],
                'C': row['Incorrect Answer 3'],
                'D': row['Correct Answer']
            }
            
            # Evaluate question
            predicted_letter, response = self.evaluate_question(row['Question'], choices)
            
            # Check correctness
            is_correct = predicted_letter == 'D'  # Since correct answer is always D in choices
            
            # Store results
            results.append({
                'question_id': idx,
                'question': row['Question'],
                'predicted_answer': predicted_letter,
                'is_correct': is_correct,
                'model_response': response
            })
            
            # Log progress
            logging.info(f"Question {idx}: {'✓' if is_correct else '✗'} (predicted {predicted_letter})")
            
        # Save results
        results_df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df.to_csv(f'results_{timestamp}.csv', index=False)
        
        # Print final accuracy
        accuracy = sum(r['is_correct'] for r in results) / len(results)
        print(f"\nFinal Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    predictor = AnswerPredictor('datasets/gpqa_diamond.csv')
    predictor.main()