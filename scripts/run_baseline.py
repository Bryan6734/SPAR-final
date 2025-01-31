import json
import logging
import os
import re
from datetime import datetime
import anthropic
from tqdm import tqdm
import random
import time

class AnswerPredictor:
    LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    
    def __init__(self, data_filename: str, examples_path: str = "prompts/chain_of_thought_examples.json"):
        self.model_name = "claude-3-5-sonnet-20241022"
        self.data_filename = data_filename
        self.last_request_time = 0
        self.min_request_interval = 60/16  # 16 requests per minute
        
        # Load few-shot examples
        with open(examples_path) as f:
            self.examples = json.load(f)["questions"]
            
        # Initialize Claude client
        self.client = anthropic.Anthropic()
        
        # Prepare cached system message and example
        example = self.examples[0]  # Use just the first example
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

    def test_single(self, question: str, choices: dict):
        """Test a single question and show exact model input/output"""
        # Format single example for one-shot prompt
        example = self.examples[0]  # Use just the first example
        few_shot_examples = f"\nQuestion: {example['question']}\n"
        few_shot_examples += "Choices:\n"
        for letter, choice in example['choices'].items():
            few_shot_examples += f"({letter}) {choice}\n"
        few_shot_examples += f"\nLet me think through this step by step:\n{example['explanation']}\n"
        few_shot_examples += f"Therefore, the answer is ({example['correct_answer']})\n\n---\n"

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
        prompt += "Here is an example:\n"  # Changed from "examples" to "example"
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
        
        # Load compressed questions
        with open('compressions_20250130_175049.json', 'r') as f:
            compressed_questions = json.load(f)
        
        # Create timestamp for files if not resuming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f'results_{timestamp}.csv'
        raw_log_path = f'raw_interactions_{timestamp}.txt'
        
        # Check for existing results to resume from
        results = []
        completed_pairs = set()  # Track (record_id, compression_idx) pairs we've done
        
        # Look for most recent results file
        existing_results = sorted([f for f in os.listdir('.') if f.startswith('results_') and f.endswith('.csv')], reverse=True)
        if existing_results:
            latest_results = existing_results[0]
            try:
                # Load existing results
                existing_df = pd.read_csv(latest_results)
                results = existing_df.to_dict('records')
                # Track what we've already done
                completed_pairs = {(r['record_id'], r['compression_index']) for r in results}
                # Use the same files to continue appending
                results_path = latest_results
                raw_log_path = f"raw_interactions_{latest_results[8:-4]}.txt"
                logging.info(f"Resuming from {latest_results} with {len(results)} existing results")
            except Exception as e:
                logging.warning(f"Error loading existing results, starting fresh: {e}")
                results = []
                completed_pairs = set()
        
        # Open raw log file in append mode
        raw_log = open(raw_log_path, 'a', encoding='utf-8')
        
        # Save results every N questions
        SAVE_FREQUENCY = 5
        
        try:
            # Process each question
            for idx, row in tqdm(df.iterrows(), total=len(df)):
                record_id = row['Record ID']
                original_question = row['Question']
                original_char_count = len(original_question)
                
                # Get all versions to evaluate (original + compressed)
                questions_to_evaluate = [original_question]  # Start with original
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
                random.shuffle(answer_letters)  # Shuffle the letters
                choices = dict(zip(answer_letters, answers))  # Map shuffled letters to answers
                
                # Find which letter has the correct answer
                correct_letter = None
                for letter, answer in choices.items():
                    if answer == row['Correct Answer']:
                        correct_letter = letter
                        break
                
                # Process each version of the question
                for compression_idx, question in enumerate(questions_to_evaluate):
                    # Skip if we've already done this pair
                    if (record_id, compression_idx) in completed_pairs:
                        continue
                        
                    compressed_char_count = len(question)
                    percent_of_original = (compressed_char_count / original_char_count) * 100
                    
                    # Evaluate question
                    predicted_letter, response, prompt = self.evaluate_question(question, choices)
                    
                    # Log raw input/output to file
                    raw_log.write(f"\n{'='*100}\n")
                    raw_log.write(f"QUESTION {idx} (Compression {compression_idx})\n")
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
                    
                    # Store result
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
                
                # Save results to CSV periodically
                if (idx + 1) % SAVE_FREQUENCY == 0 or idx == len(df) - 1:
                    pd.DataFrame(results).to_csv(results_path, index=False)
                    logging.info(f"Saved {len(results)} results to {results_path}")
                
                # Log progress
                logging.info(f"Question {idx}: Processed {len(questions_to_evaluate)} versions")
            
            # Print final accuracy
            accuracy = sum(r['is_correct'] for r in results) / len(results)
            print(f"\nFinal Accuracy: {accuracy:.2%}")
            
        finally:
            # Always close the raw log file
            raw_log.close()
            # Make sure to save one last time
            pd.DataFrame(results).to_csv(results_path, index=False)
            print(f"\nRaw interactions saved to: {raw_log_path}")
            print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    predictor = AnswerPredictor('datasets/gpqa_diamond.csv')  # Using compressed dataset
    predictor.main()