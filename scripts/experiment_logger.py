"""Logger for experiment inputs and outputs"""

import json
from datetime import datetime
import os

class ExperimentLogger:
    def __init__(self, experiment_name="experiment"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = "experiment_logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        self.log_file = os.path.join(self.log_dir, f"{experiment_name}_{self.timestamp}.jsonl")
        
    def log_interaction(self, question_number: int, original_question: dict, choices: list, versions: list):
        """Log a single question's interactions across all compression levels
        
        Args:
            question_number: The number of the question
            original_question: The original question data from the CSV
            choices: The shuffled answer choices
            versions: List of dicts containing predictions for each version
                     Each dict should have: {'version': '100/75/50/25', 'predicted_letter': str}
        """
        log_entry = {
            "question_number": question_number,
            "original_data": {
                "question": original_question["Question"],
                "correct_answer": original_question["Correct Answer"]
            },
            "choices": choices,
            "predictions": versions,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
    def get_log_file(self):
        return self.log_file
