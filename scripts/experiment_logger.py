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
        
    def log_interaction(self, question_number: int, question: str, choices: list, prompt: str, response: str):
        """Log a single model interaction"""
        log_entry = {
            "question_number": question_number,
            "question": question,
            "choices": choices,
            "model_prompt": prompt,
            "model_response": response,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
    def get_log_file(self):
        return self.log_file
