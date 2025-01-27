# GPQA Compression Evaluation Process

## Overview
This document details the evaluation process for testing the effectiveness of question compressions in the GPQA dataset. The evaluation compares model performance on original questions against their compressed versions at 75%, 50%, and 25% of the original word count.

## Input Data
- Source: `datasets/gpqa_diamond_compressed_1.csv`
- Contains:
  - Original questions and their correct answers
  - Compressed versions at 75%, 50%, and 25% compression levels
  - Word counts for each version
  - Target word counts for each compression level

## Evaluation Process

### 1. Setup and Configuration
- Uses Claude 3 Sonnet (2024-02-29 version)
- Temperature set to 0 for deterministic responses
- Rate limiting: 1 second delay between API calls
- Logging setup:
  - Detailed logs saved to `logs/evaluation_[timestamp].log`
  - Both file and console output
  - Includes timestamps and log levels

### 2. Question Evaluation
For each question (original and compressed versions):

1. **System Prompt**:
   ```
   You are an expert at answering academic questions accurately and evaluating answers.
   Your task is to:
   1. Answer the given question
   2. Compare your answer to the provided correct answer
   3. Return a tuple of (your_answer, True/False) where True means the answers match
   
   Be strict in your evaluation - answers must match in meaning, even if the wording is different.
   ```

2. **Evaluation Prompt Format**:
   ```
   Question: [question_text]
   Correct Answer: [correct_answer]
   
   Provide your answer and evaluation in this format:
   (your_answer, is_correct)
   ```

3. **Processing Order**:
   - Original question
   - 75% compression
   - 50% compression
   - 25% compression

### 3. Data Collection
For each question version, we collect:
- Predicted answer from the model
- Boolean correctness (True/False)
- Full model response text
- Word counts (actual vs. target)

### 4. Progress Tracking and Backup
- Progress saved every 10 questions to:
  - Main output file: `datasets/gpqa_diamond_compressed_2.csv`
  - Backup file: `datasets/backups/gpqa_diamond_compressed_2_backup_[timestamp].csv`
- Running accuracy statistics logged for:
  - Original questions
  - Each compression level (75%, 50%, 25%)

### 5. Output Data Format
The final CSV (`gpqa_diamond_compressed_2.csv`) contains:

Original columns:
- Question
- Correct Answer
- original_word_count
- compression_[75/50/25]
- compression_[75/50/25]_word_count
- compression_[75/50/25]_target_words

New evaluation columns:
- original_predicted_answer
- original_is_correct
- original_response
- compression_[75/50/25]_predicted_answer
- compression_[75/50/25]_is_correct
- compression_[75/50/25]_response

### 6. Error Handling
- Robust error handling for API calls
- Logging of all errors and warnings
- Graceful handling of missing or malformed data
- Backup saves to prevent data loss

### 7. Performance Considerations
- Rate limiting to prevent API throttling
- Efficient batch processing
- Regular progress saves
- Backup system for data safety

## Running the Evaluation
1. Ensure ANTHROPIC_API_KEY environment variable is set
2. Run: `python scripts/run_baseline.py`
3. Monitor progress in console and log file
4. Results will be saved to `datasets/gpqa_diamond_compressed_2.csv`

## Analysis Metrics
The evaluation tracks:
1. Accuracy per compression level
2. Word count adherence
3. Answer consistency across compression levels
4. Response patterns and failure modes

## Output Files
1. **Main Results**: `datasets/gpqa_diamond_compressed_2.csv`
2. **Logs**: `logs/evaluation_[timestamp].log`
3. **Backups**: `datasets/backups/gpqa_diamond_compressed_2_backup_[timestamp].csv`
