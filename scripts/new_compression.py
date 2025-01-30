import anthropic
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
from datetime import datetime

# Define the system prompt
system_prompt = """You are an expert at compressing questions while preserving intent, meaning, and essential information.
You will be given a long question, and your task is to generate **five versions** of it, each progressively more compressed.

### **Instructions:**
1. **Generate five questions:**
   - **Q1**: 90% of the length of Q1.
   - **Q2**: 80% of the length of Q1.
   - **Q3**: 60% of the length of Q1.
   - **Q4**: 40% of the length of Q1.
   - **Q5**: 20% of the length of Q1
2. **Preserve meaning**: The core intent must remain intact in all versions.
3. **Ensure self-containment**: Each compressed version must still make sense without needing external context. Prioritize essential details needed for computation.
4. **Apply smart compression**: Use contractions, remove redundancy, simplify wording, and eliminate non-essential phrases while maintaining clarity.
5. **Prioritize specific details**: Some problems require specific numbers/formulas to compute. Prioritize these essential details.
6. **Focus on compression over grammar**: If necessary, you do NOT need to use grammar or full sentences. Focus on capturing core ideas.
7. **Compare to Q1**: Each version should be a percnet of the length of Q1, not a percent of the length of the previous question.

### **Format:**
ONLY output the five compressed versions in the following structured format (no explanations or extra text):

Q1: [90% of the length of Q1]
Q2: [80% of the length of Q1]
Q3: [60% of the length of Q1]
Q4: [40% of the length of Q1]
Q5: [20% of the length of Q1]
"""

def process_single_question(question):
    """Process a single question using the Anthropic API"""
    client = anthropic.Anthropic()
    user_prompt = f"Question: {question}\nCompressed questions:"

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=600,
        temperature=0.5,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]
            }
        ]
    )
    return message.content[0].text

def parse_compressed_questions(response):
    """Extract the five compressed questions from the API response"""
    questions = []
    for line in response.split('\n'):
        if line.startswith('Q'):
            questions.append(line[3:].strip())
    return questions

def process_with_retry(record_id, question):
    """Process a question with retry logic and exponential backoff"""
    max_retries = 3
    base_delay = 2  # Base delay in seconds
    
    for attempt in range(max_retries):
        try:
            response = process_single_question(question)
            compressed = parse_compressed_questions(response)
            if len(compressed) == 5:  # Verify we got all 5 compressions
                return record_id, compressed
            raise Exception("Did not receive all 5 compressions")
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                print(f"Failed to process question {record_id} after {max_retries} attempts: {str(e)}")
                return record_id, None
            wait_time = base_delay * (2 ** attempt)  # Exponential backoff
            print(f"Attempt {attempt + 1} failed for {record_id}. Waiting {wait_time} seconds...")
            time.sleep(wait_time)

def process_questions_parallel(questions_df, num_processes=3):
    """Process multiple questions in parallel with rate limiting"""
    results = {}
    total_questions = len(questions_df)
    processed_count = 0
    
    # Create a timestamp for the output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'compressions_{timestamp}.json'
    
    print(f"Processing {total_questions} questions using {num_processes} parallel processes...")
    
    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        # Submit all tasks
        future_to_id = {
            executor.submit(
                process_with_retry, 
                row['Record ID'], 
                row['Question']
            ): row['Record ID'] 
            for _, row in questions_df.iterrows()
        }
        
        # Process completed tasks
        for future in as_completed(future_to_id):
            record_id, compressed = future.result()
            processed_count += 1
            
            if compressed:
                results[record_id] = compressed
                print(f"Successfully processed question {processed_count}/{total_questions} (ID: {record_id})")
            else:
                print(f"Failed to process question {processed_count}/{total_questions} (ID: {record_id})")
            
            # Save progress periodically
            if processed_count % 10 == 0:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Progress saved to {output_file}")
    
    # Save final results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nProcessing complete. Results saved to {output_file}")
    return results

if __name__ == "__main__":
    # Read questions from CSV
    df = pd.read_csv('datasets/gpqa_diamond.csv')
    
    # If only one question, use original functionality
    if len(df) == 1:
        question = "A light beam is propagating through a glass with index of refraction n. The glass is moving at constant velocity v in the same direction as the beam and toward the observer in laboratory. What is the speed of light in glass relative to the observer in laboratory? Take the speed of light in vacuum c=1."
        result = process_single_question(question)
        print(result)
    else:
        # Process multiple questions in parallel
        results = process_questions_parallel(df)