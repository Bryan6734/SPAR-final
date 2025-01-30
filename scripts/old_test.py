import anthropic

question = "A light beam is propagating through a glass with index of refraction n. The glass is moving at constant velocity v in the same direction as the beam and toward the observer in laboratory. What is the speed of light in glass relative to the observer in laboratory? Take the speed of light in vacuum c=1."

client = anthropic.Anthropic()

# Define the system prompt
system_prompt = """You are an expert at compressing questions while preserving intent, meaning, and essential information.
You will be given a long question, and your task is to generate **five versions** of it, each progressively more compressed.

### **Instructions:**
1. **Generate five questions:**
   - **Q1**: The original question (unchanged).
   - **Q2**: 20% shorter than Q1.
   - **Q3**: 40% shorter than Q1.
   - **Q4**: 60% shorter than Q1.
   - **Q5**: 80% shorter than Q1.
2. **Preserve meaning**: The core intent must remain intact in all versions.
3. **Ensure self-containment**: Each compressed version must still make sense without needing external context. Prioritize essential details needed for computation.
4. **Apply smart compression**: Use contractions, remove redundancy, simplify wording, and eliminate non-essential phrases while maintaining clarity.
5. **Prioritize specific details**: Some problems require specific numbers/formulas to compute. Prioritize these essential details.
6. **Focus on compression over grammar**: If necessary, you do NOT need to use grammar or full sentences. Focus on capturing core ideas.
7. **Compare to Q1**: Each version must be compared directly to Q1, not just to the previous version. Ensure that Q2 removes exactly 20% of Q1, Q3 removes 40% of Q1, etc.

### **Format:**
ONLY output the five compressed versions in the following structured format (no explanations or extra text):

Q1: [original question]
Q2: [20% shorter than Q1]
Q3: [40% shorter than Q1]
Q4: [60% shorter than Q1]
Q5: [80% shorter than Q1]

"""

# User prompt with the question
user_prompt = f"Question: {question}\nCompressed questions:"

# Send a single API request to generate all five compressed versions
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=600,  # Sufficient space for all outputs
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

# Print the response
response = message.content[0].text
print(response)