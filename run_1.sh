#!/bin/bash

# Activate virtual environment if you have one
source venv/bin/activate

# Run GPQA-diamond evaluation with Claude
python scripts/run_baseline.py \
    --model_name=claude-3-sonnet-20240229 \
    --hf_dataset=lighteval/gpqa-diamond \
    --hf_split=test \
    --prompt_type=zero_shot \
    --verbose=True \
    --max_examples=10  # Remove this line to run on all examples