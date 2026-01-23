#!/usr/bin/env python3
"""
Script to process JSONL files with prompt/predict/label fields.
Hashes prompts and creates a sorted dictionary of hash:prediction or hash:label pairs.
"""

import argparse
import json
import hashlib
from pathlib import Path
from collections import OrderedDict


def hash_prompt(prompt):
    """Create a hash from the prompt string."""
    return hashlib.sha256(prompt.encode('utf-8')).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description='Process JSONL file and create hash:prediction/label pairs'
    )
    parser.add_argument(
        '--save_name',
        type=str,
        required=True,
        help='Full path (including filename) where the output will be saved'
    )
    parser.add_argument(
        '--save_gt',
        action='store_true',
        default=False,
        help='If set, save hash:label pairs instead of hash:predict pairs'
    )
    parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help='Path to input JSONL file'
    )
    
    args = parser.parse_args()
    
    # Create output path and ensure parent directory exists
    output_path = Path(args.save_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Read JSONL file and create hash:value pairs
    hash_dict = {}
    
    with open(args.input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                prompt = data['prompt']
                value = data['label'] if args.save_gt else data['predict']
                
                # Hash the prompt
                prompt_hash = hash_prompt(prompt)
                hash_dict[prompt_hash] = value
                
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON on line {line_num}: {e}")
            except KeyError as e:
                print(f"Warning: Missing key {e} on line {line_num}")
    
    # Sort by hash
    sorted_dict = OrderedDict(sorted(hash_dict.items()))
    
    # Save as JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_dict, f, indent=2, ensure_ascii=False)
    
    print(f"Processed {len(hash_dict)} entries")
    print(f"Output saved to: {output_path}")


if __name__ == '__main__':
    main()

