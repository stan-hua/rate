#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone QC CLI for generating quality control files from radiology report processing results.

Usage examples:
    # Generate all QC types
    python3 src/qc_cli.py --results-path results.json --qc-dir qc_output/
    
    # Generate only specific QC types
    python3 src/qc_cli.py --results-path results.json --qc-dir qc_output/ --qc-types findings categories
    
    # Customize sample budgets
    python3 src/qc_cli.py --results-path results.json --qc-dir qc_output/ --findings-budget 50 --category-budget 15
"""

import argparse
import sys
from pathlib import Path

from .core.qc import QCGenerator

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate QC files from radiology report processing results.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument('--results-path', type=str, required=True,
                      help='Path to the JSON file containing processing results')
    parser.add_argument('--qc-dir', type=str, required=True,
                      help='Directory to save QC files')
    
    # QC type selection
    parser.add_argument('--qc-types', nargs='+', 
                      choices=['combined-findings', 'categories', 'questions', 'all'],
                      default=['all'],
                      help='Types of QC files to generate (default: all)')
    
    # Budget settings for QC
    parser.add_argument('--combined-findings-budget', type=int, default=40,
                      help='Number of samples for combined findings+no-comparison QC (default: 40)')
    parser.add_argument('--category-budget', type=int, default=10,
                      help='Number of samples per category for consolidated category QC (default: 10)')
    parser.add_argument('--questions-budget', type=int, default=50,
                      help='Total number of samples for questions QC (default: 50)')
    
    # Set random seed for reproducibility
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducible sampling (default: 42)')
    
    # Verbose output
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Enable verbose output')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup logging
    import logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set random seed for reproducibility
    import numpy as np
    np.random.seed(args.seed)
    
    print(f"QC File Generator")
    print(f"Results file: {args.results_path}")
    print(f"Output directory: {args.qc_dir}")
    print(f"Random seed: {args.seed}")
    
    # Validate input file exists
    if not Path(args.results_path).exists():
        print(f"Error: Results file not found: {args.results_path}")
        return 1
    
    # Initialize QC generator
    try:
        generator = QCGenerator(args.results_path, args.qc_dir)
    except Exception as e:
        print(f"Error initializing QC generator: {e}")
        return 1
    
    # Determine which QC types to generate
    if args.qc_types == ['all']:
        qc_types = ['combined-findings', 'categories', 'questions']
    else:
        qc_types = args.qc_types
    
    print(f"Generating QC types: {', '.join(qc_types)}")
    print()

    try:
        generated_files = generator.generate_all_qc(
            combined_findings_budget=args.combined_findings_budget,
            category_budget=args.category_budget,
            questions_budget=args.questions_budget
        )
    except Exception as e:
        print(f"Error during QC generation: {e}")
        return 1
    
    # Display results
    print(f"\nQC file generation complete!")
    print(f"Generated files saved in: {args.qc_dir}")
    print("-" * 50)
    
    total_files = 0
    for qc_type, file_paths in generated_files.items():
        if file_paths:
            print(f"{qc_type.replace('_', ' ').title()}: {len(file_paths)} file(s)")
            total_files += len(file_paths)
    
    print(f"\nTotal files generated: {total_files}")
    
    # Show what files were created
    qc_dir = Path(args.qc_dir)
    if qc_dir.exists():
        csv_files = list(qc_dir.glob("*.csv"))
        if csv_files:
            print(f"Files created:")
            for csv_file in sorted(csv_files):
                size_kb = csv_file.stat().st_size / 1024
                print(f"   {csv_file.name} ({size_kb:.1f} KB)")
    
    return 0

if __name__ == "__main__":
    exit(main()) 