#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation CLI for analyzing quality control file annotations and computing performance metrics.

Usage examples:
    # Evaluate all QC types in a directory
    python3 src/eval_cli.py --qc-dir qc_output/
    
    # Evaluate specific QC types only
    python3 src/eval_cli.py --qc-dir qc_output/ --qc-types questions categories
    
    # Save results to file and CSV of imperfect tasks
    python3 src/eval_cli.py --qc-dir qc_output/ --output-file evaluation_results.json --imperfect-csv imperfect_tasks.csv
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

class QCEvaluator:
    """Evaluator for annotated QC files."""
    
    def __init__(self, qc_dir: str, verbose: bool = False):
        """
        Initialize QC evaluator.
        
        Args:
            qc_dir: Directory containing QC files
            verbose: Enable verbose logging
        """
        self.qc_dir = Path(qc_dir)
        self.verbose = verbose
        
        if verbose:
            logging.basicConfig(level=logging.INFO)
        
        if not self.qc_dir.exists():
            raise ValueError(f"QC directory does not exist: {qc_dir}")
    
    def _convert_to_binary(self, values: pd.Series, value_type: str = "label") -> np.ndarray:
        """
        Convert various annotation formats to binary labels.
        
        Args:
            values: Series containing the values to convert
            value_type: Type of value being converted ("label" or "correct")
            
        Returns:
            Binary numpy array
        """
        binary_values = []
        
        for val in values:
            if pd.isna(val) or val == "":
                binary_values.append(np.nan)
                continue
                
            if isinstance(val, (int, float)):
                binary_values.append(1 if val > 0 else 0)
            elif isinstance(val, str):
                val_lower = val.lower().strip()
                if val_lower in ['yes', 'true', '1', 'correct', 'good'] or val_lower.startswith('y'):
                    binary_values.append(1)
                elif val_lower in ['no', 'false', '0', 'incorrect', 'bad', 'wrong'] or val_lower.startswith('n'):
                    binary_values.append(0)
                else:
                    binary_values.append(np.nan)
            else:
                binary_values.append(np.nan)
        
        return np.array(binary_values)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Calculate performance metrics manually.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            
        Returns:
            Dictionary of metrics
        """
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if not mask.any():
            return {
                "accuracy": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1_score": np.nan,
                "n_samples": 0,
                "n_annotated": 0
            }
        
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        # Calculate confusion matrix components manually
        tp = np.sum((y_true_clean == 1) & (y_pred_clean == 1))
        tn = np.sum((y_true_clean == 0) & (y_pred_clean == 0))
        fp = np.sum((y_true_clean == 0) & (y_pred_clean == 1))
        fn = np.sum((y_true_clean == 1) & (y_pred_clean == 0))
        
        # Calculate metrics manually
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # PPV
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0     # Sensitivity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0        # Negative Predictive Value
        
        # F1 score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "accuracy": accuracy,
            "precision": precision,  # PPV
            "recall": recall,        # Sensitivity
            "f1_score": f1_score,
            "specificity": specificity,
            "npv": npv,              # Negative Predictive Value
            "n_samples": len(y_true),
            "n_annotated": len(y_true_clean),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp)
            }
        }
    
    def evaluate_no_comparisons_qc(self) -> Dict[str, Any]:
        """Evaluate no-comparisons QC from combined findings file."""
        file_path = self.qc_dir / "combined_findings_qc.csv"
        
        if not file_path.exists():
            logger.warning(f"Combined findings QC file not found: {file_path}")
            return {"error": "File not found"}
        
        try:
            df = pd.read_csv(file_path)
            
            if "no_comparison_correct" not in df.columns:
                return {"error": "Missing 'no_comparison_correct' column for human annotation"}
            
            # For no-comparisons, we assume correct=1 means the extraction was good
            # This is a correctness evaluation, not binary classification
            human_labels = self._convert_to_binary(pd.Series(df["no_comparison_correct"]), "correct")
            
            annotated_mask = ~np.isnan(human_labels)
            if not annotated_mask.any():
                return {"error": "No human annotations found"}
            
            accuracy = np.mean(human_labels[annotated_mask])
            
            return {
                "file": str(file_path),
                "accuracy": accuracy,
                "n_samples": len(df),
                "n_annotated": int(annotated_mask.sum()),
                "type": "correctness_evaluation"
            }
            
        except Exception as e:
            logger.error(f"Error evaluating no-comparisons QC: {e}")
            return {"error": str(e)}
    
    def evaluate_findings_qc(self) -> Dict[str, Any]:
        """Evaluate findings QC from combined findings file."""
        file_path = self.qc_dir / "combined_findings_qc.csv"
        
        if not file_path.exists():
            logger.warning(f"Combined findings QC file not found: {file_path}")
            return {"error": "File not found"}
        
        try:
            df = pd.read_csv(file_path)
            
            if "findings_correct" not in df.columns:
                return {"error": "Missing 'findings_correct' column for human annotation"}
            
            # Similar to no-comparisons - correctness evaluation
            human_labels = self._convert_to_binary(pd.Series(df["findings_correct"]), "correct")
            
            annotated_mask = ~np.isnan(human_labels)
            if not annotated_mask.any():
                return {"error": "No human annotations found"}
            
            accuracy = np.mean(human_labels[annotated_mask])
            
            return {
                "file": str(file_path),
                "accuracy": accuracy,
                "n_samples": len(df),
                "n_annotated": int(annotated_mask.sum()),
                "type": "correctness_evaluation"
            }
            
        except Exception as e:
            logger.error(f"Error evaluating findings QC: {e}")
            return {"error": str(e)}
    
    def evaluate_category_qc(self) -> Dict[str, Any]:
        """Evaluate category QC from consolidated categories file."""
        file_path = self.qc_dir / "categories_qc.csv"
        
        if not file_path.exists():
            logger.warning(f"Categories QC file not found: {file_path}")
            return {"error": "Categories QC file not found"}
        
        try:
            df = pd.read_csv(file_path)
            
            if "correct" not in df.columns:
                return {"error": "Missing 'correct' column for human annotation"}
            
            if "category" not in df.columns:
                return {"error": "Missing 'category' column in categories QC file"}
            
            results = {}
            
            # Group by category and evaluate each one separately
            categories_group = df.groupby('category')
            
            for category_name, category_df in categories_group:
                if len(category_df) == 0:
                    continue
                    
                try:
                    # Correctness evaluation for category findings
                    human_labels = self._convert_to_binary(pd.Series(category_df["correct"]), "correct")
                    
                    annotated_mask = ~np.isnan(human_labels)
                    if not annotated_mask.any():
                        results[category_name] = {"error": "No human annotations found"}
                        continue
                    
                    accuracy = np.mean(human_labels[annotated_mask])
                    
                    results[category_name] = {
                        "file": str(file_path),
                        "accuracy": accuracy,
                        "n_samples": len(category_df),
                        "n_annotated": int(annotated_mask.sum()),
                        "type": "correctness_evaluation"
                    }
                    
                except Exception as e:
                    logger.error(f"Error evaluating category QC {category_name}: {e}")
                    results[category_name] = {"error": str(e)}
            
            if not results:
                return {"error": "No valid categories found for evaluation"}
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating categories QC: {e}")
            return {"error": str(e)}
    
    def evaluate_questions_qc(self) -> Dict[str, Any]:
        """
        Evaluate questions QC files that contain mixed questions in a consolidated format.
        
        Returns:
            Dictionary containing evaluation metrics for each question plus consolidated metrics
        """
        qc_file = self.qc_dir / "questions_qc.csv"
        
        if not qc_file.exists():
            if self.verbose:
                logger.warning(f"Questions QC file not found: {qc_file}")
            return {}
        
        try:
            df = pd.read_csv(qc_file)
            
            # Check required columns
            required_columns = ['query_id', 'question', 'predicted_label', 'human_label']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                if self.verbose:
                    logger.warning(f"Missing columns in questions QC: {missing_columns}")
                return {}
            
            # Filter out rows without human annotations
            annotated_df = df.dropna(subset=['human_label'])
            annotated_df = annotated_df[annotated_df['human_label'].astype(str).str.strip() != '']
            
            if len(annotated_df) == 0:
                if self.verbose:
                    logger.warning("No annotated samples found in questions QC")
                return {}
            
            results = {}
            
            # First, calculate consolidated metrics across ALL questions
            try:
                # Convert all labels to binary for consolidated evaluation
                y_true_all = self._convert_to_binary(pd.Series(annotated_df['human_label']), "label")
                y_pred_all = self._convert_to_binary(pd.Series(annotated_df['predicted_label']), "label")
                
                # Calculate consolidated metrics
                consolidated_metrics = self._calculate_metrics(y_true_all, y_pred_all)
                
                # Count unique questions for additional context
                unique_questions = len(pd.Series(annotated_df['question']).unique())
                
                # Store consolidated results
                results['_CONSOLIDATED'] = {
                    **consolidated_metrics,
                    'n_samples': len(annotated_df),
                    'n_annotated': len(annotated_df),
                    'n_unique_questions': unique_questions,
                    'file': str(qc_file),
                    'description': f'Consolidated evaluation across {unique_questions} unique questions',
                    'type': 'binary_classification_consolidated'
                }
                
                if self.verbose:
                    logger.info(f"Consolidated questions evaluation: {len(annotated_df)} samples across {unique_questions} questions")
                
            except Exception as e:
                if self.verbose:
                    logger.warning(f"Failed to calculate consolidated metrics: {e}")
            
            # Then, process each question individually (existing behavior)
            questions_group = annotated_df.groupby('question')
            
            for question, question_df in questions_group:
                if len(question_df) == 0:
                    continue
                    
                try:
                    # Convert labels to binary
                    y_true = self._convert_to_binary(pd.Series(question_df['human_label']), "label")
                    y_pred = self._convert_to_binary(pd.Series(question_df['predicted_label']), "label")
                    
                    # Calculate metrics
                    metrics = self._calculate_metrics(y_true, y_pred)
                    
                    # Store results with question key
                    question_key = str(question).replace('?', '').replace(' ', '_').replace('/', '_')
                    results[question_key] = {
                        **metrics,
                        'n_samples': len(question_df),
                        'n_annotated': len(question_df),
                        'file': str(qc_file),
                        'question': str(question),
                        'type': 'binary_classification'
                    }
                    
                except Exception as e:
                    if self.verbose:
                        logger.warning(f"Failed to evaluate question '{question}': {e}")
                    continue
            
            return results
            
        except Exception as e:
            if self.verbose:
                logger.error(f"Error evaluating questions QC: {e}")
            return {}
    
    def evaluate_all(self, qc_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate all available QC files.
        
        Args:
            qc_types: List of QC types to evaluate. If None, evaluates all available types.
                     Options: 'no-comparisons', 'findings', 'categories', 'questions', 'questions-consolidated'
        
        Returns:
            Dictionary containing evaluation results for all QC types
        """
        if qc_types is None:
            qc_types = ['no-comparisons', 'findings', 'categories', 'questions']
        
        results = {}
        
        # Evaluate no-comparisons QC
        if 'no-comparisons' in qc_types:
            no_comp_results = self.evaluate_no_comparisons_qc()
            if no_comp_results:
                results['no_comparisons'] = no_comp_results
        
        # Evaluate findings QC
        if 'findings' in qc_types:
            findings_results = self.evaluate_findings_qc()
            if findings_results:
                results['findings'] = findings_results
        
        # Evaluate categories QC
        if 'categories' in qc_types:
            category_results = self.evaluate_category_qc()
            if category_results:
                results['categories'] = category_results
        
        # Evaluate questions QC (separate files)
        if 'questions' in qc_types:
            questions_results = self.evaluate_questions_qc()
            if questions_results:
                results['questions'] = questions_results
        
        return results
    
    def find_imperfect_tasks(self, results: Dict[str, Any], accuracy_threshold: float = 1.0) -> List[Dict[str, Any]]:
        """
        Find tasks with accuracy below the specified threshold.
        
        Args:
            results: Evaluation results from evaluate_all()
            accuracy_threshold: Accuracy threshold for identifying imperfect tasks (default: 1.0 for 100%)
            
        Returns:
            List of dictionaries containing information about imperfect tasks
        """
        imperfect_tasks = []
        
        # Helper function to add imperfect task
        def add_imperfect_task(qc_type: str, task_name: str, task_data: Dict[str, Any], question_text: str = None):
            accuracy = task_data.get('accuracy', 0)
            if accuracy < accuracy_threshold:
                task_info = {
                    'qc_type': qc_type,
                    'task_name': task_name,
                    'accuracy': accuracy,
                    'accuracy_percent': round(accuracy * 100, 1),
                    'n_samples': task_data.get('n_samples', 0),
                    'n_annotated': task_data.get('n_annotated', 0),
                    'evaluation_type': task_data.get('type', 'unknown')
                }
                
                if question_text:
                    task_info['question_text'] = question_text
                
                # Add classification metrics if available
                for metric in ['precision', 'recall', 'f1_score', 'specificity', 'npv']:
                    if metric in task_data:
                        task_info[metric] = task_data[metric]
                
                imperfect_tasks.append(task_info)
        
        # Check each QC type
        for qc_type, qc_results in results.items():
            if not qc_results:
                continue
                
            if qc_type == 'questions':
                # Handle questions results specially (contains both consolidated and individual)
                for task_name, task_data in qc_results.items():
                    if task_name == '_CONSOLIDATED':
                        # Handle consolidated questions result
                        add_imperfect_task('questions_consolidated', 'All Questions Combined', task_data)
                    else:
                        # Handle individual question results
                        question_text = task_data.get('question', task_name)
                        add_imperfect_task('questions_individual', task_name, task_data, question_text)
                        
            elif qc_type == 'categories':
                # Handle category results
                for category_name, category_data in qc_results.items():
                    add_imperfect_task('categories', category_name, category_data)
                    
            elif qc_type in ['no_comparisons', 'findings']:
                # Handle simple results
                add_imperfect_task(qc_type, qc_type, qc_results)
        
        return imperfect_tasks
    
    def print_imperfect_tasks_summary(self, imperfect_tasks: List[Dict[str, Any]]) -> None:
        """
        Print a summary of imperfect tasks in a user-friendly format.
        
        Args:
            imperfect_tasks: List of imperfect tasks from find_imperfect_tasks()
        """
        if not imperfect_tasks:
            print("\n🎉 ALL TASKS ACHIEVED 100% ACCURACY!")
            print("No tasks need attention.")
            return
        
        print(f"\n{'='*60}")
        print(f"SUMMARY: TASKS WITH ACCURACY < 100%")
        print(f"{'='*60}")
        print(f"Found {len(imperfect_tasks)} task(s) with accuracy < 100%:")
        
        # Group tasks by type
        task_groups = {}
        for task in imperfect_tasks:
            qc_type = task['qc_type']
            if qc_type not in task_groups:
                task_groups[qc_type] = []
            task_groups[qc_type].append(task)
        
        # Display each group
        for qc_type, tasks in task_groups.items():
            if qc_type == 'questions_consolidated':
                print(f"\nQUESTIONS QC (CONSOLIDATED):")
                print("-" * 40)
                for task in tasks:
                    print(f"  • Overall Questions Performance: {task['accuracy_percent']}% accuracy "
                          f"({task['n_annotated']}/{task['n_samples']} annotated)")
                    if 'precision' in task:
                        print(f"    Precision: {task['precision']:.3f}, Recall: {task['recall']:.3f}, "
                              f"F1: {task['f1_score']:.3f}")
                
            elif qc_type == 'questions_individual':
                print(f"\nQUESTIONS QC (INDIVIDUAL):")
                print("-" * 40)
                for task in tasks:
                    question_text = task.get('question_text', task['task_name'].replace('_', ' '))
                    print(f"  • {question_text}: {task['accuracy_percent']}% accuracy "
                          f"({task['n_annotated']}/{task['n_samples']} annotated)")
                    if 'precision' in task:
                        print(f"    Precision: {task['precision']:.3f}, Recall: {task['recall']:.3f}, "
                              f"F1: {task['f1_score']:.3f}")
                
            elif qc_type == 'categories':
                print(f"\nCATEGORIES TASKS:")
                print("-" * 30)
                for task in tasks:
                    print(f"  • {task['task_name']}: {task['accuracy_percent']}% accuracy "
                          f"({task['n_annotated']}/{task['n_samples']} annotated)")
                    
            elif qc_type in ['no_comparisons', 'findings']:
                print(f"\n{qc_type.upper().replace('_', '-')} TASKS:")
                print("-" * 30)
                for task in tasks:
                    print(f"  • {task['task_name']}: {task['accuracy_percent']}% accuracy "
                          f"({task['n_annotated']}/{task['n_samples']} annotated)")
        
        print(f"\n💡 Consider reviewing these tasks for potential improvements.")
        print(f"   Use --imperfect-csv to export detailed information to CSV.")
    
    def export_imperfect_tasks_csv(self, imperfect_tasks: List[Dict[str, Any]], csv_path: str) -> None:
        """
        Export imperfect tasks to CSV file.
        
        Args:
            imperfect_tasks: List of imperfect tasks from find_imperfect_tasks()
            csv_path: Path to save CSV file
        """
        if not imperfect_tasks:
            print(f"No imperfect tasks to export.")
            return
        
        # Prepare rows for CSV
        csv_rows = []
        for task in imperfect_tasks:
            row = {
                'qc_type': task['qc_type'],
                'task_name': task['task_name'],
                'accuracy': task['accuracy'],
                'accuracy_percent': task['accuracy_percent'],
                'n_samples': task['n_samples'],
                'n_annotated': task['n_annotated'],
                'evaluation_type': task['evaluation_type']
            }
            
            # Add question text for question-based tasks
            if task['qc_type'] in ['questions_individual', 'questions_consolidated']:
                if task['qc_type'] == 'questions_consolidated':
                    row['question_text'] = 'ALL_QUESTIONS_COMBINED'
                else:
                    row['question_text'] = task.get('question_text', '')
            else:
                row['question_text'] = ''
            
            # Add classification metrics if available
            for metric in ['precision', 'recall', 'f1_score', 'specificity', 'npv']:
                row[metric] = task.get(metric, '')
            
            csv_rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_rows)
        
        # Order columns logically
        column_order = [
            'qc_type', 'task_name', 'accuracy', 'accuracy_percent', 
            'n_samples', 'n_annotated', 'evaluation_type', 'question_text',
            'precision', 'recall', 'f1_score', 'specificity', 'npv'
        ]
        df = df[column_order]
        
        # Sort by accuracy (worst first)
        df = df.sort_values('accuracy')
        
        df.to_csv(csv_path, index=False)
        
        print(f"Exported {len(imperfect_tasks)} imperfect tasks to {csv_path}")
        
        # Print summary by task type
        qc_type_counts = df['qc_type'].value_counts()
        if len(qc_type_counts) > 1:
            print("Task breakdown:")
            for qc_type, count in qc_type_counts.items():
                type_name = qc_type.replace('_', ' ').title()
                print(f"  {type_name}: {count} tasks")

def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate annotated QC files and compute performance metrics.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument('--qc-dir', type=str, required=True,
                      help='Directory containing QC files with human annotations')
    
    # QC type selection
    parser.add_argument('--qc-types', nargs='+', 
                      choices=['no-comparisons', 'findings', 'categories', 'questions', 'all'],
                      default=['all'],
                      help='Types of QC files to evaluate (default: all)')
    
    # Output options
    parser.add_argument('--output-file', type=str,
                      help='Path to save evaluation results JSON file')
    
    parser.add_argument('--imperfect-csv', type=str,
                      help='Path to save CSV file listing tasks with accuracy < 100%%')
    
    parser.add_argument('--accuracy-threshold', type=float, default=1.0,
                      help='Accuracy threshold for identifying imperfect tasks (default: 1.0 for 100%%)')
    
    # Verbose output
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Enable verbose output')
    
    return parser.parse_args()

def format_metrics(metrics: Dict[str, Any], indent: str = "  ") -> str:
    """Format metrics for display."""
    if "error" in metrics:
        return f"{indent}ERROR: {metrics['error']}"
    
    lines = []
    
    if metrics.get("type") in ["binary_classification", "binary_classification_consolidated"]:
        lines.append(f"{indent}Samples: {metrics['n_samples']} (annotated: {metrics['n_annotated']})")
        lines.append(f"{indent}Accuracy: {metrics['accuracy']:.3f}")
        lines.append(f"{indent}Precision (PPV): {metrics['precision']:.3f}")
        lines.append(f"{indent}Recall (Sensitivity): {metrics['recall']:.3f}")
        lines.append(f"{indent}Specificity: {metrics['specificity']:.3f}")
        lines.append(f"{indent}NPV: {metrics['npv']:.3f}")
        lines.append(f"{indent}F1 Score: {metrics['f1_score']:.3f}")
        
        # Add unique questions info for consolidated
        if metrics.get("type") == "binary_classification_consolidated":
            lines.append(f"{indent}Unique Questions: {metrics.get('n_unique_questions', 'N/A')}")
        
        cm = metrics['confusion_matrix']
        lines.append(f"{indent}Confusion Matrix:")
        lines.append(f"{indent}  TP: {cm['true_positive']}, FP: {cm['false_positive']}")
        lines.append(f"{indent}  FN: {cm['false_negative']}, TN: {cm['true_negative']}")
        
    elif metrics.get("type") == "correctness_evaluation":
        lines.append(f"{indent}Samples: {metrics['n_samples']} (annotated: {metrics['n_annotated']})")
        lines.append(f"{indent}Accuracy: {metrics['accuracy']:.3f}")
    
    return "\n".join(lines)

def main():
    args = parse_args()
    
    print("QC Evaluation Tool")
    print(f"QC Directory: {args.qc_dir}")
    print("="*60)
    
    try:
        evaluator = QCEvaluator(args.qc_dir, verbose=args.verbose)
        
        # Determine which QC types to evaluate
        if args.qc_types == ['all']:
            qc_types = ['no-comparisons', 'findings', 'categories', 'questions']
        else:
            qc_types = args.qc_types
        
        evaluation_results = evaluator.evaluate_all(qc_types)
        
        # Print results to console
        print(f"\nQC Evaluation Tool")
        print(f"QC Directory: {args.qc_dir}")
        print("=" * 60)
        
        if "no_comparisons" in evaluation_results:
            print(f"\nNO COMPARISONS QC EVALUATION")
            print("-" * 50)
            result = evaluation_results["no_comparisons"]
            print(format_metrics(result))
        
        if "findings" in evaluation_results:
            print(f"\nFINDINGS QC EVALUATION")
            print("-" * 50)
            result = evaluation_results["findings"]
            print(format_metrics(result))
        
        if "categories" in evaluation_results:
            print(f"\nCATEGORIES QC EVALUATION")
            print("-" * 50)
            for category, result in evaluation_results["categories"].items():
                print(f"\n{category}:")
                print(format_metrics(result))
        
        if "questions" in evaluation_results:
            print(f"\nQUESTIONS QC EVALUATION")
            print("-" * 50)
            
            questions_results = evaluation_results["questions"]
            
            # First, show consolidated results if available
            if "_CONSOLIDATED" in questions_results:
                consolidated = questions_results["_CONSOLIDATED"]
                print(f"\nCONSOLIDATED (All Questions Combined):")
                print(f"  {consolidated.get('description', 'Combined evaluation across all questions')}")
                print(format_metrics(consolidated))
            
            # Then show individual question results
            individual_questions = {k: v for k, v in questions_results.items() if k != "_CONSOLIDATED"}
            if individual_questions:
                if "_CONSOLIDATED" in questions_results:
                    print(f"\nINDIVIDUAL QUESTIONS:")
                    print("-" * 30)
                
                for question_key, result in individual_questions.items():
                    question_text = result.get('question', question_key.replace('_', ' '))
                    print(f"\n{question_text}:")
                    print(format_metrics(result))
        
        # Find and display imperfect tasks
        imperfect_tasks = evaluator.find_imperfect_tasks(evaluation_results, args.accuracy_threshold)
        evaluator.print_imperfect_tasks_summary(imperfect_tasks)
        
        # Export imperfect tasks to CSV if requested
        if args.imperfect_csv:
            evaluator.export_imperfect_tasks_csv(imperfect_tasks, args.imperfect_csv)
        
        # Save results if requested
        if args.output_file:
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(evaluation_results, f, indent=2, default=str)
            
            print(f"\nResults saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 