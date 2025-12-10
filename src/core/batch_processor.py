"""Improved batch processing with concurrent requests for high-throughput processing."""

import json
import time
import random
from typing import Dict, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .exceptions import BatchProcessingError
from .validators import ResultValidator


class BatchProcessor:
    """Handles batch processing with improved error handling and validation."""
    
    def __init__(self, client, config: Dict, logger, validator: ResultValidator):
        self.client = client
        self.config = config
        self.logger = logger
        self.validator = validator
        self.max_retries = config.get('processing', {}).get('max_retries', 2)
        self.retry_delay = config.get('processing', {}).get('retry_delay', 5)
        # Number of concurrent requests (default: 512 for high-throughput)
        self.request_concurrency = config.get('processing', {}).get('request-concurrency', 512)
    
    def process_batch(self, requests: List[Tuple[str, str]], stage_name: str) -> Dict[str, str]:
        """Process a batch of requests using concurrent requests for high throughput."""
        if not requests:
            return {}
        
        self.logger.info(f"Processing batch of {len(requests)} requests for {stage_name} using {self.request_concurrency} concurrent workers")
        
        for attempt in range(self.max_retries + 1):
            try:
                # Use concurrent requests instead of batch API
                results = self._run_requests_concurrently(requests)
                
                # Validate results
                validated_results = self.validator.validate_batch_results(results, requests, stage_name)
                
                self.logger.info(f"Successfully processed {len(validated_results)}/{len(requests)} requests")
                return validated_results
                
            except BatchProcessingError as e:
                self.logger.warning(f"Batch processing attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries:
                    self.logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    continue
                else:
                    self.logger.error(f"All {self.max_retries + 1} attempts failed for batch")
                    raise
            except Exception as e:
                self.logger.error(f"Unexpected error in batch processing: {str(e)}")
                raise BatchProcessingError(f"Unexpected batch processing error: {str(e)}")
    
    def _run_requests_concurrently(self, requests: List[Tuple[str, str]]) -> Dict[str, str]:
        """Run requests concurrently using ThreadPoolExecutor."""
        results: Dict[str, str] = {}
        errors = 0
        
        def _request_with_retries(custom_id: str, prompt: str) -> Tuple[str, str]:
            """Make a single request with retries."""
            for i in range(self.max_retries + 1):
                try:
                    return custom_id, self._request_once(prompt)
                except Exception as e:
                    if i < self.max_retries:
                        # Exponential backoff with jitter
                        backoff = self.retry_delay * (2 ** i) + random.uniform(0, 0.25)
                        self.logger.debug(f"Request {custom_id} failed (attempt {i+1}): {e}; retrying in {backoff:.2f}s")
                        time.sleep(backoff)
                    else:
                        self.logger.warning(f"Request {custom_id} failed after {self.max_retries + 1} attempts: {e}")
                        return custom_id, ""
        
        with ThreadPoolExecutor(max_workers=self.request_concurrency) as pool:
            futures = [pool.submit(_request_with_retries, rid, prompt) for rid, prompt in requests]
            
            with tqdm(total=len(requests), desc="Processing requests", unit="req") as pbar:
                for fut in as_completed(futures):
                    rid, content = fut.result()
                    # Clean up content (remove thinking tags if present)
                    if content and "</think>" in content:
                        content = content.split("</think>")[-1]
                    results[rid] = content.strip() if content else ""
                    if not results[rid]:
                        errors += 1
                    pbar.update(1)
        
        self.logger.info(f"Concurrent requests completed: {len(results)} total, {errors} empty/failed")
        
        # Log summary statistics
        non_empty_results = sum(1 for v in results.values() if v.strip())
        empty_results = len(results) - non_empty_results
        self.logger.info(f"Results: {non_empty_results} non-empty, {empty_results} empty")
        
        return results
    
    def _request_once(self, prompt: str) -> str:
        """Make a single chat completion request."""
        try:
            resp = self.client.chat.completions.create(
                model=self.config['model']['name'],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Please output in plain text without any formatting."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config['model']['temperature'],
                top_p=self.config['model']['top_p'],
                max_tokens=self.config['model']['max_tokens'],
            )
            
            # Handle response (compatible with different formats)
            try:
                choices = getattr(resp, "choices", None)
                if isinstance(choices, list) and choices:
                    return choices[0].message.content or ""
                if isinstance(choices, dict):
                    return choices.get("message", {}).get("content", "") or ""
            except Exception:
                pass
            
            # If response is dict format
            if isinstance(resp, dict):
                ch = resp.get("choices")
                if isinstance(ch, list) and ch:
                    return ch[0].get("message", {}).get("content", "") or ""
                if isinstance(ch, dict):
                    return ch.get("message", {}).get("content", "") or ""
            
            return ""
            
        except Exception as e:
            self.logger.debug(f"Request failed: {e}")
            raise
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """Get statistics about batch processing."""
        return {
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "request_concurrency": self.request_concurrency,
        } 