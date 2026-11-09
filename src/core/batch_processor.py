"""Improved batch processing with better error handling and validation."""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

from .exceptions import BatchProcessingError

# The request-body fields that control sampling, in the order a reader expects.
# `model` and `structured_output` live under the same config key and are not
# sampling, which is why this is a whitelist rather than a copy of the block.
SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "max_tokens",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
)
from .logging_config import redacted
from .validators import ResultValidator

# The free-text stages return prose, and this keeps markdown out of it.
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Please output in plain text without any formatting."
)

# The questions stage is decoded against a JSON schema, so the format needs no
# asking for. What the model is doing needs saying, because the per-question
# prompt opens on a report and never names the task.
QUESTION_SYSTEM_PROMPT = (
    "You are a radiologist reading a radiology report and answering one "
    "question about one imaging finding. When attributing your responses, you "
    "must quote the report rather than paraphrase it."
)



class BatchProcessor:
    """Handles batch processing with improved error handling and validation."""
    
    def __init__(self, client, config: Dict, logger, validator: ResultValidator):
        self.client = client
        self.config = config
        self.logger = logger
        self.validator = validator
        self.max_retries = config.get('processing', {}).get('max_retries', 2)
        self.retry_delay = config.get('processing', {}).get('retry_delay', 5)
        # Extra request-body fields, set per stage. The questions stage puts a
        # `response_format` here so the server constrains decoding to the answer
        # schema; every other stage leaves it empty and asks for free text.
        self.body_extras = {}
        # The system message, set per stage for the same reason. The default
        # tells a free-text stage to leave markdown out. It read "Please output
        # in plain text without any formatting" for the questions stage too,
        # where the grammar requires JSON -- an instruction the model could
        # only obey by disobeying the format it was decoded into.
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
    
    def process_batch(self, requests: List[Tuple[str, str]], stage_name: str) -> Dict[str, str]:
        """Process a batch of requests with improved error handling and validation."""
        if not requests:
            return {}
        
        self.logger.info(f"Processing batch of {len(requests)} requests for {stage_name}")
        
        for attempt in range(self.max_retries + 1):
            try:
                # Submit batch and wait for completion
                batch_response = self._submit_and_wait_for_batch(requests, attempt)
                
                # Extract and validate results
                results = self._extract_batch_results(batch_response, requests, stage_name)
                
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
    
    def _sampling(self) -> Dict[str, Any]:
        """Return the sampling fields the config sets, and nothing else.

        Driven by the config rather than named here, so changing how the model
        samples is a config edit. The whitelist is what keeps `name` and
        `structured_output` out of the request body, and an absent key is not
        sent at all, so the server's own default applies.

        `top_k` and `min_p` are sglang extensions rather than OpenAI fields.
        sglang 0.4.7 declares both on `ChatCompletionRequest` (`top_k: int =
        -1`, `min_p: float = 0.0`), so they are read rather than refused.
        """
        model = self.config.get('model', {})
        return {field: model[field] for field in SAMPLING_FIELDS if field in model}

    def _thinking(self) -> Dict[str, Any]:
        """Return the chat-template switch that turns Qwen3's thinking on or off.

        sglang reads `enable_thinking` out of `chat_template_kwargs`; there is
        no top-level field for it. Absent from the config, nothing is sent and
        the template's own default applies.

        **This has to agree with how the server was launched.** sglang decides
        at launch whether to wrap the grammar backend:

            if server_args.reasoning_parser and hasattr(tokenizer, "think_end_id"):
                grammar_backend = ReasonerGrammarBackend(grammar_backend, ...)

        and `ReasonerGrammarObject` starts with `is_in_reasoning = True`, so it
        masks nothing until a `</think>` token arrives. With thinking off the
        model never emits one, so a server started with `--reasoning-parser`
        would never enforce the schema -- the failure reported in sgl-project
        #6675 and #9282. Thinking off therefore means launching without
        `--reasoning-parser`, which leaves the plain grammar backend applying
        from the first token.
        """
        model = self.config.get('model', {})
        if 'enable_thinking' not in model:
            return {}
        return {"chat_template_kwargs": {"enable_thinking": bool(model['enable_thinking'])}}

    def _submit_and_wait_for_batch(self, requests: List[Tuple[str, str]], attempt: int) -> Any:
        """Submit batch and wait for completion."""
        # Create batch requests
        # A stage that asks for an object constrains decoding to its schema, so
        # the server guarantees valid JSON instead of the prompt requesting it.
        batch_requests = [
            {
                "custom_id": request_id,
                "method": "POST", 
                "url": "/chat/completions",
                "body": {
                    "model": self.config['model']['name'],
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    **self._sampling(),
                    **self._thinking(),
                    **self.body_extras,
                }
            }
            for request_id, prompt in requests
        ]
        
        # Create and submit batch file
        temp_file = None
        try:
            temp_file = self._create_batch_file(batch_requests, attempt)
            file_response = self._upload_batch_file(temp_file)
            batch_response = self._create_batch_request(file_response.id)
            
            # Wait for completion with progress tracking
            return self._wait_for_batch_completion(batch_response)
            
        finally:
            # Clean up temporary file
            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception as e:
                    self.logger.warning(f"Could not delete temp file {temp_file}: {e}")
    
    def _create_batch_file(self, batch_requests: List[Dict], attempt: int) -> Path:
        """Create temporary batch file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_file = Path(f"temp_batch_{timestamp}_attempt_{attempt}.jsonl")
        
        try:
            with open(temp_file, "w") as f:
                for req in batch_requests:
                    f.write(json.dumps(req) + "\n")
            return temp_file
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise BatchProcessingError(f"Failed to create batch file: {str(e)}")
    
    def _upload_batch_file(self, temp_file: Path) -> Any:
        """Upload batch file to API."""
        try:
            with open(temp_file, "rb") as f:
                file_response = self.client.files.create(file=f, purpose="batch")
            return file_response
        except Exception as e:
            raise BatchProcessingError(f"Failed to upload batch file: {str(e)}")
    
    def _create_batch_request(self, file_id: str) -> Any:
        """Create batch processing request."""
        try:
            batch_response = self.client.batches.create(
                input_file_id=file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            return batch_response
        except Exception as e:
            raise BatchProcessingError(f"Failed to create batch request: {str(e)}")
    
    def _wait_for_batch_completion(self, batch_response: Any) -> Any:
        """Wait for batch completion with progress tracking."""
        with tqdm(desc=f"Waiting for batch {batch_response.id[:8]}...", 
                 unit="check", 
                 bar_format='{desc}: {elapsed}') as pbar:
            
            while batch_response.status not in ["completed", "failed", "cancelled"]:
                time.sleep(3)
                try:
                    batch_response = self.client.batches.retrieve(batch_response.id)
                    pbar.update(1)
                    pbar.set_description(f"Batch {batch_response.id[:8]} - {batch_response.status}")
                except Exception as e:
                    raise BatchProcessingError(f"Failed to check batch status: {str(e)}")
        
        if batch_response.status != "completed":
            raise BatchProcessingError(f"Batch failed with status: {batch_response.status}")
        
        return batch_response
    
    def _parse_result_lines(self, result_content: str) -> Tuple[Dict[str, str], Dict[str, int]]:
        """Read a batch output file into answers, and count what went wrong.

        Separated from the download so it can be exercised without a server.
        This is the half most likely to be wrong, because the shape of a
        response depends on the sglang version and on which of the grammar
        backend and the reasoning parser got to the tokens first.

        Args:
            result_content: The batch output file, one JSON object per line.

        Returns:
            The answers keyed by `custom_id`, and the counts of empty answers
            and of answers readable only in `reasoning_content`.
        """
        lines = result_content.split("\n")
        results = {}
        # Two counts worth seeing per batch rather than at the end of a
        # four-million-request run: how many answers arrived empty, and how
        # many were readable only in `reasoning_content`.
        empty = 0
        reasoning_only = 0
        for line_num, line in enumerate(lines):
            if line.strip():
                try:
                    result = json.loads(line)
                    custom_id = result['custom_id']
                    
                    # Try parsing the sglang response format
                    content = ""
                    parsing_success = False
                    
                    # Log full structure for first result to help with debugging
                    if line_num == 0:
                        self.logger.debug(f"First result structure: {json.dumps(result, indent=2)}")
                    
                    # sglang format: choices is an object, not an array
                    if 'response' in result and 'body' in result['response']:
                        try:
                            choices = result['response']['body']['choices']
                            # Handle both object and array formats
                            message = (
                                choices['message']
                                if isinstance(choices, dict)
                                else choices[0]['message']
                            )
                            content = message.get('content') or ''
                            # A reasoning parser and a grammar together put
                            # the constrained output in `reasoning_content`
                            # and leave `content` empty, because sglang
                            # defers the grammar mask until the end-of-think
                            # token. Reading `content` alone then returns ""
                            # for every answer in the run. See sgl-project
                            # issues #6675 and #9282 and ai-dynamo #9006.
                            if not content.strip():
                                content = message.get('reasoning_content') or ''
                                if content.strip():
                                    reasoning_only += 1
                            parsing_success = True
                            if line_num == 0:
                                self.logger.debug("Successfully parsed sglang response format")
                        except (KeyError, IndexError, TypeError) as e:
                            self.logger.debug(f"Failed to parse expected format: {e}")
                    
                    # Handle error responses
                    if not parsing_success and 'error' in result:
                        self.logger.warning(f"Error in batch response for {redacted(custom_id)}: {result['error']}")
                        content = ""
                        parsing_success = True
                    
                    if not parsing_success:
                        self.logger.warning(f"Could not parse response for {redacted(custom_id)}, available keys: {list(result.keys())}")
                        content = ""
                    
                    # Clean up content (remove thinking tags if present)
                    if content and "</think>" in content:
                        content = content.split("</think>")[-1]
                    
                    content = content.strip() if content else ""
                    if not content:
                        empty += 1
                    results[custom_id] = content
                    
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Could not parse result line {line_num}: {line[:100]}... Error: {e}")
                    continue
                except Exception as e:
                    self.logger.warning(f"Unexpected error parsing line {line_num}: {e}")
                    continue
        
        if reasoning_only:
            self.logger.warning(
                f"{reasoning_only} of {len(results)} answers were empty in `content` "
                f"and read from `reasoning_content` instead. The reasoning parser and "
                f"the response schema are both on, and sglang defers the grammar mask "
                f"until the end-of-think token. Set model.structured_output to false "
                f"to drop the schema, or launch the server without --reasoning-parser."
            )
        if empty:
            self.logger.warning(
                f"{empty} of {len(results)} answers in this batch came back empty. "
                f"Stop the run and check one response before spending more GPU time."
            )

        return results, {"empty": empty, "reasoning_only": reasoning_only}

    def _extract_batch_results(self, batch_response: Any, requests: List[Tuple[str, str]], stage_name: str) -> Dict[str, str]:
        """Extract results from completed batch."""
        try:
            # Download results
            result_content = self.client.files.content(batch_response.output_file_id).read().decode("utf-8")
            
            # Log the first few lines for debugging
            lines = result_content.split("\n")
            self.logger.debug(f"Batch response has {len([l for l in lines if l.strip()])} lines")
            if lines and lines[0].strip():
                self.logger.debug(f"First response line: {lines[0][:200]}...")

            results, counts = self._parse_result_lines(result_content)

            if counts["reasoning_only"]:
                self.logger.warning(
                    f"{counts['reasoning_only']} of {len(results)} answers were empty in "
                    f"`content` and read from `reasoning_content` instead. The reasoning "
                    f"parser and the response schema are both on, and sglang defers the "
                    f"grammar mask until the end-of-think token. Set model.structured_output "
                    f"to false to drop the schema, or launch the server without "
                    f"--reasoning-parser."
                )
            if counts["empty"]:
                self.logger.warning(
                    f"{counts['empty']} of {len(results)} answers in this batch came back "
                    f"empty. Stop the run and check one response before spending more GPU time."
                )

            # Clean up API files
            try:
                self.client.files.delete(batch_response.output_file_id)
                self.client.files.delete(batch_response.input_file_id)
            except Exception as e:
                self.logger.warning(f"Could not delete API files: {e}")
            
            # Check if we got results for all requests
            request_ids = {req_id for req_id, _ in requests}
            missing_ids = request_ids - set(results.keys())
            
            if missing_ids:
                self.logger.warning(f"Missing results for {len(missing_ids)} requests: {list(missing_ids)[:5]}...")
                # Add empty results for missing IDs to maintain consistency
                for missing_id in missing_ids:
                    results[missing_id] = ""
            
            # Log summary with detailed breakdown
            non_empty_results = sum(1 for v in results.values() if v.strip())
            empty_results = len(results) - non_empty_results
            self.logger.info(f"Extracted {len(results)} results: {non_empty_results} non-empty, {empty_results} empty")
            
            # Log sample results for debugging - include some empty ones
            sample_results = list(results.items())[:5]
            empty_samples = [(k, v) for k, v in results.items() if not v.strip()][:3]
            
            self.logger.info(f"Sample non-empty results:")
            for rid, content in sample_results:
                if content.strip():
                    self.logger.info(
                        f"  {redacted(rid)}: "
                        f"'{content[:150]}{'...' if len(content) > 150 else ''}'"
                    )
            
            if empty_samples:
                self.logger.warning(f"Sample empty results:")
                for rid, content in empty_samples:
                    self.logger.warning(f"  {redacted(rid)}: EMPTY ('{content}')")
            
            # Log response patterns to identify issues
            response_patterns = {}
            for content in results.values():
                content_lower = content.strip().lower()
                if not content_lower:
                    key = "EMPTY"
                elif "no relevant findings" in content_lower:
                    key = "no_relevant_findings"
                elif "no findings" in content_lower:
                    key = "no_findings"  
                elif "normal" in content_lower:
                    key = "normal"
                elif len(content_lower) < 10:
                    key = "very_short"
                else:
                    key = "substantial"
                
                response_patterns[key] = response_patterns.get(key, 0) + 1
            
            self.logger.info(f"Response patterns: {response_patterns}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to extract batch results: {str(e)}")
            # Include more details in the error for debugging
            self.logger.error(f"Batch response status: {getattr(batch_response, 'status', 'unknown')}")
            self.logger.error(f"Batch response ID: {getattr(batch_response, 'id', 'unknown')}")
            raise BatchProcessingError(f"Failed to extract batch results: {str(e)}")
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """Get statistics about batch processing."""
        return {
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
        } 