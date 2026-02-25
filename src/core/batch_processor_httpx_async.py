"""High-performance async batch processing using aiohttp for maximum throughput.

This implementation uses:
- Native asyncio for true concurrent I/O (no thread overhead)
- aiohttp for fast async HTTP (faster than httpx for high concurrency)
- Semaphore-based concurrency limiting
- uvloop for faster event loop (when available on Linux)
- Connection pooling with keep-alive
- Optional grouped requests mode: batches multiple prompts per HTTP request
"""

import asyncio
import hashlib
import json
import logging
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import aiohttp
import uvloop
from tqdm.asyncio import tqdm as async_tqdm
from transformers import AutoTokenizer

from .exceptions import BatchProcessingError
from .validators import ResultValidator

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
EBNF_GRAMMAR = """
root ::= yesno
yesno ::= "Yes" | "No"
"""


# Set to True to force debug mode regardless of --study-ids
DEBUG_MODE_FORCE = False


@dataclass
class UsageStats:
    """Tracks usage statistics across the lifetime of a BatchProcessor instance.

    Collects min/max/mean/median/total for token usage fields and HTTP response metrics.
    Thread-safe for use with async code.
    """

    # Token usage tracking - store all values for computing statistics
    prompt_tokens: List[int] = field(default_factory=list)
    completion_tokens: List[int] = field(default_factory=list)
    total_tokens: List[int] = field(default_factory=list)
    reasoning_tokens: List[int] = field(default_factory=list)

    # HTTP metrics
    total_http_requests: int = 0
    http_status_codes: Dict[int, int] = field(default_factory=lambda: defaultdict(int))

    # Error tracking
    total_errors: int = 0

    def record_response(self, response_data: Dict[str, Any], http_status: int) -> None:
        """Record stats from a single API response.

        Args:
            response_data: The JSON response from the API
            http_status: The HTTP status code
        """
        self.total_http_requests += 1
        self.http_status_codes[http_status] += 1

        # Extract usage stats if present
        usage = response_data.get("usage", {})
        if not usage:
            return

        # Safely extract each field (don't crash if missing)
        if "prompt_tokens" in usage:
            self.prompt_tokens.append(usage["prompt_tokens"])
        if "completion_tokens" in usage:
            self.completion_tokens.append(usage["completion_tokens"])
        if "total_tokens" in usage:
            self.total_tokens.append(usage["total_tokens"])
        # reasoning_tokens may be nested under completion_tokens_details
        reasoning = usage.get("reasoning_tokens")
        if reasoning is None:
            reasoning = usage.get("completion_tokens_details", {}).get("reasoning_tokens")
        if reasoning is not None:
            self.reasoning_tokens.append(reasoning)

    def record_error(self, http_status: Optional[int] = None) -> None:
        """Record an error (failed request)."""
        self.total_http_requests += 1
        self.total_errors += 1
        if http_status is not None:
            self.http_status_codes[http_status] += 1

    @staticmethod
    def _compute_stats(values: List[int]) -> Dict[str, Any]:
        """Compute min/max/mean/median/total for a list of values."""
        if not values:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "total": 0,
            }
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": round(statistics.mean(values), 2),
            "median": round(statistics.median(values), 2),
            "total": sum(values),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all collected statistics."""
        return {
            "token_usage": {
                "prompt_tokens": self._compute_stats(self.prompt_tokens),
                "completion_tokens": self._compute_stats(self.completion_tokens),
                "total_tokens": self._compute_stats(self.total_tokens),
                "reasoning_tokens": self._compute_stats(self.reasoning_tokens),
            },
            "http_metrics": {
                "total_requests": self.total_http_requests,
                "total_errors": self.total_errors,
                "status_codes": dict(self.http_status_codes),
            },
        }

    def format_summary(self) -> str:
        """Format the summary as a human-readable string."""
        summary = self.get_summary()
        lines = [
            "",
            "=" * 70,
            "BATCH PROCESSOR LIFETIME STATISTICS",
            "=" * 70,
            "",
            "HTTP Metrics:",
            f"  Total HTTP Requests: {summary['http_metrics']['total_requests']}",
            f"  Total Errors: {summary['http_metrics']['total_errors']}",
            "  Status Codes:",
        ]

        for status, count in sorted(summary["http_metrics"]["status_codes"].items()):
            lines.append(f"    HTTP {status}: {count}")

        lines.append("")
        lines.append("Token Usage Statistics:")

        for field_name, stats in summary["token_usage"].items():
            if stats["count"] > 0:
                lines.append(f"  {field_name}:")
                lines.append(f"    count: {stats['count']}")
                lines.append(f"    min: {stats['min']}")
                lines.append(f"    max: {stats['max']}")
                lines.append(f"    mean: {stats['mean']}")
                lines.append(f"    median: {stats['median']}")
                lines.append(f"    total: {stats['total']}")
            else:
                lines.append(f"  {field_name}: (no data)")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)


class BatchProcessorHttpxAsync:
    """High-performance async batch processor using aiohttp.

    This processor can handle 10,000+ concurrent requests efficiently by using
    native async I/O instead of thread pools. It's designed for maximum throughput
    when sending many requests to an LLM server.

    Supports two modes:
    - Individual requests (grouped_requests=False): One HTTP request per prompt
    - Grouped requests (grouped_requests=True): Multiple prompts per HTTP request,
      grouped by report_id (cache_salt) for better cache locality
    """

    def __init__(
        self,
        client,
        config: Dict,
        logger,
        validator: ResultValidator,
        send_tokenized_inputs: bool = False,
    ):
        # Keep `client` param for compatibility with Engine, but we use aiohttp directly.
        # Debug mode: enabled if DEBUG_MODE_FORCE=True or --study-ids filter is specified
        self.debug_mode = DEBUG_MODE_FORCE or bool(config.get("study-ids-filter"))
        self.client = client
        self.config = config
        self.logger = logger
        self.validator = validator
        self.max_retries = config.get("processing", {}).get("max_retries", 2)
        self.retry_delay = config.get("processing", {}).get("retry_delay", 1.0)

        # Grouped requests mode: batch multiple prompts per HTTP request
        self.grouped_requests = config.get("processing", {}).get("grouped-requests", False)

        # Tokenized inputs mode: send input_ids instead of text prompts
        self.send_tokenized_inputs = send_tokenized_inputs
        self.send_tokenized_inputs = config.get("processing", {}).get("send-tokenized-inputs", False)
        # self.debug_mode = True

        # Load tokenizer from model name if not provided
        model_name = config["model"]["name"]
        self.logger.info(f"Loading tokenizer for model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.stop_strings = self.get_stop_strings(model_name, self.tokenizer)

        base_url = self.config["server"]["base_url"]
        if not base_url.startswith("http"):
            base_url = f"http://{base_url}"
        port = self.config["server"]["port"]

        # Different endpoints for different modes
        self._chat_completions_url = f"{base_url}:{port}/v1/chat/completions"
        self._completions_url = f"{base_url}:{port}/v1/completions"

        # * http / tcp settings
        self.request_concurrency = config.get("processing", {}).get("request-concurrency", 2048)
        self._timeout = config.get("processing", {}).get("timeout", 300.0)
        self.client_timeout_total = None
        self.client_timeout_connect = config.get("processing", {}).get("client-timeout-connect", 30.0)
        self.client_timeout_sock_connect = config.get("processing", {}).get("client-timeout-sock-connect", 10.0)
        self.client_timeout_sock_read = config.get("processing", {}).get("client-timeout-sock-read", 900.0)

        # TODO: Rename this, it's actually max concurrent requests **per-study_id**:
        self.max_in_flight_reports = config.get("processing", {}).get("max-in-flight-reports", 16)
        mode_str = "grouped" if self.grouped_requests else "individual"

        self.usage_stats = UsageStats()

        # Sample logging: log first N requests/responses per batch with logger.info
        self._sample_log_max = 3
        self._sample_log_count = 0
        self._sample_log_lock = None  # Created per batch in async context

        if self.debug_mode:
            self.logger.setLevel(logging.DEBUG)
            self.logger.info("BatchProcessor DEBUG MODE enabled (log level set to DEBUG)")
            self.logger.debug("BatchProcessor DEBUG MODE enabled (log level set to DEBUG)")
        else:
            self.logger.setLevel(logging.INFO)
            self.logger.info("BatchProcessor DEBUG MODE disabled (log level set to INFO)")
            self.logger.debug("BatchProcessor DEBUG MODE disabled (log level set to INFO)")
        self.logger.debug(f"max-completion-tokens: {self.config['model']['max-completion-tokens']}")
        self.logger.info(str(self.config))

        self.logger.info(
            f"BatchProcessorHttpxAsync initialized: mode={mode_str}, "
            f"concurrency={self.request_concurrency}, "
            f"grouped_requests={self.grouped_requests}, "
            f"send_tokenized_inputs={self.send_tokenized_inputs}, "
            f"request_concurrency={self.request_concurrency}, "
            f"batch_size={self.config['processing']['batch-size']}, "
            f"debug_mode={self.debug_mode}, "
            f"enable-thinking={self.config['model']['enable-thinking']}, "
        )

    def get_stop_strings(self, model_name: str, tokenizer: AutoTokenizer) -> List[str]:
        """
        Return string-based stop sequences that mark the end
        of an assistant turn, model-agnostically.
        """
        stop_strings: List[str] = []

        chat_template = getattr(tokenizer, "chat_template", None)
        if not chat_template:
            return stop_strings  # non-chat model

        # Canonical assistant-end markers used across templates
        CANDIDATES = [
            "<|im_end|>",  # Qwen / ChatML
            "</s>",  # LLaMA / Mistral
            "<|eot_id|>",  # HF chat templates
        ]

        for s in CANDIDATES:
            if s in chat_template:
                stop_strings.append(s)

        return stop_strings

    def get_sampling_params(self, endpoint: str) -> dict[str, Any]:
        """Get sampling parameters from the model configuration."""
        model_cfg = self.config.get("model", {})
        params = {
            "model": model_cfg["name"],
            "temperature": model_cfg.get("temperature", 1.0),
            "top_p": model_cfg.get("top_p", 1.0),
            "min_p": model_cfg.get("min_p", 0.0),
            "top_k": model_cfg.get("top_k", -1),
            "repetition_penalty": model_cfg.get("repetition_penalty", 1.0),
            "presence_penalty": model_cfg.get("presence_penalty", 0.0),
            "frequency_penalty": model_cfg.get("frequency_penalty", 0.0),
            # "stop": self.stop_strings,
            "seed": model_cfg.get("seed", 42),
        }
        if "/generate" in endpoint:
            # https://github.com/supa-thibaud/sglang-dry/blob/main/docs/en/sampling_params.md
            params["max_new_tokens"] = model_cfg.get("max-completion-tokens", 4096)
        elif "/v1/completions" in endpoint:
            # /v1/completions (Raw) uses max_tokens.
            params["max_tokens"] = model_cfg.get("max-completion-tokens", 4096)
        elif "/v1/chat/completions" in endpoint:
            # /v1/chat/completions uses max_completion_tokens.
            params["max_completion_tokens"] = model_cfg.get("max-completion-tokens", 4096)
        else:
            raise NotImplementedError(f"Unsupported endpoint: {endpoint}")
        return params

    def process_batch(
        self,
        requests: List[Tuple[str, str]],
        stage_name: str,
        flush_callback: Optional[callable] = None,
        flush_interval: Optional[int] = None,
    ) -> Dict[str, str]:
        """Process a batch of requests using async I/O for high throughput.

        This is a sync wrapper that runs the async processing internally.

        Args:
            requests: List of (request_id, prompt) tuples.
            stage_name: Name of the processing stage.
            flush_callback: Optional sync callable (Dict[str, str]) -> None that is called
                periodically to flush accumulated results to disk.
            flush_interval: Number of results to buffer before calling flush_callback.
                When None, no periodic flushing occurs.
        """
        if not requests:
            return {}

        # Reset sample logging counter for this batch
        self._sample_log_count = 0

        mode_str = "grouped" if self.grouped_requests else "individual"
        self.logger.info(
            f"Processing batch of {len(requests)} requests for {stage_name} "
            f"using async I/O ({mode_str} mode) with {self.request_concurrency} concurrent requests"
        )
        self.logger.info(f"flush_interval: {flush_interval}")
        self.logger.info(f"flush_callback: {flush_callback}")

        for attempt in range(self.max_retries + 1):
            try:
                self._sample_log_lock = None  # Will be created in async context
                # Run async processing
                if self.grouped_requests:
                    results = asyncio.run(
                        self._process_batch_grouped_async(
                            requests,
                            stage_name,
                            flush_callback=flush_callback,
                            flush_interval=flush_interval,
                        )
                    )
                else:
                    results = asyncio.run(
                        self._process_batch_async(
                            requests,
                            stage_name,
                            flush_callback=flush_callback,
                            flush_interval=flush_interval,
                        )
                    )

                # Validate results
                validated_results = self.validator.validate_batch_results(results, requests, stage_name)

                self.logger.info(f"Successfully processed {len(validated_results)}/{len(requests)} requests")
                return validated_results

            except BatchProcessingError as e:
                self.logger.warning(f"Batch processing attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries:
                    self.logger.info(f"Retrying in {self.retry_delay} seconds...")
                    asyncio.run(asyncio.sleep(self.retry_delay))
                    continue
                else:
                    self.logger.error(f"All {self.max_retries + 1} attempts failed for batch")
                    raise
            except Exception as e:
                self.logger.error(f"Unexpected error in batch processing: {str(e)}")
                raise BatchProcessingError(f"Unexpected batch processing error: {str(e)}") from e

        raise BatchProcessingError("All attempts failed for batch")

    # -------------------------------------------------------------------------
    # GROUPED REQUESTS MODE: Multiple prompts per HTTP request
    # -------------------------------------------------------------------------

    async def _process_batch_grouped_async(
        self,
        requests: List[Tuple[str, str]],
        stage_name: str,
        flush_callback: Optional[callable] = None,
        flush_interval: Optional[int] = None,
    ) -> Dict[str, str]:
        """Process requests in groups by cache_salt (report_id).

        Groups all prompts with the same report_id and sends them in a single
        HTTP request to /v1/completions for better cache locality.
        """
        self.logger.debug(f"[_process_batch_grouped_async()] Posting grouped batch to : {self._completions_url}")
        # Group requests by cache_salt
        groups: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for rid, prompt in requests:
            cache_salt = self._routing_cache_salt(rid, stage_name)
            # Use a default key for requests without cache_salt
            group_key = cache_salt if cache_salt else "__ungrouped__"
            groups[group_key].append((rid, prompt))

        self.logger.info(f"Grouped {len(requests)} requests into {len(groups)} groups by cache_salt")

        # Configure connection pool — cap TCP connections (see _process_batch_async)
        max_tcp_connections = min(self.request_concurrency, 8192)
        connector = aiohttp.TCPConnector(
            limit=max_tcp_connections,
            limit_per_host=max_tcp_connections,
            keepalive_timeout=60,
            enable_cleanup_closed=True,
            force_close=False,
            ttl_dns_cache=300,
        )

        timeout = aiohttp.ClientTimeout(
            total=self.client_timeout_total,  # don't count connector queue time
            connect=self.client_timeout_connect,
            sock_connect=self.client_timeout_sock_connect,
            sock_read=self.client_timeout_sock_read,  # "max inference latency"
        )

        results: Dict[str, str] = {}
        buffer: Dict[str, str] = {}
        flush_task: Optional[asyncio.Future] = None
        errors = 0

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Semaphore to limit concurrent group requests
            sem = asyncio.Semaphore(self.request_concurrency)

            async def process_group(cache_salt: str, group_requests: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
                """Process a group of requests with the same cache_salt."""
                async with sem:
                    for attempt in range(self.max_retries + 1):
                        try:
                            group_results = await self._request_group_async(
                                session, group_requests, stage_name, cache_salt
                            )
                            return group_results
                        except Exception as e:
                            if attempt < self.max_retries:
                                backoff = self.retry_delay * (2**attempt) + random.uniform(0, 0.5)
                                self.logger.debug(
                                    f"Group {cache_salt} failed (attempt {attempt + 1}): {e}; "
                                    f"retrying in {backoff:.2f}s"
                                )
                                await asyncio.sleep(backoff)
                            else:
                                self.logger.warning(
                                    f"Group {cache_salt} ({len(group_requests)} prompts) "
                                    f"failed after {self.max_retries + 1} attempts: {e}"
                                )
                                # Return empty results for all requests in the group
                                return [(rid, "") for rid, _ in group_requests]

                    return [(rid, "") for rid, _ in group_requests]

            # Create tasks for each group
            tasks = [process_group(cache_salt, group_requests) for cache_salt, group_requests in groups.items()]

            # Process with progress bar
            flush_errors = 0
            with async_tqdm(total=len(tasks), desc="Processing groups", unit="group", mininterval=1.0) as pbar:
                for coro in asyncio.as_completed(tasks):
                    group_results = await coro
                    for rid, content in group_results:
                        results[rid] = content
                        if not content:
                            errors += 1
                        elif flush_callback and flush_interval:
                            buffer[rid] = content

                    # Flush buffer (separate try/except so flush errors don't
                    # get conflated with HTTP errors).
                    if flush_callback and flush_interval and len(buffer) >= flush_interval:
                        try:
                            if flush_task is not None and not flush_task.done():
                                await flush_task
                            to_flush = buffer.copy()
                            buffer.clear()
                            self.logger.info(
                                f"Flushing buffer: {len(to_flush)} results ({len(results)}/{len(tasks)} total completed)"
                            )
                            loop = asyncio.get_running_loop()
                            flush_task = loop.run_in_executor(None, flush_callback, to_flush)
                        except Exception as e:
                            flush_errors += 1
                            self.logger.error(
                                f"Flush to disk failed ({type(e).__name__}: {e}). "
                                f"Total flush errors so far: {flush_errors}"
                            )
                    pbar.update(1)

            # Final flush of remaining buffer
            if flush_callback and buffer:
                try:
                    if flush_task is not None and not flush_task.done():
                        await flush_task
                    self.logger.info(f"Final buffer flush: {len(buffer)} remaining results")
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, flush_callback, dict(buffer))
                    buffer.clear()
                except Exception as e:
                    flush_errors += 1
                    self.logger.error(f"Final flush to disk failed ({type(e).__name__}: {e})")
            if flush_task is not None and not flush_task.done():
                await flush_task

        self.logger.info(f"Grouped async requests completed: {len(results)} total, {errors} empty/failed")

        # Log summary statistics
        non_empty_results = sum(1 for v in results.values() if v.strip())
        empty_results = len(results) - non_empty_results
        self.logger.info(f"Results: {non_empty_results} non-empty, {empty_results} empty")

        return results

    async def _request_group_async(
        self,
        session: aiohttp.ClientSession,
        group_requests: List[Tuple[str, str]],
        stage_name: str,
        cache_salt: Optional[str] = None,
    ) -> List[Tuple[str, str]]:
        """Send a group of prompts in a single HTTP request to /v1/completions.

        Uses the completions endpoint which supports multiple prompts in one request.
        Returns a list of (rid, content) tuples in the same order as input.
        """
        rids = [rid for rid, _ in group_requests]
        prompts = [prompt for _, prompt in group_requests]
        enable_thinking: bool = self.config["model"]["enable-thinking"]

        # Optionally prepend routing key for cache-aware routing
        # if stage_name in ("category_findings", "questions") and cache_salt:
        #     prompts = [f"ROUTE_KEY: '{cache_salt}'\n" + p for p in prompts]

        # Common payload fields
        model_cfg = self.config["model"]
        sampling_params = self.get_sampling_params(self._completions_url)
        payload: Dict[str, Any] = {"skip_special_tokens": True, **sampling_params}

        # Tokenize all prompts to input_ids
        assert self.tokenizer is not None, "Tokenizer must be set when send_tokenized_inputs=True"
        prompts_processed = []
        for prompt in prompts:
            # if not enable_thinking:
            #     prompt = prompt.strip() + " /no_think"
            # Build the full prompt with chat template
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Please output in plain text without any formatting.",
                },
                {"role": "user", "content": prompt},
            ]
            # Apply chat template and tokenize to get input_ids
            input_ids = self._apply_chat_template_and_tokenize(messages, self.send_tokenized_inputs)
            prompts_processed.append(input_ids)

        if self.send_tokenized_inputs:
            payload["input_ids"] = prompts_processed
        else:
            payload["prompt"] = prompts_processed

        # self.logger.info(f"payload: {payload}")
        if self.config["processing"]["apply-constrained-output"]:
            payload["ebnf"] = EBNF_GRAMMAR

        headers = {"Content-Type": "application/json"}
        if cache_salt and cache_salt != "__ungrouped__":
            headers["X-SMG-Routing-Key"] = cache_salt

        # Note: Request details are logged paired with response below (in debug mode or for samples)

        async with session.post(self._completions_url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                self.usage_stats.record_error(http_status=resp.status)
                raise aiohttp.ClientResponseError(
                    resp.request_info,
                    resp.history,
                    status=resp.status,
                    message=f"HTTP {resp.status}: {error_text}",
                )

            data = await resp.json()
            self.usage_stats.record_response(data, http_status=resp.status)

            # Sample logging: log first N request/response pairs with logger.info
            if await self._should_sample_log() or self.debug_mode:
                log_msg = self._format_request_response_log(
                    endpoint_url=self._completions_url,
                    headers=headers,
                    payload=payload,
                    response_data=data,
                    http_status=resp.status,
                    request_id=f"group:{cache_salt} ({len(rids)} prompts)",
                )
                self.logger.info(log_msg)

            # Parse response - Sort choices by index to ensure correct order
            choices = data.get("choices", [])
            choices_sorted = sorted(choices, key=lambda c: c.get("index", 0))

            # Extract text from each choice
            results = []
            for i, rid in enumerate(rids):
                if i < len(choices_sorted):
                    text = choices_sorted[i].get("text", "")
                    # Clean up the response (remove leading newlines that completions API adds)
                    text = text.strip("\n")
                    results.append((rid, text))
                else:
                    self.logger.warning(f"Missing choice for index {i} (rid={rid})")
                    results.append((rid, ""))

            return results

    # * ---------------------------------------------------------------------------------------------------------------------
    # * INDIVIDUAL REQUESTS MODE: One HTTP request per prompt
    # * ---------------------------------------------------------------------------------------------------------------------

    async def _process_batch_async(
        self,
        requests: List[Tuple[str, str]],
        stage_name: str,
        flush_callback: Optional[callable] = None,
        flush_interval: Optional[int] = None,
    ) -> Dict[str, str]:
        """Process all requests concurrently using aiohttp (one request per prompt)."""

        # Configure connection pool — limit TCP connections independently from
        # logical concurrency. HTTP keep-alive means fewer connections can serve
        # many in-flight requests. This reduces connection churn and avoids
        # uvloop fd race conditions (OSError errno 9) under high concurrency.
        max_tcp_connections = min(self.request_concurrency, 8192)
        connector = aiohttp.TCPConnector(
            limit=max_tcp_connections,
            limit_per_host=max_tcp_connections,
            keepalive_timeout=60,
            enable_cleanup_closed=True,
            force_close=False,
            ttl_dns_cache=300,
        )

        timeout = aiohttp.ClientTimeout(
            total=self.client_timeout_total,  # don't count connector queue time
            connect=self.client_timeout_connect,
            sock_connect=self.client_timeout_sock_connect,
            sock_read=self.client_timeout_sock_read,  # "max inference latency"
        )

        results: Dict[str, str] = {}
        buffer: Dict[str, str] = {}
        flush_task: Optional[asyncio.Future] = None
        errors = 0
        unhandled_errors = 0
        report_sems: dict[str, asyncio.Semaphore] = defaultdict(lambda: asyncio.Semaphore(self.max_in_flight_reports))

        # Silence uvloop bad-fd callback errors (race condition under high
        # concurrency). The affected requests will fail and be retried via the
        # normal retry path in bounded_request.
        loop = asyncio.get_running_loop()
        original_handler = loop.get_exception_handler()

        def _uvloop_exception_handler(loop, context):
            exc = context.get("exception")
            if isinstance(exc, OSError) and getattr(exc, "errno", None) == 9:
                self.logger.debug(
                    "uvloop bad fd (connection race, will be retried): %s",
                    context.get("message", ""),
                )
                return
            if original_handler:
                original_handler(loop, context)
            else:
                loop.default_exception_handler(context)

        loop.set_exception_handler(_uvloop_exception_handler)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Semaphore to limit concurrent requests
            sem = asyncio.Semaphore(self.request_concurrency)

            # Create all tasks
            tasks = [
                self.bounded_request(rid, prompt, sem, stage_name, session, report_sems) for rid, prompt in requests
            ]
            expected_rids = {rid for rid, _ in requests}

            # Process with progress bar using tqdm.asyncio
            flush_errors = 0
            with async_tqdm(total=len(tasks), desc="Processing requests", unit="req", mininterval=1.0) as pbar:
                for coro in asyncio.as_completed(tasks):
                    try:
                        rid, content = await coro
                        results[rid] = content
                        if not content:
                            errors += 1
                        elif flush_callback and flush_interval:
                            buffer[rid] = content
                    except Exception as e:
                        # Exception escaped bounded_request's retry logic
                        # (e.g. session/connector corruption from uvloop fd race).
                        # We can't determine which rid failed here, but missing
                        # rids are detected below and will be retried on the next
                        # run since they won't appear in processed_request_ids.
                        unhandled_errors += 1
                        self.logger.error(
                            f"Task raised unhandled exception ({type(e).__name__}: {e}). "
                            f"Total unhandled so far: {unhandled_errors}"
                        )

                    # Flush buffer outside the HTTP try/except so flush errors
                    # are reported separately from HTTP errors.
                    if flush_callback and flush_interval and len(buffer) >= flush_interval:
                        try:
                            if flush_task is not None and not flush_task.done():
                                await flush_task
                            to_flush = buffer.copy()
                            buffer.clear()
                            self.logger.info(
                                f"Flushing buffer: {len(to_flush)} results ({len(results)}/{len(tasks)} total completed)"
                            )
                            flush_task = loop.run_in_executor(None, flush_callback, to_flush)
                        except Exception as e:
                            flush_errors += 1
                            self.logger.error(
                                f"Flush to disk failed ({type(e).__name__}: {e}). "
                                f"Total flush errors so far: {flush_errors}"
                            )

                    pbar.update(1)

            # Final flush of remaining buffer
            if flush_callback and buffer:
                try:
                    if flush_task is not None and not flush_task.done():
                        await flush_task
                    self.logger.info(f"Final buffer flush: {len(buffer)} remaining results")
                    await loop.run_in_executor(None, flush_callback, dict(buffer))
                    buffer.clear()
                except Exception as e:
                    flush_errors += 1
                    self.logger.error(f"Final flush to disk failed ({type(e).__name__}: {e})")
            if flush_task is not None and not flush_task.done():
                await flush_task

        # Restore the original exception handler
        loop.set_exception_handler(original_handler)

        # Detect rids that were lost to unhandled exceptions. These won't appear
        # in processed_request_ids so they'll be retried on the next run.
        missing_rids = expected_rids - set(results.keys())
        if missing_rids:
            self.logger.warning(
                f"{len(missing_rids)} request(s) lost to unhandled exceptions "
                f"(will be retried on next run). First 10: {list(missing_rids)[:10]}"
            )

        self.logger.info(
            f"Async requests completed: {len(results)} total, {errors} empty/failed, "
            f"{unhandled_errors} unhandled exceptions, {len(missing_rids)} missing rids"
        )

        # Log summary statistics
        non_empty_results = sum(1 for v in results.values() if v.strip())
        empty_results = len(results) - non_empty_results
        self.logger.info(f"Results: {non_empty_results} non-empty, {empty_results} empty")

        return results

    async def bounded_request(
        self,
        rid: str,
        prompt: str,
        sem: asyncio.Semaphore,
        stage_name: str,
        session: aiohttp.ClientSession,
        report_sems: dict[str, asyncio.Semaphore],
    ) -> Tuple[str, str]:
        """Make a single request with semaphore-bounded concurrency and retries."""

        async with sem:  # global concurrency
            report_id = rid.split("_", 1)[0]
            async with report_sems[report_id]:  # per-report concurrency
                cache_salt = self._routing_cache_salt(rid, stage_name)

                for attempt in range(self.max_retries + 1):
                    try:
                        content = await self._request_once_async(session, prompt, stage_name, cache_salt)
                        return rid, content
                    except Exception as e:
                        if attempt < self.max_retries:
                            # Exponential backoff with jitter
                            backoff = self.retry_delay * (2**attempt) + random.uniform(0, 0.5)
                            self.logger.debug(
                                f"Request {rid} failed (attempt {attempt + 1}): {e}; retrying in {backoff:.2f}s"
                            )
                            await asyncio.sleep(backoff)
                        else:
                            self.logger.warning(f"Request {rid} failed after {self.max_retries + 1} attempts: {e}")
                            return rid, ""

                return rid, ""  # Fallback (shouldn't reach here)

    def _routing_cache_salt(self, request_id: str, stage_name: str, num_workers: int = 8) -> Optional[str]:
        """Use a stable per-report key for the many-requests-per-report stages.

        request IDs are:
          - category_findings:  "{report_id}_{category_name}"
          - questions:          "{report_id}_{category_name}_{question_id}"

        Returns a bucket number (0 to num_workers-1) as a string, hashed from report_id
        to distribute evenly across workers.
        """
        if stage_name not in ("category_findings", "questions"):
            return None
        report_id = request_id.split("_", 1)[0]
        # Hash the report_id and map to a bucket for even distribution
        hash_str = hashlib.md5(report_id.encode()).hexdigest()
        return hash_str
        # hash_int = int(hash_str, 16)
        # bucket = hash_int % num_workers
        # return str(bucket)

    def _apply_chat_template_and_tokenize(self, messages: List[Dict[str, str]], tokenize: bool) -> List[int]:
        """Apply chat template to messages and return token IDs.

        Some tokenizers (e.g., Qwen with thinking mode) support enable_thinking kwarg,
        while others don't. This method tries with the kwarg first, then falls back.

        Returns:
            List of token IDs (input_ids)
        """
        assert self.tokenizer is not None, "Tokenizer must be set"
        enable_thinking = self.config["model"]["enable-thinking"]
        # TODO: If we try to run with other models, they might have different ways of toggling thinking mode.
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=tokenize,
            enable_thinking=enable_thinking,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )

    async def _request_once_async(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        stage_name: str,
        cache_salt: Optional[str] = None,
    ) -> str:
        """Make a single async chat completion request."""

        # Build messages once (used by both paths)
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Please output in plain text without any formatting.",
            },
            {"role": "user", "content": prompt},
        ]

        # Common payload fields
        model_cfg = self.config.get("model", {})
        if self.send_tokenized_inputs:
            endpoint_url = self._completions_url
        else:
            endpoint_url = self._chat_completions_url
        sampling_params = self.get_sampling_params(endpoint_url)
        payload: Dict[str, Any] = {**sampling_params}

        if self.send_tokenized_inputs:
            # Apply chat template and tokenize to get input_ids
            assert self.tokenizer is not None, "Tokenizer must be set when send_tokenized_inputs=True"
            input_ids = self._apply_chat_template_and_tokenize(messages, self.send_tokenized_inputs)
            payload["prompt"] = input_ids
            payload["skip_special_tokens"] = True
        else:
            payload["messages"] = messages
            payload["chat_template_kwargs"] = {"enable_thinking": model_cfg["enable-thinking"]}

        if self.config["processing"]["apply-constrained-output"]:
            payload["ebnf"] = EBNF_GRAMMAR

        headers = {}
        if cache_salt:
            headers["X-SMG-Routing-Key"] = cache_salt

        # Note: Request details are logged paired with response below (in debug mode or for samples)

        async with session.post(endpoint_url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                # Record error with status code
                self.usage_stats.record_error(http_status=resp.status)
                raise aiohttp.ClientResponseError(
                    resp.request_info,
                    resp.history,
                    status=resp.status,
                    message=f"HTTP {resp.status}: {error_text}",
                )

            data = await resp.json()

            # Record usage stats from response
            self.usage_stats.record_response(data, http_status=resp.status)

            # Sample logging: log first N request/response pairs with logger.info
            if await self._should_sample_log() or self.debug_mode:
                log_msg = self._format_request_response_log(
                    endpoint_url=endpoint_url,
                    headers=headers,
                    payload=payload,
                    response_data=data,
                    http_status=resp.status,
                    request_id=cache_salt,
                )
                self.logger.info(log_msg)

            if self.send_tokenized_inputs:
                # /v1/completions response format: choices[0].text
                text = data.get("choices", [{}])[0].get("text", "")
                # Clean up the response (remove leading newlines that completions API adds)
                content = text.strip() if text else ""
            else:
                # /v1/chat/completions response format: choices[0].message.content
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content or ""

    async def _should_sample_log(self) -> bool:
        """Check if we should log this request/response pair as a sample.

        Thread-safe check and increment of sample counter.
        Returns True if this is one of the first N requests in the batch.
        """
        # Lazy create lock in async context
        if self._sample_log_lock is None:
            self._sample_log_lock = asyncio.Lock()

        async with self._sample_log_lock:
            if self._sample_log_count < self._sample_log_max:
                self._sample_log_count += 1
                return True
            return False

    def _format_request_response_log(
        self,
        endpoint_url: str,
        headers: Dict[str, Any],
        payload: Dict[str, Any],
        response_data: Dict[str, Any],
        http_status: int,
        request_id: Optional[str] = None,
    ) -> str:
        """Format a paired request/response for logging."""
        lines = [
            "",
            "=" * 80,
            f"SAMPLE REQUEST/RESPONSE #{self._sample_log_count}" + (f" (ID: {request_id})" if request_id else ""),
            "=" * 80,
            "",
            ">>> REQUEST:",
            f"  Endpoint: {endpoint_url}",
            f"  Headers: {headers}",
            f"  Payload: {json.dumps(payload, indent=2, default=str)}",
            "",
            "<<< RESPONSE:",
            f"  HTTP Status: {http_status}",
            f"  Body: {json.dumps(response_data, indent=2, default=str)}",
            "",
            "=" * 80,
        ]
        return "\n".join(lines)

    def get_batch_stats(self) -> Dict[str, Any]:
        """Get statistics about batch processing."""
        return {
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "request_concurrency": self.request_concurrency,
            "grouped_requests": self.grouped_requests,
            "send_tokenized_inputs": self.send_tokenized_inputs,
            "processor_type": "httpx_async",
            "lifetime_stats": self.usage_stats.get_summary(),
        }

    def get_lifetime_stats(self) -> Dict[str, Any]:
        """Get lifetime usage statistics summary."""
        return self.usage_stats.get_summary()

    def log_final_metrics(self) -> None:
        """Log the final lifetime metrics summary.

        Call this at the end of processing to display accumulated statistics.
        """
        summary_str = self.usage_stats.format_summary()
        self.logger.info(summary_str)

    def reset_stats(self) -> None:
        """Reset all lifetime statistics.

        Useful if you want to track stats for a specific subset of processing.
        """
        self.usage_stats = UsageStats()
