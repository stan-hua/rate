import hashlib
import re
import logging
from datetime import datetime
from pathlib import Path

def setup_logging(name: str, save_dir: str = "logs", filename: str = None) -> logging.Logger:
    """Setup logging configuration for a given module.
    
    Args:
        name: Name of the module/logger
        save_dir: Directory to save log files
        filename: Optional filename to include in log name
        
    Returns:
        Configured logger instance
    """
    # Create logs directory in save_dir if it doesn't exist
    log_dir = Path(save_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Build log filename with optional input filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if filename:
        # Extract base filename without extension
        base_filename = Path(filename).stem
        log_filename = f'{name}_{base_filename}_{timestamp}.log'
    else:
        log_filename = f'{name}_{timestamp}.log'
    
    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_dir / log_filename)
    
    # Create formatters and add it to handlers
    log_format = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(log_format)
    file_handler.setFormatter(log_format)
    
    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger 

# A request id opens with the report id, which for this cohort is a real
# accession number. The sample lines the batch processor logs print five ids
# and five answers per batch at INFO, so a run over the cohort would write
# some ten thousand accession numbers into a log file. `redacted` keeps the
# half that debugs anything -- the finding and the wording index -- and
# replaces the report id with a short digest, which still groups one report's
# lines together.
_REQUEST_ID = re.compile(r"^(.*?)_([a-z][a-z0-9_]*_v\d+)$")


def redacted(request_id):
    """Return a request id with the report id replaced by a short digest.

    Parameters
    ----------
    request_id : str
        `{report_id}_{finding_id}_v{variant_index}`, or any other stage's id.

    Returns
    -------
    str
        The digest, with the finding and wording kept where the id has them.
    """
    text = str(request_id)
    match = _REQUEST_ID.match(text)
    report, rest = (match.group(1), match.group(2)) if match else (text, "")
    digest = hashlib.sha256(report.encode()).hexdigest()[:8]
    return f"{digest}_{rest}" if rest else digest
