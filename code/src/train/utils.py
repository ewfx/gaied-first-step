import os
from shutil import rmtree
from config import logger  # Import logger

def clear_transformers_cache():
    """Clears the Transformers cache directory."""
    cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
    if os.path.exists(cache_dir) and os.path.isdir(cache_dir):
        try:
            rmtree(cache_dir)
            logger.info("Transformers cache cleared.")
        except Exception as e:
            logger.warning(f"Failed to clear Transformers cache: {e}")
    else:
        logger.info("Transformers cache directory not found.")