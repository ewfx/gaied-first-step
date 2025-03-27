import logging
import random
import numpy as np
import torch

# -----------------------------------------------------------------------------
# Configure logging
# -----------------------------------------------------------------------------
def logging_setup():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("EmailClassifierTraining")
    return logger

# -----------------------------------------------------------------------------
# Reproducibility: Set seeds
# -----------------------------------------------------------------------------
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Global variables
RANDOM_SEED = 42
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 2

logger = logging_setup()  # Initialize logger
set_seeds(RANDOM_SEED)      # Set seeds