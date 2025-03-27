from config import logger
from model_handling import load_model, model_init
import os

def initialize_model(model_dir):
    """Loads an existing model or initializes a new one."""
    choice = input(f"Load an existing model from '{model_dir}'? (y/n): ").strip().lower()
    model = None

    if choice == "y" and os.path.exists(model_dir):
        try:
            model = load_model(model_dir)
        except Exception as e:
            logger.error(f"Error loading model from {model_dir}: {e}. Creating a new model.", e)
            model = model_init()
    else:
        if choice == "y":
            logger.warning("No model found at '%s'. Creating a new model.", model_dir)
        else:
            logger.info("Creating a new model.")
        model = model_init()
    return model