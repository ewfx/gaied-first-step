from config import logger, MODEL_NAME
from training import train_model
from evaluation import evaluate_model
from utils import clear_transformers_cache
from data_loader import load_and_prepare_data
from data_tokenizer import tokenize_datasets
from model_initializer import initialize_model
from model_handling import save_model

def main():
    clear_transformers_cache()

    train_dataset, eval_dataset, raw_eval_dataset = load_and_prepare_data()

    if train_dataset is None:  # Handle data loading failure
        return

    train_dataset, eval_dataset, tokenizer = tokenize_datasets(train_dataset, eval_dataset)
    model = initialize_model("email_classifier_llm_latest")

    trainer = train_model(model, train_dataset, eval_dataset, tokenizer)
    evaluate_model(trainer, eval_dataset, raw_eval_dataset)
    save_model(model, tokenizer, "email_classifier_llm_latest")


if __name__ == "__main__":
    main()