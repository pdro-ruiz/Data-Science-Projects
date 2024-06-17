# utf-8

"""
evaluate.py

Evaluation of RoBERTa fine tuning

Loads fine-tuned RoBERTa and a tokenizer, evaluates the model and prints the metrics
"""

import os
import numpy as np
from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from preprocess import preprocess_data


def load_model_and_tokenizer(model_path):
    """
    Loads the model and tokenizer

    Arg:
        model_path : The path to the model and tokenizer
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Path {model_path} does not exist")
    
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    return model, tokenizer

def compute_metrics(p):
    """
    Calculates the evaluation metrics of the model predictions

    Arg:
        p: a named tuple with predictions and label_ids
    """
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary')
    acc = accuracy_score(p.label_ids, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def evaluate_model(model_path='./results'):
    """
    Evaluates the model

    This function loads the model, the tokenizer, preprocesses the test set, and evaluates the model

    Arg:
        model_path: The path to the fitted model and tokenizer
    """
    model, tokenizer = load_model_and_tokenizer(model_path)

    # Preprocess
    tokenized_datasets = preprocess_data()
    print("Dataset preprocessed")

    # Set evaluation arguments
    training_args = TrainingArguments(
        output_dir='./results',                         # Directory to save evaluation
        per_device_eval_batch_size=8,                   # Batch size for evaluation
    )

    # Create a Trainer instance for evaluation
    trainer = Trainer(
        model=model,                                    # Fine-tuned model
        args=training_args,                             # Evaluation arguments
        eval_dataset=tokenized_datasets['test'],        # Evaluation dataset
        compute_metrics=compute_metrics                 # Function to compute evaluation metrics
    )

    # Evaluate
    results = trainer.evaluate()
    print("Evaluation completed")
    print(results)

if __name__ == "__main__":
    evaluate_model()