# utf-8

"""
preprocess.py

Preprocessing for the IMDb dataset using the RoBERTa tokenizer

Load the IMDb dataset, tokenize the text using the RoBERTa tokenizer and prepare the data in format for model training 
"""

from datasets import load_dataset
from transformers import RobertaTokenizer


def preprocess_data():
    """
    Loads the IMDb dataset, tokenizes the text and prepares the data for training
    """
    # Load IMDb DF
    dataset = load_dataset("imdb")
    print("DF loaded")

    # Load RoBERTa tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    print("RoBERTa tokenizer loaded")

    def preprocess(examples):
        """
        Tokenizes input texts
        
        Arg: 
            - examples: A dictionary containing text examples
        """
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

    # Tokenize training/test df
    tokenized_datasets = dataset.map(preprocess, batched=True)

    # Rename columns to model requirements
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch", columns=['input_ids', 'attention_mask', 'labels'])
    print("Df formatted")

    return tokenized_datasets

if __name__ == "__main__":
    tokenized_datasets = preprocess_data()
    print("Preprocessing Complete")