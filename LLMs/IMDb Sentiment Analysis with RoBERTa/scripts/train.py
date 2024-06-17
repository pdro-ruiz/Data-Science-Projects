# utf-8

"""
train.py

Fine-tuning training for RoBERTa on the IMDb df
"""

from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments, RobertaTokenizer
from preprocess import preprocess_data


def train_model():
    """
    Train the model by performing fine tuning on IMDb data. 
    Evaluates the model at each epoch, and saves the trained model and the tokenizer.
    """
    
    # Loading of preprocessed data
    tokenized_datasets = preprocess_data()

    # Load the RoBERTa model and tokenizer
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Configuration of training arguments
    training_args = TrainingArguments(
        output_dir='./results',                             # Path to save the control points of the model and the final model
        evaluation_strategy="epoch",                        # evaluate the model at the end of each epoch
        learning_rate=2e-5,                                 # learning rate for the optimizer
        per_device_train_batch_size=8,                      # batch size for training
        per_device_eval_batch_size=8,                       # batch size for evaluation
        num_train_epochs=4,                                 # number of training epochs
        weight_decay=0.01,                                  # neight decay for regularization
    )

    # Create an instance for training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],          # Training df
        eval_dataset=tokenized_datasets['test'],            # Evaluation df
    )

    # Train 
    trainer.train()
    print("Training completed")

    # Saving of trained model and tokenizer
    model.save_pretrained('./results')
    tokenizer.save_pretrained('./results')
    print("Model and tokenizer saved")

if __name__ == "__main__":
    train_model()