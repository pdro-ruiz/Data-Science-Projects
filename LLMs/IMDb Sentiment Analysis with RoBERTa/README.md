# IMDb Sentiment Analysis with RoBERTa

This project was born with the intention of demonstrating the fine tuning capabilities of RoBERTa's LLM model. To do so, we will use sentiment analysis on the IMDb movie review dataset. The goal is to classify movie reviews as positive or negative.

## Project Overview

### Why RoBERTa?

We chose RoBERTa (Robustly optimized BERT approach) as the LLM model because of its improved performance over the original BERT model. This model has been pre-trained on a much larger dataset and optimized with better training strategies, and has a free access without APIKeys.

### Process

1. **Data Collection and Preprocessing**:
    - The IMDb dataset was loaded using the Hugging Face `datasets` library.
    - Preprocessing involved tokenizing the text data using the RoBERTa tokenizer. The text was truncated and padded to ensure uniform input size.

2. **Ajuste del modelo**:
    - Se ajustó un modelo RoBERTa previamente entrenado en el conjunto de datos IMDb. Para ello, se ajustaron los pesos del modelo en función de los datos etiquetados para mejorar su rendimiento en la tarea específica de análisis de sentimientos.

3. **Evaluación**:
    - puntuación F1, precisión y recuperación.

### Challenges and adaptations

- **Resource constraints**: The fine tuning of large models such as RoBERTa requires significant computing resources. This was achieved by optimizing batch sizes and using data loading techniques.

- **Class imbalance management**: Although the IMDb ensemble is relatively balanced, to ensure learning and good generalization it was necessary to monitor and adjust the training process to avoid over-fitting.

- **Terminology issues**: Managing longer reviews made it necessary to apply a tokenization process that did not truncate important information.

### Results

We managed to obtain very good results:

- **Valuation loss**: 0.2524
- **Accuracy**: 95.39%.
- **F1 score**: 0.9541
- **Accuracy**: 0.9497
- **Recovery**: 0.9586

These results indicate that RoBERTa is a well-balanced model with good performance, good accuracy and recall, which makes it quite reliable for processing review sentiment analysis.

## Conclusion

This project successfully demonstrates the power of transfer learning and fine tuning for sentiment analysis on IMDb reviews. Leveraging RoBERTa, we achieve high accuracy and balanced performance metrics. 
