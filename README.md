# Extractive Question Answering Using BERT, BM25, and DPR

A comprehensive question answering system that combines fine-tuned BERT for answer extraction with multiple retrieval methods (BM25 and Dense Passage Retrieval) for document ranking and selection.

## Overview

This project implements a two-stage question answering pipeline:
1. **Document Retrieval**: Find relevant documents using BM25 or DPR
2. **Answer Extraction**: Extract precise answers from retrieved documents using fine-tuned BERT

## Architecture

### 1. BERT Question Answering Model
- **Base Model**: `bert-base-cased`
- **Fine-tuning Dataset**: SQuAD v2
- **Task**: Extractive question answering (start/end position prediction)

### 2. Retrieval Methods

#### BM25 (Lexical Retrieval)
- Traditional keyword-based retrieval
- Uses Okapi BM25 ranking function
- Fast and interpretable

#### Dense Passage Retrieval (DPR)
- **Sentence Transformers**: `facebook-dpr-ctx_encoder-single-nq-base`
- **Facebook DPR**: Separate question and context encoders
- Semantic similarity-based retrieval using dense embeddings

## Key Features

- **Multi-Stage Pipeline**: Retrieval + Extraction for comprehensive QA
- **Multiple Retrieval Methods**: Compare BM25 vs. DPR performance
- **Fine-tuned BERT**: Optimized for answer extraction on SQuAD v2
- **Performance Visualization**: Training curves and retrieval score comparison
- **Normalized Scoring**: Fair comparison across different retrieval methods

## Dataset

- **Training**: SQuAD v2 (5,000 samples)
- **Validation**: SQuAD v2 (2,000 samples) 
- **Testing**: SQuAD v2 (800 samples)
- **Document Collection**: Custom sample documents for retrieval demo

## Requirements

```
transformers
datasets
torch
faiss-cpu
rank_bm25
sentence-transformers
matplotlib
seaborn
nltk
numpy
```

## Installation & Usage

1. **Install Dependencies**:
   ```bash
   pip install transformers datasets torch faiss-cpu rank_bm25 sentence-transformers
   ```

2. **Download NLTK Resources**:
   ```python
   nltk.download('punkt_tab')
   nltk.download('punkt')
   ```

3. **Run the Complete Pipeline**:
   - Fine-tune BERT on SQuAD v2
   - Test retrieval methods on sample documents
   - Extract answers using the fine-tuned model

## Model Performance

### BERT Fine-tuning Results
- **Training Loss**: 0.662 → 0.361 (3 epochs)
- **Validation Loss**: 0.449 → 0.394 (3 epochs)
- **Early Stopping**: Implemented to prevent overfitting

### Retrieval Comparison
The system compares three retrieval approaches:
- **BM25**: Lexical matching with TF-IDF weighting
- **Sentence Transformers DPR**: Pre-trained dense retrieval
- **Facebook DPR**: Dual-encoder architecture for questions/contexts

## Technical Implementation

### BERT Fine-tuning
```python
# Key components:
- Preprocessing: tokenize_function with start/end positions
- Training: 3 epochs with early stopping
- Optimization: AdamW with learning rate 5e-5
```

### Retrieval Pipeline
```python
# BM25 Implementation
bm25 = BM25Okapi(tokenized_docs)
scores = bm25.get_scores(query_tokens)

# DPR Implementation  
embeddings = dpr_model.encode(documents)
similarity = faiss.IndexFlatIP(embeddings)
```

### Answer Extraction
```python
qa_pipeline = pipeline("question-answering", 
                      model="./fine_tuned_bert", 
                      tokenizer=tokenizer)
result = qa_pipeline(question=query, context=retrieved_doc)
```

## Visualization Features

- **Training Curves**: Loss progression over epochs
- **Retrieval Heatmap**: Normalized scores across all retrieval methods
- **Performance Comparison**: Side-by-side model evaluation

## Example Usage

```python
# Example query and documents
query = "Where is the Eiffel Tower?"
documents = [
    "The Eiffel Tower is in Paris.",
    "The Statue of Liberty is in New York.", 
    "Mount Everest is the highest mountain."
]

# 1. Retrieve relevant document
best_doc = retrieval_method.retrieve(query, documents)

# 2. Extract answer
result = qa_pipeline(question=query, context=best_doc)
print(f"Answer: {result['answer']}")
```

## Evaluation Metrics

- **Retrieval**: Similarity scores (normalized for comparison)
- **Extraction**: Exact Match (EM) and F1 scores on SQuAD v2
- **End-to-End**: Combined retrieval + extraction accuracy

## Research Applications

- **Information Retrieval**: Compare lexical vs. semantic retrieval
- **Question Answering**: Benchmark different QA architectures
- **Hybrid Systems**: Combine multiple retrieval methods
- **Domain Adaptation**: Fine-tune for specific knowledge domains

## Future Enhancements

- **Hybrid Retrieval**: Combine BM25 and DPR scores
- **Larger Document Collections**: Scale to Wikipedia-sized corpora
- **Multi-hop QA**: Handle questions requiring multiple documents
- **Real-time API**: Deploy as a web service
- **Domain-specific Models**: Fine-tune for specialized domains (legal, medical, etc.)

## File Structure

```
├── results/              # Training outputs
├── logs/                # Training logs  
├── fine_tuned_bert/     # Saved BERT model
├── qa_system.py         # Main implementation
└── README.md           # This file
```

## Performance Notes

- **BERT Fine-tuning**: ~30 minutes on GPU for 5K samples
- **Retrieval Speed**: BM25 > Sentence Transformers > Facebook DPR
- **Memory Usage**: DPR requires more memory for embeddings
- **Accuracy Trade-off**: Dense retrieval typically more accurate than BM25
