import re
import numpy as np
import torch
import tensorflow as tf
from transformers import BertTokenizer, BertForSequenceClassification
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Preprocess and clean the text
def preprocess_text(text):
    # Remove special characters and extra spaces
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Tokenize and pad sequences using BERT tokenizer
def tokenize_and_pad(texts, tokenizer, max_length):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

# Create CNN model
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Embedding(input_dim=input_shape[0], output_dim=128, input_length=input_shape[1]),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Example text samples
text_samples = [
    "I hate the chatbot.",
    "I hate bittergrounds.",
    "Coding is fun",
    "Cats are cute.",
    "I am fine but weak",
    "I have nothing to do",
    "The weather is rainy."
]

preprocessed_texts = [preprocess_text(text) for text in text_samples]

# Tokenize and pad sequences
max_length = 50
input_ids, attention_masks = tokenize_and_pad(preprocessed_texts, tokenizer, max_length)

# Sentiment analysis using BERT
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_masks)
    predictions = np.argmax(outputs.logits, axis=1)

sentiments = ['Neutral', 'Negative', 'Positive']
for i, pred in enumerate(predictions):
    print(f"Text: {text_samples[i]}")
    print(f"Predicted Sentiment: {sentiments[pred]}\n")

# Sentiment analysis using CNN
vocab_size = len(tokenizer.get_vocab())
cnn_model = create_cnn_model(input_shape=(vocab_size, max_length), num_classes=3)
cnn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the CNN model (you would need labeled data for this)
# cnn_model.fit(input_ids.numpy(), predictions, epochs=5, batch_size=4)




