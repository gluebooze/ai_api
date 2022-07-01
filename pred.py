import pathlib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

model = load_model("models/spam-sms/spam-model.h5")

data = {}
with open("datasets/exports/spam-metadata.pkl", 'rb') as f:
    data = pickle.load(f)

labels_legend_inverted = data['labels_legend_inverted']
max_sequence = data['max_sequence']
max_words = data['max_words']
tokenizer = data['tokenizer']

def predict(text_str, max_words=280, max_sequence = 280, tokenizer=None):
  if not tokenizer:
    return None
  sequences = tokenizer.texts_to_sequences([text_str])
  x_input = pad_sequences(sequences, maxlen=max_sequence)
  y_output = model.predict(x_input)
  top_y_index = np.argmax(y_output)
  preds = y_output[top_y_index]
  labeled_preds = [{f"{labels_legend_inverted[str(i)]}": x} for i, x in enumerate(preds)]
  return labeled_preds


while(True):
  msg = input("Insert message: ")
  res = predict(msg, max_words=max_words, max_sequence=max_sequence, tokenizer=tokenizer)
  print(res)