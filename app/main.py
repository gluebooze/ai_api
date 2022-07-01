from fastapi import FastAPI
import numpy as np
from typing import Optional
import pathlib
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()

BASE_DIR = pathlib.Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR.parent / "models"
SMS_SPAM_DIR = MODEL_DIR / "spam-sms"
MODEL_PATH = SMS_SPAM_DIR / "spam-model.h5"
TOKENIZER_PATH = SMS_SPAM_DIR / "spam-classifer-tokenizer.json"
METADATA_PATH = SMS_SPAM_DIR / "spam-classifer-metadata.json"

AI_MODEL = None
AI_TOKENIZER = None
MODEL_METADATA = {}
labels_legend_inverted = {}

@app.on_event("startup")
def on_startup():
    global AI_MODEL, AI_TOKENIZER, MODEL_METADATA, labels_legend_inverted
    if MODEL_PATH.exists():
        AI_MODEL = load_model(MODEL_PATH)
    if TOKENIZER_PATH.exists():
        t_json = TOKENIZER_PATH.read_text()
        AI_TOKENIZER = tokenizer_from_json(t_json)
    if METADATA_PATH.exists():
        MODEL_METADATA = json.loads(METADATA_PATH.read_text())
        labels_legend_inverted = MODEL_METADATA['labels_legend_inverted']

def pred(query:str):
    sequences = AI_TOKENIZER.texts_to_sequences([query])
    max_len = MODEL_METADATA.get('max_sequence') or 280
    x_input=  pad_sequences(sequences, maxlen = max_len)
    # print(x_input)
    # print(x_input.shape)
    preds = AI_MODEL.predict(x_input)[0]
    top_idx_val = np.argmax(preds)
    top_pred = {
        'label': labels_legend_inverted[str(top_idx_val)],
        'confidence': float(preds[top_idx_val])
    }
    res = [{'label': labels_legend_inverted[str(i)], 'confidence': float(x)} for i, x in enumerate(list(preds))]

    # print(preds_list)
    # print(res)
    return{
        'top': top_pred , 'predictions': res
    }


@app.get("/")
def read_index(q:Optional[str] = None):
    query = q or "hello world"
    global AI_MODEL, MODEL_METADATA , labels_legend_inverted
    return{
        "query": query,
        'results': pred(query)}
