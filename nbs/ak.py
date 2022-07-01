import pathlib
import pandas as pd
import random

BASE_DIR= pathlib.Path().resolve()
DATABASE_DIR = BASE_DIR / 'datasets'
EXPORT_DIR = DATABASE_DIR / 'exports'
#EXPORT_DIR.mkdir(exist_ok=True,parents=True)
SPAM_DATASET_PATH = EXPORT_DIR / 'spam-dataset.csv'

METADATA_EXPORT_PATH = EXPORT_DIR / 'spam-metadata.pkl'
TOKENIZER_EXPORT_PATH = EXPORT_DIR / 'spam-tokenizer.json'
df = pd.read_csv(SPAM_DATASET_PATH)
df.head()
labels= df['label'].tolist()
texts= df['text'].tolist()

label_legend = {"ham": 0, "spam": 1}
label_legend_inverted = {f"{v}": k for k,v in label_legend.items()}

label_as_int = [label_legend[x] for x in labels ]

random_idx = random.randint(0,len(labels))

assert texts[random_idx]  == df.iloc[random_idx].text
assert label_legend_inverted[str(label_as_int[random_idx])]  == df.iloc[random_idx].label

from tensorflow.keras.preprocessing.text import Tokenizer

MAX_NUM_WORDS = 280

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
sequences

tokenizer.word_index

from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_SEQUENCE_LENGTH = 300

x= pad_sequences(sequences,maxlen=MAX_SEQUENCE_LENGTH)
x

import numpy as np
from tensorflow.keras.utils import to_categorical

labels_as_int_array = np.asarray(label_as_int)
labels_as_int_array

y=to_categorical(labels_as_int_array)
y

from sklearn.model_selection import train_test_split

import pickle

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.333,random_state=42)

training_data={
    "X_train" : X_train,
    "X_test" : X_test,
    "y_train" : y_train,
    "y_test" : y_test,
    "max_words" : MAX_NUM_WORDS,
    "max_sequence" : MAX_SEQUENCE_LENGTH,
    "legend" : label_legend,
    "labels_legend_inverted" : label_legend_inverted,
    "tokenizer" : tokenizer,
}

tokenizer_json = tokenizer.to_json()
TOKENIZER_EXPORT_PATH.write_text(tokenizer_json)

with open(METADATA_EXPORT_PATH,'wb') as f:
    pickle.dump(training_data,f)

data = {}

with open("datasets/exports/spam-metadata.pkl", 'rb') as f:
    data = pickle.load(f)