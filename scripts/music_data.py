import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data():
    data = {
        "happy melody": [60, 62, 64, 65, 67],  # MIDI note numbers
        "sad melody": [65, 63, 61, 60],
        "calm melody": [60, 60, 64, 64],
        "energetic melody": [67, 69, 71, 72, 74]
    }

    texts = list(data.keys())
    max_sequence_len = max([len(notes) for notes in data.values()])
    music_sequences = pad_sequences([data[text] for text in texts], maxlen=max_sequence_len)

    # Tokenize text descriptions
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    text_sequences = tokenizer.texts_to_sequences(texts)
    text_sequences = pad_sequences(text_sequences, maxlen=5)

    return tokenizer, text_sequences, music_sequences, max_sequence_len
