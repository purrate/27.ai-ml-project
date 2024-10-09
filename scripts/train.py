import numpy as np
from music_data import load_data
from music_model import create_model
from keras.utils import to_categorical

# Load the data
tokenizer, text_sequences, music_sequences, max_music_len = load_data()

# Verify the shapes of loaded data
print("Text sequences shape:", text_sequences.shape)
print("Music sequences shape:", music_sequences.shape)

# Expand dimensions for loss function if necessary
music_sequences = np.expand_dims(music_sequences, axis=-1)

# Create the model
vocab_size = len(tokenizer.word_index) + 1
num_classes = np.max(music_sequences) + 1  # Define number of classes based on music_sequences
model = create_model(vocab_size=vocab_size, max_text_len=text_sequences.shape[1], 
                     max_music_len=max_music_len, num_classes=num_classes)

# Convert music sequences to one-hot encoding
music_sequences_one_hot = to_categorical(music_sequences, num_classes=num_classes)

# Reshape for sequence prediction (4, 5, 75)
music_sequences_one_hot = music_sequences_one_hot.reshape((music_sequences_one_hot.shape[0], 
                                                            music_sequences_one_hot.shape[1], 
                                                            -1))

# Train the model
model.fit(text_sequences, music_sequences_one_hot, epochs=100, verbose=2)

# Save the trained model
model.save('text_to_music_model.h5')
print("Model saved as text_to_music_model.h5")
