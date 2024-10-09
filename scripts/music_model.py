from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, TimeDistributed

def create_model(vocab_size, max_text_len, max_music_len, num_classes):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=max_text_len))
    model.add(LSTM(128, return_sequences=True))  # Return sequences for each time step
    model.add(TimeDistributed(Dense(num_classes, activation='softmax')))  # Predict at each time step
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
