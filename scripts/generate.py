import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from music_data import load_data  # Corrected import statement
from midi_utils import save_to_midi

def generate_music(text, model, tokenizer, max_text_len):
    # Convert text to a sequence
    text_seq = tokenizer.texts_to_sequences([text])
    text_seq = pad_sequences(text_seq, maxlen=max_text_len)
    
    # Predict the music sequence (MIDI notes)
    predicted_seq = model.predict(text_seq)[0]
    generated_notes = np.argmax(predicted_seq, axis=-1)
    
    return generated_notes

if __name__ == "__main__":
    # Load trained model
    model = load_model('text_to_music_model.h5')

    # Load tokenizer and data properties
    tokenizer, _, _, max_text_len = load_data()  # Load max_text_len from data

    # Generate music for a given text
    input_text = input("Enter a description (e.g., 'happy melody'): ")
    
    if input_text.strip():  # Check for empty input
        generated_notes = generate_music(input_text, model, tokenizer, max_text_len)

        # Save the generated notes as a MIDI file
        save_to_midi(generated_notes, filename='generated_music.mid')
        print("Generated music saved as 'generated_music.mid'")
    else:
        print("Please enter a valid description.")
