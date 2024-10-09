import pretty_midi
import numpy as np

def save_to_midi(note_sequence, filename='generated_music.mid'):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
    start_time = 0
    
    for note_number in note_sequence:
        note = pretty_midi.Note(velocity=100, pitch=note_number, start=start_time, end=start_time + 0.5)
        instrument.notes.append(note)
        start_time += 0.5  # Move to the next note

    midi.instruments.append(instrument)
    midi.write(filename)
    print(f"Saved music to {filename}")
