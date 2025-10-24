import pandas as pd
import mido

def csv_to_drum_track(csv_path, target_ticks, ticks_per_beat=480, velocity=100):
    """
    Converts a generated drum CSV file into a MIDI track (channel 9 - standard drum channel).
    The generated drum track is scaled to match `target_ticks` (duration of original drum track).
    Handles multiple drum hits per row and preserves correct timing.
    """
    df = pd.read_csv(csv_path)
    drum_cols = df.columns[4:]  # First 4 columns are informational
    track = mido.MidiTrack()
    track.append(mido.MetaMessage('track_name', name='Generated Drums'))

    num_rows = len(df)
    if num_rows == 0:
        return track

    # Compute delta so that CSV spans exactly the same length as original drum track
    delta = int(target_ticks / num_rows)

    accum_delta = 0

    for _, row in df.iterrows():
        hits = [col for col in drum_cols if row[col] == 1 or row[col] == 1.0]

        if hits:
            for i, col in enumerate(hits):
                note = int(col.split('_')[0])
                # Apply accumulated delta only to the first hit in this row
                time = accum_delta if i == 0 else 0
                track.append(mido.Message('note_on', note=note, velocity=velocity, channel=9, time=time))
                track.append(mido.Message('note_off', note=note, velocity=0, channel=9, time=0))
            accum_delta = delta
        else:
            accum_delta += delta

    return track


def replace_drum_track_preserve_tempo(midi_in_path, csv_path, midi_out_path):
    """
    Replaces the drum track (channel 9) of a MIDI file with a new one generated from a CSV.
    Preserves original tempo, time signature, and other tracks.
    """
    # Load original MIDI
    mid = mido.MidiFile(midi_in_path)
    new_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)

    # --- Step 1: Extract global tempo/time/key events ---
    tempo_events = []
    for track in mid.tracks:
        for msg in track:
            if msg.is_meta and msg.type in ('set_tempo', 'time_signature', 'key_signature'):
                tempo_events.append(msg)

    tempo_track = mido.MidiTrack()
    tempo_track.append(mido.MetaMessage('track_name', name='Tempo / Global Meta'))
    for msg in tempo_events:
        tempo_track.append(msg)
    new_mid.tracks.append(tempo_track)

    # --- Step 2: Find original drum track duration ---
    drum_track_ticks = 0
    for track in mid.tracks:
        if any(not msg.is_meta and hasattr(msg, "channel") and msg.channel == 9 for msg in track):
            drum_track_ticks = sum(msg.time for msg in track)
            break

    # --- Step 3: Create new drum track ---
    new_drum_track = csv_to_drum_track(csv_path, target_ticks=drum_track_ticks, ticks_per_beat=mid.ticks_per_beat)
    drum_replaced = False

    # --- Step 4: Copy all non-drum tracks + insert new drums ---
    for track in mid.tracks:
        has_drum = any(not msg.is_meta and hasattr(msg, "channel") and msg.channel == 9 for msg in track)
        if has_drum and not drum_replaced:
            # Replace original drum track
            new_mid.tracks.append(new_drum_track)
            drum_replaced = True
        elif not has_drum:
            # Keep all non-drum tracks
            new_mid.tracks.append(track)

    # --- Step 5: If no original drum track found, append generated drums at end ---
    if not drum_replaced:
        new_mid.tracks.append(new_drum_track)

    # --- Step 6: Save new MIDI ---
    new_mid.save(midi_out_path)
    print(f"MIDI saved with replaced drums: {midi_out_path}")


# === Example usage ===
replace_drum_track_preserve_tempo(
    r"original.mid", #original midi file path
    r"generated_drums.csv", #generated drums midi file path
    r"original_with_new_drums.midi" #output path for new midi
)
