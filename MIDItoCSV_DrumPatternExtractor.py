import mido
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

def extract_drum_patterns(midi_file_path, output_csv_path, force_time_resolution=None):
    """
    Extract drum patterns from a MIDI file and save as binary pattern in CSV format.
    Automatically detects time signatures and adjusts accordingly.
    
    Args:
        midi_file_path (str): Path to the input MIDI file
        output_csv_path (str): Path to the output CSV file
        force_time_resolution (int): Override automatic resolution (slots per beat)
    
    Returns:
        dict: Summary of extraction
    """
    
    # General MIDI drum map (channel 9, notes 35-81)
    drum_map = {
        35: 'Acoustic Bass Drum', 36: 'Bass Drum 1', 37: 'Side Stick', 38: 'Acoustic Snare',
        39: 'Hand Clap', 40: 'Electric Snare', 41: 'Low Floor Tom', 42: 'Closed Hi Hat',
        43: 'High Floor Tom', 44: 'Pedal Hi-Hat', 45: 'Low Tom', 46: 'Open Hi-Hat',
        47: 'Low-Mid Tom', 48: 'Hi Mid Tom', 49: 'Crash Cymbal 1', 50: 'High Tom',
        51: 'Ride Cymbal 1', 52: 'Chinese Cymbal', 53: 'Ride Bell', 54: 'Tambourine',
        55: 'Splash Cymbal', 56: 'Cowbell', 57: 'Crash Cymbal 2', 58: 'Vibraslap',
        59: 'Ride Cymbal 2', 60: 'Hi Bongo', 61: 'Low Bongo', 62: 'Mute Hi Conga',
        63: 'Open Hi Conga', 64: 'Low Conga', 65: 'High Timbale', 66: 'Low Timbale',
        67: 'High Agogo', 68: 'Low Agogo', 69: 'Cabasa', 70: 'Maracas',
        71: 'Short Whistle', 72: 'Long Whistle', 73: 'Short Guiro', 74: 'Long Guiro',
        75: 'Claves', 76: 'Hi Wood Block', 77: 'Low Wood Block', 78: 'Mute Cuica',
        79: 'Open Cuica', 80: 'Mute Triangle', 81: 'Open Triangle'
    }
    
    try:
        # Load MIDI file
        mid = mido.MidiFile(midi_file_path)
        
        print(f"Processing MIDI file: {midi_file_path}")
        print(f"MIDI ticks per beat: {mid.ticks_per_beat}")
        
        # Detect time signatures and tempo changes
        time_signatures = []  # List of (tick, numerator, denominator)
        tempo_changes = []    # List of (tick, tempo_bpm)
        current_tick = 0
        
        # First pass: collect time signatures and tempo
        for track in mid.tracks:
            current_tick = 0
            for msg in track:
                current_tick += msg.time
                
                if msg.type == 'time_signature':
                    time_signatures.append((current_tick, msg.numerator, msg.denominator))
                    print(f"Time signature found at tick {current_tick}: {msg.numerator}/{msg.denominator}")
                
                elif msg.type == 'set_tempo':
                    # Convert microseconds per beat to BPM
                    bpm = 60000000 / msg.tempo
                    tempo_changes.append((current_tick, bpm))
                    print(f"Tempo change at tick {current_tick}: {bpm:.1f} BPM")
        
        # Default to 4/4 if no time signature found
        if not time_signatures:
            time_signatures.append((0, 4, 4))
            print("No time signature found, assuming 4/4")
        else:
            print(f"Found {len(time_signatures)} time signature change(s)")
        
        # Default tempo if none found
        if not tempo_changes:
            tempo_changes.append((0, 120.0))
            print("No tempo found, assuming 120 BPM")
        
        # Calculate time resolution based on time signature
        main_time_sig = time_signatures[0]  # Use first time signature
        numerator, denominator = main_time_sig[1], main_time_sig[2]
        
        if force_time_resolution:
            time_resolution = force_time_resolution
            print(f"Using forced time resolution: {time_resolution} slots per beat")
        else:
            # Adaptive resolution based on time signature
            if denominator == 4:  # Quarter note beat
                time_resolution = 16  # 16th notes
            elif denominator == 8:  # Eighth note beat
                time_resolution = 8   # 16th notes relative to 8th beat
            else:
                time_resolution = 16  # Default
            print(f"Auto-detected time resolution: {time_resolution} slots per beat")
        
        print(f"Main time signature: {numerator}/{denominator}")
        
        # Calculate total ticks and time slots
        total_ticks = sum(msg.time for track in mid.tracks for msg in track)
        ticks_per_slot = mid.ticks_per_beat // time_resolution
        total_slots = (total_ticks // ticks_per_slot) + 1
        
        # Calculate measures and beats
        ticks_per_measure = mid.ticks_per_beat * numerator
        slots_per_measure = time_resolution * numerator
        total_measures = total_slots / slots_per_measure
        
        print(f"Total ticks: {total_ticks}")
        print(f"Ticks per slot: {ticks_per_slot}")
        print(f"Total time slots: {total_slots}")
        print(f"Slots per measure: {slots_per_measure}")
        print(f"Estimated measures: {total_measures:.1f}")
        
        # Dictionary to store drum patterns
        drum_patterns = {}
        active_drums = set()
        
        # Second pass: extract drum patterns
        for track_num, track in enumerate(mid.tracks):
            current_tick = 0
            
            for msg in track:
                current_tick += msg.time
                
                # Check if it's a drum channel message (channel 9 = index 9)
                if hasattr(msg, 'channel') and msg.channel == 9:
                    if msg.type == 'note_on' and msg.velocity > 0:
                        # Note on event
                        note = msg.note
                        if note in drum_map:
                            slot = current_tick // ticks_per_slot
                            #Check if slot is inside the scope
                            if slot < total_slots:
                                #Initialize the note slots
                                if note not in drum_patterns:
                                    drum_patterns[note] = np.zeros(total_slots, dtype=int)
                                drum_patterns[note][slot] = 1
                                active_drums.add(note)
        
        # Filter out drums that are never played
        active_drum_patterns = {drum: pattern for drum, pattern in drum_patterns.items() if np.sum(pattern) > 0}
        
        if not active_drum_patterns:
            print("Warning: No drum patterns found in the MIDI file!")
            return {'drums_found': 0, 'total_slots': total_slots, 'output_file': output_csv_path}
        
        # Create CSV output with measure/beat information
        with open(output_csv_path, 'w', newline='') as csvfile:
            # Prepare headers
            sorted_drums = sorted(active_drum_patterns.keys())
            headers = ['Time_Slot', 'Measure', 'Beat', 'Sub_Beat'] + [f"{drum}_{drum_map.get(drum, f'Unknown_{drum}')}" for drum in sorted_drums]
            
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            
            # Write pattern data with musical context
            for slot in range(total_slots):
                measure = (slot // slots_per_measure) + 1
                beat_in_measure = ((slot % slots_per_measure) // time_resolution) + 1
                sub_beat = (slot % time_resolution) + 1
                
                row = [slot, measure, beat_in_measure, sub_beat]
                for drum in sorted_drums:
                    row.append(active_drum_patterns[drum][slot])
                writer.writerow(row)
        
        # Summary
        summary = {
            'drums_found': len(active_drum_patterns),
            'drum_list': [f"{drum}: {drum_map.get(drum, f'Unknown_{drum}')}" for drum in sorted_drums],
            'total_slots': total_slots,
            'total_hits': sum(np.sum(pattern) for pattern in active_drum_patterns.values()),
            'time_signature': f"{numerator}/{denominator}",
            'time_resolution': time_resolution,
            'slots_per_measure': slots_per_measure,
            'estimated_measures': round(total_measures, 1),
            'tempo_changes': len(tempo_changes),
            'output_file': output_csv_path
        }
        
        print(f"\nExtraction complete!")
        print(f"Time signature: {summary['time_signature']}")
        print(f"Drums found: {summary['drums_found']}")
        print(f"Total drum hits: {summary['total_hits']}")
        print(f"Estimated measures: {summary['estimated_measures']}")
        print(f"Pattern saved to: {output_csv_path}")
        
        return summary
        
    except Exception as e:
        print(f"Error processing MIDI file: {str(e)}")
        return {'error': str(e)}

def main():
    """Main function to run the drum pattern extractor"""
    
    print("MIDI Drum Pattern Extractor with Time Signature Detection")
    print("=" * 60)
    
    # Get input file
    midi_file = input("Enter the path to your MIDI file: ").strip()
    
    if not Path(midi_file).exists():
        print(f"Error: MIDI file '{midi_file}' does not exist!")
        return
    
    # Get output file
    output_file = input("Enter the path for output CSV file (or press Enter for default): ").strip()
    #Creates the csv file in the same folder where the input file is by default
    if not output_file:
        midi_path = Path(midi_file)
        output_file = midi_path.parent / f"{midi_path.stem}_drums.csv"
    
    # Get time resolution override
    try:
        resolution = input("Force time resolution (slots per beat) or press Enter for auto-detection: ").strip()
        force_resolution = int(resolution) if resolution else None
    except ValueError:
        force_resolution = None
        print("Invalid resolution, will use auto-detection")
    
    # Extract drum patterns
    print(f"\nExtracting drum patterns...")
    result = extract_drum_patterns(midi_file, output_file, force_time_resolution=force_resolution)
    
    if 'error' not in result:
        print(f"\nSUMMARY:")
        print(f"- Time signature: {result['time_signature']}")
        print(f"- Time resolution: {result['time_resolution']} slots per beat")
        print(f"- Slots per measure: {result['slots_per_measure']}")
        print(f"- Drums extracted: {result['drums_found']}")
        print(f"- Total drum hits: {result['total_hits']}")
        print(f"- Estimated measures: {result['estimated_measures']}")
        print(f"- Output saved to: {result['output_file']}")
        
        if 'drum_list' in result:
            print(f"\nDrums found in the file:")
            for drum_info in result['drum_list']:
                print(f"  • {drum_info}")

if __name__ == "__main__":
    main()