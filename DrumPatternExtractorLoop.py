import subprocess
from pathlib import Path

# Base directory = where this script lives
BASE_DIR = Path(__file__).parent.resolve()

# Input and output folders (always relative to script location)
INPUT_DIR = BASE_DIR / "RockMIDI_Dataset"
OUTPUT_DIR = BASE_DIR / "RockBinary_Dataset"

def batch_process(input_path=INPUT_DIR, output_path=OUTPUT_DIR):
    output_path.mkdir(parents=True, exist_ok=True)

    midi_files = list(input_path.glob("*.midi"))

    if not midi_files:
        print(f"No .midi files found in {input_path}")
        return

    for midi_file in midi_files:
        output_file = output_path / f"{midi_file.stem}_drums.csv"

        print(f"\nProcessing {midi_file} → {output_file}")

        # Feed both input and output paths to the extractor script
        subprocess.run(
            ["python", str(BASE_DIR / "MIDItoCSV_DrumPatternExtractor.py")],
            input=f"{midi_file}\n{output_file}\n\n",  # input, output, skip resolution
            text=True
        )

if __name__ == "__main__":
    batch_process()
