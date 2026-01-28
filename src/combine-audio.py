import os
import sys
import re
from pydub import AudioSegment
from tqdm import tqdm

# --- Configuration ---
# This should match the OUTPUT_DIR from the generation script.
SOURCE_DIR = "output2"
ERROR_SUFFIX = "-error"
# Duration of silence to add between each audio part, in milliseconds.
# Set to 0 to have no pause.
PAUSE_BETWEEN_SEGMENTS_MS = 200
# MP3 export settings
MP3_BITRATE = "192k"
# --- End Configuration ---


def combine_audio_parts(base_name: str):
    """
    Finds, verifies, and combines WAV audio parts into a single MP3 file,
    adding a configurable pause between each part.

    Args:
        base_name (str): The base name of the files to process (e.g., "kapitel1").
    """
    print(f"--- Starting combination process for '{base_name}' ---")

    # 1. Get all .wav files from the source directory, except error files
    print(f"1. Searching for valid WAV files in '{SOURCE_DIR}'...")
    try:
        all_files = os.listdir(SOURCE_DIR)
    except FileNotFoundError:
        print(f"Error: Source directory '{SOURCE_DIR}' not found. Did you run the generation script first?")
        return

    # Regex to match files like "kapitel-1-001.wav"
    file_pattern = re.compile(re.escape(base_name) + r'-(\d{3,})\.wav')

    valid_files = []
    for filename in all_files:
        if filename.startswith(base_name) and filename.endswith('.wav'):
            if ERROR_SUFFIX not in filename:
                if file_pattern.match(filename):
                    valid_files.append(filename)

    if not valid_files:
        print("Error: No valid WAV files found for this base name. Nothing to do.")
        return

    # Sort files alphabetically to ensure correct order
    valid_files.sort()
    print(f"Found {len(valid_files)} valid parts.")

    # 2. Verify that no parts are missing
    print("\n2. Verifying part sequence...")
    part_numbers = []
    for filename in valid_files:
        match = file_pattern.match(filename)
        if match:
            # Extract the number part and convert it to an integer
            part_numbers.append(int(match.group(1)))

    # Check for gaps in the sequence
    expected_parts = set(range(1, len(part_numbers) + 1))
    actual_parts = set(part_numbers)

    missing_parts = sorted(list(expected_parts - actual_parts))

    if min(part_numbers) != 1:
        print(f"Error: Sequence is invalid. Part 1 is missing.")
        return

    if missing_parts:
        print(f"Error: Sequence is incomplete! Missing parts: {missing_parts}")
        return

    print("Sequence verification successful. All parts are present.")

    # 3. Concatenate all files together with pauses
    if PAUSE_BETWEEN_SEGMENTS_MS > 0:
        print(f"\n3. Concatenating audio parts with a {PAUSE_BETWEEN_SEGMENTS_MS}ms pause between them...")
        pause = AudioSegment.silent(duration=PAUSE_BETWEEN_SEGMENTS_MS)
    else:
        print("\n3. Concatenating audio parts...")
        pause = None # No pause will be added

    # Initialize with the first audio file
    first_file_path = os.path.join(SOURCE_DIR, valid_files[0])
    combined_audio = AudioSegment.from_wav(first_file_path)

    # Add the rest of the files in a loop with a progress bar
    for filename in tqdm(valid_files[1:], desc="Combining"):
        # Add the pause *before* the next segment
        if pause:
            combined_audio += pause

        # Add the next audio segment
        file_path = os.path.join(SOURCE_DIR, filename)
        next_segment = AudioSegment.from_wav(file_path)
        combined_audio += next_segment

    # 4. Convert to MP3 and save
    output_mp3_path = f"{base_name}.mp3"
    print(f"\n4. Exporting to MP3: '{output_mp3_path}'...")

    try:
        combined_audio.export(
            output_mp3_path,
            format="mp3",
            bitrate=MP3_BITRATE,
            tags={'artist': 'TTS-Chatterbox', 'title': base_name.replace('_', ' ').title()}
        )
        final_duration_minutes = len(combined_audio) / 60000
        print("\n--- Success! ---")
        print(f"File '{output_mp3_path}' created successfully.")
        print(f"Total duration: {final_duration_minutes:.2f} minutes.")

    except Exception as e:
        print("\n--- Error during export! ---")
        print("An error occurred while saving the MP3 file.")
        print("Please ensure FFmpeg is installed and accessible in your system's PATH.")
        print(f"Details: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <base_name>")
        print("Example: python combine_audio.py kapitel-1")
        sys.exit(1)

    # Get the base name from the first argument
    input_base_name = sys.argv[1]
    combine_audio_parts(input_base_name)