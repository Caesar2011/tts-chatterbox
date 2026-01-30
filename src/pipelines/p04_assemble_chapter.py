import os
import re
from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm
import shutil

from utils.config import PROJECT_CONFIG
from utils.audio import run_loudness_normalization

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    """Sorts strings with numbers in a natural, human-friendly order."""
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]

def run(chapter_name: str):
    """
    Assembles a chapter from synthesized audio segments and pause markers.

    This function reads all 'segment_*.wav' files from a chapter's temporary
    directory. It distinguishes between actual audio parts and 0-byte pause
    markers based on their filenames, then concatenates them in the correct
    order to create the full chapter audio.
    """
    root_path = Path(__file__).parent.parent.parent
    temp_dir = root_path / "temp"
    temp_chapter_dir = temp_dir / chapter_name
    output_dir = root_path / "output"
    os.makedirs(output_dir, exist_ok=True)

    if not temp_chapter_dir.exists():
        raise FileNotFoundError(f"Temporary directory for chapter '{chapter_name}' not found. Run synthesis first.")

    # Define pause durations in milliseconds.
    pause_config = PROJECT_CONFIG.get('pauses', {
        'small': 500,    # Short thinking pause
        'medium': 1200,  # Pause for a new scene or topic change
        'large': 2500    # Pause for a major section break
    })

    print("1. Searching for audio segments and pause markers...")
    all_files = [f for f in os.listdir(temp_chapter_dir) if f.startswith('segment_') and f.endswith('.wav')]

    if not all_files:
        print("No audio segments or pause markers found. Nothing to assemble.")
        return

    all_files.sort(key=natural_sort_key)
    print(f"Found {len(all_files)} files (segments and pauses) to assemble.")

    print("2. Concatenating audio segments and inserting pauses...")
    combined_audio = AudioSegment.empty()
    pause_pattern = re.compile(r"segment_\d+_pause_(\w+)\.wav")

    for filename in tqdm(all_files, desc="Assembling"):
        file_path = temp_chapter_dir / filename
        pause_match = pause_pattern.match(filename)

        if pause_match:
            pause_type = pause_match.group(1)
            duration_ms = pause_config.get(pause_type, pause_config['medium']) # Default to medium
            if duration_ms > 0:
                combined_audio += AudioSegment.silent(duration=duration_ms)
        else:
            try:
                segment = AudioSegment.from_wav(file_path)
                combined_audio += segment
            except Exception as e:
                print(f"Warning: Could not load segment {filename}. Skipping. Error: {e}")

    processing_temp_dir = temp_dir / "_processing"
    os.makedirs(processing_temp_dir, exist_ok=True)
    raw_combined_path = processing_temp_dir / f"{chapter_name}_raw.wav"

    print(f"3. Exporting raw combined WAV to {raw_combined_path}...")
    combined_audio.export(raw_combined_path, format="wav")

    current_file = raw_combined_path
    if PROJECT_CONFIG.get('use_loudness_normalization', False):
        print("4. Applying post-processing...")
        normalized_file = processing_temp_dir / f"{chapter_name}_normalized.wav"
        success, error_msg = run_loudness_normalization(current_file, normalized_file)
        if success:
            current_file = normalized_file
            print("Loudness normalization successful.")
        else:
            print(f"Warning: Loudness normalization failed. Using raw combined audio. FFMPEG Error:\n{error_msg}")
    else:
        print("4. Skipping post-processing as per configuration.")

    final_mp3_path = output_dir / f"{chapter_name.replace('_', ' ').title()}.mp3"
    print(f"5. Exporting final MP3 to '{final_mp3_path}'...")

    final_audio = AudioSegment.from_wav(current_file)
    final_audio.export(
        final_mp3_path,
        format="mp3",
        bitrate=PROJECT_CONFIG.get('mp3_bitrate', '192k'),
        tags={
            'artist': PROJECT_CONFIG.get('mp3_artist', 'Audiobook Producer'),
            'title': chapter_name.replace('_', ' ').title()
        }
    )

    print("6. Cleaning up temporary processing files...")
    shutil.rmtree(processing_temp_dir)

    duration_min = len(final_audio) / 60000
    print("\n--- Assembly Complete! ---")
    print(f"Final file: {final_mp3_path}")
    print(f"Total duration: {duration_min:.2f} minutes.")