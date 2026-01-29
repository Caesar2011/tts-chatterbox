import os
import sys
import re
from pydub import AudioSegment
from tqdm import tqdm
import subprocess

# --- Optional Dependency Checks ---
try:
    import pyrnnoise
    import soundfile as sf

    _PYRNNOISE_AVAILABLE = True
except ImportError:
    _PYRNNOISE_AVAILABLE = False

# --- Configuration ---
# All user-configurable parameters are here.
CONFIG = {
    # --- File & Path Settings ---
    "SOURCE_DIR": "output",

    # --- Audio Properties ---
    # MODIFIED: Set this to the sample rate of your TTS model (e.g., 24000, 48000).
    # This is CRITICAL for preventing audio corruption.
    "SAMPLE_RATE": 48000,

    # --- Post-Processing Pipeline ---
    # Set these to False to disable a step.
    "USE_DENOISER": False,  # Requires pyrnnoise-clone and soundfile
    "USE_AUTO_EDITOR": False,  # Requires auto-editor
    "USE_LOUDNESS_NORMALIZATION": True,  # Requires ffmpeg

    # --- auto-editor Settings ---
    "AE_SILENCE_THRESHOLD": "5%",  # Volume below which is considered silence.
    "AE_MARGIN": "0.2s",  # Margin to leave around detected audio.

    # --- EBU R128 Loudness Normalization Settings ---
    "EBU_TARGET_LOUDNESS": -23.0,  # Target Integrated Loudness in LUFS (for spoken word).
    "EBU_TRUE_PEAK": -2.0,  # Max True Peak in dBTP.
    "EBU_LOUDNESS_RANGE": 7.0,  # Loudness Range in LU.

    # --- MP3 Export Settings ---
    "MP3_BITRATE": "192k",
    "MP3_ARTIST": "TTS-Chatterbox",
}


# --- End Configuration ---


def run_denoiser(input_path: str, output_path: str) -> bool:
    """
    Applies RNNoise denoiser to an audio file.
    Returns True on success, False on failure.
    """
    if not _PYRNNOISE_AVAILABLE:
        print("Warning: Denoising skipped. 'pyrnnoise-clone' or 'soundfile' not installed.")
        return False

    print("Applying RNNoise denoiser...")
    try:
        denoiser = pyrnnoise.RNNoise(sample_rate=CONFIG["SAMPLE_RATE"])
        for _ in denoiser.denoise_wav(input_path, output_path):
            pass
        return True
    except Exception as e:
        print(f"Error during denoising: {e}")
        return False


def run_auto_editor(input_path: str, output_path: str):
    """
    Applies auto-editor to intelligently trim silence from an audio file.
    """
    print("Applying auto-editor for intelligent silence trimming...")
    command = [
        "auto-editor",
        input_path,
        #"--edit", f"audio:threshold={CONFIG['AE_SILENCE_THRESHOLD']}",
        #"--margin", CONFIG['AE_MARGIN'],
        #"--sample-rate", str(CONFIG["SAMPLE_RATE"]),
        "-o", output_path
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError("auto-editor not found. Please install it (`pip install auto-editor`).")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"auto-editor failed with error:\n{e.stderr}")


def run_loudness_normalization(input_path: str, output_path: str):
    """
    Applies EBU R128 loudness normalization using ffmpeg.
    """
    print("Applying EBU R128 loudness normalization with ffmpeg...")
    loudnorm_filter = (
        f"loudnorm=I={CONFIG['EBU_TARGET_LOUDNESS']}:"
        f"TP={CONFIG['EBU_TRUE_PEAK']}:"
        f"LRA={CONFIG['EBU_LOUDNESS_RANGE']}:"
    )
    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-af", loudnorm_filter,
        "-ar", str(CONFIG["SAMPLE_RATE"]),
        output_path
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please ensure it is installed and in your system's PATH.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg normalization failed with error:\n{e.stderr}")


def combine_audio_parts(base_name: str):
    """
    Finds, verifies, combines, and post-processes audio parts into a final file.
    """
    print(f"--- Starting combination process for '{base_name}' ---")

    # 1. Find and verify all generated chunk files
    print(f"1. Searching for valid WAV files in '{CONFIG['SOURCE_DIR']}'...")
    try:
        all_files = os.listdir(CONFIG['SOURCE_DIR'])
    except FileNotFoundError:
        print(f"Error: Source directory '{CONFIG['SOURCE_DIR']}' not found.")
        return

    file_pattern = re.compile(re.escape(base_name) + r'-(\d{3,})\.wav')
    valid_files = [f for f in all_files if file_pattern.match(f)]
    valid_files.sort()

    if not valid_files:
        print("Error: No valid WAV files found for this base name. Nothing to do.")
        return

    print(f"Found {len(valid_files)} valid parts. Verifying sequence...")
    part_numbers = [int(file_pattern.match(f).group(1)) for f in valid_files]
    if min(part_numbers) != 1 or set(part_numbers) != set(range(1, len(part_numbers) + 1)):
        missing = sorted(list(set(range(1, len(part_numbers) + 1)) - set(part_numbers)))
        print(f"Error: Sequence is incomplete! Expected {len(part_numbers)} parts. Missing: {missing}")
        return
    print("Sequence verification successful.")

    # 2. Concatenate all files into a single raw WAV
    print("\n2. Concatenating audio parts...")
    combined_audio = AudioSegment.empty()
    for filename in tqdm(valid_files, desc="Combining"):
        segment = AudioSegment.from_wav(os.path.join(CONFIG['SOURCE_DIR'], filename))
        combined_audio += segment

    # 3. Run the post-processing pipeline on temporary files
    print("\n3. Starting post-processing pipeline...")
    temp_dir = os.path.join(CONFIG['SOURCE_DIR'], "temp_processing")
    os.makedirs(temp_dir, exist_ok=True)

    current_file = os.path.join(temp_dir, f"{base_name}_raw_combined.wav")
    combined_audio.export(current_file, format="wav")

    if CONFIG['USE_DENOISER']:
        denoised_file = os.path.join(temp_dir, f"{base_name}_denoised.wav")
        if run_denoiser(current_file, denoised_file):
            current_file = denoised_file
        else:
            print("Skipping denoise step due to error.")

    if CONFIG['USE_AUTO_EDITOR']:
        trimmed_file = os.path.join(temp_dir, f"{base_name}_trimmed.wav")
        try:
            run_auto_editor(current_file, trimmed_file)
            current_file = trimmed_file
        except Exception as e:
            print(f"Warning: auto-editor step failed. Continuing without it. Error: {e}")

    if CONFIG['USE_LOUDNESS_NORMALIZATION']:
        normalized_file = os.path.join(temp_dir, f"{base_name}_normalized.wav")
        try:
            run_loudness_normalization(current_file, normalized_file)
            current_file = normalized_file
        except Exception as e:
            print(f"Warning: Loudness normalization step failed. Continuing without it. Error: {e}")

    # 4. Export the final, processed audio to MP3
    output_mp3_path = f"{base_name}.mp3"
    print(f"\n4. Exporting final processed audio to MP3: '{output_mp3_path}'...")

    try:
        final_audio = AudioSegment.from_wav(current_file)
        final_audio.export(
            output_mp3_path,
            format="mp3",
            bitrate=CONFIG['MP3_BITRATE'],
            tags={'artist': CONFIG['MP3_ARTIST'], 'title': base_name.replace('_', ' ').title()}
        )
        final_duration_minutes = len(final_audio) / 60000
        print("\n--- Success! ---")
        print(f"File '{output_mp3_path}' created successfully.")
        print(f"Total duration: {final_duration_minutes:.2f} minutes.")
    except Exception as e:
        print(f"\n--- Error during final export! --- \nDetails: {e}")
    finally:
        # 5. Clean up temporary processing files
        print("Cleaning up temporary files...")
        for file_in_temp in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file_in_temp))
        os.rmdir(temp_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <base_name>")
        print("Example: python combine_audio.py kapitel-1")
        sys.exit(1)

    input_base_name = sys.argv[1]
    combine_audio_parts(input_base_name)