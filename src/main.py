import os
import sys
import torch
import torchaudio
from tqdm import tqdm
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import re
import random
import numpy as np
import string
import difflib
import whisper
import shutil

# --- Configuration ---
# All user-configurable parameters are here.
CONFIG = {
    # --- File & Path Settings ---
    "OUTPUT_DIR": "output",
    "AUDIO_PROMPT_PATH": "data/demo-beck-2.wav",
    "TARGET_LANG": "de",

    # --- Text Splitting Settings ---
    "MIN_CHUNK_LEN": 200,
    "MAX_CHUNK_LEN": 300,

    # --- Generation & Validation Settings ---
    "BASE_SEED": 42,
    "MAX_CANDIDATES": 9,
    "EARLY_ABORT_SUCCESS_COUNT": 2,
    "WHISPER_MODEL_NAME": "base",
    "SCORE_THRESHOLD": 0.90,
}
# --- End Configuration ---


class OptimalTextSplitter:
    """
    Splits text into optimally sized chunks by leveraging preferred split points.
    """
    INFINITY = sys.maxsize

    def __init__(self, min_len=200, max_len=300, costs=None):
        self.min_len = min_len
        self.max_len = max_len
        if costs is None:
            self.costs = {
                'double_newline': 0,
                'single_newline': 10,
                'sentence_end': 100,
                'word_end': 1000,
                'start': 0
            }
        else:
            self.costs = costs

    def _find_potential_splits(self, text):
        splits = {0: 'start'}
        for i in range(1, len(text)):
            if text[i - 1:i + 1] == '\n\n':
                splits[i] = 'double_newline'
            elif text[i - 1] == '\n' and i not in splits:
                splits[i] = 'single_newline'
            elif text[i - 1] in '.!?' and text[i].isspace() and i not in splits:
                splits[i] = 'sentence_end'
            elif text[i - 1].isspace() and not text[i].isspace() and i not in splits:
                splits[i] = 'word_end'
        splits[len(text)] = 'word_end'
        return splits

    def split(self, text):
        if not text:
            return []
        potential_splits = self._find_potential_splits(text)
        split_indices = sorted(potential_splits.keys())
        min_costs = {idx: self.INFINITY for idx in split_indices}
        best_prev_split = {idx: 0 for idx in split_indices}
        min_costs[0] = 0

        for i in range(1, len(split_indices)):
            current_pos = split_indices[i]
            for j in range(i - 1, -1, -1):
                prev_pos = split_indices[j]
                chunk_len = current_pos - prev_pos
                if chunk_len > self.max_len + 100:
                    break
                length_cost = 0
                if chunk_len < self.min_len:
                    length_cost = (self.min_len - chunk_len) ** 2
                elif chunk_len > self.max_len:
                    length_cost = (chunk_len - self.max_len) ** 2
                split_type = potential_splits[current_pos]
                split_cost = self.costs[split_type]
                total_cost = min_costs[prev_pos] + length_cost + split_cost
                if total_cost < min_costs[current_pos]:
                    min_costs[current_pos] = total_cost
                    best_prev_split[current_pos] = prev_pos
        chunks = []
        current_pos = split_indices[-1]
        while current_pos > 0:
            prev_pos = best_prev_split[current_pos]
            chunk = text[prev_pos:current_pos].strip()
            if chunk:
                chunks.insert(0, chunk)
            current_pos = prev_pos
        return chunks


# --- Helper Functions ---

def set_seed(seed: int, device: str):
    """Sets a seed for reproducibility."""
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def derive_seed(base_seed: int, chunk_idx: int, cand_idx: int) -> int:
    """Deterministically derives a unique seed for each generation attempt."""
    if base_seed == 0:
        return random.randint(1, 2 ** 31 - 1)
    mix = (np.uint64(base_seed) * np.uint64(1000003)
           + np.uint64(chunk_idx) * np.uint64(10007)
           + np.uint64(cand_idx) * np.uint64(10009))
    return int(mix & np.uint64(0x7FFFFFFF))


def normalize_for_comparison(text: str) -> str:
    """Normalizes text for Whisper comparison by removing punctuation and lowercasing."""
    text = text.lower().strip()
    return re.sub(f"[{re.escape(string.punctuation)}]", '', text)


def whisper_check(model, audio_path: str, target_text: str) -> tuple[float, str]:
    """Transcribes audio and returns a similarity score against the target text."""
    try:
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1024:
            size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 'N/A'
            return 0.0, f"[ERROR: Invalid audio file at '{audio_path}' (size: {size} bytes)]"

        result = model.transcribe(audio_path, language=CONFIG["TARGET_LANG"], fp16=torch.cuda.is_available())
        transcribed_text = result['text']

        norm_transcribed = normalize_for_comparison(transcribed_text)
        norm_target = normalize_for_comparison(target_text)

        score = difflib.SequenceMatcher(None, norm_transcribed, norm_target).ratio()
        return score, transcribed_text

    except Exception as e:
        return 0.0, f"[ERROR: Whisper check failed - {e}]"


# --- Main Logic ---

def main():
    """Main function to generate and validate audio chunks from a text file."""
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <path_to_text_file>")
        sys.exit(1)

    text_file_path = sys.argv[1]
    if not os.path.exists(text_file_path):
        print(f"Error: Input file not found at {text_file_path}")
        sys.exit(1)

    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    temp_dir = os.path.join(CONFIG["OUTPUT_DIR"], "temp")
    os.makedirs(temp_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Patches torch.load to map tensors to the selected device, avoiding errors when a model
    # was saved on a different device (e.g., loading a CUDA-saved model on a CPU machine).
    original_torch_load = torch.load
    def patched_torch_load(f, map_location=None, **kwargs):
        return original_torch_load(f, map_location=device, **kwargs)
    torch.load = patched_torch_load

    print("Loading TTS model...")
    tts_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    tts_model.prepare_conditionals(CONFIG["AUDIO_PROMPT_PATH"])
    print("TTS model loaded and conditionals prepared.")

    print(f"Loading Whisper model '{CONFIG['WHISPER_MODEL_NAME']}'...")
    whisper_device = "cpu" if device == "mps" else device
    if whisper_device != device:
        print(f"Note: Using {whisper_device} for Whisper to avoid MPS compatibility issues.")
    whisper_model = whisper.load_model(CONFIG["WHISPER_MODEL_NAME"], device=whisper_device)
    print("Whisper model loaded.")

    base_name = os.path.splitext(os.path.basename(text_file_path))[0]
    with open(text_file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    # Regex replace all apostrophes with "
    full_text = re.sub(r"[‚‘’„“]", '"', full_text)

    print("Splitting text into optimal chunks...")
    splitter = OptimalTextSplitter(min_len=CONFIG["MIN_CHUNK_LEN"], max_len=CONFIG["MAX_CHUNK_LEN"])
    text_chunks = splitter.split(full_text)
    print(f"Text split into {len(text_chunks)} chunks. Starting generation...")

    for i, chunk in enumerate(tqdm(text_chunks, desc="Processing Chunks")):
        part_num = i + 1
        final_output_path = os.path.join(CONFIG["OUTPUT_DIR"], f"{base_name}-{part_num:03d}.wav")

        if os.path.exists(final_output_path):
            tqdm.write(f"Skipping Part {part_num}: Final file already exists.")
            continue

        tqdm.write(f"\n--- Part {part_num}/{len(text_chunks)} ---")
        tqdm.write(f"Text (len {len(chunk)}): \"{chunk[:80].replace(chr(10), ' ')}...\"")

        generated_candidates = []
        temp_files_this_chunk = []
        successful_finds = 0

        for cand_idx in range(CONFIG["MAX_CANDIDATES"]):
            tqdm.write(f"  Generating candidate {cand_idx + 1}/{CONFIG['MAX_CANDIDATES']}...")
            seed = derive_seed(CONFIG["BASE_SEED"], i, cand_idx)
            set_seed(seed, device)

            temp_path = os.path.join(temp_dir, f"part_{part_num:03d}_cand_{cand_idx + 1}.wav")
            temp_files_this_chunk.append(temp_path)

            wav_data = tts_model.generate(chunk, language_id=CONFIG["TARGET_LANG"])
            torchaudio.save(temp_path, wav_data, tts_model.sr)
            duration = len(wav_data[0]) / tts_model.sr

            score, transcript = whisper_check(whisper_model, temp_path, chunk)
            ratio = score ** 2 / duration if duration > 0 else 0

            generated_candidates.append({
                "path": temp_path, "duration": duration, "score": score,
                "ratio": ratio, "transcript": transcript
            })

            tqdm.write(f"  > Candidate {cand_idx + 1} | Score: {score:.3f} | Ratio: {ratio:.3f} | Duration: {duration:.2f}s")
            if score >= CONFIG["SCORE_THRESHOLD"]:
                successful_finds += 1

            if successful_finds >= CONFIG["EARLY_ABORT_SUCCESS_COUNT"]:
                tqdm.write(f"  > Aborting early: {successful_finds} candidates passed the threshold.")
                break

        if not generated_candidates:
            tqdm.write(f"ERROR: Part {part_num} | No candidates were generated. Skipping.")
            continue

        generated_candidates.sort(key=lambda x: x['ratio'], reverse=True)
        best_candidate = generated_candidates[0]

        if best_candidate['score'] < CONFIG['SCORE_THRESHOLD']:
            tqdm.write(f"WARNING: Part {part_num} | Best candidate failed threshold.")

        tqdm.write(
            f"SUCCESS: Part {part_num} | Selected best candidate with ratio {best_candidate['ratio']:.3f} "
            f"(Score: {best_candidate['score']:.3f}, Duration: {best_candidate['duration']:.2f}s)."
        )

        try:
            shutil.move(best_candidate['path'], final_output_path)
        except Exception as e:
            tqdm.write(f"ERROR: Could not move best candidate file: {e}. Skipping chunk.")
            continue
        finally:
            for path in temp_files_this_chunk:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError as e:
                        tqdm.write(f"Warning: Could not remove temp file {path}: {e}")

    try:
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)
    except OSError:
        pass

    print("\nAll parts have been processed successfully.")


if __name__ == "__main__":
    main()