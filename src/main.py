import os
import sys
import pync
import torch
import torchaudio as ta
from tqdm import tqdm
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# A very large number to represent infinity for costs
INFINITY = sys.maxsize


class OptimalTextSplitter:
    """
    Splits text into optimally sized chunks by leveraging preferred split points.
    """

    def __init__(self, min_len=200, max_len=300, costs=None):
        self.min_len = min_len
        self.max_len = max_len
        if costs is None:
            self.costs = {
                'double_newline': 0,
                'single_newline': 10,
                'sentence_end': 50,
                'word_end': 200,
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
        min_costs = {idx: INFINITY for idx in split_indices}
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


def main():
    # --- Configuration ---
    # File Generation
    OUTPUT_DIR = "output"
    AUDIO_PROMPT_PATH = "data/demo-beck-2.wav"
    TARGET_LANG = "de"
    ERROR_SUFFIX = "-error"

    # Text Splitting
    MIN_CHUNK_LEN = 200
    MAX_CHUNK_LEN = 300

    # Error Handling & Retries
    MAX_RETRIES = 3
    SIZE_THRESHOLD_PERCENT = 20
    NORMAL_CHARS_FOR_SIZE = 280
    NORMAL_SIZE_MB = 1.9
    # --- End Configuration ---

    # Check for command-line argument
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <path_to_text_file>")
        sys.exit(1)

    text_file_path = sys.argv[1]
    if not os.path.exists(text_file_path):
        print(f"Error: Input file not found at {text_file_path}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Monkey patch torch.load to default to 'cpu'
    original_torch_load = torch.load

    def patched_torch_load(f, map_location=None, **kwargs):
        if map_location is None: map_location = 'cpu'
        return original_torch_load(f, map_location=map_location, **kwargs)

    torch.load = patched_torch_load

    # Setup model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("Loading TTS model...")
    multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device='cpu')
    print("Model loaded.")

    base_name = os.path.splitext(os.path.basename(text_file_path))[0]
    with open(text_file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    # Split text
    print("Splitting text into optimal chunks...")
    splitter = OptimalTextSplitter(min_len=MIN_CHUNK_LEN, max_len=MAX_CHUNK_LEN)
    text_chunks = splitter.split(full_text)
    print(f"Text split into {len(text_chunks)} chunks. Starting generation...")

    # Process each chunk with a tqdm progress bar
    for i, chunk in enumerate(tqdm(text_chunks, desc="Generating Audio Parts")):
        part_num = i + 1
        part_str = f"{part_num:03d}"

        output_filename = f"{base_name}-{part_str}.wav"
        error_filename = f"{base_name}-{part_str}{ERROR_SUFFIX}.wav"

        output_path = os.path.join(OUTPUT_DIR, output_filename)
        error_path = os.path.join(OUTPUT_DIR, error_filename)

        # Skip if a final version (normal or error) already exists
        if os.path.exists(output_path):
            tqdm.write(f"Skipping Part {part_num}: {output_filename} already exists.")
            continue
        if os.path.exists(error_path):
            tqdm.write(f"Skipping Part {part_num}: Error file {error_filename} exists. Please resolve manually.")
            continue

        tqdm.write(f"--- Processing Part {part_num}/{len(text_chunks)} ---")
        tqdm.write(f"Text (length {len(chunk)}): \"{chunk[:80].replace(chr(10), ' ')}...\"")

        # Retry loop for generation and size check
        for attempt in range(1, MAX_RETRIES + 1):
            tqdm.write(f"Attempt {attempt}/{MAX_RETRIES}...")

            wav_german = multilingual_model.generate(
                chunk,
                language_id=TARGET_LANG,
                audio_prompt_path=AUDIO_PROMPT_PATH
            )
            ta.save(output_path, wav_german, multilingual_model.sr)

            # File Size Check
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            expected_size_mb = (len(chunk) / NORMAL_CHARS_FOR_SIZE) * NORMAL_SIZE_MB
            max_allowed_size_mb = expected_size_mb * (1 + SIZE_THRESHOLD_PERCENT / 100)

            if file_size_mb <= max_allowed_size_mb:
                tqdm.write(f"SUCCESS: Part {part_num} generated successfully (Size: {file_size_mb:.2f} MB).")
                send_notification("Speech Generation Success", f"Part {part_num} saved to {output_filename}")
                break
            else:
                tqdm.write(
                    f"WARNING: Attempt {attempt} failed. Size {file_size_mb:.2f} MB > max allowed {max_allowed_size_mb:.2f} MB.")
                if attempt == MAX_RETRIES:
                    tqdm.write(f"ERROR: Part {part_num} failed after {MAX_RETRIES} attempts. Renaming to error file.")
                    os.rename(output_path, error_path)
                    send_notification("Speech Generation FAILED", f"Part {part_num} saved as {error_filename}")

    print("\nAll parts have been processed.")


def send_notification(title, message):
    """Sends a desktop notification on macOS."""
    try:
        pync.notify(
            message,
            title=title,
            appIcon='https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/960px-Python-logo-notext.svg.png'
        )
    except Exception:
        # Silently fail if notification doesn't work, but log to console.
        # tqdm.write is not available here, so we use print.
        # print(f"Could not send notification: {e}")
        pass


if __name__ == "__main__":
    main()
