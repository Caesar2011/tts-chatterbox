import os
import sys
import pync
import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# A very large number to represent infinity for costs
INFINITY = sys.maxsize


class OptimalTextSplitter:
    """
    Splits text into optimally sized chunks (e.g., 200-300 characters)
    by leveraging preferred split points like paragraphs, sentences, and words.

    This class uses a dynamic programming algorithm to find a globally optimal
    solution, rather than a locally optimal one (greedy approach).
    """

    def __init__(self, min_len=200, max_len=300, costs=None):
        """
        Initializes the splitter with target length and cost heuristics.

        Args:
            min_len (int): The minimum desired length for a chunk.
            max_len (int): The maximum desired length for a chunk.
            costs (dict, optional): A dictionary to override the default costs for split types.
        """
        self.min_len = min_len
        self.max_len = max_len

        # The heuristic: costs for different types of splits. Lower costs are better.
        if costs is None:
            self.costs = {
                'double_newline': 0,  # Perfect split at a paragraph break
                'single_newline': 10,  # Good split at a line break
                'sentence_end': 50,  # Acceptable split at a sentence end
                'word_end': 200,  # Last resort split between words
                'start': 0  # Cost for the very beginning of the text
            }
        else:
            self.costs = costs

    def _find_potential_splits(self, text):
        """
        Analyzes the text to find all potential split points.

        Returns:
            dict: A dictionary mapping character index to split type.
        """
        splits = {0: 'start'}  # The start of the text is always a potential starting point

        for i in range(1, len(text)):
            # Important order: check for more specific cases first!
            if text[i - 1:i + 1] == '\n\n':
                splits[i] = 'double_newline'
            elif text[i - 1] == '\n' and i not in splits:
                splits[i] = 'single_newline'
            elif text[i - 1] in '.!?' and text[i].isspace() and i not in splits:
                # Split after the punctuation mark
                splits[i] = 'sentence_end'
            elif text[i - 1].isspace() and not text[i].isspace() and i not in splits:
                # Split at the start of a word (i.e., after a whitespace)
                splits[i] = 'word_end'

        splits[len(text)] = 'word_end'  # The end of the text is always a potential endpoint
        return splits

    def split(self, text):
        """
        Performs the optimal splitting of the text into chunks.

        Args:
            text (str): The input text to be split.

        Returns:
            list: A list of text chunks.
        """
        if not text:
            return []

        potential_splits = self._find_potential_splits(text)
        # We work with a sorted list of indices for the DP algorithm
        split_indices = sorted(potential_splits.keys())

        # DP arrays:
        # min_costs[i] stores the minimum cost to split the text up to the i-th split point.
        # best_prev_split[i] stores the index of the previous split that led to this minimum cost.
        min_costs = {idx: INFINITY for idx in split_indices}
        best_prev_split = {idx: 0 for idx in split_indices}

        min_costs[0] = 0

        # Dynamic Programming: Find the optimal predecessor for each split point
        for i in range(1, len(split_indices)):
            current_pos = split_indices[i]

            # Iterate backwards through possible previous split points
            for j in range(i - 1, -1, -1):
                prev_pos = split_indices[j]
                chunk_len = current_pos - prev_pos

                # Optimization: If the chunk gets too long, we don't need to look back further
                if chunk_len > self.max_len + 100:  # A buffer
                    break

                # 1. Calculate length cost (quadratic penalty for deviations)
                length_cost = 0
                if chunk_len < self.min_len:
                    length_cost = (self.min_len - chunk_len) ** 2
                elif chunk_len > self.max_len:
                    length_cost = (chunk_len - self.max_len) ** 2

                # 2. Calculate split cost
                split_type = potential_splits[current_pos]
                split_cost = self.costs[split_type]

                # 3. Calculate total cost for this path
                total_cost = min_costs[prev_pos] + length_cost + split_cost

                # 4. If we found a better path, store it
                if total_cost < min_costs[current_pos]:
                    min_costs[current_pos] = total_cost
                    best_prev_split[current_pos] = prev_pos

        # Reconstruction: Build the chunks from the stored best splits, going backwards
        chunks = []
        current_pos = split_indices[-1]  # Start at the end of the text
        while current_pos > 0:
            prev_pos = best_prev_split[current_pos]
            chunk = text[prev_pos:current_pos].strip()
            if chunk:
                chunks.insert(0, chunk)  # Insert at the beginning of the list
            current_pos = prev_pos

        return chunks


def main():
    # Monkey patch torch.load to default to 'cpu' if no map_location is provided
    original_torch_load = torch.load

    def patched_torch_load(f, map_location=None, **kwargs):
        if map_location is None:
            map_location = 'cpu'
        return original_torch_load(f, map_location=map_location, **kwargs)

    torch.load = patched_torch_load

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print("Loading TTS model...")
    multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device='cpu')
    print("Model loaded.")

    # Read German text from data/text.txt
    text_file_path = "data/text.txt"
    if not os.path.exists(text_file_path):
        print(f"Error: {text_file_path} not found.")
        return

    with open(text_file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    # Split the text into optimal chunks
    print("Splitting text into optimal chunks...")
    splitter = OptimalTextSplitter(min_len=200, max_len=300)
    text_chunks = splitter.split(full_text)
    print(f"Text split into {len(text_chunks)} chunks. Starting generation...")

    # Process each chunk individually and save it to a separate file
    for i, chunk in enumerate(text_chunks):
        output_path = f"test-german-part-{i + 1}.wav"
        print(f"\n--- Generating speech for Part {i + 1}/{len(text_chunks)} ---")
        print(f"Text (length {len(chunk)}): \"{chunk[:80].replace(chr(10), ' ')}...\"")

        # Generate speech
        wav_german = multilingual_model.generate(
            chunk,
            language_id="de",
            audio_prompt_path="data/demo-beck-2.wav"
        )

        # Save to file
        ta.save(output_path, wav_german, multilingual_model.sr)
        print(f"Saved speech to {output_path}")
        send_notification("Speech Generation Complete", f"Part {i + 1} saved to {output_path}")

    print("\nAll parts have been processed successfully.")


def send_notification(title, message):
    """Sends a desktop notification on macOS."""
    try:
        pync.notify(
            message,
            title=title,
            appIcon='https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/960px-Python-logo-notext.svg.png'
        )
        print("Notification sent successfully.")
    except Exception as e:
        print(f"Could not send notification (is 'pync' installed and are you on macOS?): {e}")


if __name__ == "__main__":
    main()