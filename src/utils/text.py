import sys
import re
import string
import difflib

class OptimalTextSplitter:
    """
    Splits text into optimally sized chunks by leveraging preferred split points.
    Adapted from the original tts-chatterbox project.
    """
    INFINITY = sys.maxsize

    def __init__(self, min_len=200, max_len=300, costs=None):
        self.min_len = min_len
        self.max_len = max_len
        self.costs = costs or {
            'double_newline': 0,
            'single_newline': 10,
            'sentence_end': 100,
            'word_end': 1000,
            'start': 0
        }

    def _find_potential_splits(self, text):
        splits = {0: 'start'}
        # Prioritize sentence endings with space after
        sentence_end_pattern = re.compile(r'([.!?])(\s+)')
        for match in sentence_end_pattern.finditer(text):
            splits[match.start(1)] = 'sentence_end'

        # Add other split points if not already a sentence end
        for i in range(1, len(text)):
            if text[i - 1:i + 1] == '\n\n' and i not in splits:
                splits[i] = 'double_newline'
            elif text[i - 1] == '\n' and i not in splits:
                splits[i] = 'single_newline'
            elif text[i - 1] in ',;:' and text[i].isspace() and i not in splits:
                splits[i] = 'word_end' # Treat commas like word ends

        # Finally, add all word ends
        for i in range(1, len(text)):
            if text[i-1].isspace() and not text[i].isspace() and i not in splits:
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
                if chunk_len > self.max_len + 100:  # Optimization
                    break

                length_cost = 0
                if chunk_len < self.min_len:
                    length_cost = (self.min_len - chunk_len) ** 2
                elif chunk_len > self.max_len:
                    length_cost = (chunk_len - self.max_len) ** 2

                split_type = potential_splits.get(current_pos, 'word_end')
                split_cost = self.costs.get(split_type, 1000)

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

def normalize_for_comparison(text: str) -> str:
    """Normalizes text for Whisper comparison."""
    text = text.lower().strip()
    return re.sub(f"[{re.escape(string.punctuation)}]", '', text)

def calculate_similarity(text_a: str, text_b: str) -> float:
    """Calculates the similarity ratio between two normalized strings."""
    norm_a = normalize_for_comparison(text_a)
    norm_b = normalize_for_comparison(text_b)
    return difflib.SequenceMatcher(None, norm_a, norm_b).ratio()
