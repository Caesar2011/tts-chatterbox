import json
import os
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
from utils.models import load_voice_design_model

def run():
    model = load_voice_design_model()

    root_path = Path(__file__).parent.parent.parent
    characters_path = root_path / "input/characters.json"
    voices_dir = root_path / "voices"
    os.makedirs(voices_dir, exist_ok=True)

    with open(characters_path, 'r', encoding='utf-8') as f:
        characters = json.load(f)

    for name, data in tqdm(characters.items(), desc="Generating Voice References"):
        output_path = root_path / data['reference_audio_path']

        if os.path.exists(output_path):
            tqdm.write(f"Skipping '{name}': Reference audio already exists at '{output_path}'.")
            continue

        tqdm.write(f"Generating voice for '{name}'...")

        lang = "German"

        wavs, sr = model.generate_voice_design(
            text=data['text'],
            language=lang,
            instruct=data['instruct'],
        )

        sf.write(str(output_path), wavs[0], sr)
        tqdm.write(f"Successfully saved reference for '{name}' to '{output_path}'.")
