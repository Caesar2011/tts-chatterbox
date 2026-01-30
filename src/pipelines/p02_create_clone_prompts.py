import json
import os
from pathlib import Path
from tqdm import tqdm
import torch
from utils.models import load_voice_clone_model

def run():
    model = load_voice_clone_model()

    root_path = Path(__file__).parent.parent.parent
    characters_path = root_path / "input/characters.json"

    with open(characters_path, 'r', encoding='utf-8') as f:
        characters = json.load(f)

    for name, data in tqdm(characters.items(), desc="Creating Clone Prompts"):
        clone_prompt_path_str = data['clone_prompt_path']
        output_path = root_path / clone_prompt_path_str

        if os.path.exists(output_path):
            tqdm.write(f"Skipping '{name}': Clone prompt already exists at '{clone_prompt_path_str}'.")
            continue

        ref_audio_path = str(root_path / data['reference_audio_path'])
        if not os.path.exists(ref_audio_path):
            tqdm.write(f"ERROR for '{name}': Reference audio '{ref_audio_path}' not found. Run generate-voices first.")
            continue

        tqdm.write(f"Processing reference for '{name}'...")

        prompt_items = model.create_voice_clone_prompt(
            ref_audio=ref_audio_path,
            ref_text="""Moody löste nun ganz gelassen die Schnüre der großen Säcke, die er mitgebracht hatte. Als er sich wieder aufrichtete, standen sechs Harry Potters keuchend und schnaufend vor ihm. Fred und George wandten sich einander zu und sagten, »Wow, wir sind absolut gleich!« »Ich weiß nicht, aber ich glaube, er sähe immer noch besser aus,« sagte Fred, während er sein Spiegelbild im Wasserkessel musterte. »Maaaaaah!« sagte Fleur, die sich in der Klappe der Mikrowelle begutachtete. »Bill, sieh mich nicht an! Ich bin hässlich!« »Wem seine Klamotten ein wenig zu weit sind, ich hab' hier kleinere«, sagte Moody und deutete auf den ersten Sack. Und umgekehrt. Vergiss nicht, die Brillen in der Seitentasche sind sechs Stück, und wenn ihr angezogen seid, findet ihr in dem anderen Sack Reisegepäck.""",
            #x_vector_only_mode=True
        )

        torch.save(prompt_items, output_path)
        tqdm.write(f"Successfully saved clone prompt for '{name}' to '{clone_prompt_path_str}'.")
