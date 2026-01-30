import json
import os
import shutil
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from lxml import etree
import soundfile as sf

from utils.config import PROJECT_CONFIG
from utils.models import load_voice_clone_model
from utils.text import OptimalTextSplitter
from utils.audio import load_whisper_model, whisper_check, get_audio_duration

def derive_seed(base_seed, segment_idx, part_idx, cand_idx):
    """Deterministically derives a unique seed for each generation attempt."""
    if base_seed == 0:
        return random.randint(1, 2**31 - 1)
    mix = (np.uint64(base_seed) * np.uint64(1000003) +
           np.uint64(segment_idx) * np.uint64(10007) +
           np.uint64(part_idx) * np.uint64(10009) +
           np.uint64(cand_idx) * np.uint64(10011))
    return int(mix & np.uint64(0x7FFFFFFF))

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def run(chapter_xml_path_str: str):
    """
    Synthesizes an entire chapter from an XML file.

    This function processes an XML file, ensuring no text content is lost. It handles
    speakable tags (any tag with a 'speaker' attribute) and pause tags.

    Expected XML structure:
    <chapter>
      <narration speaker="narrator">This is the chapter title.</narration>
      <pause type="large"/>
      <narration speaker="narrator">An opening sentence for the scene.</narration>
      <dialogue speaker="char1">"A line of dialogue," he said.</dialogue>
      <pause type="small"/>
      <narration speaker="narrator">A concluding sentence.</narration>
    </chapter>

    - Speakable tags like <narration>, <dialogue>, or any custom tag must have a 'speaker' attribute.
    - Pause tags <pause type="..."/> are used to insert silence. Supported types: 'small', 'medium', 'large'.
    """
    root_path = Path(__file__).parent.parent.parent
    chapter_xml_path = Path(chapter_xml_path_str)

    if not chapter_xml_path.exists():
        raise FileNotFoundError(f"Chapter XML not found: {chapter_xml_path}")

    chapter_name = chapter_xml_path.stem
    temp_chapter_dir = root_path / "temp" / chapter_name
    os.makedirs(temp_chapter_dir, exist_ok=True)

    # Load models
    tts_model = load_voice_clone_model()
    whisper_model = load_whisper_model()

    # Load character data and clone prompts
    with open(root_path / "input/characters.json", 'r', encoding='utf-8') as f:
        characters = json.load(f)

    clone_prompts = {}
    print("Loading voice clone prompts...")
    for name, data in characters.items():
        prompt_path = root_path / data['clone_prompt_path']
        if prompt_path.exists():
            clone_prompts[name] = torch.load(prompt_path, map_location=PROJECT_CONFIG['device'], weights_only=False)
        else:
            print(f"Warning: Clone prompt not found for '{name}' at '{prompt_path}'. This speaker will be unavailable.")

    # Parse XML and prepare segments
    tree = etree.parse(str(chapter_xml_path))
    xml_elements = list(tree.getroot())

    splitter = OptimalTextSplitter(
        min_len=PROJECT_CONFIG['min_chunk_len'],
        max_len=PROJECT_CONFIG['max_chunk_len']
    )

    # Pre-calculate total number of synthesis chunks for the progress bar
    total_chunks = 0
    for element in xml_elements:
        if element.get("speaker") is not None and element.text and element.text.strip():
            total_chunks += len(splitter.split(element.text.strip()))

    print(f"Chapter contains {len(xml_elements)} XML elements, with {total_chunks} synthesis chunks to generate.")

    # Main synthesis loop
    with tqdm(total=total_chunks, desc="Synthesizing Chunks") as pbar:
        for seg_idx, element in enumerate(xml_elements):
            seg_num = seg_idx + 1
            tag = element.tag

            # --- Handle Pause Tags ---
            if tag == 'pause':
                pause_type = element.get('type', 'medium')
                pause_marker_path = temp_chapter_dir / f"segment_{seg_num:04d}_pause_{pause_type}.wav"
                if not pause_marker_path.exists():
                    pause_marker_path.touch() # Create 0-byte marker file
                continue

            # --- Handle Speakable Tags ---
            speaker = element.get("speaker")
            text = element.text

            if not speaker or not text or not text.strip():
                if tag != 'pause':
                    tqdm.write(f"Info: Skipping segment {seg_num} (<{tag}>) - no speaker or text content.")
                continue

            if speaker not in clone_prompts:
                tqdm.write(f"ERROR: Speaker '{speaker}' for segment {seg_num} has no clone prompt. Skipping.")
                continue

            text_chunks = splitter.split(text.strip())

            for part_idx, chunk_text in enumerate(text_chunks):
                part_num = part_idx + 1
                pbar.set_description(f"S{seg_num:03d}-P{part_num:02d} ({speaker})")

                final_output_path = temp_chapter_dir / f"segment_{seg_num:04d}_part_{part_num:03d}.wav"
                if final_output_path.exists():
                    pbar.update(1)
                    continue

                generated_candidates = []
                successful_finds = 0

                for cand_idx in range(PROJECT_CONFIG['max_candidates_per_chunk']):
                    seed = derive_seed(PROJECT_CONFIG.get('base_seed', 42), seg_idx, part_idx, cand_idx)
                    set_seed(seed)

                    temp_path = temp_chapter_dir / f"_temp_cand_{seg_idx}_{part_idx}_{cand_idx}.wav"

                    wavs, sr = tts_model.generate_voice_clone(
                        text=chunk_text,
                        language="German",
                        voice_clone_prompt=clone_prompts[speaker]
                    )
                    torch.cuda.empty_cache()
                    sf.write(str(temp_path), wavs[0], sr)

                    score, transcript = whisper_check(whisper_model, str(temp_path), chunk_text)
                    duration = get_audio_duration(str(temp_path))

                    generated_candidates.append({
                        "path": temp_path, "duration": duration, "score": score,
                        "transcript": transcript
                    })

                    if score >= PROJECT_CONFIG['validation_score_threshold']:
                        successful_finds += 1
                    if successful_finds >= PROJECT_CONFIG['early_abort_success_count']:
                        break

                if not generated_candidates:
                    tqdm.write(f"ERROR: No candidates generated for S{seg_num}-P{part_num}. Skipping.")
                    pbar.update(1)
                    continue

                generated_candidates.sort(key=lambda x: x['score'], reverse=True)
                best_candidate = generated_candidates[0]

                if best_candidate['score'] < PROJECT_CONFIG['validation_score_threshold']:
                    tqdm.write(f"WARNING: Best score for S{seg_num}-P{part_num} is {best_candidate['score']:.2f}, below threshold.")
                    tqdm.write(f"  - Target:     '{chunk_text[:80]}...'")
                    tqdm.write(f"  - Transcript: '{best_candidate['transcript'][:80]}...'")

                shutil.move(best_candidate['path'], final_output_path)

                for cand in generated_candidates:
                    if cand['path'] != best_candidate['path'] and os.path.exists(cand['path']):
                        os.remove(cand['path'])

                pbar.update(1)
