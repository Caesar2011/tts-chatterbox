import argparse
import os
import sys

# Ensure the src directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipelines import p01_generate_voice_references, p02_create_clone_prompts, p03_synthesize_chapter, p04_assemble_chapter

def main():
    parser = argparse.ArgumentParser(
        description="Qwen Audiobook Producer: A pipeline for generating multi-speaker audiobooks."
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Command: generate-voices
    parser_gen_voices = subparsers.add_parser(
        "generate-voices",
        help="Step 1 & 2: Generate reference WAVs and cloneable prompts for all characters in characters.json."
    )
    parser_gen_voices.set_defaults(func=run_voice_generation)

    # Command: synthesize
    parser_synth = subparsers.add_parser(
        "synthesize",
        help="Step 3: Synthesize all audio segments for a specific chapter from its XML file."
    )
    parser_synth.add_argument("chapter_xml", type=str, help="Path to the processed chapter XML file (e.g., 'processed_xml/kapitel-01.xml').")
    parser_synth.set_defaults(func=run_synthesize)

    # Command: assemble
    parser_assemble = subparsers.add_parser(
        "assemble",
        help="Step 4: Assemble all synthesized audio segments for a chapter into a final MP3 file."
    )
    parser_assemble.add_argument("chapter_name", type=str, help="Base name of the chapter to assemble (e.g., 'kapitel-01').")
    parser_assemble.set_defaults(func=run_assemble)

    args = parser.parse_args()
    args.func(args)

def run_voice_generation(args):
    print("--- Running Pipeline Step 1: Generating Voice Reference Audio ---")
    p01_generate_voice_references.run()
    print("\n--- Running Pipeline Step 2: Creating Reusable Clone Prompts ---")
    p02_create_clone_prompts.run()
    print("\n--- Voice Generation Pipeline Complete ---")

def run_synthesize(args):
    print(f"--- Running Pipeline Step 3: Synthesizing Chapter '{args.chapter_xml}' ---")
    p03_synthesize_chapter.run(args.chapter_xml)
    print("\n--- Chapter Synthesis Complete ---")

def run_assemble(args):
    print(f"--- Running Pipeline Step 4: Assembling Chapter '{args.chapter_name}' ---")
    p04_assemble_chapter.run(args.chapter_name)
    print("\n--- Chapter Assembly Complete ---")

if __name__ == "__main__":
    main()
