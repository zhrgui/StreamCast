"""Extract PERSON named entities from transcript segments.

For each JSON file in --subtitle_dir (a list of {"text": ...} entries, as
produced by transcribe_parakeet.py / transcribe_whisperx.py), runs a spaCy
NER model and writes a parallel file under --output_dir keeping only entries
that contain at least one PERSON mention, with an added "names" list.
"""

import argparse
import json
import os

import spacy
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subtitle_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--spacy_model", type=str, default="en_core_web_trf",
                        help='e.g. "en_core_web_trf" or "zh_core_web_trf".')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    model = spacy.load(args.spacy_model)

    for filename in sorted(os.listdir(args.subtitle_dir)):
        with open(os.path.join(args.subtitle_dir, filename), "r") as f:
            subtitles = json.load(f)

        save_data = []
        for entry in tqdm(subtitles, desc=filename, leave=False):
            doc = model(entry.get("text", ""))
            names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            if names:
                entry = dict(entry, names=names)
                save_data.append(entry)

        with open(os.path.join(args.output_dir, filename), "w") as f:
            json.dump(save_data, f, indent=4)


if __name__ == "__main__":
    main()
