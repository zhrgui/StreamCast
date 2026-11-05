import argparse
import json
import os
import torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoConfig
from dataloader import Name_Linking_Dataset_Video
from utils import _parse_json_from_text, _load_existing_indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, default="")
    parser.add_argument("--video_dir", type=str, default="")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-4B-Thinking")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_jsonl", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=7168)
    parser.add_argument("--prompt_id", type=int, default=4)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.checkpoint is None:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(args.model)
    else:
        from_pretrained_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        if args.thinking:
            from_pretrained_kwargs["config"] = AutoConfig.from_pretrained(args.model)
        model = Qwen3VLForConditionalGeneration.from_pretrained(args.checkpoint, **from_pretrained_kwargs)
        processor = AutoProcessor.from_pretrained(args.model if args.thinking else args.checkpoint)

    dataset = Name_Linking_Dataset_Video(args.video_dir, args.csv_file, args.prompt_id)
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    existing_indices = _load_existing_indices(args.output_jsonl)

    indices_to_process = [idx for idx in range(len(dataset)) if idx not in existing_indices]
    out_f = open(args.output_jsonl, "a", encoding="utf-8")
    results = []

    for idx in tqdm(indices_to_process, total=len(indices_to_process), desc="Inference"):
        video_path, prompt = dataset[idx]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        text = output_text[0] if output_text else ""
        parsed_text = text.split("</think>")[-1] if args.thinking else text
        parsed = _parse_json_from_text(parsed_text)
        item = {
            "index": idx,
            "video_path": video_path,
            "response_text": text,
            "response_json": parsed,
        }
        results.append(item)
        out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
        out_f.flush()

    out_f.close()

if __name__ == "__main__":
    main()
