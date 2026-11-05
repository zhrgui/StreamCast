# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration, Qwen3VLForConditionalGeneration

from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainerModified
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from datasets import Dataset, DatasetDict

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    temporal: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using temporal GRPO"},
    )
    len_control: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using length reward"},
    )
    data_root: Optional[str] = field(
        default=None,
        metadata={"help": "Directory that the dataset's video paths are relative to"},
    )


def _parse_json_from_text(text):
    """Extract JSON dict from text, following inference_qwen3vl_instruct.py logic."""
    if text is None:
        return None

    s = text.strip()

    # 1) Try to extract a ```json ... ``` block (even if it's not at start).
    m = re.search(r"```(?:json)?\s*\n([\s\S]*?)(?:\n```|$)", s, flags=re.IGNORECASE)
    candidate = m.group(1).strip() if m else s

    # 2) Try loading JSON; if it yields a string that itself is JSON, load again.
    obj = _loads_json_maybe_nested(candidate)
    if isinstance(obj, dict):
        return obj

    # 3) Fallback: find the LAST {...} span anywhere (often the final answer).
    blob = _last_braced_object(candidate) or _last_braced_object(s)
    if blob:
        obj = _loads_json_maybe_nested(blob)
        if isinstance(obj, dict):
            return obj

    return None


def _loads_json_maybe_nested(x):
    if x is None:
        return None
    y = x.strip()
    y = y.strip("`").strip()

    try:
        obj = json.loads(y)
    except Exception:
        return None

    # Handle double-encoded JSON
    if isinstance(obj, str):
        z = obj.strip()
        if (z.startswith("{") and z.endswith("}")) or (z.startswith("[") and z.endswith("]")):
            try:
                return json.loads(z)
            except Exception:
                return obj

    return obj


def _last_braced_object(s):
    """Return the last {...} substring using a simple brace stack scan."""
    stack = []
    spans = []
    for i, ch in enumerate(s):
        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            spans.append((start, i + 1))
    if not spans:
        return None
    start, end = spans[-1]
    return s[start:end].strip()


def _extract_answer_from_text(text):
    """Extract answer from model output after </think>, trying JSON parsing."""
    if not text:
        return ""

    think_match = re.search(r"</think>\s*(.*)", text, re.DOTALL)
    after_think = think_match.group(1).strip() if think_match else text.strip()

    parsed = _parse_json_from_text(after_think)
    if isinstance(parsed, dict):
        for key in ("answer", "prediction", "result", "response"):
            if key in parsed and parsed[key] is not None:
                val = parsed[key]
                return str(val).strip() if isinstance(val, (str, int, float)) else str(val)

    parsed_full = _parse_json_from_text(text)
    if isinstance(parsed_full, dict):
        for key in ("answer", "prediction", "result", "response"):
            if key in parsed_full and parsed_full[key] is not None:
                val = parsed_full[key]
                return str(val).strip() if isinstance(val, (str, int, float)) else str(val)

    return after_think


def accuracy_reward(completions, solution, **kwargs):

    def normalize_number(num_str):
        try:
            num_str = num_str.replace(',', '')
            return float(num_str)
        except Exception as e:
            print(f"Error converting '{num_str}' to float: {e}")
            return None

    def wer(reference, hypothesis):
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        m = len(ref_words)
        n = len(hyp_words)
        d = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1):
            d[i][0] = i
        for j in range(n+1):
            d[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
        return d[m][n] / max(1, m)


    def compute_rouge_score(reference, hypothesis, use_stemmer=True):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
        scores = scorer.score(reference, hypothesis)
        average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
        return average_fmeasure
    

    question_type = kwargs['problem_type'][0]
    
    contents = [completion[0]["content"] for completion in completions]
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []

    for content, sol in zip(contents, solution):
    
        try:
            output_ans = _extract_answer_from_text(content)
            gt_ans = _extract_answer_from_text(sol)
            if question_type == "name_linking":
                # Answer format: {"character_name": "color"} per character
                # For each character in gt, +1 if predicted color matches, else 0. Max = len(gt)
                try:
                    pred_dict = _parse_json_from_text(content) or _parse_json_from_text(output_ans)
                    if not isinstance(pred_dict, dict):
                        pred_dict = {}
                    sol_str = str(sol)
                    gt_dict = {}
                    if sol_str:
                        try:
                            decoded = json.loads(sol_str)
                            gt_dict = json.loads(decoded) if isinstance(decoded, str) else decoded
                        except Exception:
                            gt_dict = _parse_json_from_text(sol_str) or {}
                    if not isinstance(gt_dict, dict):
                        gt_dict = {}
                    reward = 0
                    for char, gt_color in gt_dict.items():
                        if pred_dict.get(char) == gt_color:
                            reward += 1
                    reward = float(reward)  # max = len(gt_dict)
                except Exception:
                    reward = 0.0
            elif question_type == "multiple choice":
                reward = 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
            elif question_type == "numerical":
                gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                out_has_decimal = ("." in output_ans) or ("," in output_ans)
                if gt_has_decimal != out_has_decimal:
                    reward = 0.0
                else:
                    gt_number = normalize_number(gt_ans)
                    out_number = normalize_number(output_ans)
                    if gt_number is None or out_number is None:
                        reward = 0.0
                    else:
                        reward = 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
            elif question_type == "OCR":
                error_rate = wer(gt_ans, output_ans)
                reward = 1 - error_rate
                reward = max(0.0, min(1.0, reward))
            elif question_type == "free-form":
                score = compute_rouge_score(gt_ans, output_ans)
                reward = max(0.0, min(1.0, score))
            elif question_type == "regression":
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    reward = 0.0
                rel_diff = (abs(out_number - gt_number) + 1e-9) / (abs(gt_number) + 1e-9)
                rel_diff = min(1.0, max(0.0, rel_diff))
                reward = 1 - rel_diff
            else:
                reward = 0.0
        except Exception as e:
            print(f"Error in reward_fn for question_type '{question_type}': {e}")
            reward = 0.0
    
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
            
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has the expected format:
    ...reasoning...</think> ```json\n"{...}"\n``` or ...reasoning...</think> {...}
    """
    pattern = r".*</think>\s*(?:```(?:json)?\s*)?\s*\"?\{.*\}\"?\s*(?:```\s*)?"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process is enclosed within <think> </think> tags. After </think>, output ONLY the final answer."
)


def main(script_args, training_args, model_args):
    if not script_args.data_root or not os.path.isdir(script_args.data_root):
        raise ValueError(f"--data_root must be an existing directory, got {script_args.data_root!r}")

    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)


    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    
    QUESTION_TEMPLATE = (
        "{Question}\n"
        "Reason step-by-step before answering. "
        "After </think>, output the final answer."
    )

    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags.",
        "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "name_linking": "",
    }

    def make_conversation_image(example):
        
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }
    
        
    def make_conversation_video(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
    }
        
    def make_conversation_image_and_video(example):
        if example["problem_type"] == 'multiple choice':
            question = example['problem'] + "Options:\n"
            for op in example["options"]:
                question += op + "\n"
        else:
            question = example['problem']

        
        msg ={
            "prompt": 
               [{
                    "role": "user",
                    "content": [
                        {
                            "type": example['data_type'],
                            # example['data_type']: os.getcwd() + "/Video-R1-data" + example['path'][1:]
                        },
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[example['problem_type']]
                        }
                        ]
                }]
            }
        
        return msg

    
    dataset = dataset.map(make_conversation_image_and_video)

    
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )
    
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
