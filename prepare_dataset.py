import json
import os
import numpy as np
from flan.v2 import task_configs

# These datasets for whatever reason throw various errors or must be downloaded manually.
# I'll handle them outside this script.
MANUALLY_CURATED_DATASETS = ["aeslc", "newsroom", "samsum", "xsum", "story_cloze", "glue_mrpc", "glue_qqp", "winogrande", "para_crawl_enes",
"wmt14_enfr", "wmt16_translate_deen", "wmt16_translate_csen", "wmt16_translate_fien", "wmt16_translate_roen", "wmt16_translate_ruen",
"wmt16_translate_tren", "common_gen", "dart", "e2e_nlg", "web_nlg_en", "cola", "sst2", "mnli_matched", "mnli_mismatched", "qnli",
"wnli", "stsb", "wiki_dialog", "qrecc"]

# Cursed.
NONSERIALIZABLE_CONVERSIONS = {
    bytes: lambda x : x.decode('utf-8'),
    np.int32: int,
    np.int64: int,
    np.float32: float,
    np.float64: float,
}

ALL_CANDIDATE_TASK_CONFIGS = task_configs.ALL_CANDIDATE_TASK_CONFIGS

def _serialize_values(text_dict: dict) -> dict:
    '''Handles non-serializable inputs in the values of a dictionary'''
    for k, v in text_dict.items():
        v_type = type(v)
        # I can't believe this actually works.
        if v_type in NONSERIALIZABLE_CONVERSIONS.keys():
            text_dict[k] = NONSERIALIZABLE_CONVERSIONS[v_type](v)
        elif v_type == np.ndarray:
            # TODO: This is hardcoded for bool_q. Make this not hardcoded for bool_q
            text_dict[k] = [i.decode('utf-8') for i in list(v)]
        elif v_type == dict:
            # Recursion
            text_dict[k] = _serialize_values(v)
    return text_dict

# Create separate folder for data
os.makedirs("train", exist_ok=True)
# Go through all datasets
for name, task in ALL_CANDIDATE_TASK_CONFIGS.items():
    if name in MANUALLY_CURATED_DATASETS:
        print(f"\nDataset {name} must be manually curated otherwise there will be an error, skipping...")
        continue

    print(f"Processing task {name}...")
    dataset = task.source.get_dataset(split="train")
    # Preprocess dataset
    for fn in task.preprocessors:
        dataset = fn(dataset)
    # Dump to .jsonl
    print(f"\nWriting to file {name}.jsonl...\n")
    with open(f'train/{name}.jsonl', 'w', encoding="utf-8") as f:
        for i in dataset.as_numpy_iterator():
            f.write(json.dumps(_serialize_values(i)) + "\n")