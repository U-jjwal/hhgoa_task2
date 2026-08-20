"""
Download a small subset of Hindi data from MSMARCO-XI.
Streams only the Hindi parquet file, ensuring fast download and low disk/memory usage.
"""

import json
import os
import sys
from datasets import load_dataset


def download_hindi_subset(target_count=1000, output_path="data/hi_subset_5000.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Streaming Hindi data directly from hintrain.parquet...")
    print(f"Target count: {target_count}")
    
    try:
        # Pass data_files directly to load only the Hindi parquet file in streaming mode
        dataset = load_dataset(
            "ai4bharat/MSMARCO-XI",
            data_files={"train": "train/hintrain.parquet"},
            split="train",
            streaming=True
        )
        
        all_samples = []
        for i, example in enumerate(dataset):
            if i >= target_count:
                break
                
            passages = example.get("passages", {})
            all_samples.append({
                "query": str(example.get("query", "")),
                "Answer": str(example.get("Answer", "")),
                "query_id": int(example.get("query_id", 0)),
                "query_type": str(example.get("query_type", "")),
                "source_lang": str(example.get("source_lang", "")),
                "target_lang": str(example.get("target_lang", "")),
                "Eng_Query": str(example.get("Eng_Query", "")),
                "Eng_Answer": str(example.get("Eng_Answer", "")),
                "passages": {
                    "Translated_passages": list(passages.get("Translated_passages", [])),
                    "English_passages": list(passages.get("English_passages", [])),
                    "is_selected": [int(s) for s in passages.get("is_selected", [])],
                },
            })
            
            if (i + 1) % 100 == 0:
                print(f"  Loaded {i + 1} samples...")
                
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)
            
        print(f"✅ Successfully downloaded {len(all_samples)} Hindi samples to {output_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to download Hindi subset: {e}")
        return False


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    download_hindi_subset(target_count=count)
