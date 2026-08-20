"""
Dataset exploration script for MSMARCO-XI Hindi.

The dataset has per-language parquet files.
We use streaming mode to avoid downloading the entire 3.7GB Hindi file.
We also use trust_remote_code=True to use the custom loading script.
"""

from datasets import load_dataset
import json
import os


def explore_dataset(language="hi", num_samples=5, save_sample=True):
    """Load and explore the MSMARCO-XI dataset using streaming."""
    
    print("=" * 60)
    print(f"MSMARCO-XI Dataset Exploration (Language: {language})")
    print("=" * 60)
    
    # Try loading with config name (uses the custom loading script)
    print(f"\nLoading {num_samples} samples via streaming...")
    try:
        dataset = load_dataset(
            "ai4bharat/MSMARCO-XI",
            language,
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Config-based loading failed: {e}")
        print("Trying default config with streaming...")
        dataset = load_dataset(
            "ai4bharat/MSMARCO-XI",
            split="train",
            streaming=True,
        )
    
    # Collect samples
    samples = []
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        samples.append(example)
        
        print(f"\n{'='*40}")
        print(f"=== Sample {i+1} ===")
        print(f"{'='*40}")
        
        # Print all top-level keys
        for key in example.keys():
            if key == "passages":
                passages = example["passages"]
                print(f"\nPassages:")
                print(f"  Keys: {list(passages.keys())}")
                
                translated = passages.get("Translated_passages", [])
                is_selected = passages.get("is_selected", [])
                english = passages.get("English_passages", [])
                
                print(f"  Number of passages: {len(translated)}")
                for j in range(min(3, len(translated))):
                    selected = is_selected[j] if j < len(is_selected) else 0
                    passage = translated[j] if j < len(translated) else ""
                    eng_passage = english[j] if j < len(english) else ""
                    status = "✅ SELECTED" if selected else "  "
                    preview = passage[:200] + "..." if len(passage) > 200 else passage
                    print(f"  [{j}] {status} ({len(passage)} chars): {preview}")
            elif key == "meta":
                print(f"\nmeta: {example[key]}")
            else:
                value = example[key]
                if isinstance(value, str) and len(value) > 300:
                    value = value[:300] + "..."
                print(f"\n{key}: {value}")
    
    # Save sample for prototyping
    if save_sample and samples:
        os.makedirs("data", exist_ok=True)
        output_path = f"data/sample_{language}.json"
        
        # Convert to serializable format
        serializable_samples = []
        for example in samples:
            sample = {}
            for key, value in example.items():
                if key == "passages":
                    sample["passages"] = {
                        "Translated_passages": value.get("Translated_passages", []),
                        "English_passages": value.get("English_passages", []),
                        "is_selected": [int(s) for s in value.get("is_selected", [])],
                    }
                elif key == "meta":
                    sample["meta"] = {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in value.items()}
                else:
                    sample[key] = value
            serializable_samples.append(sample)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_samples, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Sample saved to {output_path}")
    
    # Statistics from collected samples
    if samples:
        print(f"\n{'='*60}")
        print("DATASET STATISTICS (from collected samples)")
        print(f"{'='*60}")
        
        total_passages = 0
        selected_passages = 0
        passage_lengths = []
        query_lengths = []
        
        for example in samples:
            query_lengths.append(len(example.get("query", "")))
            translated = example["passages"]["Translated_passages"]
            is_selected = example["passages"]["is_selected"]
            total_passages += len(translated)
            selected_passages += sum(1 for s in is_selected if s)
            passage_lengths.extend([len(p) for p in translated])
        
        print(f"  Samples: {len(samples)}")
        print(f"  Total passages: {total_passages}")
        print(f"  Selected passages: {selected_passages}")
        print(f"  Avg passages per query: {total_passages / len(samples):.1f}")
        if passage_lengths:
            print(f"  Avg passage length: {sum(passage_lengths) / len(passage_lengths):.0f} chars")
            print(f"  Min passage length: {min(passage_lengths)} chars")
            print(f"  Max passage length: {max(passage_lengths)} chars")
        if query_lengths:
            print(f"  Avg query length: {sum(query_lengths) / len(query_lengths):.0f} chars")
    
    return samples


def download_subset(language="hi", num_samples=5000):
    """Download a subset of the dataset for local use."""
    
    print(f"Downloading {num_samples} samples of {language} data...")
    
    try:
        dataset = load_dataset(
            "ai4bharat/MSMARCO-XI",
            language,
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception:
        dataset = load_dataset(
            "ai4bharat/MSMARCO-XI",
            split="train",
            streaming=True,
        )
    
    samples = []
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        samples.append(example)
        if (i + 1) % 500 == 0:
            print(f"  Downloaded {i + 1} samples...")
    
    os.makedirs("data", exist_ok=True)
    output_path = f"data/{language}_subset_{num_samples}.json"
    
    # Convert to serializable
    serializable = []
    for ex in samples:
        item = {}
        for key, value in ex.items():
            if key == "passages":
                item["passages"] = {
                    "Translated_passages": value.get("Translated_passages", []),
                    "English_passages": value.get("English_passages", []),
                    "is_selected": [int(s) for s in value.get("is_selected", [])],
                }
            elif key == "meta":
                item["meta"] = {k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in value.items()}
            else:
                item[key] = value
        serializable.append(item)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False)
    
    print(f"✅ Saved {len(samples)} samples to {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    
    return output_path


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        num = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
        download_subset(language="hi", num_samples=num)
    else:
        explore_dataset(language="hi", num_samples=5)
