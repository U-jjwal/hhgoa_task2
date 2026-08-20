"""Quick check of MSMARCO-XI dataset structure."""
from datasets import load_dataset

# Load without specifying a config
print("Loading with 'default' config...")
dataset = load_dataset("ai4bharat/MSMARCO-XI", split="train[:3]")

print(f"\nFeatures: {dataset.features}")
print(f"Column names: {dataset.column_names}")

for i, example in enumerate(dataset):
    print(f"\n=== Sample {i} ===")
    for key, value in example.items():
        if isinstance(value, dict):
            print(f"  {key} (dict, keys={list(value.keys())})")
            for k, v in value.items():
                if isinstance(v, list):
                    print(f"    {k}: list of {len(v)} items")
                    if v:
                        print(f"      [0]: {str(v[0])[:200]}")
                else:
                    print(f"    {k}: {str(v)[:200]}")
        elif isinstance(value, list):
            print(f"  {key}: list of {len(value)} items")
            if value:
                print(f"    [0]: {str(value[0])[:200]}")
        elif isinstance(value, str):
            print(f"  {key}: {value[:200]}")
        else:
            print(f"  {key}: {value}")
