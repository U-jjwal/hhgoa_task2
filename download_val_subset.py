"""
Download a small subset of Hindi data from the MSMARCO-XI validation split.
The validation parquet file (hinval.parquet) is ~461MB, which is small enough to download,
extract 2000-5000 rows, and then delete.
"""

import json
import os
import sys
import pandas as pd
import httpx


VAL_PARQUET_URL = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/hinval.parquet"
TEMP_FILE = "data/temp_hinval.parquet"


def download_val_subset(target_count=5000, output_path="data/hi_subset_5000.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1. Download parquet file
    print(f"Downloading remote validation parquet ({VAL_PARQUET_URL})...")
    
    try:
        with httpx.Client(timeout=300.0, follow_redirects=True) as client:
            with client.stream("GET", VAL_PARQUET_URL) as response:
                if response.status_code != 200:
                    print(f"❌ Failed to download parquet file: HTTP {response.status_code}")
                    return False
                
                total_bytes = int(response.headers.get("content-length", 0))
                bytes_downloaded = 0
                
                with open(TEMP_FILE, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=1024*1024):
                        f.write(chunk)
                        bytes_downloaded += len(chunk)
                        if total_bytes > 0:
                            percent = (bytes_downloaded / total_bytes) * 100
                            if int(percent) % 10 == 0:
                                print(f"  Downloaded: {percent:.1f}% ({bytes_downloaded / 1024 / 1024:.1f} MB / {total_bytes / 1024 / 1024:.1f} MB)")
                        else:
                            print(f"  Downloaded: {bytes_downloaded / 1024 / 1024:.1f} MB")
        
        print("✅ Download finished.")
        
        # 2. Read with pandas
        print("Reading parquet file...")
        df = pd.read_parquet(TEMP_FILE)
        print(f"Loaded parquet. Total rows in file: {len(df)}")
        
        # 3. Extract subset
        df_subset = df.head(target_count)
        print(f"Processing {len(df_subset)} rows...")
        
        all_samples = []
        for idx, row in df_subset.iterrows():
            passages = row.get("passages", {})
            
            translated = None
            english = None
            is_selected = None
            
            if isinstance(passages, dict):
                translated = passages.get("Translated_passages")
                english = passages.get("English_passages")
                is_selected = passages.get("is_selected")
            elif hasattr(passages, "get"):
                translated = passages.get("Translated_passages")
                english = passages.get("English_passages")
                is_selected = passages.get("is_selected")
                
            all_samples.append({
                "query": str(row.get("query", "")),
                "Answer": str(row.get("Answer", "")),
                "query_id": int(row.get("query_id", 0)),
                "query_type": str(row.get("query_type", "")),
                "source_lang": str(row.get("source_lang", "")),
                "target_lang": str(row.get("target_lang", "")),
                "Eng_Query": str(row.get("Eng_Query", "")),
                "Eng_Answer": str(row.get("Eng_Answer", "")),
                "passages": {
                    "Translated_passages": list(translated) if translated is not None and len(translated) > 0 else [],
                    "English_passages": list(english) if english is not None and len(english) > 0 else [],
                    "is_selected": [int(s) for s in is_selected] if is_selected is not None and len(is_selected) > 0 else [],
                },
            })
            
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)
            
        print(f"✅ Successfully extracted {len(all_samples)} samples and saved to {output_path}")
        return True
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Error during download/processing: {e}")
        return False
        
    finally:
        # Clean up temporary file
        if os.path.exists(TEMP_FILE):
            print("Cleaning up temporary parquet file...")
            os.remove(TEMP_FILE)
            print("Temporary file deleted.")


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    download_val_subset(target_count=count)
