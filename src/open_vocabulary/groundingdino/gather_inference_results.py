#!/usr/bin/env python3

import os
import json
from glob import glob

def main(results_dir="./results", output_path="./combined_results.json"):
    # This will accumulate the final data
    final_data = {}

    # Collect all partial result files
    result_files = sorted(glob(os.path.join(results_dir, "results_caption_*.json")))

    for rf in result_files:
        print(f"Merging {rf}...")
        with open(rf, "r", encoding="utf-8") as f:
            partial_data = json.load(f)  # e.g. { image_id: { caption_id: {...} } }

        # Merge partial_data into final_data
        for image_id, caption_dict in partial_data.items():
            if image_id not in final_data:
                final_data[image_id] = {}
            # caption_dict is { caption_id: {...}, caption_id2: {...}, ... }
            for c_id, c_info in caption_dict.items():
                final_data[image_id][c_id] = c_info

    # Write out the merged data
    try:
        with open(output_path, "w") as out:
            json.dump(final_data, out, ensure_ascii=True)
    except:
        with open(output_path, "w", encoding="utf-8") as out:
            json.dump(final_data, out, ensure_ascii=False)

    print(f"Combined results saved to {output_path}")

if __name__ == "__main__":
    main()
