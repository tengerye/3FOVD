import argparse
import json
import os
import subprocess
import warnings
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_size', type=int, required=True,
                        help='Total number of machines/nodes.')
    parser.add_argument('--node_idx', type=int, required=True,
                        help='Index of the current machine (0-based).')
    parser.add_argument('--text_embed_file', type=str, required=True,
                        help='Path to the text embeddings JSON file.')
    parser.add_argument('--image_base_dir', type=str, required=True,
                        help='Base directory for images.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size per node for inference.')
    parser.add_argument('--out_dir', type=str, default='./distributed_out',
                        help='Directory to store this node\'s output JSON files.')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load caption IDs
    with open(args.text_embed_file, 'r') as f:
        text_embeddings = json.load(f)

    caption_id_list = list(text_embeddings.keys())
    total_length = len(caption_id_list)
    print(f"Total caption candidates: {total_length}")
    
    # Select only relevant candidates for this node
    selected_captions = [caption_id for i, caption_id in enumerate(caption_id_list) if i % args.node_size == args.node_idx]
    selected_count = len(selected_captions)
    print(f"[Node {args.node_idx}] Processing {selected_count} captions out of {total_length}.")
    
    # Progress bar setup
    for caption_id in tqdm(selected_captions, desc=f"Node {args.node_idx} Progress", unit="caption"):
        out_file_this_caption = os.path.join(args.out_dir, f"{caption_id}.json")
        
        # Skip if file already exists
        if os.path.exists(out_file_this_caption):
            print(f"{out_file_this_caption} already exists. Skipping...")
            continue

        # Build the command to run main.py for this single caption
        cmd = [
            "python", "main.py",
            "--caption_idx", str(caption_id),
            "--image_base_dir", args.image_base_dir,
            "--out_file", out_file_this_caption,
            "--batch_size", str(args.batch_size),
            "--text_embed_file", args.text_embed_file,
        ]

        # Suppress warnings and run the command
        try:
            print(f"Executing {' '.join(cmd)}")
            subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"[Node {args.node_idx}] Error processing caption {caption_id}: {e}")
    
    print(f"[Node {args.node_idx}] Processing complete.")

if __name__ == "__main__":
    # Suppress warnings globally
    warnings.filterwarnings("ignore")
    main()
