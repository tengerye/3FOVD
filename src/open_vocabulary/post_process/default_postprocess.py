"""
The prediction of all models has the following format:
```
{
    image_id: {
        caption_id_1: {
            "boxes": [[x1, y1, x2, y2], ...],
            "words": [...],
            "scores": [...],
            "word_idx": [...],
        },
        caption_id_2: {...},
        ...
    },
    image_id_2: {...},
    ...
}
```

We need to convert this format to the standard COCO-style detection format:
```
[
  {
    "image_id": 1,
    "category_id": 3,
    "bbox": [x1, y1, w, h],
    "score": 0.86,
    "word": "apple",
  },
  ...
]
```
"""
import json
from typing import List


def convert_custom_predictions_to_coco(
    custom_preds,
    top_k=None,
) -> List[dict]:
    """
    Convert a custom prediction format into standard COCO-style detection results.

    Args:
        custom_preds (dict): A dictionary of the form:

        top_k (int): Maximum number of boxes to keep per (image_id, caption_id).

    Returns:
        List[dict]: A list of COCO-style detection dicts.
    """
    coco_results = []

    # Loop over each image
    for img_id, caption_dict in custom_preds.items():

        bbox_array = []
        score_array = []
        words_array = []
        # Loop over each caption in this image
        for cap_id, cap_data in caption_dict.items():
            boxes = cap_data.get("boxes", [])
            scores = cap_data.get("scores", [])
            words = cap_data.get("words", [])

            # If there's a mismatch, skip or handle gracefully
            assert len(boxes) == len(scores), f"Number of boxes and scores must match!"
            bbox_array.extend(boxes)
            score_array.extend(scores)
            words_array.extend(words)

        # Combine boxes and scores for sorting
        zipped_data = list(zip(bbox_array, score_array, words_array))

        # Sort by descending score
        zipped_data.sort(key=lambda x: x[1], reverse=True)

        # Keep only top K
        if top_k is None:
            top_k = len(zipped_data)
        top_data = zipped_data[:top_k]

        # Convert each bounding box from [x1, y1, x2, y2] to [x, y, w, h]
        for (x1, y1, x2, y2), scr, word in top_data:
            w = x2 - x1
            h = y2 - y1

            # Prepare the COCO-format detection
            detection = {
                "image_id": int(img_id),
                # Convert caption_id to integer if possible
                "category_id": int(cap_id),
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(scr),
                "word": word,
            }
            coco_results.append(detection)

    return coco_results


def evaluate():
    pass


# Example usage:
if __name__ == "__main__":
    with open("datasets/3FOVD-RP/test/vild_product_test_prediction_small.json", "r") as f:
        raw_preds = json.load(f)

    pred_coco_format = convert_custom_predictions_to_coco(raw_preds, top_k=300)
    with open("datasets/3FOVD-RP/test/vild_product_test_prediction_small_coco.json", "w") as f:
        json.dump(pred_coco_format, f)
