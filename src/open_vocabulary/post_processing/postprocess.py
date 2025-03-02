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
from functools import partial
import json
from typing import List, Tuple


def remove_covered_boxes(
        bboxes, 
        scores, 
        threshold=0.5,
    ) -> List[int]:
    """
    Remove boxes that are covered by a higher-confidence box beyond the given threshold.
    Also returns the original indices of the kept boxes.
    
    Args:
        bboxes (list of lists or tuples): Each element is [x1, y1, x2, y2].
        scores (list of float): Confidence scores, must be the same length as bboxes.
        threshold (float): If a box is covered by another box with a higher score 
                           beyond this fraction of its area, it will be removed.

    Returns:
        (final_bboxes, final_scores, final_indices):
            final_bboxes: Filtered bounding boxes.
            final_scores: Corresponding confidence scores.
            final_indices: Indices (in the original order) of the kept bounding boxes.
    """
    assert len(bboxes) == len(scores), \
        "Length of bboxes and scores must match."

    def area(box):
        """Compute area of a bounding box [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)

    def intersection_area(box_a, box_b):
        """Compute intersection area between two boxes."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        
        return inter_w * inter_h

    # Sort indices by descending confidence
    indices_sorted_by_score = sorted(range(len(bboxes)), key=lambda i: scores[i], reverse=True)
    
    keep_indices = []
    
    for idx in indices_sorted_by_score:
        box = bboxes[idx]
        box_area = area(box)
        
        covered = False
        for kept_idx in keep_indices:
            inter_area = intersection_area(bboxes[kept_idx], box)
            cover_ratio = inter_area / float(box_area) if box_area > 0 else 0
            
            # If current box is covered beyond threshold by a higher confidence box, we skip it
            if cover_ratio >= threshold and scores[kept_idx] >= scores[idx]:
                covered = True
                break
        
        if not covered:
            keep_indices.append(idx)
    
    # Sort 'keep_indices' so that final results follow the original input order
    keep_indices.sort()

    return keep_indices


def convert_custom_predictions_to_coco(
    custom_preds,
    pp_func: callable = None,
    top_k=None,
) -> List[dict]:
    """
    Convert a custom prediction format into standard COCO-style detection results.

    Args:
        custom_preds (dict): A dictionary of the form:
        pp_func (callable): A post-processing function to return indices of kept boxes.
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

            if pp_func is not None:
                # Advanced post-processing.
                indices = remove_covered_boxes(boxes, scores, threshold=0.8)
                boxes = [boxes[i] for i in indices]
                words = [words[i] for i in indices]
                scores = [scores[i] for i in indices]

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
    with open("datasets/3FOVD-RP/test/vild_product_test_prediction.json", "r") as f:
        raw_preds = json.load(f)

    pred_coco_format = convert_custom_predictions_to_coco(raw_preds, pp_func=partial(remove_covered_boxes, threshold=0.8))
    with open("datasets/3FOVD-RP/test/vild_product_test_prediction_coco_pp.json", "w") as f:
        json.dump(pred_coco_format, f)
