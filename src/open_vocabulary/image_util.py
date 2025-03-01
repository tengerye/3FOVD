from collections import defaultdict, Counter
import json
import random
from tkinter import W
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from pyparsing import annotations


def visualize_predictions_vanilla(
    image: np.ndarray,
    bboxes: List,
    labels: List,
    scores: List,
    label_colors: Dict = None,
) -> np.ndarray:
    """
    Visualize prediction results using OpenCV.
    args:
        bboxes: list of [x1, y1, x2, y2].
    """
    # We will map each label to a unique color
    if label_colors is None:
        label_colors = {}

    # Loop through each detected instance
    for idx, (bbox, label, score) in enumerate(zip(bboxes, labels, scores)):

        # If this label doesn't have a color yet, assign a random color
        if label not in label_colors:
            # (B, G, R) format in OpenCV, each in [0..255]
            label_colors[label] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )

        # Convert bounding box coordinates to int
        x1, y1, x2, y2 = map(int, bbox)

        # Draw the bounding box
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color=label_colors[label],
            thickness=6
        )

        # Prepare text: "label (score)"
        text = f"{label} ({score:.2f})"
        # For the text location, we’ll offset a bit from the top-left corner of the bounding box
        text_position = (x1, y1 - 5)

        # Draw text (label + score) above the bounding box
        cv2.putText(
            image,
            text,
            text_position,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=label_colors[label],
            thickness=3
        )


def visualize_predictions_grouped(
    image: np.ndarray,
    bboxes: List[List[float]],
    labels: List[str],
    scores: List[float],
    label_colors: Dict[str, Tuple[int, int, int]] = None,
) -> np.ndarray:
    """
    Visualize prediction results by grouping bounding boxes that have
    the same position and size. For each group, draw a single box,
    and put text as: "NUM_OF_BBOXES | label1, label2, label3" (up to 3 labels).

    Args:
        image (np.ndarray): BGR image array to draw on.
        bboxes (List[List[float]]): List of boxes in [x1, y1, x2, y2] format.
        labels (List[str]): List of labels corresponding to each box.
        scores (List[float]): List of scores corresponding to each box.
        label_colors (Dict[str, Tuple[int,int,int]]):
            Optional dictionary mapping label -> (B, G, R) color.

    Returns:
        np.ndarray: Annotated image (in place).
    """
    if label_colors is None:
        label_colors = {}

    # Step 1: Group bounding boxes by identical coordinates
    # We'll use a dict with key = (x1, y1, x2, y2) and value = list of (label, score)
    grouped_data = defaultdict(list)

    for bbox, label, score in zip(bboxes, labels, scores):
        x1, y1, x2, y2 = map(int, bbox)
        grouped_data[(x1, y1, x2, y2)].append((label, score))

    # Step 2: For each group, draw one bounding box, and build the text
    for (x1, y1, x2, y2), items in grouped_data.items():
        # We'll pick the color based on the *first* label in the group
        # (You could choose to do something else, like a random color for each group)
        first_label = items[0][0]
        if first_label not in label_colors:
            label_colors[first_label] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
        color = label_colors[first_label]

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=2)

        # Build the text:
        # Example: "3| cat, dog, horse" if we have 3 boxes, with labels cat, dog, horse
        # We'll only show up to 3 labels to keep it short
        num_boxes = len(items)
        # Collect the distinct labels or just all labels in order:
        label_list = [itm[0] for itm in items]
        # If you want them distinct: label_list = list(set(label_list))
        # Show only the first three in the text
        top_three = Counter(label_list).most_common(3)
        label_text = ", ".join([label for label, _ in top_three])
        text = f"{num_boxes}| {label_text}"

        text_position = (x1, max(0, y1 - 5))
        # Put the text
        cv2.putText(
            image,
            text,
            text_position,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=color,
            thickness=2
        )

    return image


def visualize_coco_predictions_on_image(
    image: np.ndarray,
    predictions: List[Dict],
    category_id_to_name: Dict[int, str],
    label_colors: Dict = None,
) -> np.ndarray:
    """
    Visualize all prediction results for a single image in COCO format.

    Args:
        image_path (str): Path to the image file.
        predictions (List[Dict]): A list of detections in COCO format, each dict containing:
            {
                "image_id": int,
                "category_id": int,
                "bbox": [x, y, w, h],
                "score": float
            }
        category_id_to_name (Dict[int, str]): Mapping from category_id to category name.
        label_colors (Dict): Optionally pass a dict to specify or store colors. 
            { "cat_label": (B, G, R), ... }
        show (bool): Whether to display the image in a window via cv2.imshow. 
            If False, returns the annotated image without showing.

    Returns:
        np.ndarray: The annotated image array (BGR).
    """
    # Prepare arrays for bounding boxes, labels, scores
    bboxes = []
    labels = []
    scores = []

    # Convert each COCO prediction (bbox in [x, y, w, h]) to the form [x1, y1, x2, y2]
    for pred in predictions:
        # COCO boxes are [x, y, w, h]
        x, y, w, h = pred["bbox"]
        x1, y1, x2, y2 = x, y, x + w, y + h

        bboxes.append([x1, y1, x2, y2])

        # Map category_id to label string
        cat_id = pred["category_id"]
        if cat_id in category_id_to_name:
            label = category_id_to_name[cat_id]
        else:
            # fallback if no name is found
            label = f"cat_{cat_id}"

        labels.append(label)
        scores.append(pred["score"])

    # Now call your existing visualization function
    annotated_image = visualize_predictions_vanilla(
        image=image,
        bboxes=bboxes,
        labels=labels,
        scores=scores,
        label_colors=label_colors,
    )

    return annotated_image


def test_visualize_gt():
    # Example of the JSON structure you might have:
    with open('/home/ubuntu/workspace/tpu/instances_rp_test.json', 'r') as f:
        results = json.load(f)

    for img_info in results['images']:
        if img_info['file_name'] != '2024-01-21 151056.jpg':
            continue

        img_id = img_info['id']
        file_name = img_info['file_name']

    bboxes = []
    labels = []
    scores = []
    for anno_info in results['annotations']:
        if anno_info['image_id'] != img_id:
            continue

        x, y, w, h = anno_info['bbox']
        bboxes.append([x, y, x + w, y + h])
        labels.append(anno_info['category_id'])
        scores.append(1.0)
        print(anno_info)

    image = cv2.imread(f"/home/ubuntu/workspace/tpu/images/{file_name}")
    # pil_image = Image.open(f"/home/ubuntu/workspace/tpu/images/{file_name}").convert('RGB')
    # image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    # Visualize results
    visualize_predictions_vanilla(
        image,
        bboxes,
        labels,
        scores
    )

    cv2.imwrite('out.jpg', image)


def test_visualize_coco_grouped(
    image_id: int,
):
    # Get the image path.
    with open('/home/ubuntu/workspace/3FOVD/datasets/3FOVD-RP/test/instances_rp_test.json', 'r') as f:
        annotations = json.load(f)

    for img_info in annotations['images']:
        if img_info['id'] != image_id:
            continue

        file_name = img_info['file_name']

    assert file_name is not None, f"Image ID {image_id} not found in annotations."


    # Load the COCO results for this image.
    with open('/home/ubuntu/workspace/3FOVD/datasets/3FOVD-RP/test/vild_product_test_prediction_small_coco.json', 'r') as f:
        coco_results = json.load(f)


    bboxes = []
    labels = []
    scores = []
    for box_info in coco_results:
        if box_info['image_id'] != image_id:
            continue

        x, y, w, h = box_info['bbox']
        bboxes.append([x, y, x + w, y + h])
        labels.append(box_info['word'])
        scores.append(box_info['score'])

    image = cv2.imread(f"/home/ubuntu/workspace/tpu/images/{file_name}")
    # pil_image = Image.open(f"/home/ubuntu/workspace/tpu/images/{file_name}").convert('RGB')
    # image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    # Visualize results
    visualize_predictions_grouped(
        image,
        bboxes,
        labels,
        scores
    )

    cv2.imwrite('out.jpg', image)


if __name__ == "__main__":
    test_visualize_coco_grouped(11006)
