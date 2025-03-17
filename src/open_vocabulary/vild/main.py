import argparse
from collections import defaultdict
from curses import raw
from glob import glob
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ExifTags
import tensorflow.compat.v1 as tf
import tqdm

from get_text_embeddings import get_text_embedding
from image_util import visualize_predictions_vanilla, product_label_colors, rotate_image_and_bboxes


def extract_exif_orientation(image_path: str) -> Tuple[int, int, int]:
    """
    When OpenCV reads an image it rotates the image based on the EXIF information. And the annotation is based on the rotated image. 
    However, the ViLD uses Pillow Image compatiable format which does not rotate the image. 
    Therefore, we need to rotate the predicted bounding boxes.
    """
    # Extract the EXIF orientation if it exists
    with open(image_path, 'rb') as f:
        img = Image.open(f)
        w, h = img.size  # (width, height)
        orientation = 1  # Default (no rotation)
        try:
            exif = img._getexif()
            if exif is not None:
                # Pillow <=9 places orientation under tag 274
                for tag, value in exif.items():
                    decoded = ExifTags.TAGS.get(tag, tag)
                    if decoded == 'Orientation':
                        orientation = value
                        break
        except Exception:
            # EXIF might be absent or unreadable; default orientation=1
            print(f"Error reading EXIF orientation for {image_path}")
            raise

    return orientation, w, h


def adjust_boxes_for_exif(
        orientation: int, width: int, height: int, boxes: np.ndarray
    ) -> np.ndarray:
    """
    Adjust a bounding box given EXIF orientation.
    """
    # Unpack the original box
    boxes_out = np.zeros_like(boxes)

    if orientation == 1:
        # Orientation 1: No rotation
        # Just swap to [xmin, ymin, xmax, ymax]
        boxes_out[:, 0] = boxes[:, 1]
        boxes_out[:, 1] = boxes[:, 0]
        boxes_out[:, 2] = boxes[:, 3]
        boxes_out[:, 3] = boxes[:, 2]

    elif orientation == 3:
        # Orientation 3: 180° rotation
        # (y, x) -> (h - y - 1, w - x - 1)
        # Then swap to [xmin, ymin, xmax, ymax]
        # For each box: [ymin, xmin, ymax, xmax]
        new_ymin = height - boxes[:, 2] - 1
        new_ymax = height - boxes[:, 0] - 1
        new_xmin = width - boxes[:, 3] - 1
        new_xmax = width - boxes[:, 1] - 1

        # Assign in [xmin, ymin, xmax, ymax] order
        boxes_out[:, 0] = new_xmin
        boxes_out[:, 1] = new_ymin
        boxes_out[:, 2] = new_xmax
        boxes_out[:, 3] = new_ymax

    elif orientation == 6:
        # Orientation 6: Rotate 90° clockwise
        # (y, x) -> (x, h - y - 1)
        new_ymin = boxes[:, 1]                # old xmin -> new ymin
        new_ymax = boxes[:, 3]                # old xmax -> new ymax
        new_xmin = height - boxes[:, 2] - 1        # h - old ymax - 1 -> new xmin
        new_xmax = height - boxes[:, 0] - 1        # h - old ymin - 1 -> new xmax
        
        new_w, new_h = height, width  # After 90° CW, image dims swap

        # Assign in [xmin, ymin, xmax, ymax] order
        boxes_out[:, 0] = new_xmin
        boxes_out[:, 1] = new_ymin
        boxes_out[:, 2] = new_xmax
        boxes_out[:, 3] = new_ymax

    elif orientation == 8:
        # Orientation 8: Rotate 90° counter-clockwise
        # (y, x) -> (w - x - 1, y)
        new_ymin = width - boxes[:, 3] - 1        # w - old xmax - 1
        new_ymax = width - boxes[:, 1] - 1        # w - old xmin - 1
        new_xmin = boxes[:, 0]                # old ymin
        new_xmax = boxes[:, 2]                # old ymax

        new_w, new_h = height, width  # After 90° CCW, image dims swap

        # Assign in [xmin, ymin, xmax, ymax] order
        boxes_out[:, 0] = new_xmin
        boxes_out[:, 1] = new_ymin
        boxes_out[:, 2] = new_xmax
        boxes_out[:, 3] = new_ymax

    else:
        # Other possible EXIF orientation codes (2, 4, 5, 7) can be handled here if needed.
        # For now, treat them as "no rotation".
        boxes_out[:, 0] = boxes[:, 1]
        boxes_out[:, 1] = boxes[:, 0]
        boxes_out[:, 2] = boxes[:, 3]
        boxes_out[:, 3] = boxes[:, 2]

    return boxes_out

def nms(dets, scores, thresh, max_dets=1000):
    """Non-maximum suppression.
    Args:
      dets: [N, 4]
      scores: [N,]
      thresh: iou threshold. Float
      max_dets: int.
    """
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0 and len(keep) < max_dets:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        overlap = intersection / \
            (areas[i] + areas[order[1:]] - intersection + 1e-12)

        inds = np.where(overlap <= thresh)[0]
        order = order[inds + 1]
    return keep


def run_inference_single_image(
        tf_session: tf.Session,
        image_path: str,
        nms_threshold: float = 0.6,
        min_rpn_score_thresh: float = 0.9,
        min_box_area: float = 220,
    )-> Dict[str, Any]:
    """
    Runs the ViLD (TF-based) model inference on a single image and returns
    the region proposals, scaled boxes, confidence scores, and image features.
    
    Args:
        tf_session: A TensorFlow session loaded with the ViLD SavedModel.
        image_path: Path to the input image (string).
        
    Returns:
        A dictionary containing:
            - roi_boxes: [N, 4] numpy array of region proposal bounding boxes
              (absolute coordinates).
            - roi_scores: [N, ] numpy array of confidence scores for each
              region proposal.
            - detection_boxes: [N, 4] boxes used for second-stage detection
              (not yet scaled to the original image).
            - box_outputs: Model-specific box refinements (optional).
            - mask_outputs: Model-specific mask predictions (optional).
            - visual_features: [N, D] feature vectors for each region proposal.
            - image_info: [1, 4, 2] array containing scaling information
              (among other metadata).
            - rescaled_detection_boxes: [N, 4] bounding boxes scaled to
              the original input image size.
    """

    # 2) Run the TF session to get the relevant outputs
    #    (names match those from the ViLD saved_model).
    fetch_names = [
        'RoiBoxes:0',         # region proposal boxes
        'RoiScores:0',        # region proposal scores
        '2ndStageBoxes:0',    # detection boxes
        '2ndStageScoresUnused:0',
        'BoxOutputs:0',       # bounding box refinements
        'MaskOutputs:0',      # instance masks (if any)
        'VisualFeatOutputs:0',# features of each region
        'ImageInfo:0',        # scaling and other image info
    ]

    (roi_boxes, roi_scores,
     detection_boxes, _,
     box_outputs, mask_outputs,
     visual_features, image_info) = tf_session.run(
         fetch_names,
         feed_dict={'Placeholder:0': [image_path]}
    )

    rotation, width, height = extract_exif_orientation(image_path)

    # Remove the batch dimension (if present)
    roi_boxes = np.squeeze(roi_boxes, axis=0)        # [N, 4]
    roi_scores = np.squeeze(roi_scores, axis=0)      # [N, ]
    detection_boxes = np.squeeze(detection_boxes, axis=(0, 2))  # [N, 4]
    box_outputs = np.squeeze(box_outputs, axis=0)               # [N, 4]
    mask_outputs = np.squeeze(mask_outputs, axis=0)             # [N, H, W] or similar
    visual_features = np.squeeze(visual_features, axis=0)       # [N, D]
    image_info = np.squeeze(image_info, axis=0)                 # [4, 2], e.g. shape info

    # 3) Compute the rescaled detection boxes to match the original image size
    #    image_info[2, :] holds the scale factors for this model’s preprocessing
    #    so we tile it to match [1, 2].
    image_scale = np.tile(image_info[2:3, :], (1, 2))  
    rescaled_detection_boxes = detection_boxes / image_scale

    nmsed_indices = nms(
        detection_boxes,
        roi_scores,
        thresh=nms_threshold
    )

    # Compute RPN box size.
    box_sizes = (rescaled_detection_boxes[:, 2] - rescaled_detection_boxes[:, 0]) * (
        rescaled_detection_boxes[:, 3] - rescaled_detection_boxes[:, 1])
    
    # Filter out invalid rois (nmsed rois)
    valid_indices = np.where(
        np.logical_and(
            np.isin(np.arange(len(roi_scores), dtype=np.int32), nmsed_indices),
            np.logical_and(
                np.logical_not(np.all(roi_boxes == 0., axis=-1)),
                np.logical_and(
                    roi_scores >= min_rpn_score_thresh,
                    box_sizes > min_box_area
                )
            )
        )
    )[0]

    detection_roi_scores = roi_scores[valid_indices]
    detection_boxes = detection_boxes[valid_indices]
    detection_visual_feat = visual_features[valid_indices]
    rescaled_detection_boxes = rescaled_detection_boxes[valid_indices]

    # Change from [ymin, xmin, ymax, xmax] to [xmin, ymin, xmax, ymax].
    # rescaled_detection_boxes = rescaled_detection_boxes[:, [1, 0, 3, 2]]
    rescaled_detection_boxes = adjust_boxes_for_exif(
        rotation, 
        width, 
        height, 
        rescaled_detection_boxes
    )

    # 4) Return all relevant outputs in a structured dictionary
    return {
        "roi_boxes": roi_boxes,
        "roi_scores": detection_roi_scores,
        "detection_boxes": detection_boxes,
        "box_outputs": box_outputs,
        "mask_outputs": mask_outputs,
        "visual_features": detection_visual_feat,
        "image_info": image_info,
        "rescaled_detection_boxes": rescaled_detection_boxes,
    }


def run_an_image_with_all_captions(
        tf_session: tf.Session,
        captions: List[str],
        caption_idx_array: List[int],
        unique_words: List[str],
        text_embeddings: List[np.ndarray],
        image_path: str,
        max_boxes_to_draw: float = 25,
    ) -> List[Dict[str, Any]]:
    """
    Runs the ViLD model on a single image, then computes similarity between
    each region's visual features and all caption embeddings.

    Args:
        tf_session: A TensorFlow session loaded with the ViLD SavedModel.
        text_embeddings: A 2D numpy array of shape (N, 512), containing N
                         caption embeddings (e.g., CLIP-derived).
        image_path: Path to the input image.

    Returns:
        A dictionary (indexed by caption index) -> {
            "boxes": List of bounding boxes (each box is [ymin, xmin, ymax, xmax]),
            "labels": List of caption labels (strings),
            "scores": List of similarity scores (floats)
        }
        Each list is in parallel, meaning boxes[i], labels[i], scores[i]
        all refer to the same detected region.
    """
    # 1) Run the TF-based inference on the image to obtain region proposals & features
    inference_results = run_inference_single_image(tf_session, image_path)

    roi_boxes = inference_results["roi_boxes"]                # [R, 4]
    roi_scores = inference_results["roi_scores"]              # [R, ]
    detection_boxes = inference_results["detection_boxes"]    # [R, 4]
    rescaled_detection_boxes = inference_results["rescaled_detection_boxes"]  # [R, 4]
    visual_features = inference_results["visual_features"]    # [R, 512]


    # If no valid proposals remain, return empty results for all captions
    if len(visual_features) == 0:
        return {
            i: {"boxes": [], "labels": [], "scores": []}
            for i in range(text_embeddings.shape[0])
        }

    # 3) Construct the output dictionary
    #    For each caption index i, store bounding boxes, repeated "caption_i", and similarity scores
    num_captions = len(text_embeddings)
    result_list = []

    for i in range(num_captions):
        raw_scores = visual_features.dot(text_embeddings[i].T)
        raw_scores = raw_scores.tolist()
        # Scores for this caption across all region proposals
        max_idx_of_word_per_box = np.argmax(raw_scores, axis=1)

        boxes_list = [box.tolist() for c, box in zip(max_idx_of_word_per_box, rescaled_detection_boxes) if c > 0]
        caption_scores = [raw_scores[r][c] for r, c in enumerate(max_idx_of_word_per_box) if c > 0]
        # The first word, 'background', is ignored from the unique_words.
        labels_list = [unique_words[i][c-1] for r, c in enumerate(max_idx_of_word_per_box) if c > 0]

        result_list.append({
            "boxes": boxes_list,         # list of [ymin, xmin, ymax, xmax]
            "labels": labels_list,       # repeated caption label
            "scores": caption_scores,     # float similarity scores
            "labels_idx": max_idx_of_word_per_box.tolist(),
        })

    return result_list


def main(
        tf_session: tf.Session, 
        text_embed_file: str,
        image_dir: str, 
        caption_idx: int = 0,
        anno_file: str = None, 
        vis_target_dir: str = None,
        limit: int = None,
        out_file_name: str = "vild_product_test_prediction.json",
    ) -> None:
    """
    Runs inference for a specific caption index on all images found under 'images/*.jpg'
    using the run_an_image_with_all_captions() function.
    
    Visualizes the results for the single specified caption_idx, writes
    the bounding-box-overlaid images to target_dir, and accumulates the
    inference results in a dictionary keyed by:
      img_and_cap_to_instances[image_path][caption_idx] -> {
        "boxes":  [...],
        "labels": [...],
        "scores": [...]
      }
    Finally, it writes the entire dictionary to 'img_path_to_instances.json'.

    Args:
        tf_session: A TensorFlow session loaded with the ViLD SavedModel.
        text_embed_file: Path to a JSON file containing all caption embeddings,
                         e.g.:
                         {
                             "0": {"embedding": [512 floats], "caption": "some text"},
                             "1": {"embedding": [512 floats], "caption": "another text"},
                             ...
                         }
        caption_idx: Integer index specifying which caption embedding we want
                     to visualize. (We do run inference with *all* captions but
                     only visualize/store the results for this specific one.)
        target_dir: A directory in which to save the visualized images with
                    bounding boxes drawn.
    """

    # 1) Ensure output directory exists
    if vis_target_dir is not None:
        os.makedirs(vis_target_dir, exist_ok=True)

    # 2) Load and parse all caption embeddings into a list of [(N, 512), ...] array
    with open(text_embed_file, 'r') as f:
        raw_json = json.load(f)

    caption_array = []
    text_embedding_array = []
    unique_words_array = []
    caption_idx_array = []

    for caption_idx in raw_json: # caption_idx is not continuous.
        caption_idx_array.append(caption_idx)
        caption = raw_json[str(caption_idx)]
        text_embedding = raw_json[str(caption_idx)]["embedding"]
        unique_words = raw_json[str(caption_idx)]["words"]

        caption_array.append(caption)
        text_embedding = np.array(text_embedding, dtype=np.float32)
        text_embedding_array.append(text_embedding)
        unique_words_array.append(unique_words)

    # 3) Gather images
    with open(anno_file, 'r') as f:
        anno_json = json.load(f)

    # image_paths = sorted(glob(f"{image_dir}/*.jpg"))

    # 4) Prepare a structure to store final results
    #    We'll index by [image_path][caption_idx]
    img_and_cap_to_instances = defaultdict(dict)

    # 5) For each image, run multi-caption inference, then select the sub-result for caption_idx
    if limit is None:
        limit = len(anno_json['images'])

    # for img_path in tqdm.tqdm(image_paths[:limit]):

    for img_info in tqdm.tqdm(anno_json['images'][:limit]):
        if img_info['id'] != 11006:
            continue
        img_path = os.path.join(image_dir, img_info['file_name'])
        # This function returns a dict: {cap_idx: {"boxes": [...], "labels": [...], "scores": [...]}, ...}
        pred_dict_per_caption = run_an_image_with_all_captions(
            tf_session, 
            caption_array,
            caption_idx_array,
            unique_words_array,
            text_embedding_array,
            img_path
        )

        if vis_target_dir is not None:
            # 6) Visualize the bounding boxes on the image
            bboxes = pred_dict_per_caption[73]["boxes"]
            labels = pred_dict_per_caption[73]["labels"]
            scores = pred_dict_per_caption[73]["scores"]
            labels_idx = pred_dict_per_caption[73]["labels_idx"]

            image = cv2.imread(img_path)
            # image, bboxes = rotate_image_and_bboxes(image, bboxes, 1)
            visualize_predictions_vanilla(
                image=image,
                bboxes=bboxes,
                labels=labels,
                scores=scores,
                label_colors=product_label_colors,
            )

            # Save the visualization result
            out_path = os.path.join(vis_target_dir, os.path.basename(img_path))
            cv2.imwrite(out_path, image)

        # 7) Store these results in the dictionary
        for caption_idx, pred_dict in zip(caption_idx_array, pred_dict_per_caption):
            assert len(pred_dict["boxes"]) == len(pred_dict["labels"]) == len(pred_dict["scores"]), \
                f"Length mismatch in: {img_info['id']}, {caption_idx}!"
            img_and_cap_to_instances[img_info['id']][str(caption_idx)] = {
                "boxes": pred_dict["boxes"],
                "words": pred_dict["labels"],
                "scores": pred_dict["scores"],
                "word_idx": pred_dict["labels_idx"],
            }

    # 8) Write out the final dictionary to JSON for safekeeping
    with open(out_file_name, 'w') as f:
        json.dump(img_and_cap_to_instances, f, indent=2)


if __name__ == "__main__":
    # Load the ViLD SavedModel.
    tf_model_dir = "/home/ubuntu/workspace/tpu/models/official/detection/projects/vild/image_path_v2"
    session = tf.Session(graph=tf.Graph())
    _ = tf.saved_model.loader.load(session, ['serve'], tf_model_dir)

    main(
        tf_session=session,
        text_embed_file='/home/ubuntu/workspace/tpu/vild_rp_text_embeddings.json',  # random embeddings for 10 captions
        image_dir='/home/ubuntu/workspace/tpu/images',
        anno_file='/home/ubuntu/workspace/tpu/instances_rp_test.json',
        # image_dir = '/home/ubuntu/workspace/tpu/models/official/detection/projects/vild/examples/',
        vis_target_dir='test_vis',
    )
