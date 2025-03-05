import argparse
import json
import os
import sys
import copy

import cv2
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from line_profiler import profile

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.models.GroundingDINO.groundingdino import precompute_caption_embeddings, precompute_image_embeddings
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span
from image_util import visualize_predictions_vanilla


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    logits_filt = logits.cpu().clone()
    boxes_filt = boxes.cpu().clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


def get_grounding_output_with_embeddings(model, precomputed_image_embeddings, device, box_threshold, text_threshold=None, precomputed_text_dict=None, captions=None):

    with torch.no_grad():
        srcs, masks, poss = precomputed_image_embeddings
        outputs = model.forward_with_embeddings(srcs, masks, poss, text_dict=precomputed_text_dict, unset_image_tensor=False)
        model.unset_image_tensor()  # Unset image tensor after the forward pass

    boxes_filt_array = []
    pred_phrases_array = []
    scores_array = []

    # logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    # boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    for logits, boxes, caption in zip(
        outputs["pred_logits"].sigmoid(),
        outputs["pred_boxes"],
        captions,
    ):
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption).to(device)
        # build pred
        pred_phrases = []
        scores = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            scores.append(logit.max().item())
            pred_phrases.append(pred_phrase)

        boxes_filt_array.append(boxes_filt)
        pred_phrases_array.append(pred_phrases)
        scores_array.append(scores)
    
    return boxes_filt_array, pred_phrases_array, scores_array

@profile
def inference_for_all_images_and_captions(
    config_file: str,
    checkpoint_path: str,
    box_threshold: float,
    text_threshold: float,
    device,              # torch device (e.g., 'cuda' or 'cpu')
    anno_file,           # annotation JSON (COCO-format)
    images_path,         # path to folder containing images
    csv_file,            # path to car_categories.csv
    output_file,          # path to the JSON file to store final results
    vis_target_dir: str = None,
    limit = None,
    caption_row_idx: int = None,
):
    """
    1. Model is assumed to be loaded and passed in. Set it to eval mode, move to device.
    2. Read car_categories.csv with pandas to get (caption_id, caption_text).
    3. Precompute text embeddings for all captions via precompute_caption_embeddings().
    4. For each image in the annotation file, precompute image embeddings.
    5. For each (image, caption) pair, call get_grounding_output_with_embeddings() 
       to get bounding boxes, phrases, and scores.
    6. Store results in the desired JSON structure and write to output_file.
    """
    device = "cuda"
    if vis_target_dir is not None:
        os.makedirs(vis_target_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Prepare model
    model = load_model(config_file, checkpoint_path, cpu_only=False)
    model.eval()
    model.to(device)

    # -----------------------------------------------------------------------
    # Step 2: Read CSV with pandas
    # Assuming your CSV has two columns: [caption_id, caption_text]
    # If your file has a header, use header=0 or rename columns as needed.
    df = pd.read_csv(csv_file, header=0)  # or header=0 if you have a header row
    # Example: df.columns = ['caption_id', 'caption_text'] if you like clarity
    # df.columns = ['caption_id', 'caption_text']  # Uncomment if helpful
    if caption_row_idx is None:
        caption_ids = df.iloc[:, 0].tolist()    # first column -> caption id
        caption_texts = df.iloc[:, -1].tolist()  # second column -> text
    else:
        caption_ids = [df.iloc[caption_row_idx, 0]]    # first column -> caption id
        caption_texts = [df.iloc[caption_row_idx, -1]]  # second column -> text

    # -----------------------------------------------------------------------
    # Step 3: Precompute caption embeddings
    # Store them in a dict or list keyed by their caption_id
    caption_embeddings = {}
    for c_id, c_text in zip(caption_ids, caption_texts):
        with torch.no_grad():
            text_embed = precompute_caption_embeddings(model, [c_text], device)
        caption_embeddings[c_id] = text_embed

    # -----------------------------------------------------------------------
    # Step 4: Create data structure for final results
    results = {}
    with open(anno_file, 'r') as f:
        anno_json = json.load(f)

    # Loop through all images in annotation file

    images_info = anno_json['images']
    if limit is None:
        limit = len(images_info)
    for img_info in tqdm(images_info[:limit], desc="Processing images"):
        image_id = img_info['id']
        file_name = img_info['file_name']

        # Prepare sub-dict for this image in results
        results[image_id] = {}

        # Load image
        img_path = f"{images_path}/{file_name}"
        image_pil, image = load_image(img_path)
        W, H = image_pil.size
        image = image.to(device)

        # Step 5: Precompute image embeddings
        with torch.no_grad():
            image_embed = precompute_image_embeddings(model, image[None])

        # -------------------------------------------------------------------
        # For each caption, run get_grounding_output_with_embeddings
        for c_id, c_text in zip(caption_ids, caption_texts):
            text_embed = copy.deepcopy(caption_embeddings[c_id])  # or retrieve from dict

            with torch.no_grad():
                boxes, phrases, scores = get_grounding_output_with_embeddings(
                    model,
                    image_embed,
                    device,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    precomputed_text_dict=text_embed,
                    captions=[c_text]
                )
                boxes = boxes[0]
                scores = scores[0]
                phrases = phrases[0]

                # If your boxes are in xywh format, you may need to convert
                # them to (x1, y1, x2, y2). For example:
                # x1 = box[0], y1 = box[1]
                # x2 = box[0] + box[2], y2 = box[1] + box[3]
                # Adjust below if your function already returns [x1, y1, x2, y2].
                converted_boxes = []
                for box in boxes:
                    box = box * torch.Tensor([W, H, W, H])
                    box[:2] -= box[2:] / 2
                    box[2:] += box[:2]
                    x1, y1, x2, y2 = box.tolist()
                    converted_boxes.append([float(x1), float(y1), float(x2), float(y2)])

            # Store in results: 
            # results[image_id][caption_id] = {"boxes": [...], "phrases": [...], "scores": [...]}
            results[image_id][str(c_id)] = {
                "boxes": converted_boxes,            # list of [x1, y1, x2, y2]
                "words": phrases,                  # list of string phrases
                "scores": [float(s) for s in scores] # ensure JSON-serializable
            }

            if vis_target_dir is not None:
                vis_image = np.array(image_pil)
                visualize_predictions_vanilla(
                    vis_image,
                    converted_boxes,
                    phrases,
                    [float(s) for s in scores],
                )

                out_path = os.path.join(vis_target_dir, os.path.basename(img_path)[:-4] + f'_{caption_row_idx}.jpg')
                cv2.imwrite(out_path, vis_image)

    # -----------------------------------------------------------------------
    # Step 6: Write out JSON
    with open(f'{output_file[:-5]}_{caption_row_idx}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f)


def visualize_demo(config_file, checkpoint_path, image_path, text_prompt, output_dir, box_threshold, text_threshold):
    cpu_only = False
    token_spans = None
    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil, image = load_image(image_path)
    # load model
    model = load_model(config_file, checkpoint_path, cpu_only=cpu_only)
    
    # Determine the device
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)

    # Precompute text embeddings
    precomputed_text_dict = precompute_caption_embeddings(model, [text_prompt], device)
    precomputed_image_embeddings = precompute_image_embeddings(model, image[None])

    # run model
    # boxes_filt, pred_phrases = get_grounding_output(
    #     model, image, text_prompt, box_threshold, text_threshold, cpu_only, token_spans=token_spans
    # )
    boxes_filt_array, pred_phrases_array, scores_array = get_grounding_output_with_embeddings(
        model, precomputed_image_embeddings, device, box_threshold, text_threshold, precomputed_text_dict=precomputed_text_dict, captions=[text_prompt]
    )
    boxes_filt = boxes_filt_array[0]
    pred_phrases = pred_phrases_array[0]
    scores = scores_array[0]

    # visualize pred
    size = image_pil.size
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": [f"{l}: {s}" for l, s in zip(pred_phrases, scores)],
    }
    image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
    image_with_box.save(os.path.join(output_dir, os.path.basename(image_path)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for images given captions.")
    parser.add_argument(
        "caption_row_idx",
        nargs="?",
        type=int,
        default=None,
        help="CSV row index to process. If not provided, processes all rows."
    )

    args = parser.parse_args()

    config_file = '/home/ubuntu/workspace/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
    checkpoint_path = '/home/ubuntu/workspace/GroundingDINO/weights/groundingdino_swint_ogc.pth'
    box_threshold = 0.3
    text_threshold = 0.25

    # # Call the function with the parsed arguments
    # visualize_demo(
    #     '/home/ubuntu/workspace/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py', 
    #     '/home/ubuntu/workspace/GroundingDINO/weights/groundingdino_swint_ogc.pth', 
    #     '/home/ubuntu/workspace/GroundingDINO/car.jpg', 
    #     "The car model is a sedan, with a red circular emblem embedded with the letters FIAT. The front lights of the car are separate, narrow, and angled like a boat. It has a rectangular air intake duct, equipped with a horizontal bar-shaped air intake grille or an inverted trapezoidal air intake. There are rectangular fog lights. The hood contains crimp lines protruding on either side, slightly flat with a slight curve. The overall outline of the car body is smooth and streamlined.",
    #     'test_vis', 
    #     box_threshold, 
    #     text_threshold,
    # )

    inference_for_all_images_and_captions(
        config_file,
        checkpoint_path,
        box_threshold,
        text_threshold,
        'cuda',              
        '/home/ubuntu/workspace/datasets/cars/val/instances_car_val.json',           # annotation JSON (COCO-format)
        '/home/ubuntu/workspace/datasets/cars/val/images',         # path to folder containing images
        '/home/ubuntu/workspace/GroundingDINO/car_categories.csv',            # path to car_categories.csv
        'predict_jsons/groundingdino_car_test_prediction.json',          # path to the JSON file to store final results
        # 'vis_test/',
        # limit=10,
        caption_row_idx=args.caption_row_idx,
    )
