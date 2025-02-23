import argparse
from collections import defaultdict
from glob import glob
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Tuple

import cv2
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling import build_model
import detectron2.data.transforms as T
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm

sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.modeling.utils import reset_cls_test
from detic.config import add_detic_config
from image_util import visualize_predictions


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", help="Take inputs from webcam.")
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--vocabulary",
        default="custom",
        choices=['lvis', 'openimages', 'objects365', 'coco', 'custom'],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="",
        help="",
    )
    parser.add_argument("--pred_all_class", action='store_true')
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"],
        nargs=argparse.REMAINDER,
    )
    return parser


def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


class PreprocessedDataset(Dataset):
    """
    Loads images from a folder, applies transforms, and returns a
    list/dict that the detectron2 model can accept directly.
    """
    def __init__(self, folder_path, aug):
        super().__init__()
        self.image_files = sorted(glob(os.path.join(folder_path, "*.jpg")))
        self.aug = aug

    def __len__(self) -> int:
        # return len(self.image_files)
        return len(self.image_files)

    def __getitem__(self, idx) -> Dict[str, Any]:
        img_path = self.image_files[idx]
        try:
            original_image = read_image(img_path, format="BGR")
        except:
            print(f"Image corrupted: {img_path}")
            # raise ValueError(f"Image corrupted: {img_path}")

        original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]

        image = self.aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width, "img_path": img_path}
        return inputs


def identity_collate_fn(batch) -> Any:
    return batch

def run_inference_directory_with_preprocessing(
        rank: int,
        world_size: int,
        cfg,
        folder_path: str,
        text_embeddings: dict,
        model_output_file: str,
        caption_idx: int,
        batch_size=8,
    ) -> list:
    """
    Build a model from cfg, build a dataset that does all
    preprocessing inside, then run inference on the entire folder.
    """
    
    text_embedding = text_embeddings[f'{caption_idx}']['embedding']
    caption = text_embeddings[f'{caption_idx}']['caption']
    unique_words = text_embeddings[f'{caption_idx}']['words']

    setup(rank, world_size)
    torch.cuda.set_device(rank)

    # Example “resize” transform that matches typical DefaultPredictor behavior
    # (adjust as needed).
    # If your config uses e.g. build_transform_gen(...) or a specific approach,
    # just replicate it here.
    aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )
    # Initialize the dataset
    dataset = PreprocessedDataset(folder_path, aug=aug)

    # Build a DataLoader. 
    # The collate function can default if we just return a list of dicts.
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=world_size * 8,
        sampler=sampler,
        collate_fn=identity_collate_fn,  # keep them as a list of dicts
        pin_memory=True,
    )

    # Build model & load weights
    model = build_model(cfg)

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    num_classes = len(unique_words)
    classifier = torch.tensor(text_embedding)
    reset_cls_test(model, classifier, num_classes=num_classes)
    model.eval()

    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        broadcast_buffers=False
    )

    local_results = []
    # In typical usage, put model on GPU
    device = torch.device(cfg.MODEL.DEVICE)  # e.g. "cuda" or "cuda:0"
    model.to(device)

    since = time.time()
    # loader_iter = tqdm(data_loader, desc=f"Rank {rank}", disable=(rank != 0))
    with torch.no_grad():
        for batch_inputs in data_loader:
            for inp in batch_inputs:
                inp["image"] = inp["image"].to(device)
            outputs = ddp_model(batch_inputs)

            for inp_dict, out_dict in zip(batch_inputs, outputs):
                instances = out_dict["instances"].to(torch.device("cpu"))
                boxes = instances.pred_boxes.tensor.tolist()    # list of [x1, y1, x2, y2]
                labels_idx = instances.pred_classes.tolist()    # list of numeric class indices
                scores = instances.scores.tolist()             # list of confidence scores

                local_results.append((inp_dict["img_path"], boxes, labels_idx, scores))

    all_results_list = [None for _ in range(world_size)]
    dist.all_gather_object(all_results_list, local_results)

    # On rank 0, merge everything
    merged_results = []
    if rank == 0:
        for partial in all_results_list:
            merged_results.extend(partial)

        print(f"Time elapsed: {time.time() - since:.2f} seconds for {len(dataset)} images.")

        if not os.path.exists(model_output_file):
            img_and_cap_to_instances = defaultdict(dict)
        else:
            with open(model_output_file, 'r') as f:
                img_and_cap_to_instances = json.load(f)
            img_and_cap_to_instances = defaultdict(dict, img_and_cap_to_instances)

        for img_path, bboxes, labels_idx, scores in merged_results:
            labels = [unique_words[i] for i in labels_idx]

            img_filename = os.path.basename(img_path)
            img_and_cap_to_instances[caption_idx][img_filename] = {
                "boxes": bboxes,
                "labels_idx": labels_idx,
                "labels": labels,
                "scores": scores,
            }

        with open(model_output_file, 'w') as f:
            json.dump(img_and_cap_to_instances, f)

    dist.barrier()
    dist.destroy_process_group()


def run_inference_single_image(
    loaded_model,
    image, 
    class_names
)-> Dict[str, Any]:
    """
    Args:
        loaded_model: An instance of VisualizationDemo or DefaultPredictor (or similar)
                      that has a .model attribute (Detic/Detectron2 model).
        image: np.array, read from OpenCV.
        class_names: A list of strings specifying your vocabulary classes.
    
    Returns:
        A dictionary containing bounding boxes, labels, and scores:
        {
          "boxes": [ [x1, y1, x2, y2], ... ],
          "labels": [ "class1", "class2", ... ],
          "scores": [ 0.98, 0.72, ... ]
        }
    """

    # 3) Run inference

    predictions = loaded_model(image)

    # 4) Convert img_result into a convenient format
    instances = predictions["instances"].to(torch.device("cpu"))
    boxes = instances.pred_boxes.tensor.tolist()    # list of [x1, y1, x2, y2]
    labels_idx = instances.pred_classes.tolist()    # list of numeric class indices
    scores = instances.scores.tolist()             # list of confidence scores

    # Map numeric labels back to text class names
    labels = [class_names[i] for i in labels_idx]

    # 5) Return bounding box img_result in a dictionary
    result = {
        "boxes": boxes,
        "labels": labels,
        "scores": scores
    }
    return result


def run_a_caption_on_images(
    predictor: DefaultPredictor,
    unique_words: List[str],
    text_embedding: List[List[float]],
    image_paths: List[str],
)-> List[Any]:
    """
    Args:
        loaded_model: An instance of the Detic predictor (VisualizationDemo or DefaultPredictor).
        caption: A single string containing your entire vocabulary, e.g. "car,headlights,door,..."
        image_paths: List of image file paths.

    Returns:
        A Python dict object containing bounding box img_result for each image, structured as:
        {
          "img_result": [
            {
              "image_path": "/path/to/image1.jpg",
              "instances": [
                {
                  "bbox": [x1, y1, x2, y2],
                  "label": "some_class",
                  "score": 0.98
                },
                ...
              ]
            },
            ...
          ]
        }
        You can convert this dict to JSON with json.dumps(...).
    """

    num_classes = len(unique_words)
    classifier = torch.tensor(text_embedding)


    reset_cls_test(predictor.model, classifier, num_classes=num_classes)
    predictor.model.eval()

    # 3) Run inference on each image
    all_img_result = []
    for img_path in tqdm.tqdm(image_paths):
        image_result = {
            "image_path": img_path,
            "instances": []
        }

        image = read_image(img_path, format="BGR")
       
        if image is None:
            raise ValueError(f"Could not read image from {img_path}")

        # Inference
        instances = run_inference_single_image(predictor, image, unique_words)

        # Convert predictions to a convenient format
        boxes = instances['boxes']
        labels = instances['labels']
        scores = instances['scores']

        # Map numeric labels to text
        for b, label, sc in zip(boxes, labels, scores):
            instance_data = {
                "bbox": b,
                "label": label,
                "score": float(sc)
            }
            image_result["instances"].append(instance_data)

        all_img_result.append(image_result)

    return all_img_result


def visualize_demo(
        text_embed_file: str = "rp_text_embeddings.json", 
        caption_idx = 1
    ) -> None:

    with open(text_embed_file, 'r') as f:
        text_embeddings = json.load(f)
    
    text_embedding = text_embeddings[f'{caption_idx}']['embedding']
    caption = text_embeddings[f'{caption_idx}']['caption']
    unique_words = text_embeddings[f'{caption_idx}']['words']

    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    img_and_cap_to_instances = defaultdict(dict)

    # image_paths = ["230608_769.jpg", "car.jpg"]
    image_paths = glob("images/*.jpg", recursive=True)[:10]
    predictor = DefaultPredictor(cfg)
    since = time.time()
    img_results = run_a_caption_on_images(predictor, unique_words, text_embedding, image_paths)
    print(f"Time elapsed: {time.time() - since:.2f} seconds for {len(image_paths)} images.")

    for img_path, img_result in zip(image_paths, img_results):
        image = cv2.imread(img_path)
        bboxes = [instance["bbox"] for instance in img_result["instances"]]
        labels = [instance["label"] for instance in img_result["instances"]]
        scores = [instance["score"] for instance in img_result["instances"]]
        # Visualize img_result
        visualize_predictions(
            image,
            bboxes,
            labels,
            scores
        )
        cv2.imwrite(f'test_vis/{os.path.basename(img_path)}', image)
        img_and_cap_to_instances[caption_idx][img_path] = img_result
    
    with open("img_path_to_instances.json", 'w') as f:
        json.dump(img_and_cap_to_instances, f)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def batch_demo(
        text_embed_file: str = "rp_text_embeddings.json",
        model_output_file: str = "detic_product_test.json",
    ) -> None:
    """
    This function is a main function to run when you have multiple GPUs.
    Because the running can be interrupted, we need to store the results after getting the results of
    all images for a caption.
    """

    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    with open(text_embed_file, 'r') as f:
        text_embeddings = json.load(f)

    world_size = torch.cuda.device_count()

    rest_embed_indices = find_rest_text_embed_indices(text_embed_file, model_output_file)

    for caption_idx in tqdm.tqdm(rest_embed_indices):
        mp.spawn(
            run_inference_directory_with_preprocessing,
            nprocs=world_size,
            args=(world_size, cfg, "images", text_embeddings, model_output_file, caption_idx),
        )


def find_rest_text_embed_indices(
    text_embed_file: str = "vehicle_text_embeddings.json",
    model_output_file: str = "img_path_to_instances.json",
) -> List[int]:
    assert os.path.exists(text_embed_file), f"{text_embed_file} does not exist!"
    with open(text_embed_file, 'r') as f:
        text_embeddings = json.load(f)
    text_embed_indices = set(text_embeddings.keys())

    if not os.path.exists(model_output_file):
        return [int(i) for i in text_embed_indices]
    
    with open(model_output_file, 'r') as f:
        model_outputs = json.load(f)
    exist_text_embed_indices = set(model_outputs.keys())
    rest_indices = text_embed_indices - exist_text_embed_indices
    rest_indices = [int(i) for i in rest_indices]
    return rest_indices


if __name__ == "__main__":
    # visualize_demo()
    batch_demo()
