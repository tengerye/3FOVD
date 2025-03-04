The input to the post-processing is the raw prediction json file which is the prediction results of the model. There are two steps in the post-processing stage: convert into COCO predict format (`src/open_vocabulary/post_processing/postprocess.py`), and calculate the metrics (`src/open_vocabulary/post_processing/coco_eval.py`).

The raw prediction json file has the naming convension `{METHOD}_{DATABASE}_{SET}_prediction.json`. For example, the file `vild_product_test_prediction.json` is the prediction results of ViLD on the product test set. The content of the file follows the format:

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

When we need to perform custom post-processing method, we usually insert it into this stage by passing the function into `def convert_custom_predictions_to_coco()` using parameter `pp_func`. The input to the `pp_func` are `bboxes`, `scores`, `labels` and you need to return the indices to remain.

After that, you need to run `python src/open_vocabulary/post_processing/coco_eval.py` for COCO evaluation results.