from glob import glob
import json
import os

import tensorflow.compat.v1 as tf
from tqdm import tqdm


def dump_image_embeddings(
    image_root_path: str,
    saved_model_dir: str,
    output_file: str,
):
    image_paths = glob(os.path.join(image_root_path, '*.jpg'))

    session = tf.Session(graph=tf.Graph())
    _ = tf.saved_model.loader.load(session, ['serve'], saved_model_dir)

    roi_boxes_array = []
    roi_scores_array = []
    detection_boxes_array = []
    box_outputs_array = []
    visual_features_array = []
    image_info_array = []

    for image_path in tqdm(image_paths):
        roi_boxes, roi_scores, detection_boxes, _, box_outputs, _, visual_features, image_info = session.run(
            ['RoiBoxes:0', 'RoiScores:0', '2ndStageBoxes:0', '2ndStageScoresUnused:0',
                'BoxOutputs:0', 'MaskOutputs:0', 'VisualFeatOutputs:0', 'ImageInfo:0'],
            feed_dict={'Placeholder:0': [image_path]})

        roi_boxes_array.append(roi_boxes)
        roi_scores_array.append(roi_scores)
        detection_boxes_array.append(detection_boxes)
        box_outputs_array.append(box_outputs)
        visual_features_array.append(visual_features)
        image_info_array.append(image_info)


    with open(output_file, 'w') as f:
        json.dump({
            'image_paths': image_paths,
            'roi_boxes': roi_boxes_array,
            'roi_scores': roi_scores_array,
            'detection_boxes': detection_boxes_array,
            'box_outputs': box_outputs_array,
            'visual_features': visual_features_array,
            'image_info': image_info_array,
        }, f)


if __name__ == '__main__':
    dump_image_embeddings(
        '/home/ubuntu/workspace/tpu/images',
        '/home/ubuntu/workspace/tpu/models/official/detection/projects/vild/image_path_v2',
        'vild_product_test_image_embed.json',
    )
