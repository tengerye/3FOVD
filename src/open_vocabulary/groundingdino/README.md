1. Install `pytorch`, `torchvision` and `transformers` with compatible with your `CUDA`. For example, in `python==3.11.11` and `CUDA-12.2`, a feasible version of packages is: `torch==2.1.0`, `torchvision==0.16.0`, and `transformers==4.49.0`.

2. Follow the [official instruction](https://github.com/IDEA-Research/GroundingDINO) and make sure you can run the demo successfully.

3. Copy all python files in the `src/open_vocabulary/groundingdino` and `src/open_vocabulary/image_util.py` to the `demo/` directory of the GroundingDiNO.

4. Install the following packages `pandas`.
5. Replace the `groundingdino.py` file.

6. The following is an example to distribute inference
```
python demo/distribute_inference.py --num_gpus 4 --max_processes_per_gpu 8 --csv_path /home/ubuntu/workspace/GroundingDINO/car_categories.csv --output_dir pred_json --node_size 8 --node_idx 0
```
