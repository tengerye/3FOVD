1. Follow the [official install instruction](https://github.com/facebookresearch/Detic/blob/main/docs/INSTALL.md) to install the environment. Let us denote the root directory of it is `DETIC_ROOT`. 

2. Follow the [official readme](https://github.com/facebookresearch/Detic/tree/main) to run the first demo. **Note**: the link to the original image (`https://eecs.engin.umich.edu/~fouhey/fun/desk/desk.jpg`) for the demo may have been expired.

3. In your conda environment `detic`, install `pandas` using: `pip install pandas==2.0.3`.

4. Copy all python files in the directory `src/open_vocabulary/detic` into the `DETIC_ROOT`.

5. In the `DETIC_ROOT` directory. To evaluation the 3FOVD-RP, run `python -W ignore main.py`. If you would like to see the visualization results (`detic_${DATASET}_test.json`), use the function `visualize_demo()`.

6. The following is an example to run it on multiple machines:
```
python distribute_inference.py --node_size 10 --node_idx 0 --text_embed_file detic_car_text_embeddings.json --image_base_dir /home/ubuntu/workspace/datasets/cars/test/images/ --batch_size 8 --out_dir detic_car_test_pred/
```