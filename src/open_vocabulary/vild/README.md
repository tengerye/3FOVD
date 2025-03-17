1. Follow the [official install instruction](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild) to install the environment. Let us denote the root directory of it is `VILD_ROOT`. For example, you can use `python==3.11` and `tensorflow==2.17.0`.

2. Try to run the [demo](https://colab.research.google.com/github/tensorflow/tpu/blob/master/models/official/detection/projects/vild/ViLD_demo.ipynb). If you encounter the issue `INVALID_ARGUMENT: No DNN in stream executor.`, you need to check the compatibility of your tensorflow and CUDA version.

3. Install `tqdm`, `pandas`, `torch`, `torchvision` and `clip`.
4. Copy all python files in the directory `src/open_vocabulary/vild` into the `${DETIC_ROOT}/models/official/detection/projects/vild`.

5. Generate image embeddings by running `python get_image_embedding.py`. You may need to change the path of the parameters of the function `dump_image_embeddings()`.

6. 