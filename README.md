We put the sampled images and corresponding annotation files into `datasets` folder. The full data will be released immediately after the paper is accepted.

# Datasets

The structure of the 3FOVD-RP is like the follows:
```
- train
    - images.zip
    - instances_rp_train.json
- val
    - images.zip
    - instances_rp_val.json
- test
    - images.zip
    - instances_rp_val.json

- rp_categories.csv
```

The structure of the 3FOVD-V is similar as that of 3FOVD-RP, but replacing `rp` with `car` from the above.


# Codebase
The structure of this codebase is as follows:
```
- datasets
    - README.md: instructions to download our datasets.

- src: codes for repeating experiments.
    - supervised: codes for traditional object detectors (section V-B in our paper).
    - open_vocabulary: codes for benchmarking open-vocabulary object detectors (section V-C in our paper).
        - cora
        - detic
        - gdino
        - vild

    - post_process: codes for improving the open-vocabulary object detectors (section V-D).
```