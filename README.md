We put the sampled images and corresponding annotation files into `datasets` folder. The full data will be released immediately after the paper is accepted.

# Datasets

[here](./path/to/other/README.md)


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


# Reference

> BibTeX Style Citation

```
@article{liu2025fine,
  title={Fine-Grained Open-Vocabulary Object Detection with Fined-Grained Prompts: Task, Dataset and Benchmark},
  author={Liu, Ying and Hua, Yijing and Chai, Haojiang and Wang, Yanbo and Ye, TengQi},
  journal={arXiv preprint arXiv:2503.14862},
  year={2025}
}
```