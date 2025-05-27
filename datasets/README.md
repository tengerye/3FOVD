# Introduction
The NEU-171K dataset supports both supervised and open-vocabulary object detection (3F-OVD) protocols. Each image is annotated using a COCO-style format with split files for training, validation, and test sets.

It has **145,825 images**, **676,471 bounding boxes**, **719 fine-grained classes** and contains the following two sub-sets.

## NEU-171K-C
NEU-171K-C contains cars in real-world traffic scenes. 
![NEU-171K-C](../figures/car_cover.jpg)

## NEU-171K-RP
NEU-171K-RP is a subset of the NEU-171K dataset, focusing on fine-grained object detection in the domain of retail products. It includes 56,462 high-resolution images captured under controlled warehouse-like conditions, annotated with 360,377 bounding boxes across 121 distinct product classes.

Compared to NEU-171K-C, this subset offers clearer object boundaries and visually consistent backgrounds, making it ideal for studying visual details that distinguish fine-grained categories.

![NEU-171K-RP](../figures/rp_cover.jpg)


# Structure
The structure of the `NEU-171K-RP` is like the follows:
```
- train
    - images.zip
    - instances_rp_train.json
- val
    - images.zip
    - instances_rp_val.json
- test
    - images.zip
    - instances_rp_test.json

- rp_categories.csv
```

The structure of the `NEU-171K-C` is similar as that of `NEU-171K-RP`, but replacing `rp` with `car` from the above.

The dataset can only be used for research only. This dataset is strictly for research purposes only. If you believe that our dataset violates your privacy, please feel free to contact us.

# Download
## Kaggle
[NEU-171K-C](https://www.kaggle.com/datasets/tenger/neu-171k-c).

## Huggingface
[NEU-171K](https://huggingface.co/datasets/tengerye/NEU-171K) contains both NEU-171K-C and NEU-171K-RP as its sub-folders in the root directory.

## Dropbox
[NEU-171K-RP](https://www.dropbox.com/scl/fo/fsx0dhfyr9w64wxagkrqe/AOOQngNC1s46bydC57vlqSo?rlkey=19i3923jpr249ktdxdo4gqh3f&st=pvn1s9hq&dl=1), [NEU-171K-C](https://www.dropbox.com/scl/fo/oonbrky5na7wq8xkvg7g0/AHpBMT-SpR5Jr6G4GOgaDM8?rlkey=1cgj15uubaehmciabeyh1oncq&st=06f8dl1v&dl=1).

## Baidu Netdisk
[NEU-171K-RP](https://pan.baidu.com/s/1wXNpf1UJ37VmVeubeJeftw?pwd=57j5), [NEU-171K-C](https://pan.baidu.com/s/1q4WzFO2UBOen3SJV2xcl5g?pwd=4j6v).