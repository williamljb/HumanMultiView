# Shape-Aware Human Pose and Shape Reconstruction Using Multi-View Images

Junbang Liang, Ming C. Lin
ICCV 2019

[Project Page](https://gamma.umd.edu/researchdirections/virtualtryon/humanmultiview)

### Requirements
- Python 2.7
- [TensorFlow](https://www.tensorflow.org/) tested on version 1.3

### Demo

1. Download the [pre-trained models](https://drive.google.com/file/d/1grEX6HmqL6CKittCyl_N6nggqIRIEOCt/view?usp=sharing)

2. Run the demo
```
python -m demo --img_paths ${your_image_paths_separated_by_commas}
```

Images should be cropped so that the height of the person is roughly 2/3 of the image height. Please check demo.py for more details.

### Training and Data

Please see doc/train.md.

### Citation
If you use this code for your research, please consider citing:
```
@inProceedings{liang2019shape,
  title={Shape-Aware Human Pose and Shape Reconstruction Using Multi-View Images},
  author = {Junbang Liang
  and Ming C. Lin},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2019}
}
```

### Acknowledgement
This project is derived from [HMR](https://github.com/akanazawa/hmr). If you have any question, feel free to refer to the original help doc or email liangjb@cs.umd.edu. This work is supported by National Science Foundation and Elizabeth S. Iribe Professorship.
