## Pre-reqs

### Download required models

Download the SMPL model and pre-trained [params](https://drive.google.com/file/d/1grEX6HmqL6CKittCyl_N6nggqIRIEOCt/view?usp=sharing).

Store this as `HumanMultiView/models/`.

### Download datasets.
Download these datasets somewhere.

- [LSP](http://sam.johnson.io/research/lsp_dataset.zip) and [LSP extended](http://sam.johnson.io/research/lspet_dataset.zip)
- [COCO](http://cocodataset.org/#download) we used 2014 Train. You also need to
  install the [COCO API](https://github.com/cocodataset/cocoapi) for python.
- [MPII](http://human-pose.mpi-inf.mpg.de/#download)
- [MPI-INF-3DHP](http://human-pose.mpi-inf.mpg.de/#download)
- [Our synthetic dataset](https://drive.google.com/file/d/1nQEPCVY7VOXV-KOxeCX7hIQ9I4LLumWm/view?usp=sharing)

If you use the datasets above, please consider citing their original papers.

## Training and Evaluation

We have similar training and evalutaion scripts to HMR. Please refer to the original [HMR help doc](https://github.com/akanazawa/hmr/blob/master/doc/train.md) for details. Note that you can specify the multi-view dataset IDs in src_ortho/data_loader.py.
