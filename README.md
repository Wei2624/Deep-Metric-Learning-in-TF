# Feature_Embed_GoogLeNet
This repo has reconstructed the model according to [GoogLeNet V1](https://arxiv.org/abs/1409.4842) in Tensorflow. If you want to directly check the model structure, you can jump to `GoogLeNet_V1_Model.py` file. 

The weights are directly from the `.caffemodel` file according to [Caffe Repo](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet). In addition, the performance has been tested against with `train_val.prototext` in the repo. As a result, the Tensorflow version in this repo can mirror the performance of caffe model. 

This Tensorflow version of GoogLeNet V1 has also been tested on Stanford Online Product Dataset for metric learning task according to the [paper](http://cvgl.stanford.edu/papers/song_cvpr16.pdf) where Caffe is used. On the validation set, the comparison between Tensorflow version and Caffe version is shown below. 

| Tables        | Tensorflow Version          | Caffe Version  |
| ------------- |:-------------:| -----:|
| F1 Measure      | 0.11 | 0.109 |
| NMI     | 0.839     |   0.833 |

**NOTE: there still exists little difference between the two because there is a function called Local Response Normalization which is implemented in different ways in Tensorflow and Caffe.**

This repo is stil under development. Coming next: **Recover the results of the [paper](https://arxiv.org/abs/1703.07464)**

