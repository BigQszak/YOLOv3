# YOLOv3 in PyTorch
Implementation of YOLOv3 object detector from scratch, based on the popular paper "[YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)" by Joseph Redmon, Ali Farhadi.
```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```

## Installation

### Clone and install requirements
```bash
$ git clone https://github.com/BigQszak/YOLOv3.git
$ cd YOLOv3/
$ pip install requirements.txt
```

### Training
Edit the config.py file to match the setup you want to use. 
Then run train.py

### Results
TO DO

## Main takes
1. Yolov3 predicts at three different scales - grid sizes, for different object (bounding boxes) sizes
2. Implements ResNet like skip connections
3. Uses concatenations & additions of skip channels
4. Each prediction scale is proceeded by several convolutional layers
5. Implements bilinear interpolation upscaling after each prediction
6. Opts 3 different bboxes for each grid cell - the ability to predict more (smaller) objects
7. Anchor boxes - predefined bboxes of different scales & sizes => adjusting those boxes instead of coming up with completely new ones 
    (9 in total, 3 for each grid cell at each prediction scale)

## Useful links
[How YOLOv3 Works?](https://www.youtube.com/watch?v=MKF1NHGgFfk&list=WL&index=42&t=778s)

## TO DO
- [ ] Publish trained models
- [x] Update paths 
- [ ] Implement interactive scripts for argument parsing
- [x] Debug training
