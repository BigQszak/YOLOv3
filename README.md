# YOLOv3
 Implementing YOLOv3 objet detector from scratch, based on the popular paper "[YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)" by Joseph Redmon, Ali Farhadi

- yolov3 predicts at three different scales - grid sizes, for different object (bounding boxes) sizes
- ResNet like skip connections 
- concatenations & additions of skip channels
- each predictions scale is proceeded by severl conv layers
- bilinear interpolation upscaling after each predictions
- 3 different bboxes for each grid cell - ability to predict more (smaller) objects
- anchor boxes - predefined bboxes of different scale & size => adjusting those boxes instead fo coming up with completely new ones 
    (9 in total, 3 for each grid cell at each prediction scale)