import sys
import os
from loss import check_loss

# This works
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
# sys.path.append("C:/Users/krzak/OneDrive/Pulpit/Code/DeepLearning/YOLO_v3")

# This alone doesn't work
from utils.accuracy import check_accuracy
from utils import config

if __name__ == "__main__":
    check_accuracy()
    check_loss()
    config.check_config()
