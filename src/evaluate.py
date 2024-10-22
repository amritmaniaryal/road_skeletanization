# Will host MSE, node-precision, etc.

from .data.dataset import RoadDataset
from .model import TinyUNet  
from src.metrics import mse, node_stats  

def dummy():
    print("Evaluation module placeholder")
