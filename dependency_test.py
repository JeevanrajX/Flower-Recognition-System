
import torch
import torchvision
import cv2
import numpy
import PIL
import flask
import sklearn

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("OpenCV:", cv2.__version__)
print("NumPy:", numpy.__version__)
print("Flask:", flask.__version__)
print("CUDA:", torch.cuda.is_available())

print("\nAll dependencies imported successfully!")
