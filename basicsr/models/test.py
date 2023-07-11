import torch
import cv2 as cv
from torch.distributions.laplace import Laplace
import numpy
from basicsr.utils import get_root_logger, imwrite, tensor2img, img2tensor

img = cv.imread("C:/Users/yzb/Desktop/1.png")
laplace = Laplace(torch.Tensor([1.0]), torch.Tensor([1.0]))
img = img2tensor(img)
img = laplace.icdf(img)
img = tensor2img(img)
img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.namedWindow("winname", cv.ACCESS_READ)
cv.imshow("winname", img)
cv.waitKey(0)
img = tensor2img(img)

