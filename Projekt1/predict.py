import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from PIL import Image
from keras import backend as K

def prepare_plot(origImage, origMask, predMask):
	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(50, 50))
	ax[0].imshow(origImage)
	ax[1].imshow(origMask)
	ax[2].imshow(predMask)

	ax[0].set_title("Image")
	ax[1].set_title("Original Mask")
	ax[2].set_title("Predicted Mask")

	figure.tight_layout()
	plt.show()

def make_predictions(model, imagePath):
	model.eval()

	with torch.no_grad():

		image = np.asarray(Image.open(imagePath))
		image = image.astype("float32") / 255.0
		orig = image.copy()

		filename = imagePath.split(os.path.sep)[-1]
		groundTruthPath = os.path.join(config.MASK_DATASET_PATH,filename.split('.')[0]+'.tiff')
		gtMask = np.asarray(Image.open(groundTruthPath))

		image = np.transpose(image, (2, 0, 1))
		image = np.expand_dims(image, 0)
		image = torch.from_numpy(image).to(config.DEVICE)

		predMask = model(image).squeeze()
		predMask = torch.sigmoid(predMask)
		predMask = predMask.cpu().numpy()
		predMask = (predMask > config.THRESHOLD) * 255
		predMask = predMask.astype(np.uint8)

		prepare_plot(orig, gtMask, predMask)

def iou(y_true, y_pred, smooth = 100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.square(y_true), axis = -1) + K.sum(K.square(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac
print("[INFO] loading up test image paths...")
imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
imagePaths = np.random.choice(imagePaths, size=10)

print("[INFO] load up model...")
unet = torch.load(config.MODEL_PATH).to(config.DEVICE)

for path in imagePaths:
	make_predictions(unet, path)