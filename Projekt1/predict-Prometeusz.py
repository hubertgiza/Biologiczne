from re import T
import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from PIL import Image
<<<<<<< HEAD
from keras import backend as K

def prepare_plot(origImage, origMask, predMask):
=======
from torchvision.transforms import Resize

def prepare_plot(original_image, original_mask, predicted_mask):
>>>>>>> 007f2fece20e0cb0caa06981df3518b892d37f78
	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(50, 50))
	ax[0].imshow(original_image)
	ax[1].imshow(original_mask)
	ax[2].imshow(predicted_mask)

	ax[0].set_title("Image")
	ax[1].set_title("Original Mask")
	ax[2].set_title("Predicted Mask")

	figure.tight_layout()
	plt.show()

def iou(original_mask, predicted_mask, smooth = 100):
	predicted_mask = np.apply_along_axis(np.vectorize(lambda x: 1 if x!=0 else 0),0,predicted_mask)
	intersection = np.sum(original_mask * predicted_mask)
	sum_ = np.sum(np.apply_along_axis(np.vectorize(lambda x: 1 if x==2 else x),0,(original_mask+predicted_mask)))
	jac = (intersection + smooth) / (sum_ - intersection + smooth)
	return jac

def make_predictions(model, imagePath,calculate_accuracy=False):
	model.eval()

	with torch.no_grad():
		image = np.asarray(Image.open(imagePath))
		image = image.astype("float32") / 255.0
		original_image = image.copy()

		filename = imagePath.split(os.path.sep)[-1]
		ground_truth_path = os.path.join(config.MASK_DATASET_PATH,filename.split('.')[0]+'.tiff')
		ground_truth_mask = np.asarray(Image.open(ground_truth_path).resize((config.INPUT_IMAGE_HEIGHT,config.INPUT_IMAGE_WIDTH)))

		image = np.transpose(image, (2, 0, 1))
		image = np.expand_dims(image, 0)
		image = torch.from_numpy(image).to(config.DEVICE)

		predicted_mask = model(image).squeeze()
		predicted_mask = torch.sigmoid(predicted_mask)
		predicted_mask = predicted_mask.cpu().numpy()
		predicted_mask = (predicted_mask > config.THRESHOLD) * 255
		predicted_mask = predicted_mask.astype(np.uint8)
		if calculate_accuracy:
			scores.append(iou(ground_truth_mask, predicted_mask))
		else:
			prepare_plot(original_image, ground_truth_mask, predicted_mask)

def iou(y_true, y_pred, smooth = 100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.square(y_true), axis = -1) + K.sum(K.square(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac
print("[INFO] loading up test image paths...")
image_paths = open(config.TEST_PATHS).read().strip().split("\n")

# Uncomment below if want to see results only for 10 samples
# image_paths = np.random.choice(image_paths, size=10)

print("[INFO] load up model...")
unet = torch.load(config.MODEL_PATH).to(config.DEVICE)
scores=[]
for path in image_paths:
	make_predictions(unet, path, calculate_accuracy=True)

print(np.mean(scores))