# import the necessary packages
import os
import config

from torch.utils.data import Dataset
from PIL import Image
from numpy import asarray


class SegmentationDataset(Dataset):
	def __init__(self, imagePaths, maskPaths, transforms):
		self.imagePaths = imagePaths
		self.maskPaths = maskPaths
		self.transforms = transforms
	def __len__(self):
		return len(self.imagePaths)
	def __getitem__(self, idx):
		image = asarray(Image.open(self.imagePaths[idx]))
		mask = asarray(Image.open(self.maskPaths[idx]))

		if self.transforms is not None:
			image = self.transforms(image)
			mask = self.transforms(mask)
			
		return (image, mask)