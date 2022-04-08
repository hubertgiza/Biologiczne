import config
import matplotlib.pyplot as plt
import torch
import time
import os

from dataset import SegmentationDataset
from model import UNet
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm


imagePaths = [os.path.join(config.IMAGE_DATASET_PATH,image) for image in sorted(os.listdir(config.IMAGE_DATASET_PATH))]
maskPaths = [os.path.join(config.MASK_DATASET_PATH,mask) for mask in sorted(os.listdir(config.MASK_DATASET_PATH))]

split = train_test_split(imagePaths, maskPaths,test_size=config.TEST_SPLIT, random_state=42)

(trainImages, testImages) = split[:2]
(trainMasks, testMasks) = split[2:]

print("[INFO] saving testing image paths...")
f = open(config.TEST_PATHS, "w")
f.write("\n".join(testImages))
f.close()

transforms = transforms.Compose([transforms.ToPILImage(),
                                 transforms.Resize((config.INPUT_IMAGE_HEIGHT,config.INPUT_IMAGE_WIDTH)),
                                 transforms.ToTensor()])


trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,transforms=transforms)
testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,transforms=transforms)

print(f"[INFO] found {len(trainDS)} examples in the training set...")

trainLoader = DataLoader(trainDS, shuffle=True,batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY, num_workers=os.cpu_count())
testLoader = DataLoader(testDS, shuffle=False,batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,num_workers=os.cpu_count())

unet = UNet().to(config.DEVICE)

lossFunc = BCEWithLogitsLoss()
opt = Adam(unet.parameters(), lr=config.INIT_LR)

trainSteps = len(trainDS) // config.BATCH_SIZE
testSteps = len(testDS) // config.BATCH_SIZE

H = {"train_loss": [], "test_loss": []}


print("[INFO] training the network...")
startTime = time.time()
for e in tqdm(range(config.NUM_EPOCHS)):
	unet.train()
	totalTrainLoss = 0
	totalTestLoss = 0

	for (i, (x, y)) in enumerate(trainLoader):
		(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

		pred = unet(x)
		loss = lossFunc(pred, y)

		opt.zero_grad()
		loss.backward()
		opt.step()

		totalTrainLoss += loss

	with torch.no_grad():
		unet.eval()
		for (x, y) in testLoader:
			(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
			pred = unet(x)
			totalTestLoss += lossFunc(pred, y)

	avgTrainLoss = totalTrainLoss / trainSteps
	avgTestLoss = totalTestLoss / testSteps

	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["test_loss"].append(avgTestLoss.cpu().detach().numpy())

	print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
	print("Train loss: {:.6f}, Test loss: {:.4f}".format(avgTrainLoss, avgTestLoss))

endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))

plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["test_loss"], label="test_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.PLOT_PATH)

torch.save(unet, config.MODEL_PATH)