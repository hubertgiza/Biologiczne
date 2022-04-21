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


image_paths = [os.path.join(config.IMAGE_DATASET_PATH,image) for image in sorted(os.listdir(config.IMAGE_DATASET_PATH))]
mask_paths = [os.path.join(config.MASK_DATASET_PATH,mask) for mask in sorted(os.listdir(config.MASK_DATASET_PATH))]

split = train_test_split(image_paths, mask_paths,test_size=config.TEST_SPLIT, random_state=42)

(train_images, test_images) = split[:2]
(trains_masks, test_masks) = split[2:]

print("[INFO] saving testing image paths...")
f = open(config.TEST_PATHS, "w")
f.write("\n".join(test_images))
f.close()

transforms = transforms.Compose([transforms.ToPILImage(),
                                 transforms.Resize((config.INPUT_IMAGE_HEIGHT,config.INPUT_IMAGE_WIDTH)),
                                 transforms.ToTensor()])


trains_ds = SegmentationDataset(imagePaths=train_images, maskPaths=trains_masks,transforms=transforms)
test_ds = SegmentationDataset(imagePaths=test_images, maskPaths=test_masks,transforms=transforms)

print(f"[INFO] found {len(trains_ds)} examples in the training set...")

train_loader = DataLoader(trains_ds, shuffle=True,batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY, num_workers=os.cpu_count())
test_loader = DataLoader(test_ds, shuffle=False,batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,num_workers=os.cpu_count())

unet = UNet().to(config.DEVICE)

loss_function = BCEWithLogitsLoss()
optimizer = Adam(unet.parameters(), lr=config.INIT_LR)

train_steps = len(trains_ds) // config.BATCH_SIZE
test_steps = len(test_ds) // config.BATCH_SIZE

H = {"train_loss": [], "test_loss": []}


print("[INFO] training the network...")
start_time = time.time()
for e in tqdm(range(config.NUM_EPOCHS)):
	unet.train()
	total_train_loss = 0
	total_test_loss = 0

	for (i, (x, y)) in enumerate(train_loader):
		(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

		pred = unet(x)
		loss = loss_function(pred, y)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		total_train_loss += loss

	with torch.no_grad():
		unet.eval()
		for (x, y) in test_loader:
			(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
			pred = unet(x)
			total_test_loss += loss_function(pred, y)

	average_train_loss = total_train_loss / train_steps
	average_test_loss = total_test_loss / test_steps

	H["train_loss"].append(average_train_loss.cpu().detach().numpy())
	H["test_loss"].append(average_test_loss.cpu().detach().numpy())

	print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
	print("Train loss: {:.6f}, Test loss: {:.4f}".format(average_train_loss, average_test_loss))

end_time = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(end_time - start_time))

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