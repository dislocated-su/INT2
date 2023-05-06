# Find mean and std of each image as discibed https://www.geeksforgeeks.org/how-to-normalize-images-in-pytorch/


transformToTensor = transforms.Compose([
    transforms.ToTensor()
])

trainsetNotNorm = datasets.Flowers102("/data/", split="train", download=True, transform=transformToTensor)

means = [0, 0, 0]
stds = [0, 0, 0]
for img, _ in trainsetNotNorm:
  means = [sum(i) for i in zip(means, img.mean([1, 2]))]
  stds =  [sum(i) for i in zip(stds, img.std([1, 2]))]

means = np.array(means) / len(trainsetNotNorm)
stds = np.array(stds) / len(trainsetNotNorm)