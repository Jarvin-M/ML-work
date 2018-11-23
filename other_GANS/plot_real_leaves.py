import numpy as np
import matplotlib.pyplot as plt

x = np.load('./datasets/swedish_leaf32x32pix_images.npy')
y = np.load('./datasets/swedish_leaf32x32pix_labels.npy')

fig, axs = plt.subplots(1, 15)
for cls in range(1, 16):
    img = x[np.where(y == cls)[0][0]]
    axs[cls-1].imshow(img)
    axs[cls-1].axis('off')
fig.savefig("./acgan/images/real_leaves.png")
plt.close()
