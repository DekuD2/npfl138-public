import numpy as np
from skimage.filters import gabor_kernel
from matplotlib import pyplot as plt

thetas = [np.pi / 4, np.pi / 2]
phases = [0, np.pi / 2]

fig, axes = plt.subplots(1, 4)
i = 0

for t in thetas:
    for p in phases:
        scale = np.sqrt(2) if t > np.pi / 4 else 1
        kernel = gabor_kernel(frequency=0.1 / 6, theta=t, offset=p, n_stds=4 / scale)
        axes[i].title.set_text(f"({t:.2f}, {p:.2f})")
        axes[i].axis("off")
        axes[i].imshow(kernel.real, cmap='gray')
        i += 1

plt.savefig("kernels.png", dpi=460)
