from keras.api.datasets import fashion_mnist
import pandas as pd
import numpy as np
data = fashion_mnist.load_data()
X_train, y_train = data[0][0], data[0][1]
X_test, y_test = data[1][0], data[1][1]
X_train = np.expand_dims(X_train, axis = -1)
X_test = np.expand_dims(X_test, axis = -1)
y_train = pd.get_dummies(pd.Categorical(y_train)).values
y_test = pd.get_dummies(pd.Categorical(y_test)).values
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['font.size'] = 48
def show_pictures(arrs):
    arr_cnt = arrs.shape[0]
    fig, axes = plt.subplots(1, arr_cnt, figsize=(5 * arr_cnt, arr_cnt))
    for axis, pic in zip(axes, arrs):
        axis.imshow(pic.squeeze(), cmap='gray')
        axis.axis('off')
    fig.tight_layout()
    return fig  



demo_images = X_train[:10, ..., 0]
show_pictures(demo_images).suptitle("Zdjęcia pierwotne")
odbicia_poziome = demo_images[..., ::-1]
show_pictures(odbicia_poziome).suptitle("Odbicia poziome")
odbicia_pionowe = demo_images[..., ::-1, :]
show_pictures(odbicia_pionowe).suptitle("Odbicia pionowe")
from PIL import Image
rotated_images = demo_images.copy()
img_size = demo_images.shape[1:]
angles = np.random.randint(-30, 30, len(rotated_images))
for i, img in enumerate(rotated_images):
    angle = np.random.randint(-30, 30)
    img = Image.fromarray(img).rotate(angle, expand=1).resize(img_size)
    rotated_images[i] = np.array(img)
show_pictures(rotated_images)
from PIL import Image
rotated_images = demo_images.copy()
img_size = demo_images.shape[1:]
for i, img in enumerate(rotated_images):
    angle = np.random.randint(-30, 30)
    left, upper = np.random.randint(0, 5, 2)
    right, lower = np.random.randint(23, 28, 2)
    img = Image.fromarray(img).crop((left, upper, right, lower)).resize(img_size)
    rotated_images[i] = np.array(img)
show_pictures(rotated_images)
