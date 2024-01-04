if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero.models import VGG16
import dezero
from PIL import Image

'''
model = VGG16(pretrained=True)

x = np.random.randn(1, 3, 224, 224).astype(np.float32)
model.plot(x)
'''

url = 'https://github.com/oreilly-japan/deep-learning-from-scratch-3/raw/images/zebra.jpg' # pic of zebra
img_path = dezero.utils.get_file(url)
img = Image.open(img_path)
# img.show()
x = VGG16.preprocess(img) # preprocess - static method, called by class not instance
# preprocess - image -> (3, 224, 224) size ndarray + other preprocess
x = x[np.newaxis] # add axis for batch (3, 224, 224) -> (1, 3, 224, 224)

model = VGG16(pretrained=True)
with dezero.test_mode():
    y = model(x)
predict_id = np.argmax(y.data)

model.plot(x, to_file='vgg.pdf') # calculation graph
labels = dezero.datasets.ImageNet.labels() # dic - key: object ID, value: label
print(labels[predict_id])