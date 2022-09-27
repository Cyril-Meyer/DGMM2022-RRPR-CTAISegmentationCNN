import os
import numpy as np
import skimage.io
import scipy.signal

INPUT_FOLDER = './DGMM2022-MEYER-DATA/I3_IMAGE_172-251/'
OUTPUT_FOLDER = './DGMM2022-MEYER-DATA/I3_IMAGE_172-251_LOWPASS/'

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
image = np.array(skimage.io.imread_collection(f'{INPUT_FOLDER}*.pgm'))
print(image.shape, image.dtype, image.min(), image.max())

pad = 16
image = np.pad(image, ((0, 0), (pad, pad), (pad, pad)), mode='edge')
image = scipy.signal.convolve(image, np.ones((1, 9, 9))/(9*9), mode='same', method='direct')
image = np.round(image).astype(np.uint8)
image = image[:, pad:-pad, pad:-pad]

print(image.shape, image.dtype, image.min(), image.max())
for z in range(image.shape[0]):
    skimage.io.imsave(f'{OUTPUT_FOLDER}i3_{z:04d}.pgm', image[z])
