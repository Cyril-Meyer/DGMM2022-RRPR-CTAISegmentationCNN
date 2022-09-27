import os
import numpy as np
import skimage.io
import tifffile

INPUT_FOLDER_IMAGE = './DGMM2022-MEYER-DATA/I3_IMAGE_172-251'
INPUT_FOLDER_PREPROCESS = './DGMM2022-MEYER-DATA/I3_IMAGE_172-251_LOWPASS'
INPUT_FOLDER_LABEL_MI = './DGMM2022-MEYER-DATA/I3_MI_172-251'
INPUT_FOLDER_LABEL_ER = './DGMM2022-MEYER-DATA/I3_ER_172-251'
OUTPUT_FOLDER_IMAGE = './DGMM2022-MEYER-DATA/I3_IMAGE'
OUTPUT_FOLDER_LABEL = './DGMM2022-MEYER-DATA/I3_LABEL'
if not os.path.exists(OUTPUT_FOLDER_IMAGE):
    os.makedirs(OUTPUT_FOLDER_IMAGE)

if not os.path.exists(OUTPUT_FOLDER_LABEL):
    os.makedirs(OUTPUT_FOLDER_LABEL)

image = np.array(skimage.io.imread_collection(f'{INPUT_FOLDER_IMAGE}/*.pgm'))
print(image.shape, image.dtype, image.min(), image.max())

label_mi = np.array(skimage.io.imread_collection(f'{INPUT_FOLDER_LABEL_MI}/*.pgm'))
label_er = np.array(skimage.io.imread_collection(f'{INPUT_FOLDER_LABEL_ER}/*.pgm'))
print(label_mi.shape, label_mi.dtype, label_mi.min(), label_mi.max())
print(label_er.shape, label_er.dtype, label_er.min(), label_er.max())

x = 900
x_size = 1536
y = 0
y_size = 1408

image = image[:, y:y+y_size, x:x+x_size]
label_mi = label_mi[:, y:y+y_size, x:x+x_size]
label_er = label_er[:, y:y+y_size, x:x+x_size]

tifffile.imwrite(f'{OUTPUT_FOLDER_IMAGE}/I3.tiff', image)
tifffile.imwrite(f'{OUTPUT_FOLDER_LABEL}/I3_MI.tiff', label_mi)
tifffile.imwrite(f'{OUTPUT_FOLDER_LABEL}/I3_ER.tiff', label_er)


for folder in [x[0] for x in os.walk(INPUT_FOLDER_PREPROCESS)]:
    image = np.array(skimage.io.imread_collection(f'{folder}/*.pgm'))
    image = image[:, y:y + y_size, x:x + x_size]
    print(image.shape, image.dtype, image.min(), image.max(), folder)
    tifffile.imwrite(f'{OUTPUT_FOLDER_IMAGE}/{folder.split("/")[-1]}.tiff', image)
