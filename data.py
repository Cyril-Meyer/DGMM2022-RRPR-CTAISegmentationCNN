import numpy as np
import tifffile


# ---------- DATA LOADING -------------------------------------------
def get(DATA_FOLDER, DATASET, SETUP):
    if 'I3' not in DATASET:
        raise NotImplementedError

    # ---------- IMAGE
    image = np.array(
            tifffile.imread(f'{DATA_FOLDER}/I3_IMAGE/I3.tiff') / 255.0, dtype=np.float32)

    if SETUP == 'BASELINE':
        pass
    elif SETUP == 'AREA_2048_1048576':
        image_attr = np.array(
            tifffile.imread(DATA_FOLDER + '/I3_IMAGE/AREA_2048_1048576.tiff') / 255.0, dtype=np.float32)
    elif SETUP == 'AREA_4096':
        image_attr = np.array(
            tifffile.imread(DATA_FOLDER + '/I3_IMAGE/AREA_4096.tiff') / 255.0, dtype=np.float32)
    elif SETUP == 'AREA_DUAL_256_262144':
        image_attr = np.array(
            tifffile.imread(DATA_FOLDER + '/I3_IMAGE/AREA_DUAL_256_262144.tiff') / 255.0, dtype=np.float32)
    elif SETUP == 'COMPACITY_MAX_50-MAX_AREA_D_AREAN_H_D':
        image_attr = np.array(
            tifffile.imread(DATA_FOLDER + '/I3_IMAGE/COMPACITY_MAX_50-MAX_AREA_D_AREAN_H_D.tiff') / 255.0, dtype=np.float32)
    elif SETUP == 'COMPLEXITY-MAX_AREA_D_AREAN_H_D-LIMIT_AREA':
        image_attr = np.array(
            tifffile.imread(DATA_FOLDER + '/I3_IMAGE/COMPLEXITY-MAX_AREA_D_AREAN_H_D-LIMIT_AREA.tiff') / 255.0, dtype=np.float32)
    elif SETUP == 'CONTRAST_10_150':
        image_attr = np.array(
            tifffile.imread(DATA_FOLDER + '/I3_IMAGE/CONTRAST_10_150.tiff') / 255.0, dtype=np.float32)
    elif SETUP == 'CONTRAST_DUAL_10_150':
        image_attr = np.array(
            tifffile.imread(DATA_FOLDER + '/I3_IMAGE/CONTRAST_DUAL_10_150.tiff') / 255.0, dtype=np.float32)
    elif SETUP == 'CONTRAST-MAX_AREA_D_AREAN_H_D-LIMIT_AREA':
        image_attr = np.array(
            tifffile.imread(DATA_FOLDER + '/I3_IMAGE/CONTRAST-MAX_AREA_D_AREAN_H_D-LIMIT_AREA.tiff') / 255.0, dtype=np.float32)
    elif SETUP == 'MGB-MAX_MGB-LIMIT_AREA':
        image_attr = np.array(
            tifffile.imread(DATA_FOLDER + '/I3_IMAGE/MGB-MAX_MGB-LIMIT_AREA.tiff') / 255.0, dtype=np.float32)
    elif SETUP == 'VOLUME-MAX_AREA_D_AREAN_H_D-LIMIT_AREA':
        image_attr = np.array(
            tifffile.imread(DATA_FOLDER + '/I3_IMAGE/VOLUME-MAX_AREA_D_AREAN_H_D-LIMIT_AREA.tiff') / 255.0, dtype=np.float32)
    else:
        raise NotImplementedError

    # channel last
    if SETUP == 'BASELINE':
        image = np.stack([image], axis=-1)
    else:
        image = np.stack([image, image_attr], axis=-1)

    train_image = image[0:40]
    valid_image = image[40:60]
    test_image = image[60:80]

    # ---------- LABEL
    if DATASET == 'I3':
        label = np.stack([np.array(tifffile.imread(DATA_FOLDER + '/I3_LABEL/I3_MI.tiff'), dtype=np.float32),
                          np.array(tifffile.imread(DATA_FOLDER + '/I3_LABEL/I3_ER.tiff'), dtype=np.float32)], axis=-1)
    elif DATASET == 'I3_MI':
        label = np.stack([np.array(tifffile.imread(DATA_FOLDER + '/I3_LABEL/I3_MI.tiff'), dtype=np.float32)], axis=-1)
    elif DATASET == 'I3_ER':
        label = np.stack([np.array(tifffile.imread(DATA_FOLDER + '/I3_LABEL/I3_ER.tiff'), dtype=np.float32)], axis=-1)
    else:
        raise NotImplementedError

    train_label = label[0:40]
    valid_label = label[40:60]
    test_label = label[60:80]

    return train_image, valid_image, test_image, train_label, valid_label, test_label
