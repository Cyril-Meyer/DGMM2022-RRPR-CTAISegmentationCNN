import argparse
import datetime
import os
import numpy as np
import tensorflow as tf
import tifffile
import tism
import data
import patch
import loss

print(tf.config.list_physical_devices('CPU'))
print(tf.config.list_physical_devices('GPU'))
print(tf.test.gpu_device_name())

# ---------- ARGUMENT PARSING -----------------------------
parser = argparse.ArgumentParser()
# Experimentation parameters
parser.add_argument('ID', type=str)
parser.add_argument('DATA_FOLDER', type=str)
parser.add_argument('OUTPUT_FOLDER', type=str)
parser.add_argument('DATASET', type=str) # I3, I3_MI or I3_ER
parser.add_argument('SETUP', type=str) # BASELINE or one of the 10 experimentation setup
'''
'AREA_2048_1048576', 'AREA_4096', 'AREA_DUAL_256_262144', 'COMPACITY_MAX_50-MAX_AREA_D_AREAN_H_D',
'COMPLEXITY-MAX_AREA_D_AREAN_H_D-LIMIT_AREA', 'CONTRAST_10_150', 'CONTRAST_DUAL_10_150',
'CONTRAST-MAX_AREA_D_AREAN_H_D-LIMIT_AREA', 'MGB-MAX_MGB-LIMIT_AREA', 'VOLUME-MAX_AREA_D_AREAN_H_D-LIMIT_AREA'
'''
# Default training parameters
parser.add_argument('--MODEL', type=str, default='UNET-R64')
parser.add_argument('--LOSS', type=str, default='DICE')
parser.add_argument('--EPOCHS', type=int, default=128)
parser.add_argument('--STEPS_PER_EPOCH', type=int, default=512)
parser.add_argument('--VALIDATION_STEPS', type=int, default=128)
parser.add_argument('--BATCH_SIZE', type=int, default=8)
parser.add_argument('--VERBOSE', type=int, default=0)

args = parser.parse_args()

ID = args.ID
DATA_FOLDER = args.DATA_FOLDER
OUTPUT_FOLDER = args.OUTPUT_FOLDER
DATASET = args.DATASET
SETUP = args.SETUP
MODEL = args.MODEL
LOSS = args.LOSS
EPOCHS = args.EPOCHS
STEPS_PER_EPOCH = args.STEPS_PER_EPOCH
VALIDATION_STEPS = args.VALIDATION_STEPS
BATCH_SIZE = args.BATCH_SIZE
VERBOSE = args.VERBOSE
DT = datetime.datetime.today().strftime("%j%H%M%S-%f")[:-4]

# ---------- EXPERIMENTATION FOLDER ---------------------------------
EXP_NAME = "CTAI_SEG_CNN"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

EXP_NAME = (EXP_NAME + "_" + str(DATASET) + "_" + str(SETUP) + "_" + str(LOSS) + "_" + str(ID)).replace(" ", "")
print(EXP_NAME)

# ---------- DATA LOADING -------------------------------------------
train_image, valid_image, test_image, train_label, valid_label, test_label = data.get(DATA_FOLDER, DATASET, SETUP)

# ---------- PATCH GENERATOR
if LOSS == 'DICE':
    train = patch.gen_patch_batch((256, 256), train_image, train_label, BATCH_SIZE, augmentation=True)
    valid = patch.gen_patch_batch((256, 256), valid_image, valid_label, BATCH_SIZE, augmentation=True)
    test  = patch.gen_patch_batch((256, 256), test_image,  test_label,  BATCH_SIZE, augmentation=True)
else:
    raise NotImplementedError

# ---------- MODEL PREPARATION --------------------------------------
# ---------- INPUT / OUTPUT
if SETUP == 'BASELINE':
    input_shape = (None, None, 1)
else:
    input_shape = (None, None, 2)

if DATASET == 'I3_ER' or DATASET == 'I3_MI':
    output_classes = 1
elif DATASET == 'I3':
    output_classes = 2
else:
    raise NotImplementedError

# ---------- LOSS
if output_classes == 1:
    if LOSS == 'DICE':
        output_activation = 'sigmoid'
        loss_function = loss.dice_coef_tf
    else:
        raise NotImplementedError
elif output_classes == 2:
    if LOSS == 'DICE':
        output_activation = 'sigmoid'
        loss_function = loss.dice_coef_tf_2_classes
    else:
        raise NotImplementedError

# ---------- U-NET
if MODEL == 'UNET-R64':
    # 32.4M params
    be = tism.backbone.VGG(initial_block_depth=64, initial_block_length=2, normalization="BatchNormalization")
    bd = tism.backbone.VGG(initial_block_depth=64, initial_block_length=2, normalization="BatchNormalization")

    model = tism.model.get(architecture=tism.architecture.UNet(input_shape=input_shape,
                                                               depth=5,
                                                               output_classes=output_classes,
                                                               output_activation=output_activation,
                                                               op_dim=2,
                                                               dropout=0.50,
                                                               pool_size=2,
                                                               multiple_outputs=False),
                           backbone_encoder=tism.backbone.ResBlock(backbone=be),
                           backbone_decoder=tism.backbone.ResBlock(backbone=bd))
elif MODEL == 'UNET-ResNet50':
    raise NotImplementedError
    base_model = tf.keras.applications.ResNet50(input_shape=input_shape, include_top=False, weights=None)
elif MODEL == 'UNET-ResNet50-P':
    raise NotImplementedError
    base_model = tf.keras.applications.ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
else:
    raise NotImplementedError

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
model.compile(optimizer=optimizer, loss=loss_function)
reducelrplateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
# ---------- SAVE BEST MODEL
checkpoint_path = f'{OUTPUT_FOLDER}/{EXP_NAME}/MODEL_BEST.h5'
savebestmodel = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=False,
                                                   monitor='val_loss', mode='min', save_best_only=True)

# ---------- TRAINING -----------------------------------------------
fit_history = model.fit(train, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS,
                        validation_data=valid, validation_steps=VALIDATION_STEPS,
                        verbose=VERBOSE,
                        callbacks=[reducelrplateau, savebestmodel])

# ---------- SAVE LAST MODEL
model.save(f'{OUTPUT_FOLDER}/{EXP_NAME}/MODEL_LAST.h5')

# ---------- END
eval_train = model.evaluate(train, steps=VALIDATION_STEPS)
eval_valid = model.evaluate(valid, steps=VALIDATION_STEPS)
eval_test = model.evaluate(test, steps=VALIDATION_STEPS)
print(eval_train, eval_valid, eval_test)
print('NO_ERROR_FINISHED', EXP_NAME)
