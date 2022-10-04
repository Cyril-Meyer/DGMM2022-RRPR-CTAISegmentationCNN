import argparse
import numpy as np
import tensorflow as tf
import tifffile
import medpy.metric.binary
import data
import loss
import metrics

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
parser.add_argument('SETUP', type=str)
parser.add_argument('BEST_OR_LAST', type=str) # BEST or LAST

# Default parameters
parser.add_argument('--THRESHOLD', type=int, default=0.5)
parser.add_argument('--MODEL', type=str, default='UNET-R64')
parser.add_argument('--LOSS', type=str, default='DICE')

args = parser.parse_args()

ID = args.ID
DATA_FOLDER = args.DATA_FOLDER
OUTPUT_FOLDER = args.OUTPUT_FOLDER
DATASET = args.DATASET
SETUP = args.SETUP
BEST_OR_LAST = args.BEST_OR_LAST
THRESHOLD = args.THRESHOLD
MODEL = args.MODEL
LOSS = args.LOSS

# ---------- DATA LOADING -------------------------------------------
train_image, valid_image, test_image, train_label, valid_label, test_label = data.get(DATA_FOLDER, DATASET, SETUP)

# ---------- MODEL --------------------------------------------------
EXP_NAME = "CTAI_SEG_CNN"
EXP_NAME = (EXP_NAME + "_" + str(DATASET) + "_" + str(SETUP) + "_" + str(LOSS) + "_" + str(ID)).replace(" ", "")
MODEL_FILE = f'{OUTPUT_FOLDER}/{EXP_NAME}/MODEL_{BEST_OR_LAST}.h5'
custom_objects = {'dice_coef_tf': loss.dice_coef_tf, 'dice_coef_tf_2_classes': loss.dice_coef_tf_2_classes}
with tf.keras.utils.custom_object_scope(custom_objects):
    model = tf.keras.models.load_model(MODEL_FILE)

# ---------- EVALUATION ---------------------------------------------
valid_pred = np.zeros(valid_label.shape, dtype=np.float32)
test_pred = np.zeros(test_label.shape, dtype=np.float32)

for z in range(valid_label.shape[0]):
    valid_pred[z] = model.predict(valid_image[z:z+1, :, :, :])

for z in range(test_label.shape[0]):
    test_pred[z] = model.predict(test_image[z:z+1, :, :, :])


# ---------- OUTPUT -------------------------------------------------
EXP_NAME = "CTAI_SEG_CNN"
EXP_NAME = (EXP_NAME + "_" + str(DATASET) + "_" + str(SETUP) + "_" + str(LOSS) + "_" + BEST_OR_LAST + "_" + str(ID)).replace(" ", "")

file = open(f"{OUTPUT_FOLDER}/{EXP_NAME}.csv", "w")
file.write("EXP_NAME,ID,DATASET,SETUP,THRESHOLD,MODEL,BEST_OR_LAST,LOSS,")
for c in range(test_pred.shape[-1]):
    file.write(f'PRECISION_{c + 1},RECALL_{c + 1},F1_{c + 1},IOU_{c + 1},ASSD_{c + 1},')
for c in range(valid_pred.shape[-1]):
    file.write(f'VAL_PRECISION_{c + 1},VAL_RECALL_{c + 1},VAL_F1_{c + 1},VAL_IOU_{c + 1},VAL_ASSD_{c + 1}')
file.write('\n')

file.write(f'{EXP_NAME},{ID},{DATASET},{SETUP},{THRESHOLD},{MODEL},{BEST_OR_LAST},{LOSS},')

for c in range(test_pred.shape[-1]):
    label = test_label[..., c] > THRESHOLD
    label_flat = label.flatten()
    pred = test_pred[..., c] > THRESHOLD
    pred_flat = pred.flatten()

    precision = metrics.precision(label_flat, pred_flat)
    recall    = metrics.recall(label_flat, pred_flat)
    f1        = metrics.f1(label_flat, pred_flat)
    iou       = metrics.iou(label_flat, pred_flat)

    assd = np.inf
    try:
        assd = medpy.metric.binary.assd(label, pred)
    except:
        assd = np.inf
    file.write(f'{precision},{recall},{f1},{iou},{assd},')

for c in range(valid_pred.shape[-1]):
    label = valid_label[..., c] > THRESHOLD
    label_flat = label.flatten()
    pred = valid_pred[..., c] > THRESHOLD
    pred_flat = pred.flatten()

    precision = metrics.precision(label_flat, pred_flat)
    recall    = metrics.recall(label_flat, pred_flat)
    f1        = metrics.f1(label_flat, pred_flat)
    iou       = metrics.iou(label_flat, pred_flat)

    assd = np.inf
    try:
        assd = medpy.metric.binary.assd(label, pred)
    except:
        assd = np.inf
    file.write(f'{precision},{recall},{f1},{iou},{assd}')

file.write('\n')
file.close()
