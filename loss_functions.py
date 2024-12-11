## Loss

from metrics import dice_coef, iou_coef, tversky
from tensorflow.keras import backend as K

## Taken from https://gist.github.com/CarloSegat/1a2816676c48607dac9dda38afe4f3d9

def wbce_loss():
    def wbce(y_true, y_pred, weight1=0.9, weight0=0.1 ) :
        y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
        y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
        logloss = -(y_true * K.log(y_pred) * weight1 + (1 - y_true) * K.log(1 - y_pred) * weight0 )
        return K.mean( logloss, axis=-1)

    return wbce

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def modified_loss(y_true, y_pred):
    return -(0.4*dice_coef(y_true, y_pred)+0.6*iou_coef(y_true, y_pred))

def focal_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    BCE = K.binary_crossentropy(y_true_f, y_pred_f)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(0.8 * K.pow((1-BCE_EXP), 2.) * BCE)
    return focal_loss

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def seg_loss(y_true, y_pred):
    return -(0.4*dice_coef(y_true, y_pred)+0.6*iou_coef(y_true, y_pred))