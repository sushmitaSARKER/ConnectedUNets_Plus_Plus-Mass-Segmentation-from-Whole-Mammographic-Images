## Metrics

import tensorflow as tf
from tensorflow.keras import backend as K

def dice_coef(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """
    Calculates the Dice Coefficient for evaluating segmentation overlap.

    The Dice Coefficient is a common metric for comparing the pixel-wise agreement
    between a predicted segmentation and its corresponding ground truth.
    The formula is:
        $$ \text{Dice} = \frac{2 \cdot |A \cap B|}{|A| + |B|} $$
    where A is the ground truth mask and B is the predicted mask.

    Args:
        y_true: The ground truth segmentation mask (Tensor).
        y_pred: The predicted segmentation mask (Tensor).
        smooth: A small constant added to the numerator and denominator to
                ensure numerical stability and prevent division by zero.

    Returns:
        The Dice Coefficient score as a TensorFlow tensor.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    numerator = 2. * intersection + smooth
    denominator = K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    return numerator / denominator


def jaccard_index(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """
    Calculates the Jaccard Index, also known as Intersection over Union (IoU).

    The Jaccard Index is a metric used to gauge the similarity and diversity of
    sample sets. It is defined as the size of the intersection divided by the
    size of the union of the two sets:
        $$ \text{Jaccard} = \frac{|A \cap B|}{|A \cup B|} $$
    where A is the ground truth mask and B is the predicted mask.

    Args:
        y_true: The ground truth segmentation mask (Tensor).
        y_pred: The predicted segmentation mask (Tensor).
        smooth: A small constant added to the numerator and denominator to
                ensure numerical stability and prevent division by zero.

    Returns:
        The Jaccard Index (IoU) score as a TensorFlow tensor.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)