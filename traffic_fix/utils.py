import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def swap_xy(boxes):
    """
    Swaps order of the x and y coordinates of boxes.

    Args:
        boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes.

    Returns:
        Swapped boxes with original shape.
    """
    return tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)


def convert_to_xywh(boxes):
    """
    Changes the box format to center, width and height.

    Args:
        boxes: A tensor of rank 2 or higher with shape `(..., num_boxes, 4)` \
            representing bounding boxes where each box is of the format `[xmin, ymin, xmax, ymax]`.

    Returns:
        Converted boxes with original shape.
    """
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]], axis=-1
    )


def convert_to_corners(boxes):
    """
    Changes the box format to corner coordinates.

    Args:
        boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)` \
            representing bounding boxes where each box is of the format `[x, y, width, height]`.

    Return:
        Converted boxes with original shape.
    """
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0], axis=-1
    )


def compute_iou(boxes1, boxes2):
    """
    Computes pairwise Intersection Over Union (IOU) matrix for given boxes.

    Args:
        boxes1: A tensor with shape `(N, 4)` representing bounding boxes \
            where each box is of the format `[x, y, width, height]`.
        boxes2: A tensor with shape `(M, 4)` representing bounding boxes \
            where each box is of the format `[x, y, width, height]`.

    Returns:
        Pairwise IOU matrix with shape `(N, M)`, where the value at ith row and jth column
        holds the IOU between ith box and jth box from boxes1 and boxes2 respectively
    """
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = convert_to_corners(boxes2)
    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = tf.maximum(boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8)

    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)


def visualize_detections(
    image, boxes, classes, scores, figsize=(7, 7), linewidth=1, color=[0, 0, 1]
):
    """Visualizes detections."""
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        text = f"{_cls}: {score:.2f}"
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle([x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth)
        ax.add_patch(patch)
        ax.text(
            x1, y1, text, bbox={"facecolor": color, "alpha": 0.4}, clip_box=ax.clipbox, clip_on=True
        )
    plt.show()

    return ax
