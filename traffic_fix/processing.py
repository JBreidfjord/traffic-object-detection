import tensorflow as tf

from traffic_fix.utils import compute_iou, convert_to_corners, convert_to_xywh, swap_xy


class AnchorBox:
    """
    Generates anchor boxes.

    This class has operations to generate anchor boxes for feature maps at
    strides `[8, 16, 32, 64, 128]`, where each anchor box is of the format `[x, y, width, height]`.

    Attrs:
        aspect_ratios: A list of float values representing the aspect ratios of \
            the anchor boxes at each location on the feature map.
        scales: A list of float values representing the scale of the anchor boxes \
            at each location on the feature map.
        num_anchors: The number of anchor boxes at each location on the feature map.
        areas: A list of float values representing the areas of the anchor \
            boxes for each feature map in the feature pyramid.
        strides: A list of float values representing the strides for each feature \
            map in the feature pyramid.
    """

    def __init__(self):
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.scales = [2**x for x in [0, 1 / 3, 2 / 3]]

        self._num_anchors = len(self.aspect_ratios) * len(self.scales)
        self._strides = [2**i for i in range(3, 8)]
        self._areas = [x**2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
        self._anchor_dims = self._compute_dims()

    def _compute_dims(self):
        """
        Computes anchor box dimensions for all ratios and scales at all levels
        of the feature pyramid.
        """
        anchor_dims_all = []
        for area in self._areas:
            anchor_dims = []
            for ratio in self.aspect_ratios:
                anchor_height = tf.math.sqrt(area / ratio)
                anchor_width = area / anchor_height
                dims = tf.reshape(tf.stack([anchor_width, anchor_height], axis=-1), [1, 1, 2])
                for scale in self.scales:
                    anchor_dims.append(scale * dims)
            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))

        return anchor_dims_all

    def _get_anchors(self, feature_height, feature_width, level):
        """
        Generates anchor boxes for a given feature map size and level.

        Args:
            feature_height: An integer representing the height of the feature map.
            feature_width: An integer representing the width of the feature map.
            level: An integer representing the level of the feature map in the \
                feature pyramid.
                
        Returns:
            Anchor boxes with shape `(feature_height * feature_width * num_anchors, 4)`.
        """
        rx = tf.range(feature_width, dtype=tf.float32) + 0.5
        ry = tf.range(feature_height, dtype=tf.float32) + 0.5
        centers = tf.stack(tf.meshgrid(rx, ry), axis=-1) * self._strides[level - 3]
        centers = tf.expand_dims(centers, axis=-2)
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1])
        dims = tf.tile(self._anchor_dims[level - 3], [feature_height, feature_width, 1, 1])
        anchors = tf.concat([centers, dims], axis=-1)

        return tf.reshape(anchors, [feature_height * feature_width * self._num_anchors, 4])

    def get_anchors(self, image_height, image_width):
        """
        Generates anchor boxes for all feature maps of the feature pyramid.

        Args:
            image_height: Height of the input image.
            image_width: Width of the input image.

        Returns:
            Anchor boxes for all feature maps, stacked as a single tensor
            with shape `(total_anchors, 4)`.
        """
        anchors = [
            self._get_anchors(
                tf.math.ceil(image_height / 2**i), tf.math.ceil(image_width / 2**i), i
            )
            for i in range(3, 8)
        ]
        return tf.concat(anchors, axis=0)


class LabelEncoder:
    """
    Transforms the raw labels into targets for training.

    This class has operations to generate targets for a batch of sameples which
    is made up of the input images, bounding boxes for the objects present and
    their class ids.

    Attrs:
        anchor_box: Anchor box generator to encode the bounding boxes.
        box_variance: The scaling factors used to scale the bounding box targets.
    """

    def __init__(self):
        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor([0.1, 0.1, 0.2, 0.2], dtype=tf.float32)

    def _match_anchor_boxes(self, anchor_boxes, gt_boxes, match_iou=0.5, ignore_iou=0.4):
        """
        Matches ground truth boxes to anchor boxes based on IOU.
        
        - Calculates the pairwise IOU for the M `anchor_boxes` and N `gt_boxes` \
            to get a matrix of shape `(M, N)`.
        - The ground truth box with the maximum IOU in each row is assigned to the \
            anchor box provided the IOU is greater than `match_iou`.
        - If the maximum IOU in a row is less than `ignore_iou`, the anchor box is \
            assigned with the background class id.
        - The remaining anchor boxes that do not have any class assigned are ignored \
            during training.
            
        Args:
            anchor_boxes: A float tensor of shape `(total_anchors, 4)` representing \
                all anchor boxes for a given input image shape, where each anchor box \
                is of the format `[x, y, width, height]`.
            gt_boxes: A float tensor of shape `(num_objects, 4)` representing the ground \
                truth boxes, where each box is of the format `[x, y, width, height]`.
            match_iou: The IOU threshold used to match ground truth boxes to anchor boxes.
            ignore_iou: The IOU threshold used to assign background class id to anchor boxes.
            
        Returns:
            matched_gt_idx: Index of the matched object.
            positive_mask: A mask for anchor boxes that have been assigned ground truth boxes.
            ignore_mask: A mask for anchor boxes that need to be ignored during training.
        """
        iou_matrix = compute_iou(anchor_boxes, gt_boxes)
        max_iou = tf.reduce_max(iou_matrix, axis=1)
        matched_gt_idx = tf.argmax(iou_matrix, axis=1)
        positive_mask = tf.greater_equal(max_iou, match_iou)
        negative_mask = tf.less(max_iou, ignore_iou)
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))

        return (
            matched_gt_idx,
            tf.cast(positive_mask, dtype=tf.float32),
            tf.cast(ignore_mask, dtype=tf.float32),
        )

    def _compute_box_target(self, anchor_boxes, matched_gt_boxes):
        """Transforms the ground truth boxes into targets for training."""
        box_target = tf.concat(
            [
                (matched_gt_boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:],
                tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:]),
            ],
            axis=-1,
        )
        box_target = box_target / self._box_variance

        return box_target

    def _encode_sample(self, image_shape, gt_boxes, cls_ids):
        """Creates box and classification targets for a single sample."""
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        cls_ids = tf.cast(cls_ids, dtype=tf.float32)
        matched_gt_idx, positive_mask, ignore_mask = self._match_anchor_boxes(
            anchor_boxes, gt_boxes
        )
        matches_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)
        box_target = self._compute_box_target(anchor_boxes, matches_gt_boxes)
        matched_gt_cls_ids = tf.gather(cls_ids, matched_gt_idx)
        cls_target = tf.where(tf.not_equal(positive_mask, 1.0), -1.0, matched_gt_cls_ids)
        cls_target = tf.where(tf.equal(ignore_mask, 1.0), -2.0, cls_target)
        cls_target = tf.expand_dims(cls_target, axis=-1)
        label = tf.concat([box_target, cls_target], axis=-1)

        return label

    def encode_batch(self, batch_images, gt_boxes, cls_ids):
        """Creates box and classification targets for a batch of samples."""
        images_shape = tf.shape(batch_images)
        batch_size = images_shape[0]

        labels = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=True)
        for i in range(batch_size):
            label = self._encode_sample(images_shape, gt_boxes[i], cls_ids[i])
            labels = labels.write(i, label)
        batch_images = tf.keras.applications.resnet.preprocess_input(batch_images)

        return batch_images, labels.stack()


class DecodePredictions(tf.keras.layers.Layer):
    """
    A keras layer to decode predictions of the RetinaNet model.

    Attrs:
        num_classes: Number of classes in the dataset.
        confidence_threshold: Minimum probability to consider a detection.
        nms_iou_threshold: IOU threshold for non-max suppression.
        max_detections_per_class: Maximum number of detections to retain per class.
        max_detections: Maximum number of detections to retain across all classes.
        box_variance: The scaling factors used to scale bounding box predictions.
    """

    def __init__(
        self,
        num_classes=80,
        confidence_threshold=0.05,
        nms_iou_threshold=0.5,
        max_detections_per_class=100,
        max_detections=100,
        box_variance=[0.1, 0.1, 0.2, 0.2],
        **kwargs,
    ):
        super(DecodePredictions, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections

        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(box_variance, dtype=tf.float32)

    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        boxes = box_predictions * self._box_variance
        boxes = tf.concat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
                tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
            ],
            axis=-1,
        )
        boxes_transformed = convert_to_corners(boxes)

        return boxes_transformed

    def call(self, images, predictions):
        image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        box_predictions = predictions[:, :, :4]
        cls_predictions = tf.nn.sigmoid(predictions[:, :, 4:])
        boxes = self._decode_box_predictions(anchor_boxes[None, ...], box_predictions)

        return tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2),
            cls_predictions,
            self.max_detections_per_class,
            self.max_detections,
            self.nms_iou_threshold,
            self.confidence_threshold,
            clip_boxes=False,
        )


def random_flip_horizontal(image, boxes):
    """
    Flips image and boxes horizontally with 50% chance.
    
    Args:
        image: A 3D tensor of shape `(height, width, channels)` representing an image.
        boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes, \
            having normalized coordinates.
            
    Returns:
        Randomly flipped image and boxes.
    """
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack([1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], axis=-1)
    return image, boxes


def resize_and_pad_image(image, min_side=800.0, max_side=1333.0, jitter=[640, 1024], stride=128.0):
    """
    Resizes and pads image while preserving aspect ratio.
    
    - Resizes images so that the shorter side is equal to `min_side`
    - If longer side is greater than `max_side`, resizes image so that longer side is \
        equal to `max_side`
    - Pad with zeros on right and bottom to make the image shape divisible by `stride`
    
    Args:
        image: A 3D tensor of shape `(height, width, channels)` representing an image.
        min_side: The shorter side of the image is resized to this value, \
            if `jitter` is set to None.
        max_side: If the longer side of the image exceeds this value after resizing, \
            the image is resized such that the longer side now equals this value.
        jitter: A list of floats containing minimum and maximum size for scale jittering. \
            If available, the shorter side of the image will be resized to a random value \
            in this range.
        stride: The stride of the smallest feature map in the feature pyramid. \
            Can be calculated using `image_size / feature_map_size`.
            
    Returns:
        image: Resized and padded image.
        image_shape: Shape of the image before padding.
        ratio: The scaling factor used to resize the image.
    """
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    if jitter is not None:
        min_side = tf.random.uniform((), jitter[0], jitter[1], dtype=tf.float32)
    ratio = min_side / tf.reduce_min(image_shape)
    if ratio * tf.reduce_max(image_shape) > max_side:
        ratio = max_side / tf.reduce_max(image_shape)
    image_shape = ratio * image_shape
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
    padded_image_shape = tf.cast(tf.math.ceil(image_shape / stride) * stride, dtype=tf.int32)
    image = tf.image.pad_to_bounding_box(image, 0, 0, padded_image_shape[0], padded_image_shape[1])

    return image, image_shape, ratio


def preprocess_data(sample):
    """
    Applies preprocessing step to a single sample.
    
    Args:
        sample: A dictionary representing a single training sample.
        
    Returns:
        image: Resized and padded image with random horizontal flipping applied.
        bbox: Bounding boxes with shape `(num_objects, 4)` where each box is \
            of the format `[x, y, width, height]`.
        class_id: A tensor represnting the class id of the objects, having shape \
            `(num_objects,)`.
    """
    image = sample["image"]
    bbox = swap_xy(sample["objects"]["bbox"])
    class_id = tf.cast(sample["objects"]["label"], dtype=tf.int32)

    image, bbox = random_flip_horizontal(image, bbox)
    image, image_shape, _ = resize_and_pad_image(image)

    bbox = tf.stack(
        [
            bbox[:, 0] * image_shape[1],
            bbox[:, 1] * image_shape[0],
            bbox[:, 2] * image_shape[1],
            bbox[:, 3] * image_shape[0],
        ],
        axis=-1,
    )
    bbox = convert_to_xywh(bbox)

    return image, bbox, class_id
