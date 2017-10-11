import os
import tensorflow as tf

from .object_detection_dataset import ObjectDetectionDataset


class TFRecordDatasetTfApi(ObjectDetectionDataset):
    """
    Attributes:
        context_features (dict): Context features used to parse fixed sized
            tfrecords.
        sequence_features (dict): Sequence features used to parse the variable
            sized part of tfrecords (for ground truth bounding boxes).

    """
    def __init__(self, config, name='tfrecord_dataset', **kwargs):
        """
        Args:
            config: Configuration file used in session.
        """
        super(TFRecordDatasetTfApi, self).__init__(config, name=name, **kwargs)

        self._context_features = {
            'image_raw': tf.FixedLenFeature([], tf.string),
            'filename': tf.FixedLenFeature([], tf.string),
            'width': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
        }

        self._sequence_features = {
            'label': tf.VarLenFeature(tf.int64),
            'xmin': tf.VarLenFeature(tf.int64),
            'xmax': tf.VarLenFeature(tf.int64),
            'ymin': tf.VarLenFeature(tf.int64),
            'ymax': tf.VarLenFeature(tf.int64),
        }

    def _build(self):
        """Returns a tuple containing image, image metadata and label.

        Does not receive any input since it doesn't depend on anything inside
        the graph and it's the starting point of it.

        Returns:
            dequeue_dict ({}): Dequeued dict returning and image, bounding
                boxes, filename and the scaling factor used.

        TODO: Join filename, scaling_factor (and possible other fields) into a
        metadata.
        """

        # Find split file from which we are going to read.
        split_path = os.path.join(
            self._dataset_dir, '{}.tfrecords'.format(self._split)
        )

        dataset = tf.contrib.data.TFRecordDataset(split_path)
        dataset = dataset.map(self.parser)
        dataset = dataset.batch(self._batch_size)
        dataset = dataset.repeat(self._num_epochs)
        iterator = dataset.make_one_shot_iterator()
        return iterator

    def parser(self, record):

        context_example, sequence_example = tf.parse_single_sequence_example(
            record,
            context_features=self._context_features,
            sequence_features=self._sequence_features
        )

        image_raw = tf.image.decode_jpeg(context_example['image_raw'])

        image = tf.cast(image_raw, tf.float32)
        height = tf.cast(context_example['height'], tf.int32)
        width = tf.cast(context_example['width'], tf.int32)
        image_shape = tf.stack([height, width, 3])
        image = tf.reshape(image, image_shape)

        label = self.sparse_to_tensor(sequence_example['label'])
        xmin = self.sparse_to_tensor(sequence_example['xmin'])
        xmax = self.sparse_to_tensor(sequence_example['xmax'])
        ymin = self.sparse_to_tensor(sequence_example['ymin'])
        ymax = self.sparse_to_tensor(sequence_example['ymax'])

        # Stack parsed tensors to define bounding boxes of shape (num_boxes, 5)
        bboxes = tf.stack([xmin, ymin, xmax, ymax, label], axis=1)

        # Resize images (if needed)
        image, bboxes, scale_factor = self._resize_image(image, bboxes)

        image, bboxes, applied_augmentations = self._augment(image, bboxes)

        filename = tf.cast(context_example['filename'], tf.string)

        return {'image': image, 'bboxes': bboxes, 'filename': filename}

    def sparse_to_tensor(self, sparse_tensor, dtype=tf.int32, axis=[1]):
        return tf.squeeze(
            tf.cast(tf.sparse_tensor_to_dense(sparse_tensor), dtype), axis=axis
        )
