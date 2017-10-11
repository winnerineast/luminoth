import os
import json
import tensorflow as tf

from luminoth.utils.dataset import (
    read_image, to_int64, to_string, to_bytes
)
from .dataset import DatasetTool, InvalidDataDirectory


class Cavani(DatasetTool):
    def __init__(self, data_dir):
        super(Cavani, self).__init__()
        self._data_dir = data_dir
        self._imagesets_path = os.path.join(data_dir, 'ImageSets')
        self._images_path = os.path.join(data_dir, 'Data')
        self._annotations_path = os.path.join(data_dir, 'Annotations')

    def is_valid(self):
        if not tf.gfile.Exists(self.data_dir):
            raise InvalidDataDirectory(
                '"{}" does not exist.'.format(self._data_dir)
            )

        if not tf.gfile.Exists(self._imagesets_path):
            raise InvalidDataDirectory('ImageSets path is missing')

        if not tf.gfile.Exists(self._images_path):
            raise InvalidDataDirectory('Images path is missing')

        if not tf.gfile.Exists(self._annotations_path):
            raise InvalidDataDirectory('Annotations path is missing')

    def read_classes(self):
        return ['cavani', 'notcavani']

    def get_split_path(self, split):
        return os.path.join(self._imagesets_path, '{}.txt'.format(split))

    def get_image_path(self, image_id):
        return os.path.join(self._images_path, '{}.jpg'.format(image_id))

    def load_split(self, split='train'):
        split_path = self.get_split_path(split)

        if not tf.gfile.Exists(split_path):
            raise ValueError('"{}" not found'.format(split))

        with tf.gfile.GFile(split_path) as f:
            for line in f:
                yield line.strip()

    def get_split_size(self, split):
        total_records = 0
        for line in self.load_split(split):
            total_records += 1

        return total_records

    def get_image_annotation(self, image_id):
        return os.path.join(self._annotations_path, '{}.json'.format(image_id))

    def image_to_example(self, classes, image_id):
        annotation_path = self.get_image_annotation(image_id)
        image_path = self.get_image_path(image_id)

        # Read both the image and the annotation into memory.
        annotation = json.load(tf.gfile.GFile(annotation_path))
        image = read_image(image_path)

        obj_vals = {
            'label': [],
            'xmin': [],
            'ymin': [],
            'xmax': [],
            'ymax': [],
        }

        objects = annotation.get('objects')
        if objects is None:
            # If there's no bounding boxes, we don't want it
            return
        for b in annotation['objects']:
            try:
                label_id = classes.index(b['id'])
            except ValueError:
                continue
            obj_vals['label'].append(to_int64(label_id))
            obj_vals['xmin'].append(to_int64(b['bbox']['xmin']))
            obj_vals['ymin'].append(to_int64(b['bbox']['ymin']))
            obj_vals['xmax'].append(to_int64(b['bbox']['xmax']))
            obj_vals['ymax'].append(to_int64(b['bbox']['ymax']))

        if len(obj_vals['label']) == 0:
            # No bounding box matches the available classes.
            return

        object_feature_lists = {
            'label': tf.train.FeatureList(feature=obj_vals['label']),
            'xmin': tf.train.FeatureList(feature=obj_vals['xmin']),
            'ymin': tf.train.FeatureList(feature=obj_vals['ymin']),
            'xmax': tf.train.FeatureList(feature=obj_vals['xmax']),
            'ymax': tf.train.FeatureList(feature=obj_vals['ymax']),
        }

        object_features = tf.train.FeatureLists(
            feature_list=object_feature_lists
        )

        sample = {
            'width': to_int64(annotation['size']['width']),
            'height': to_int64(annotation['size']['height']),
            'depth': to_int64(3),
            'filename': to_string(str(image_id)),
            'image_raw': to_bytes(image),
        }

        # Now build an `Example` protobuf object and save with the writer.
        context = tf.train.Features(feature=sample)
        example = tf.train.SequenceExample(
            feature_lists=object_features, context=context
        )

        return example
