import numpy as np
import tensorflow as tf
import tempfile
import os


from easydict import EasyDict
from luminoth.datasets import TFRecordDataset


def _int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


class TFRecordDatasetTest(tf.test.TestCase):
    def setUp(self):
        image = np.random.randint(low=0, high=255, size=(1, 600, 800, 3))
        object_feature = tf.train.FeatureLists(feature_list={
            'label': tf.train.FeatureList(feature=[_int64([0])]),
            'xmin': tf.train.FeatureList(feature=[_int64([10])]),
            'xmax': tf.train.FeatureList(feature=[_int64([100])]),
            'ymin': tf.train.FeatureList(feature=[_int64([10])]),
            'ymax': tf.train.FeatureList(feature=[_int64([100])]),
        })
        context = tf.train.Features(feature={
            'width': _int64([600]),
            'height': _int64([600]),
            'depth': _int64([3]),
            'filename': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=['test'.encode('utf-8')])),
            'image_raw': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[image.tobytes()]))
        })
        example = tf.train.SequenceExample(
            feature_lists=object_feature, context=context)

        dirname = tempfile.mkdtemp()
        filename = 'test.tfrecords'
        path = os.path.join(dirname, filename)
        writer = tf.python_io.TFRecordWriter(path)
        writer.write(example.SerializeToString())
        writer.close()

        print('Saved tmp file in {}'.format(path))

        self.config = EasyDict({
            'dataset': {
                'dir': dirname,
                'split': 'test',
                'image_preprocessing': {
                    'min_size': 100,
                    'max_size': 1024
                },
                'data_augmentation': []
            },
            'train': {
                'num_epochs': 1,
                'batch_size': 1,
                'random_shuffle': False
            }
        })

    def _run_tfrecord(self):
        dataset = TFRecordDataset(self.config)
        results = dataset()

        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())

            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(sess=sess, coord=coord)
            results = sess.run(results)
            return results['image']

    def testBasic(self):
        results = self._run_tfrecord()
        print(results)


if __name__ == '__main__':
    tf.test.main()
