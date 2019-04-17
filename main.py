import yaml
import os
import tensorflow as tf
from model import CycleGan
from style_classifier import Classifer


def main(_):
    with open('config.yaml', 'r') as fh:
        config = yaml.safe_load(fh)

    tf.set_random_seed(config['random_seed'])

    for key, val in config['saving'].items():
        if 'dir' in key:
            ensure_dir(val)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:

        if config['arch']['type'] == 'cyclegan':
            model = CycleGan(sess, config)
            model.train() if config['phase'] == 'train' else model.test()

        if config['arch']['type'] == 'classifier':
            classifier = Classifer(sess, config)
            classifier.train(config) if config['phase'] == 'train' else classifier.test()


def ensure_dir(f):
    if not os.path.exists(f):
        os.makedirs(f)
    return f


if __name__ == '__main__':
    tf.app.run()
