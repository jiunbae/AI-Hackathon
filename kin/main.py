# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
import tensorflow as tf

from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Embedding, LSTM, GRU, Conv1D, Conv2D, GlobalMaxPool1D, Dense, Dropout


import nsml
from nsml import DATASET_PATH, HAS_DATASET, IS_ON_NSML
from dataset import KinQueryDataset, preprocess

# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(sess, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *args):
        # directory
        os.makedirs(dir_name, exist_ok=True)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(dir_name, 'model'))

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *args):
        saver = tf.train.Saver()
        # find checkpoint
        ckpt = tf.train.get_checkpoint_state(dir_name)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(dir_name, checkpoint))
        else:
            raise NotImplemented('No checkpoint!')
        print('Model loaded')

    def infer(raw_data, **kwargs):
        """

        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        preprocessed_data = preprocess(raw_data, config.strmaxlen)
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        pred = sess.run(output_sigmoid, feed_dict={x: preprocessed_data})
        clipped = np.array(pred > config.threshold, dtype=np.int)
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 확률의 값은 영향을 미치지 않습니다
        return list(zip(pred.flatten(), clipped.flatten()))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


def _batch_loader(iterable, n=1):
    """
    데이터를 배치 사이즈만큼 잘라서 보내주는 함수입니다. PyTorch의 DataLoader와 같은 역할을 합니다

    :param iterable: 데이터 list, 혹은 다른 포맷
    :param n: 배치 사이즈
    :return:
    """
    length = len(iterable)
    for n_idx in range(0, length, n):
        yield iterable[n_idx:min(n_idx + n, length)]


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


class KerasCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        nsml.save(epoch)
        nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=0,
                        train__loss=0, step=0)

    def on_batch_end(self, batch, logs={}):
        pass
        
callback = KerasCallback()

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=10)
    args.add_argument('--batch', type=int, default=2000)
    args.add_argument('--strmaxlen', type=int, default=400)
    args.add_argument('--embedding', type=int, default=8)
    args.add_argument('--threshold', type=float, default=0.5)
    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../sample_data/kin/'

    # 모델의 specification
    input_size = config.embedding*config.strmaxlen
    output_size = 1
    hidden_layer_size = 200
    learning_rate = 0.001
    character_size = 251

    x = tf.placeholder(tf.int32, [None, config.strmaxlen])
    y_ = tf.placeholder(tf.float32, [None, output_size])

    # 임베딩
    char_embedding = tf.get_variable('char_embedding', [character_size, config.embedding])
    embedded = tf.nn.embedding_lookup(char_embedding, x)

    # 첫 번째 레이어
    first_layer_weight = weight_variable([input_size, hidden_layer_size])
    first_layer_bias = bias_variable([hidden_layer_size])
    hidden_layer = tf.matmul(tf.reshape(embedded, (-1, input_size)),
                             first_layer_weight) + first_layer_bias

    # 두 번째 (아웃풋) 레이어
    second_layer_weight = weight_variable([hidden_layer_size, output_size])
    second_layer_bias = bias_variable([output_size])
    output = tf.matmul(hidden_layer, second_layer_weight) + second_layer_bias
    output_sigmoid = tf.sigmoid(output)

    # loss와 optimizer
    binary_cross_entropy = tf.reduce_mean(-(y_ * tf.log(output_sigmoid)) - (1-y_) * tf.log(1-output_sigmoid))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(binary_cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # DONOTCHANGE: Reserved for nsml
    bind_model(sess=sess, config=config)

    # DONOTCHANGE: Reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        # 데이터를 로드합니다.
        dataset = KinQueryDataset(DATASET_PATH, config.strmaxlen)
        dataset_len = len(dataset)
        
        trained = model.fit([X_train['left'], X_train['right']], Y_train,
                   batch_size=config.batch, epochs=config.epochs, callbacks=[callback],
                   validation_data=([X_validation['left'], X_validation['right']], Y_validation))

    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.3, 0), (0.7, 1), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            queries = f.readlines()
        res = []
        for batch in _batch_loader(queries, config.batch):
            temp_res = nsml.infer(batch)
            res += temp_res
        print(res)