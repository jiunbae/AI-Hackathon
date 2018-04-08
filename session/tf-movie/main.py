import os
import numpy as np
import tensorflow as tf
from gensim.models import Doc2Vec

import nsml
import dataset
from model import TacotronEncoder
from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML

flags = tf.app.flags
flags.DEFINE_float('lr', 1e-3, 'Float, learning rate, default 1e-4')
flags.DEFINE_float('b1', 0.9, 'Float, beta1 value for Adam, default 0.9')
flags.DEFINE_integer('vector_size', 300, 'Int, vector size of the word embedding vector, default 300')
flags.DEFINE_integer('max_len', 200, 'Int, maximum length of the sentence, default 200')
flags.DEFINE_integer('embed_size', 128, 'Int, size of the encoded vector, default 128')
flags.DEFINE_integer('block_num', 16, 'Int, size of the conv1d bank, default 16')
flags.DEFINE_integer('highway_num', 4, 'Int, number of the highway layer, default 4')
flags.DEFINE_integer('batch_size', 32, 'Int, size of batch, default 32')
flags.DEFINE_integer('n_epoch', 100, 'Int, number of epoch, default 100')
flags.DEFINE_string('doc2vec_model', './doc2vec_model/doc2vec_twitter_kowiki_300000_docs.model', 'String, path of the doc2vec model.')
flags.DEFINE_string('model_name', 'default', 'String, name of the model, default default')
flags.DEFINE_string('iteration', '0', 'nsml reserved')
flags.DEFINE_string('mode', 'train', 'nsml reserved')
flags.DEFINE_integer('pause', 0, 'nsml reserved')
FLAGS = flags.FLAGS


# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(model, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *args):
        os.makedirs(dir_name, exist_ok=True)
        model.dump(os.path.join(dir_name, config.model_name))

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *args):
        model.load(os.path.join(dir_name, config.model_name))

    def infer(raw_data, **kwargs):
        """
        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        preprocessed_data = dataset.preproc(raw_data, config.model, config.max_len)
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        point = model.pred(preprocessed_data)
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(confidence interval, 포인트)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 confidence interval의 값은 영향을 미치지 않습니다
        return list(zip(np.zeros(len(point)), point))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


def main(_):
    sess = tf.Session()
    model = TacotronEncoder(sess,
                            11, # 0 ~ 10
                            FLAGS.vector_size, 
                            FLAGS.max_len, 
                            FLAGS.embed_size, 
                            FLAGS.block_num, 
                            FLAGS.highway_num, 
                            FLAGS.lr, 
                            FLAGS.b1)

    doc2vec_model = Doc2Vec.load(FLAGS.doc2vec_model)
    FLAGS.model = doc2vec_model

    bind_model(model, FLAGS)

    if FLAGS.pause:
        nsml.paused(scope=locals())
    
    if FLAGS.mode == 'train':
        data = dataset.MovieDatset(DATASET_PATH, FLAGS.model, FLAGS.max_len)
        batch = data.get_batcher(FLAGS.batch_size)

        for epoch in range(FLAGS.n_epoch):
            loss = 0
            acc = 0
            for _ in range(batch.iter_per_epoch):
                batch_x, batch_y = batch()
                model.train(batch_x, batch_y)

                n_loss, n_acc = model.inference([model.loss, model.metric], batch_x, batch_y)
                loss += n_loss
                acc += n_acc
            
            loss /= batch.iter_per_epoch
            acc /= batch.iter_per_epoch

            val_loss, val_acc = model.inference([model.loss, model.metric], data.val_data, data.val_label)

            print('Epoch {} / train_loss {} / val_loss {} / train_acc {} / val_acc {}'.format(epoch, loss, val_loss, acc, val_acc))
            nsml.report(summary=True, scope=locals(), epoch=epoch, total_epoch=FLAGS.n_epoch, val_acc=val_acc,
                        train_acc=acc, train__loss=loss, val__loss=val_loss, step=epoch)
            nsml.save(epoch)

    if FLAGS.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            reviews = f.readlines()
        
        print(nsml.infer(reviews))

if __name__ == '__main__':
    tf.app.run()
