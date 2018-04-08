import json
import numpy as np
import tensorflow as tf

class TacotronEncoder(object):
    def __init__(self,
                 sess,
                 y_dim,
                 vector_size,
                 max_sent_len,
                 embed_size,
                 block_num,
                 highway_num,
                 learning_rate,
                 beta1):
        self.sess = sess
        self.y_dim = y_dim

        self.vector_size = vector_size
        self.max_sent_len = max_sent_len

        self.embed_size = embed_size
        self.block_num = block_num
        self.highway_num = highway_num

        self.learning_rate = learning_rate
        self.beta1 = beta1

        self.plc_embed = tf.placeholder(tf.float32, [None, max_sent_len, vector_size])
        self.plc_y = tf.placeholder(tf.int32, [None])

        self.plc_training = tf.placeholder(tf.bool)
        self.plc_dropout = tf.placeholder(tf.float32)

        self.model = self._get_model()
        self.loss = self._get_loss()

        self.optimize = tf.train.AdamOptimizer(self.learning_rate, self.beta1).minimize(self.loss)
        self.summary = tf.summary.scalar('Metric', self.loss)
        self.ckpt = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

    def train(self, x, y, dropout=0.5):
        self.sess.run(self.optimize, feed_dict={self.plc_embed: x, self.plc_y: y, self.plc_dropout: dropout, self.plc_training: True})

    def pred(self, x):
        out = tf.argmax(self.model, axis=1) + 1
        return self.sess.run(out, feed_dict={self.plc_embed: x, self.plc_dropout: 1.0, self.plc_training: False})

    def inference(self, obj, x, y=None):
        feed_dict = {self.plc_embed: x, self.plc_dropout: 1.0, self.plc_training: False}
        if y is not None:
            feed_dict[self.plc_y] = y

        return self.sess.run(obj, feed_dict=feed_dict)

    def dump(self, path):
        self.ckpt.save(self.sess, path + '.ckpt')

        with open(path + '.json', 'w') as f:
            dump = json.dumps(
                {
                    'y_dim': self.y_dim,
                    'vector_size': self.vector_size,
                    'max_sent_len': self.max_sent_len,
                    'embed_size': self.embed_size,
                    'block_num': self.block_num,
                    'highway_num': self.highway_num,
                    'learning_rate': self.learning_rate,
                    'beta1': self.beta1
                }
            )

            f.write(dump)

    def load(self, path):
        with open(path + '.json') as f:
            param = json.loads(f.read())

        self.__init__(tf.Session(),
                      param['y_dim'],
                      param['vector_size'],
                      param['max_sent_len'],
                      param['embed_size'],
                      param['block_num'],
                      param['highway_num'],
                      param['learning_rate'],
                      param['beta1'])

        self.ckpt.restore(self.sess, path + '.ckpt')

    def _get_model(self):
        prenet = tf.layers.dense(self.plc_embed, self.embed_size, activation=tf.nn.relu)
        prenet = tf.nn.dropout(prenet, keep_prob=self.plc_dropout)

        prenet = tf.layers.dense(prenet, self.embed_size // 2, activation=tf.nn.sigmoid)
        prenet = tf.nn.dropout(prenet, keep_prob=self.plc_dropout)

        output = tf.layers.conv1d(prenet, self.embed_size // 2, 1, 1, padding='SAME')
        for n in range(2, self.block_num + 1):
            tmp = tf.layers.conv1d(prenet, self.embed_size // 2, n, 1, padding='SAME')
            output = tf.concat((output, tmp), axis=-1)
        
        output = tf.layers.batch_normalization(output, training=self.plc_training)
        output = tf.nn.relu(output)

        pool = tf.layers.max_pooling1d(output, 2, 1, padding='SAME')

        conv1 = tf.layers.conv1d(pool, self.embed_size // 2, 3, 1, padding='SAME')
        conv1 = tf.layers.batch_normalization(conv1, training=self.plc_training)
        conv1 = tf.nn.relu(conv1)

        conv2 = tf.layers.conv1d(conv1, self.embed_size // 2, 3, 1, padding='SAME')
        conv2 = tf.layers.batch_normalization(conv2, training=self.plc_training)

        highway = conv2 + prenet
        for _ in range(self.highway_num):
            h = tf.layers.dense(highway, self.embed_size // 2, activation=tf.nn.relu)
            t = tf.layers.dense(highway, self.embed_size // 2, activation=tf.nn.sigmoid)
            highway = h * t + h * (1. - t)

        fw = tf.nn.rnn_cell.GRUCell(self.embed_size // 2)
        bw = tf.nn.rnn_cell.GRUCell(self.embed_size // 2)
        o, _ = tf.nn.bidirectional_dynamic_rnn(fw, bw, tf.transpose(highway, (1, 0, 2)), dtype=tf.float32, time_major=True)
        o = tf.transpose(tf.concat(o, axis=-1), (1, 0, 2))

        attn = tf.layers.dense(o, self.max_sent_len) / (self.max_sent_len ** 0.5)
        attn = tf.matmul(tf.nn.softmax(attn), o)

        conv = tf.layers.conv1d(attn, 1, 1, 1, padding='SAME')
        conv = tf.layers.batch_normalization(conv, training=self.plc_training)
        conv = tf.nn.relu(conv)

        reshaped = tf.reshape(conv, (-1, self.max_sent_len))
        return tf.layers.dense(reshaped, self.y_dim)

    def _get_loss(self):
        onehot = tf.one_hot(self.plc_y, self.y_dim)
        return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.model, labels=onehot))
