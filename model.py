# -*- coding: utf-8 -*-

import os
import time
import logging
import tensorflow as tf 
import numpy as np

import artf
import artf.loss as loss
import artf.rnn as rnn
from artf.highway import Highway
from artf.conv import Conv
from artf.attention import bi_attention

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Bimpm(object):
    def __init__(self, vocab, config):
        self.logger = logging.getLogger('bimpm')

        self.vocab = vocab
        self.config = config
        self.num_classes = 2

        self.learning_rate = config.learning_rate
        self.optim_type = config.optim
        self.algo = config.algo

        # session info
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = False
        self.sess = tf.Session(config=sess_config)

        self._build_graph()

         # save info
        self.saver = tf.train.Saver()

        # initialize the model
        self.sess.run(tf.global_variables_initializer())

    def _build_graph(self):
        start_t = time.time()

        self._set_placeholders()
        self._word_represent()
        self._context_represent()
        self._matching()
        self._aggregation()
        self._prediction()
        self._train_op()

        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))
        param_num = artf.total_params(tf.trainable_variables())
        self.logger.info('There are {} parameters in the model'.format(param_num))

    def _set_placeholders(self):
        self.p = tf.placeholder(tf.int32, [None, self.config.max_p_len], "max_p_len")
        self.q = tf.placeholder(tf.int32, [None, self.config.max_q_len], "max_q_len")
        self.ph = tf.placeholder(tf.int32, [None, self.config.max_p_len, self.config.max_ch_len], "p_char")
        self.qh = tf.placeholder(tf.int32, [None, self.config.max_q_len, self.config.max_ch_len], "q_char")
        self.y = tf.placeholder(tf.int32, [None], "y_true")
        
        self.p_mask = tf.cast(self.p, tf.bool)
        self.q_mask = tf.cast(self.q, tf.bool)
        
        self.p_len = tf.reduce_sum(tf.cast(self.p_mask, tf.int32), axis = 1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis = 1)

        self.dropout = tf.placeholder(tf.float32, name="dropout")
        self.is_train = tf.get_variable("is_train", shape=[], dtype=tf.bool, trainable=False)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

    def _word_represent(self):
        with tf.variable_scope("Word_Representation_Layer"):
            self.fix_word_mat = tf.get_variable("fix_word_mat", 
                              [self.vocab.word_size(), self.vocab.word_embed_dim], 
                              dtype=tf.float32,
                              initializer=tf.constant_initializer(self.vocab.word_embeddings, 
                                                                dtype=tf.float32),
                              trainable=False)

            self.char_mat = tf.get_variable(
                'char_mat',
                shape=[self.vocab.char_size(), self.vocab.char_embed_dim],
                initializer=tf.constant_initializer(self.vocab.char_embeddings),
                trainable=True
            )

            p_fix_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.fix_word_mat, self.p), 1.0 - self.dropout)
            q_fix_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.fix_word_mat, self.q), 1.0 - self.dropout)

            # char embedding
            N, PL, QL, CL, d, dc = self.config.batch_size, self.config.max_p_len, self.config.max_q_len, self.config.max_ch_len, self.config.hidden_size, self.char_mat.get_shape()[-1]

            ph_emb = tf.reshape(tf.nn.embedding_lookup( 
                self.char_mat, self.ph), [N*PL, CL, dc])
            qh_emb = tf.reshape(tf.nn.embedding_lookup(
                self.char_mat, self.qh), [N*QL, CL, dc])

            ph_emb = tf.nn.dropout(ph_emb, 1.0 - 0.5 * self.dropout)
            qh_emb = tf.nn.dropout(qh_emb, 1.0 - 0.5 * self.dropout)

            conv = Conv(bias=True, activation=tf.nn.relu, kernel_size=3)
            ph_emb = conv(ph_emb, d, scope="char_conv", reuse = None)
            qh_emb = conv(qh_emb, d, scope="char_conv", reuse = True)

            ph_emb = tf.reduce_max(ph_emb, axis = 1)
            qh_emb = tf.reduce_max(qh_emb, axis = 1)

            ph_emb = tf.reshape(ph_emb, [N, PL, -1])
            qh_emb = tf.reshape(qh_emb, [N, QL, -1])

            self.p_emb = tf.concat([p_fix_emb, ph_emb], axis=-1)
            self.q_emb = tf.concat([q_fix_emb, qh_emb], axis=-1)

            #highway = Highway(activation=tf.nn.relu, kernel='conv', num_layers=2, dropout=self.dropout)
            #self.p_emb = highway(self.p_emb, scope="highway", reuse=None)
            #self.q_emb = highway(self.q_emb, scope="highway", reuse=True)

    def _context_represent(self):
        with tf.variable_scope("Context_Rerepresent_Layer"):
            N, PL, QL, CL, d, dc = self.config.batch_size, self.config.max_p_len, self.config.max_q_len, self.config.max_ch_len, self.config.hidden_size, self.char_mat.get_shape()[-1]

            if self.config.use_cudnn:
                self.logger.info("use cudnn rnn to accelerate")
                lstm = rnn.BiCudnnRNN(num_layers=self.config.num_layers, num_units=d, batch_size=N, 
                                  input_size=self.p_emb.get_shape().as_list()[-1], kernel='lstm',  
                                  dropout=self.dropout)
            else:
                lstm = rnn.BiRNN(num_layers=self.config.num_layers, num_units=d, batch_size=N, 
                                  input_size=self.p_emb.get_shape().as_list()[-1], kernel='lstm',  
                                  dropout=self.dropout)

            self.p_enc, self.p_enc_c, self.p_enc_h = lstm(self.p_emb, self.p_len)
            self.q_enc, self.q_enc_c, self.q_enc_h = lstm(self.q_emb, self.q_len)
    
    def _matching(self):
        with tf.variable_scope("Matching_Layer"):
            p2q, q2p = bi_attention(self.p_enc, self.q_enc, self.p_mask, self.q_mask, kernel='bilinear')
            self.G =  tf.concat([self.p_enc, p2q, self.p_enc * p2q, self.p_enc * q2p], axis=-1)

    def _aggregation(self):
            with tf.variable_scope("Aggregation_Layer"):
                N, PL, QL, CL, d, dc = self.config.batch_size, self.config.max_p_len, self.config.max_q_len, self.config.max_ch_len, self.config.hidden_size, self.char_mat.get_shape()[-1]

                if self.config.use_cudnn:
                    lstm = rnn.BiCudnnRNN(num_layers=self.config.num_layers, num_units=d, batch_size=N, 
                                      input_size=self.G.get_shape().as_list()[-1], kernel='lstm',  
                                      dropout=self.dropout)
                else:
                    lstm = rnn.BiRNN(num_layers=self.config.num_layers, num_units=d, batch_size=N, 
                                  input_size=self.G.get_shape().as_list()[-1], kernel='lstm',  
                                  dropout=self.dropout)

                self.GM, c, h  = lstm(self.G, self.p_len)
                self.GM = tf.squeeze(tf.layers.dense(self.GM, 1, name="G_projection"), -1)
                #self.GM = tf.reduce_max(self.GM, axis=1)

    def _prediction(self):
            with tf.variable_scope("Prediction_Layer"):
                hidden = tf.layers.dense(self.GM, self.config.max_p_len // 2, name="logits_hidden")
                #hidden = tf.layers.batch_normalization(hidden, training=self.is_train)
                #hidden = tf.tanh(hidden)
                hidden = tf.nn.relu(hidden)
                self.logits = tf.layers.dense(hidden, self.num_classes, name="logits")

                self.pred_probs = tf.nn.softmax(self.logits)
                self.y_pred = tf.cast(tf.argmax(tf.nn.softmax(self.logits, -1), axis=1), tf.int32)
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_pred, self.y), tf.float32))


                if self.config.loss_type == "focal_loss":
                    self.loss = loss.focal_loss(self.y, self.logits)
                else:
                    self.loss = loss.cross_entropy(self.y, self.logits)

                if self.config.loss_mask:
                    loss_mask = tf.to_float(tf.less(self.loss, 1e10))
                    self.loss = tf.reduce_sum(self.loss * loss_mask) / (tf.reduce_sum(loss_mask) + 1e-30)

                if self.config.l2_norm is not None:
                    variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                    l2_loss = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(scale = 3e-7), variables)
                    self.loss += l2_loss

                if self.config.decay is not None:
                    self.var_ema = tf.train.ExponentialMovingAverage(self.config.decay)
                    ema_op = self.var_ema.apply(tf.trainable_variables())
                    with tf.control_dependencies([ema_op]):
                        self.loss = tf.identity(self.loss)

                        self.shadow_vars = []
                        self.global_vars = []
                        for var in tf.global_variables():
                            v = self.var_ema.average(var)
                            if v:
                                self.shadow_vars.append(v)
                                self.global_vars.append(var)
                        self.assign_vars = []
                        for g,v in zip(self.global_vars, self.shadow_vars):
                            self.assign_vars.append(tf.assign(g,v))

                self.all_params = tf.trainable_variables()

    def _train_op(self):

        self.lr = tf.minimum(self.learning_rate, self.learning_rate / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1))

        if self.optim_type == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.lr)
        elif self.optim_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        elif self.optim_type == 'rprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.lr)
        elif self.optim_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))

        trainable_vars = tf.trainable_variables()
        grads = self.optimizer.compute_gradients(self.loss, var_list=trainable_vars)
        gradients, variables = zip(*grads)
        capped_grads, _ = tf.clip_by_global_norm(
            gradients, self.config.grad_clipper)
        self.train_op = self.optimizer.apply_gradients(
            zip(capped_grads, variables), global_step=self.global_step)


    def _train_epoch(self, train_batches, dropout):
        total_num, total_loss, total_accu = 0, 0, 0.0
        log_every_n_batch, n_batch_loss, n_batch_accu = 200, 0, 0.0
        for bitx, batch in enumerate(train_batches, 1):
            feed_dict = {self.p: batch['p_token_ids'],
                         self.q: batch['q_token_ids'],
                         self.ph: batch['p_char_ids'],
                         self.qh: batch["q_char_ids"],
                         self.y: batch['label_id'],
                         self.is_train: True,
                         self.dropout: dropout}


            [_, loss, global_step, logits, y_pred, accuracy] = self.sess.run([self.train_op, self.loss, self.global_step, self.logits, self.y_pred, self.accuracy], feed_dict)

            real_batch_size = len(batch['raw_data'])
            total_loss += loss * real_batch_size
            total_accu += accuracy * real_batch_size
            total_num += real_batch_size
            n_batch_loss += loss
            n_batch_accu += accuracy

            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                self.logger.info('Average loss from batch {} to {} is {}, avg accuracy is {}'.format(
                    bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch, n_batch_accu / log_every_n_batch))
                n_batch_loss = 0
                n_batch_accu = 0.0
        print(total_loss, total_num)
        return 1.0 * total_loss / total_num, 1.0 * total_accu / total_num

    def train(self, data, epochs, batch_size, save_dir, save_prefix,
              dropout=0.0, evaluate=True):

        pad_id = self.vocab.get_word_id(self.vocab.pad_token)
        pad_char_id = self.vocab.get_char_id(self.vocab.pad_token)
        max_accu = 0

        for epoch in range(1, epochs + 1):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            train_batches = data.next_batch('train', batch_size, pad_id, pad_char_id, shuffle=True)
            train_loss, train_accu = self._train_epoch(train_batches, dropout)
            self.logger.info('Average train loss for epoch {} is {}, avg accuracy is {}'.format(epoch, train_loss, train_accu))

            if evaluate:
                self.logger.info('Evaluating the model after epoch {}'.format(epoch))
                if data.dev_set is not None:
                    eval_batches = data.next_batch('dev', batch_size, pad_id, pad_char_id, shuffle=False)
                    eval_loss, eval_metrics = self.evaluate(eval_batches)
                    self.logger.info('Dev eval loss {}'.format(eval_loss))
                    self.logger.info('Dev eval accu: {}'.format(eval_metrics))

                    if eval_metrics['acc'] > max_accu:
                        self.save(save_dir)
                        max_accu = eval_metrics['acc']
                else:
                    self.logger.warning('No dev set is loaded for evaluation in the dataset!')
            else:
                self.save(save_dir, 'epoch_' + str(epoch))

    def evaluate(self, eval_batches):

        total_loss, total_num, total_accu = 0, 0, 0.0

        y_trues, y_preds = [], []

        for b_itx, batch in enumerate(eval_batches):
            real_batch_size = len(batch['raw_data'])
            feed_dict = {self.p: batch['p_token_ids'],
                         self.q: batch['q_token_ids'],
                         self.ph: batch['p_char_ids'],
                         self.qh: batch["q_char_ids"],
                         self.y: batch['label_id'],
                         self.is_train: False,
                         self.dropout: 0.0}


            loss, y_pred, accuracy = self.sess.run([self.loss, self.y_pred, self.accuracy], feed_dict)

            y_trues += batch['label_id'][:real_batch_size]
            y_preds += y_pred.tolist()[:real_batch_size]

            total_accu += accuracy * real_batch_size
            total_loss += loss * real_batch_size
            total_num += real_batch_size

        ave_loss = 1.0 * total_loss / total_num
        ave_accu = 1.0 * total_accu / total_num

        acc = accuracy_score(y_trues, y_preds)
        p   = precision_score(y_trues, y_preds, average='macro')
        r   = recall_score(y_trues, y_preds, average='macro')
        f1  = f1_score(y_trues, y_preds, average='macro')

        metrics = {"acc":acc, "p":p, "r":r, "f1":f1}

        return ave_loss, metrics

    def save(self, model_dir, model_prefix='bimpm'):
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model saved in {}, with prefix {}.'.format(model_dir, model_prefix))

    def restore(self, model_dir, model_prefix='bimpm'):
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        if self.config.decay != None and self.config.decay < 1.0:
            self.sess.run(self.assign_vars)
        self.logger.info('Model restored from {}, with prefix {}'.format(model_dir, model_prefix))

