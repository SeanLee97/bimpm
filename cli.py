# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf 
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import json
import pickle
import logging
import argparse
from dataloader import DataLoader
from vocab import Vocab
from model import Bimpm
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser('Reading Comprehension on hfltekRC dataset')
    parser.add_argument('--prepro', action='store_true',
                        help='create the directories, prepare to process the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')
    parser.add_argument('--use_cudnn', action='store_true',
                        help='wether to use cudnn')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--algo', type=str, default='bimpm',
                                help='algorithm [bimpm]')
    train_settings.add_argument('--loss_type', type=str, default='focal_loss',
                                help='cross_entropy focal loss')
    train_settings.add_argument('--loss_mask', type=bool, default=False,
                                help='wether to use loss mask')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--num_layers', type=int, default=1,
                                help='layers of lstm')
    train_settings.add_argument('--decay', type=float, default=0.9999,
                                help='decay')
    train_settings.add_argument('--l2_norm', type=float, default=1e-7,
                                help='l2 norm')
    train_settings.add_argument('--grad_clipper', type=float, default=5.0,
                                help='max norm grad')
    train_settings.add_argument('--dropout', type=float, default=0.5,
                                help='dropout rate')
    train_settings.add_argument('--batch_size', type=int, default=64,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=20,
                                help='train epochs')

    model_settings = parser.add_argument_group('model settings')
    
    model_settings.add_argument('--word_embed_size', type=int, default=100,
                                help='size of the word embeddings')
    model_settings.add_argument('--char_embed_size', type=int, default=8,
                                help='size of the char embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=128,
                                help='size of hidden units')
    model_settings.add_argument('--max_p_len', type=int, default=50,
                                help='max length of passage')
    model_settings.add_argument('--max_q_len', type=int, default=50,
                                help='max length of question')
    model_settings.add_argument('--max_ch_len', type=int, default=12,
                                help='max length of character of a word')


    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_files', nargs='+',
                               default=['./data/toy/train.txt'],
                               help='list of files that contain the preprocessed train data')
    path_settings.add_argument('--dev_files', nargs='+',
                               default=['./data/toy/dev.txt'],
                               help='list of files that contain the preprocessed dev data')
    path_settings.add_argument('--test_files', nargs='+',
                               default=['./data/toy/test.txt'],
                               help='list of files that contain the preprocessed test data')

    path_settings.add_argument('--save_dir', default='./data/bimpm',
                               help='the dir with preprocessed bimpm data')
    path_settings.add_argument('--vocab_dir', default='./data/vocab/',
                               help='the dir to save vocabulary')
    path_settings.add_argument('--model_dir', default='./data/models/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='./data/results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='./data/summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')
 
    path_settings.add_argument('--word_pretrained_kernel',default='sgns',
                               help='gensim or others')
    path_settings.add_argument('--pretrained_word_path',default=None,
                               help='path of the log file. If not set, logs are printed to console')
    path_settings.add_argument('--pretrained_char_path',default=None,
                               help='path of the log file. If not set, logs are printed to console')
    return parser.parse_args()

"""
:description: prepare to process data including building vocab
"""
def prepro(args):
    logger = logging.getLogger(args.algo)
    logger.info("====== preprocessing ======")
    logger.info('Checking the data files...')
    for data_path in args.train_files + args.dev_files + args.test_files:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)

    logger.info('Preparing the directories...')
    for dir_path in [args.vocab_dir, args.model_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    logger.info('Building vocabulary...')
    dataloader = DataLoader(args.max_p_len, args.max_q_len, args.max_ch_len, 
                          args.train_files, args.dev_files, args.test_files)

    vocab = Vocab(lower=True)
    for word in dataloader.word_iter('train'):
        vocab.add_word(word)
        [vocab.add_char(ch) for ch in word]

    unfiltered_vocab_size = vocab.word_size()
    vocab.filter_words_by_cnt(min_cnt=1)
    filtered_num = unfiltered_vocab_size - vocab.word_size()
    logger.info('After filter {} tokens, the final vocab size is {}, char size is {}'.format(filtered_num,
                                                                            vocab.word_size(), vocab.char_size()))

    unfiltered_vocab_char_size = vocab.char_size()
    vocab.filter_chars_by_cnt(min_cnt=1)
    filtered_char_num = unfiltered_vocab_char_size - vocab.char_size()
    logger.info('After filter {} chars, the final char vocab size is {}'.format(filtered_char_num,
                                                                            vocab.char_size()))

    logger.info('Assigning embeddings...')
    if args.pretrained_word_path is not None:
        logger.info('pretrained word...')
        vocab.load_pretrained_word_embeddings(args.pretrained_word_path, kernel=args.word_pretrained_kernel)
    else:
        vocab.randomly_init_word_embeddings(args.word_embed_size)
    
    if args.pretrained_char_path is not None:
        logger.info('pretrained char...')
        vocab.load_pretrained_char_embeddings(args.pretrained_char_path)
    else:
        vocab.randomly_init_char_embeddings(args.char_embed_size)

    logger.info('Saving vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'wb') as fout:
        pickle.dump(vocab, fout)

    logger.info('====== Done with preparing! ======')

"""
:description: train
"""
def train(args):
    logger = logging.getLogger(args.algo)
    logger.info("====== training ======")

    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)

    dataloader = DataLoader(args.max_p_len, args.max_q_len, args.max_ch_len,
                          args.train_files, args.dev_files)

    logger.info('Converting text into ids...')
    dataloader.convert_to_ids(vocab)

    logger.info('Initialize the model...')
    model = Bimpm(vocab, args)

    logger.info('Training the model...')
    model.train(dataloader, args.epochs, args.batch_size, save_dir=args.model_dir, save_prefix=args.algo, dropout=args.dropout)

    logger.info('====== Done with model training! ======')


"""
:descriptoin: predict answers
"""
def predict(args):
    logger = logging.getLogger(args.algo)

    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)

    assert len(args.test_files) > 0, 'No test files are provided.'
    dataloader = DataLoader(args.max_p_len, args.max_q_len, args.max_ch_len, 
                          test_files=args.test_files)

    logger.info('Converting text into ids...')
    dataloader.convert_to_ids(vocab)
    logger.info('Restoring the model...')

    model = Bimpm(vocab, args)
    model.restore(model_dir=args.model_dir)

    logger.info('Predicting answers for test set...')
    test_batches = dataloader.next_batch('test', args.batch_size, vocab.get_word_id(vocab.pad_token), vocab.get_char_id(vocab.pad_token), shuffle=False)

    ave_loss, ave_accu = model.evaluate(test_batches)
    print("avg accuracy", ave_accu)


def run():
    args = parse_args()

    logger = logging.getLogger(args.algo)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.prepro:
        prepro(args)
    if args.train:
        train(args)
    if args.evaluate:
        evaluate(args)
    if args.predict:
        predict(args)

if __name__ == '__main__':
    run()



