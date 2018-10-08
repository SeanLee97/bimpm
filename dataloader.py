# -*- coding:utf8 -*-

import os
import json
import logging
import numpy as np
from collections import Counter

def word_tokenize(sent):
    sent = sent.lower()
    return sent.split()

class DataLoader(object):

    def __init__(self, max_p_len, max_q_len, max_char_len,
                 train_files=[], dev_files=[], test_files=[]):
        self.logger = logging.getLogger("bimpm")
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len
        self.max_char_len = max_char_len

        self.train_set, self.dev_set, self.test_set = [], [], []
        if train_files:
            for train_file in train_files:
                self.logger.info('---train file-----{}'.format(train_file))
                self.train_set += self._load_dataset(train_file, train=True)
            self.logger.info('Train set size: {} qs.'.format(len(self.train_set)))

        if dev_files:
            for dev_file in dev_files:
                self.logger.info('---dev file-----{}'.format(dev_file))
                self.dev_set += self._load_dataset(dev_file, train=True)
            self.logger.info('Dev set size: {} qs.'.format(len(self.dev_set)))

        if test_files:
            for test_file in test_files:
                self.test_set += self._load_dataset(test_file)
            self.logger.info('Test set size: {} qs.'.format(len(self.test_set)))

    def _load_dataset(self, data_path, train=False):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        examples = []
        with open(data_path, "r") as fh, open("./data/failed.json", "w") as wf:
            for line in fh:
                line = line.strip()
                arr = line.split("\t")
                if len(arr) != 3:
                    continue
                sample = {}
                sample['p_tokens'] = word_tokenize(arr[1])
                sample['q_tokens'] = word_tokenize(arr[2])
                sample['label_id'] = int(arr[0])

                examples.append(sample)
        return examples

    def _one_mini_batch(self, data, indices, pad_id, pad_char_id, batch_size=16):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        batch_data = {'raw_data': [data[i] for i in indices],
                      'q_token_ids': [],
                      'q_char_ids' : [],
                      'q_length': [],
                      'p_token_ids': [],
                      'p_length': [],
                      'p_char_ids': [],
                      'label_id': [],
                    }

        for sidx, sample in enumerate(batch_data['raw_data']):
            batch_data['q_token_ids'].append(sample['q_token_ids'])
            batch_data['q_char_ids'].append(sample['q_char_ids'])
            batch_data['q_length'].append(len(sample['q_token_ids']))

            batch_data['p_token_ids'].append(sample['p_token_ids'])
            batch_data['p_char_ids'].append(sample['p_char_ids'])
            batch_data['p_length'].append(min(len(sample['p_token_ids']), self.max_p_len))
            

            batch_data['label_id'].append(sample['label_id'])

        diff = batch_size - len(batch_data['raw_data'])
        if diff > 0:
            for _ in range(diff):
                batch_data['q_token_ids'].append([])
                batch_data['q_length'].append(0)
                batch_data['q_char_ids'].append([[]])
                batch_data['p_token_ids'].append([])
                batch_data['p_length'].append(0)
                batch_data['p_char_ids'].append([[]])
                batch_data['label_id'].append(0)

        batch_data, padded_p_len, padded_q_len = self._dynamic_padding(batch_data, pad_id, pad_char_id)


        return batch_data

    def _dynamic_padding(self, batch_data, pad_id, pad_char_id):
        """
        Dynamically pads the batch_data with pad_id
        """
        pad_char_len = self.max_char_len
        pad_p_len = self.max_p_len
        pad_q_len = self.max_q_len
        batch_data['p_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                           for ids in batch_data['p_token_ids']]
        for index, char_list in enumerate(batch_data['p_char_ids']):
            #print(batch_data['p_char_ids'])
            for char_index in range(len(char_list)):
                if len(char_list[char_index]) >= pad_char_len:
                    char_list[char_index] = char_list[char_index][:self.max_char_len]
                else:
                    char_list[char_index] += [pad_char_id]*(pad_char_len - len(char_list[char_index]))
            batch_data['p_char_ids'][index] = char_list
        batch_data['p_char_ids'] = [(ids + [[pad_char_id]*pad_char_len]*(pad_p_len-len(ids)))[:pad_p_len]
                                        for ids in batch_data['p_char_ids']]

        # print(np.array(batch_data['p_char_ids']).shape, "==========")

        batch_data['q_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                            for ids in batch_data['q_token_ids']]
        for index, char_list in enumerate(batch_data['q_char_ids']):
            for char_index in range(len(char_list)):
                if len(char_list[char_index]) >= pad_char_len:
                    char_list[char_index] = char_list[char_index][:self.max_char_len]
                else:    
                    char_list[char_index] += [pad_char_id]*(pad_char_len - len(char_list[char_index]))
            batch_data['q_char_ids'][index] = char_list
        batch_data['q_char_ids'] = [(ids + [[pad_char_id]*pad_char_len]*(pad_q_len-len(ids)))[:pad_q_len]
                                        for ids in batch_data['q_char_ids']]

        return batch_data, pad_p_len, pad_q_len

    def word_iter(self, set_name=None):
        """
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                for token in sample['q_tokens']:
                    yield token
                for token in sample['p_tokens']:
                    yield token


    def convert_to_ids(self, vocab):
        """
        Convert the q and p in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """
        for data_set in [self.train_set, self.dev_set, self.test_set]:
            if data_set is None:
                continue
            for sample in data_set:
                sample['q_token_ids'] = vocab.convert_word_to_ids(sample['q_tokens'])
                sample["q_char_ids"] = vocab.convert_char_to_ids(sample['q_tokens'])
                sample['p_token_ids'] = vocab.convert_word_to_ids(sample['p_tokens'])
                sample["p_char_ids"] = vocab.convert_char_to_ids(sample['p_tokens'])

    def next_batch(self, set_name, batch_size, pad_id, pad_char_id, data=None, shuffle=True):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        elif set_name == 'demo':
            data = data
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)

        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices, pad_id, pad_char_id, batch_size=batch_size)
