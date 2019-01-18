#!/cs/puls/Projects/business_c_test/env/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import os, sys
import numpy as np
from tqdm import trange
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from sklearn.metrics import classification_report
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec


class BiLSTM():
    def __init__(self, params=None, encoded_dataset=None, raw_dataset=None, word2Idx = {}):
        self.embeddings = None
        self.word2Idx = word2Idx
        self.charEmbeddings = None
        self.char2Idx = {}
        self.session = None
        self.datagenerator = None
        self.featureMap = {'tokens': self.tokenInput, 'casing': self.casingInput,
                           'character': self.charInput, 'pos': self.posInput}
        # Hyperparameters for the network
        defaultParams = {'epoch': 1, 'miniBatchSize': 50,
                         'modelSavePath': '/cs/puls/Resources/models/English',
                         'dropout': (0.25, 0.25),
                         'embedding': "/cs/puls/Resources/embeddings/Finnish/fin-word2vec-lemma-100.bin",
                         'classifier': 'crf',
                         'crf': {
                             'learn-mode': 'join',
                             'test-mode': 'marginal'
                         },
                         'LSTM-Size': (100, 100),
                         'casingEntries': ['PADDING', 'other', 'numeric', 'mainly_numeric', 'allLower',
                                           'allUpper', 'mainly_allUpper', 'initialUpper', 'contains_upper',
                                           'contains_digit'],
                         'posEntries': ['PADDING', 'other', 'Noun', 'Verb', 'Adj', 'Adv', 'Pron', 'Conj', 'Interj',
                                        'Num', 'Punct', 'UNKNOWN'],
                         'charEntries': " 0123456789abcdefghijklmnopqrstuvwxyzäöåABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÅ" + \
                                        ".,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|\u2013\u2014\u201C\u201D",
                         'labelEntries': ['B-PER', 'B-LOC', 'B-ORG', 'B-PRO', 'B-OTH',
                                          'I-PER', 'I-LOC', 'I-ORG', 'I-PRO', 'I-OTH', 'O'],
                         'character': {
                             'charEmbeddings': 'cnn',
                             'charEmbeddingsSize': 30,
                             'charFilterSize': 30,
                             'charFilterLength': 3,
                             'charLSTMSize': 30,
                             'maxCharLength': 50,
                         },
                         'casing': {
                             'num_units': 30,
                             'activation': 'relu',
                             'dropout': 0.2
                         },
                         'pos': {
                             'num_units': 30,
                             'activation': 'relu',
                             'dropout': 0.2
                         },
                         'optimizer': {
                             'type': 'adam',
                             'clipvalue': 0,
                             'clipnorm': 1,
                         },
                         'earlyStopping': 5,
                         'featureNames': ['tokens', 'casing', 'character', 'pos'],
                         'addFeatureDimensions': 10}

        self.activation_map = {
                                'relu': tf.nn.relu,
                                'tanh': tf.nn.tanh,
                                'sigmoid': tf.nn.sigmoid,
                                'softmax': tf.nn.softmax,
                               }

        if params != None:
            for k, v in params.items():
                if type(v) == dict:
                    for kk, vv in v.items():
                        defaultParams[k][kk] = vv
                else:
                    defaultParams[k] = v

        self.params = defaultParams
        self.dataset = {}
        if 'tokens' in self.params['featureNames'] and not len(self.word2Idx):
            self.loadWordEmbedding()
        elif len(self.word2Idx):
            self.char2Idx = {"PADDING": 0, "UNKNOWN": 1}
            self.char2Idx.update({char:i+len(self.char2Idx)
                                  for i,char in enumerate(self.params['charEntries'])})
            return
        self.loadCharEmbedding()
        if raw_dataset and encoded_dataset is None:
            self.setRawDataset(raw_dataset)
            self.buildModel()
        if encoded_dataset:
            self.setDataset(encoded_dataset)
            self.buildModel()

    def loadCharEmbedding(self):
        self.char2Idx = {"PADDING": 0, "UNKNOWN": 1}
        self.charEmbeddings = np.zeros(self.params['character']['charEmbeddingsSize'])
        limit = np.sqrt(3.0 / self.params['character']['charEmbeddingsSize'])  # Why?
        self.charEmbeddings = np.vstack((self.charEmbeddings,
                                         np.random.uniform(-limit, limit,
                                                           self.params['character']['charEmbeddingsSize'])))
        for _ in self.params['charEntries']:
            self.char2Idx[_] = len(self.char2Idx)
            self.charEmbeddings = np.vstack((self.charEmbeddings,
                                             np.random.uniform(-limit, limit,
                                                               self.params['character']['charEmbeddingsSize'])))

    def loadWordEmbedding(self):
        print('Loading Embedding Matrix...')
        if 'glove' in self.params['embedding'].lower():
            glove_file = datapath(self.params['embedding'])
            tmp_file = get_tmpfile('glove2vec.txt')
            glove2word2vec(glove_file, tmp_file)
            embedding = KeyedVectors.load_word2vec_format(tmp_file, binary=False)
        else:
            embedding = KeyedVectors.load_word2vec_format(self.params['embedding'], binary=True)
        # Add padding+unknown
        self.word2Idx["PADDING_TOKEN"] = len(self.word2Idx)
        self.embeddings = np.zeros(embedding.syn0.shape[1])
        self.word2Idx["AMBIGUOUS_TOKEN"] = len(self.word2Idx)
        self.embeddings = np.vstack((np.random.uniform(-0.25, 0.25, embedding.syn0.shape[1]), self.embeddings))
        self.word2Idx["UNKNOWN_TOKEN"] = len(self.word2Idx)
        self.embeddings = np.vstack((np.random.uniform(-0.25, 0.25, embedding.syn0.shape[1]), self.embeddings))
        self.embeddings = np.vstack((self.embeddings, embedding.syn0))
        temp = len(self.word2Idx)
        self.word2Idx.update({v:k + temp for k,v in enumerate(embedding.index2word)})


    def tokenInput(self, sentence_length=None):
        tokens_input = tf.placeholder(tf.int32, [None, None], name='tokens_input')
        W = tf.Variable(self.embeddings, trainable=False, name="W_token")
        tokens = tf.nn.embedding_lookup(W, tokens_input, name='tokens')
        del self.embeddings
        tokens = tf.cast(tokens, tf.float64)
        print('Embedding Shape:', tokens.shape)
        '''
        tokens_input = Input(shape=(None,), name='words_input')
        tokens = Embedding(input_dim=self.embeddings.shape[0], output_dim=self.embeddings.shape[1],
                           weights=[self.embeddings], trainable=False, name='word_embeddings')(tokens_input)
        '''
        return tokens_input, tokens

    def casingInput(self, sentence_length=None):
        casing_input = tf.placeholder(tf.int32, [None, None],name='casing_input')
        W = tf.Variable(tf.random_uniform([len(self.params['casingEntries']),
                                           self.params['addFeatureDimensions']], -1.0, 1.0), name="W_case")
        casings = tf.nn.embedding_lookup(W, casing_input, name='casings')
        casings = tf.cast(casings, tf.float64)
        print('Casing Shape:', casings.shape)
        '''
        casing_input = Input(shape=(None, ),name='casing_input')
        casings = Embedding(input_dim=len(self.params['casingEntries']),
                            output_dim=self.params['addFeatureDimensions'], name='casings')(casing_input)
        '''
        return casing_input, casings

    def posInput(self, sentence_length=None):
        pos_input = tf.placeholder(tf.int32, [None, None, len(self.params['posEntries'])], name='pos_input')
        #pos = tf.reshape(pos_input, [-1, len(self.params['posEntries'])])
        pos = tf.layers.Dense(self.params['pos']['num_units'],
                              activation=self.activation_map[self.params['casing']['activation']],
                              name='pos_dense')(pos_input)
        if self.params['pos'].get('dropout'):
            pos = tf.layers.Dropout(self.params['casing'].get('dropout'), name='pos_dropout')(pos)
        pos = tf.cast(pos, tf.float64)
        print('POS Shape:', pos.shape)
        '''
        pos_input = Input(shape=(None, len(self.params['posEntries'])), name='pos_input')
        pos = Dense(self.params['pos']['num_units'],
                        activation=self.params['casing']['activation'], name='pos_dense')(pos_input)
        if self.params['pos'].get('dropout'):
            pos = Dropout(self.params['casing'].get('dropout'), name='pos_dropout')(pos)
        '''
        return pos_input, pos

    def charInput(self, sentence_length=None):
        chars_input = tf.placeholder(tf.int32, [None, None, self.params['character']['maxCharLength']], name='chars_input')
        # chars_input = Input(shape=(None, self.params['character']['maxCharLength']), name='char_input')
        W = tf.Variable(tf.random_uniform([self.charEmbeddings.shape[0],
                                           self.charEmbeddings.shape[1]], -1.0, 1.0), name="W_char")
        chars = tf.nn.embedding_lookup(W, chars_input, name='char_emd')
        if self.params['character']['charEmbeddings'].lower() == 'lstm':
            chars = tf.reshape(chars, [-1, self.params['character']['maxCharLength'], self.charEmbeddings.shape[1]])
            lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.params['character']['charLSTMSize'])
            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.params['character']['charLSTMSize'])
            (output_fw, output_bw), _ = \
                tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, chars, 
                    sequence_length=sentence_length,dtype=tf.float32)
            chars = tf.concat([output_fw, output_bw], axis=-1)
            chars = tf.reshape(chars, [-1, sentence_length, self.params['character']['charLSTMSize']*2])
            '''
            # Use LSTM for char embeddings from Lample et al., 2016
            chars = TimeDistributed(
            Embedding(input_dim=self.charEmbeddings.shape[0], output_dim=self.charEmbeddings.shape[1],
                      weights=[self.charEmbeddings], trainable=True, mask_zero=True), name='char_emd')(chars_input)
            charLSTMSize = self.params['character']['charLSTMSize']
            chars = TimeDistributed(Bidirectional(LSTM(charLSTMSize, return_sequences=False)), name="char_lstm")(chars)
            '''
        else:
            charFilterSize = self.params['character']['charFilterSize']
            charFilterLength = self.params['character']['charFilterLength']
            charsFilter = tf.Variable(tf.random_normal([1, charFilterLength,
                                                        self.charEmbeddings.shape[1], charFilterSize]))
            chars = tf.nn.conv2d(chars, charsFilter, strides=[1, 1, 1, 1], padding='SAME', name='char_cnn')
            chars = tf.reduce_max(chars, axis=-2, name="char_pooling")
            '''
            # Use CNNs for character embeddings from Ma and Hovy, 2016
            chars = TimeDistributed(
            Embedding(input_dim=self.charEmbeddings.shape[0], output_dim=self.charEmbeddings.shape[1],
                      weights=[self.charEmbeddings], trainable=True, mask_zero=False), name='char_emd')(chars_input)
            charFilterSize = self.params['character']['charFilterSize']
            charFilterLength = self.params['character']['charFilterLength']
            chars = TimeDistributed(Conv1D(charFilterSize, charFilterLength, padding='same'), name="char_cnn")(chars)
            chars = TimeDistributed(GlobalMaxPooling1D(), name="char_pooling")(chars)
            chars = TimeDistributed(Masking(mask_value=0), name="char_mask")(chars)
            '''
        chars = tf.cast(chars, tf.float64)
        print('CharEmbedding Shape:', chars.shape)
        del self.charEmbeddings
        #chars = tf.reshape(chars, [-1, tf.shape(chars_input)[-2], tf.shape(chars)[-1]])
        return chars_input, chars

    def buildModel(self):
        tf.reset_default_graph()
        label = tf.placeholder(tf.int32, [None, None], name='label')
        sentence_length = tf.placeholder(tf.int32, [None], name='sentence_length')
        input_nodes = [self.featureMap[_](sentence_length=sentence_length)
                       for _ in self.params['featureNames'] if _ in self.featureMap.keys()]
        merged = tf.concat([_[1] for _ in input_nodes], axis=-1)
        merged_input_shape = tf.shape(merged)
        print('Feature Concatnated:', merged.shape)
        cnt = 1
        for size in self.params['LSTM-Size']:
            lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(size, name="merged_fw_lstm_"+ str(cnt))
            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(size, name="merged_bw_lstm_"+ str(cnt))
            if isinstance(self.params['dropout'], (list, tuple)):
                lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell,
                                                             input_keep_prob=1 - self.params['dropout'][0],
                                                             output_keep_prob=1 - self.params['dropout'][1])
                lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell,
                                                             input_keep_prob=1 - self.params['dropout'][0],
                                                             output_keep_prob=1 - self.params['dropout'][1])
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, merged,
                                               sequence_length=sentence_length, dtype=tf.float64)
                merged = tf.concat([output_fw, output_bw], axis=-1)
                '''
                merged_input = Bidirectional(LSTM(size, return_sequences=True, dropout=self.params['dropout'][0],
                                                  recurrent_dropout=self.params['dropout'][1]),
                                             name='shared_varLSTM_' + str(cnt))(merged_input)
                '''
            else:
                """ Naive dropout """
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, merged,
                                                sequence_length=sentence_length,  dtype=tf.float64)
                merged = tf.concat([output_fw, output_bw], axis=-1)
                merged = tf.layers.Dropout(self.params['dropout'],
                                                 name='shared_dropout_'+ str(cnt))(merged)
                '''
                merged_input = Bidirectional(LSTM(size, return_sequences=True), name='shared_LSTM_' + str(cnt))(
                    merged_input)
                if self.params['dropout'] > 0.0:
                    merged_input = TimeDistributed(Dropout(self.params['dropout']),
                                                   name='shared_dropout_' + str(self.params['dropout']) + "_" + str(
                                                       cnt))(merged_input)
                '''
            print(cnt, 'BiLSTM Shape:', merged.shape)
            cnt += 1
        merged = tf.reshape(merged, [-1, self.params['LSTM-Size'][-1]*2])
        if self.params['classifier'].lower() == 'softmax':
            merged = tf.layers.Dense(len(self.params['labelEntries']),
                                     activation=self.activation_map['softmax'], name='output')(merged)
            merged = tf.reshape(merged, [-1, merged_input_shape[-2], len(self.params['labelEntries'])])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=merged)
            output = tf.argmax(merged, axis=-1, name="softmax_output")
            '''
            output = TimeDistributed(Dense(len(self.params['labelEntries']),
                                           activation='softmax'), name='output')(merged_input)
            lossFct = 'sparse_categorical_crossentropy'
            acc = 'sparse_categorical_accuracy'
            '''
        elif self.params['classifier'].upper() == 'CRF':
            merged = tf.layers.Dense(len(self.params['labelEntries']), name="hidden_lin_layer")(merged)
            merged = tf.reshape(merged, [-1, merged_input_shape[-2], len(self.params['labelEntries'])])
            merged = tf.cast(merged, tf.float32)
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(merged,
                                                                                  label, sentence_length)
            loss = -log_likelihood

            output, viterbi_score = tf.contrib.crf.crf_decode(merged, transition_params, sentence_length)
            output = tf.identity(output, name="crf_output")
            score = tf.identity(viterbi_score, name="score")
            '''
            output = TimeDistributed(Dense(len(self.params['labelEntries']), activation=None),
                                     name='hidden_lin_layer')(merged_input)
            crf = CRF(len(self.params['labelEntries']), learn_mode=self.params['crf']['learn-mode'],
                      test_mode=self.params['crf']['test-mode'], sparse_target=True, name='output')
            output = crf(output)
            lossFct = crf.loss_function
            acc = crf.accuracy
            '''
        print('Output Shape:', output.shape)
        lossFct = tf.reduce_mean(loss, name='loss')
        precision = tf.metrics.precision(label, output, name='precision')
        recall = tf.metrics.recall(label, output, name='recall')
        optimizerParams = {k: v for k, v in self.params['optimizer'].items() if k not in ['type', 'clipnorm', 'clipvalue']}
        optimizerParams['name'] = 'train_op'
        if self.params['optimizer']['type'].lower() == 'adam':
            opt = tf.train.AdamOptimizer(**optimizerParams)
        elif self.params['optimizer']['type'].lower() == 'nadam':
            opt = tf.contrib.opt.NadamOptimizer(**optimizerParams)
        elif self.params['optimizer']['type'].lower() == 'rmsprop':
            opt = tf.train.RMSPropOptimizer(**optimizerParams)
        elif self.params['optimizer']['type'].lower() == 'adadelta':
            opt = tf.train.AdadeltaOptimizer(**optimizerParams)
        elif self.params['optimizer']['type'].lower() == 'adagrad':
            opt = tf.train.AdagradOptimizer(**optimizerParams)
        elif self.params['optimizer']['type'].lower() == 'sgd':
            opt = tf.train.GradientDescentOptimizer(**optimizerParams)

        grad_vars = opt.compute_gradients(lossFct)
        grad_vars = [
            (tf.clip_by_norm(grad, self.params['optimizer']['clipnorm']), var)
            if grad is not None else (grad, var)
            for grad, var in grad_vars]
        grad_vars = [
            (tf.clip_by_value(grad, -abs(self.params['optimizer']['clipvalue']),
                              abs(self.params['optimizer']['clipvalue'])), var)
            if grad is not None else (grad, var)
            for grad, var in grad_vars]
        opt.apply_gradients(grad_vars)

    def printSummary(self):
        pass

    def setRawDataset(self, dataset):
        self.dataEncoding(dataset['data'])
        self.params['labelEntries'] = dataset['labelEntries']
        self.dataset = dataset
        self.datagenerator=self.dataGenerator()

    def setDataset(self, dataset):
        self.dataset = dataset
        self.params['labelEntries'] = self.dataset['labelEntries']
        self.datagenerator = self.dataGenerator()

    def dataEncoding(self, data, inBatch=True, hasLabel=True):
        # Encoding data
        def getCasing(word):
            """Returns the casing for a word"""
            casing = 'other'

            numDigits = 0
            numUpper = 0
            for char in word:
                if char.isdigit():
                    numDigits += 1
                if char.isupper():
                    numUpper += 1

            digitFraction = numDigits / float(len(word))
            upperFraction = numUpper / float(len(word))

            if word.isdigit():  # Is a digit
                casing = 'numeric'
            elif digitFraction > 0.5:
                casing = 'mainly_numeric'
            elif upperFraction > 0.5:
                casing = 'mainly_allUpper'
            elif word.islower():  # All lower case
                casing = 'allLower'
            elif word.isupper():  # All upper case
                casing = 'allUpper'
            elif word[0].isupper():  # is a title, initial char upper, then all lower
                casing = 'initialUpper'
            elif upperFraction > 0:
                casing = 'contains_upper'
            elif numDigits > 0:
                casing = 'contains_digit'

            return casing

        def encode(array, anything2Idx):
            encoded = [0 for _ in anything2Idx]
            for _ in array:
                encoded[_] = 1
            return encoded

        limit = (self.params['character']['maxCharLength'] - 2) // 2
        casing2Idx = {v: k for k, v in enumerate(self.params['casingEntries'])}
        pos2Idx = {v: k for k, v in enumerate(self.params['posEntries'])}
        label2Idx = {v: k for k, v in enumerate(self.params['labelEntries'])}
        groups = data
        if not inBatch:
            groups = [data]
        for group in groups:
            for sent in group:
                for word in sent:
                    char = word.pop('surface') if 'surface' in word.keys() else word.pop('token')
                    surface = char
                    # Padding
                    if char == 'PADDING_TOKEN':
                        word['char'] = [self.char2Idx['PADDING']
                                        for _ in range(self.params['character']['maxCharLength'])]
                        #word['casing'] = encode([casing2Idx['PADDING']], casing2Idx)
                        word['casing'] = casing2Idx['PADDING']
                        if 'tokens' in self.params['featureNames']:
                            word['lemma'] = self.word2Idx['PADDING_TOKEN']
                        word['pos'] = encode([pos2Idx['PADDING']], pos2Idx)
                        #word['label'] = encode([label2Idx[word['label']]], label2Idx)
                        word['label'] = [label2Idx[word['label']]]
                        continue
                    # Casing Encoding
                    # word['casing'] = encode([casing2Idx.get(getCasing(char), casing2Idx['other'])], casing2Idx)
                    word['casing'] = casing2Idx.get(getCasing(char), casing2Idx['other'])
                    # Char Encoding
                    if len(char) > self.params['character']['maxCharLength'] - 2:
                        char = char[:limit] + char[-limit:]
                    char = [self.char2Idx.get(_, self.char2Idx['UNKNOWN']) for _ in char]
                    char.insert(0, self.char2Idx['PADDING'])
                    char.extend([self.char2Idx['PADDING']
                                 for _ in range(len(char), self.params['character']['maxCharLength'])])
                    word['char'] = char
                    # Lemma and POS Encoding
                    word['lemma'] = self.word2Idx.get(word.get('lemma')) or self.word2Idx['UNKNOWN_TOKEN']

                    analyses = word.pop('analyses')
                    if len(analyses) == 0:
                        if 'tokens' in self.params['featureNames']:
                            word['lemma'] = self.word2Idx.get(word.get('lemma')) or\
                                            self.word2Idx.get(surface) or\
                                            self.word2Idx['UNKNOWN_TOKEN']
                        word['pos'] = encode([pos2Idx['UNKNOWN']], pos2Idx)
                    else:
                        if 'tokens' in self.params['featureNames']:
                            lemmas = set([_[0].get('canon') or _[0]['base'] for _ in analyses])
                            if len(set([_.replace('+', '') for _ in lemmas])) > 1:
                                word['lemma'] = self.word2Idx['AMBIGUOUS_TOKEN']
                            else:
                                word['lemma'] = self.word2Idx['UNKNOWN_TOKEN']
                                for _ in lemmas:
                                    lemma = _.replace('+', '|')
                                    if lemma in self.word2Idx.keys():
                                        word['lemma'] = self.word2Idx[lemma]
                                        break
                                if word['lemma'] == self.word2Idx['UNKNOWN_TOKEN']:
                                    for _ in lemmas:
                                        lemma = _.replace('-', '|')
                                        if lemma in self.word2Idx.keys():
                                            word['lemma'] = self.word2Idx[lemma]
                                            break
                                if word['lemma'] == self.word2Idx['UNKNOWN_TOKEN']:
                                    for analysis in analyses:
                                        if analysis[-1]['base'] in self.word2Idx.keys():
                                            word['lemma'] = self.word2Idx[analysis[-1]['base']]
                                            break
                        POS = set([pos2Idx.get(analysis[-1]['pos'], pos2Idx['other']) for analysis in analyses])
                        word['pos'] = encode(POS, pos2Idx)
                    # Label Encoding
                    # word['label'] = encode([label2Idx[word['label']]], label2Idx)
                    if hasLabel:
                        try:
                            word['label'] = [label2Idx[word['label']]]
                        except KeyError:
                            print(word['label'])
                            print('Label entries not matched with model')
                            return

    def dataGenerator(self):
        k = 0
        i = 0
        while True:
            k = k % len(self.dataset['data'])
            j = min(i + self.params['miniBatchSize'], len(self.dataset['data'][k]))
            data = self.dataset['data'][k][i:j]
            x = {}
            if 'tokens' in self.params['featureNames']:
                x['tokens_input:0'] = np.array([[_['lemma'] for _ in sent] for sent in data])
            if 'character' in self.params['featureNames']:
                x['chars_input:0'] = np.array([[_['char'] for _ in sent] for sent in data])
            if 'pos' in self.params['featureNames']:
                x['pos_input:0'] = np.array([[_['pos'] for _ in sent] for sent in data])
            if 'casing' in self.params['featureNames']:
                x['casing_input:0'] = np.array([[_['casing'] for _ in sent] for sent in data])
            x.update({'label:0': np.array([[_['label'][0] for _ in sent] for sent in data])})
            x.update({'sentence_length:0': np.array([len(sent) for sent in data])})
            yield x
            if j == len(self.dataset['data'][k]):
                i = 0
                k += 1
            else:
                i = j

    def model_predict(self, dict):
        if self.params['classifier'].lower() == 'softmax':
            result = self.session.run('softmax_output:0', feed_dict=dict)
        elif self.params['classifier'].upper() == 'CRF':
            result, score = self.session.run(['crf_output:0', 'score:0'], feed_dict=dict)

        return result

    def fit(self):
        if self.session is None:
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
            self.session.run(tf.local_variables_initializer())
        progress = trange(self.params['epoch'])
        for _ in progress:
            batch = next(self.datagenerator)
            self.session.run('train_op', feed_dict=batch)
            loss = self.session.run('loss:0', feed_dict=batch)
            precision = self.session.run('precision/update_op:0', feed_dict=batch)
            recall = self.session.run('recall/update_op:0', feed_dict=batch)
            if precision == 0 and recall == 0:
                score = 0
            else:
                score = precision * recall / (precision + recall)
            progress.set_postfix_str(f'loss: %f, prec: %f, rec: %f, f1-score: %f'%(loss, precision, recall, score))


    def predict(self, tokens, toTag=False):
        sents = []
        for sent in tokens:
            x = {}
            if 'tokens' in self.params['featureNames']:
                x['tokens_input:0'] = np.array([[_['lemma'] for _ in sent]])
            if 'character' in self.params['featureNames']:
                x['chars_input:0'] = np.array([[_['char'] for _ in sent]])
            if 'pos' in self.params['featureNames']:
                x['pos_input:0'] = np.array([[_['pos'] for _ in sent]])
            if 'casing' in self.params['featureNames']:
                x['casing_input:0'] = np.array([[_['casing'] for _ in sent]])
            x.update({'sentence_length:0': np.array([len(sent)])})
            y_pred = self.model_predict(x)[0]
            if not toTag:
                sents.append(y_pred)
            else:
                y_pred = [_.argmax(axis=-1) for _ in y_pred]
                sents.append([self.params['labelEntries'][_] for _ in y_pred])
        return sents

    def evaluate(self, dataset, sortedBySize=False):
        y_true = [w['label'][0] for g in dataset['data'] for s in g for w in s]
        if sortedBySize:
            data = dataset['data']
            y_pred = []
            for g in data:
                x = {}
                if 'tokens' in self.params['featureNames']:
                    x['tokens_input:0'] = np.array([[_['lemma'] for _ in sent] for sent in g])
                if 'character' in self.params['featureNames']:
                    x['chars_input:0'] = np.array([[_['char'] for _ in sent] for sent in g])
                if 'pos' in self.params['featureNames']:
                    x['pos_input:0'] = np.array([[_['pos'] for _ in sent] for sent in g])
                if 'casing' in self.params['featureNames']:
                    x['casing_input:0'] = np.array([[_['casing'] for _ in sent] for sent in g])
                x.update({'sentence_length:0': np.array([len(sent) for sent in g])})
                y_g_pred = self.model_predict(x)
                y_pred.extend([pred.argmax() for sent in y_g_pred for pred in sent])
        else:
            y_pred = [w.argmax() for g in dataset['data'] for s in self.predict(g) for w in s]
        validLabels = [_ for _ in self.params['labelEntries'] if _ != 'O']
        print(classification_report(y_true, y_pred, labels=[_ for _ in range(len(validLabels))],
                                    target_names=validLabels))

    def saveModel(self, path=None, initial_save=False, global_step=None):
        import json, h5py
        directory = path or self.params['modelSavePath']
        self.params['modelSavePath'] = directory
        if not os.path.exists(directory):
            os.makedirs(directory)
        if initial_save:
            self.saver = tf.train.Saver(max_to_keep=5)
        self.saver.save(self.session, os.path.join(directory, 'model'),
                        global_step=global_step, write_meta_graph=initial_save)
        #self.model.save(os.path.join(directory, 'model.h5'), True)
        with h5py.File(os.path.join(directory, 'config.h5'), 'w') as f:
            f.attrs['params'] = json.dumps(self.params)
            f.attrs['word2Idx'] = json.dumps(self.word2Idx)

    @staticmethod
    def loadModel(path='bilstm-fin-ner'):
        import tensorflow as tf
        import json, h5py, os, sys
        sys.path.append(os.path.abspath(os.path.dirname(__file__)))
        with h5py.File(os.path.join(path, 'config.h5'), 'r') as f:
            params = json.loads(f.attrs['params'])
            word2Idx = json.loads(f.attrs['word2Idx'])
        params['modelSavePath'] = path
        model = BiLSTM(params=params, word2Idx=word2Idx)
        model.session = tf.Session()
        saved_model = tf.train.import_meta_graph(os.path.join(path, 'model.meta'))
        saved_model.restore(model.session, tf.train.latest_checkpoint(path))
        #model.model = keras.models.load_model(os.path.join(path, 'model.h5'), custom_objects=create_custom_objects())
        return model


if __name__ == '__main__':
    import pickle

    # from BiLSTM import BiLSTM

    sample = pickle.load(open('finnish_sample.pkl', 'rb'))
    model = BiLSTM(raw_dataset=sample)
    model.fit()
    #model.saveModel(path='test', initial_save=True)