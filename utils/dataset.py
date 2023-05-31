import os
import json
import h5py
import torch

import numpy as np
import utils.utils as utils
import utils.config as config
from torch.utils.data import Dataset

torch.utils.data.ConcatDataset.__getattr__ = lambda self, attr: getattr(self.datasets[0], attr)


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(
            ',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx[w])
        return tokens

    def dump_to_file(self, path):
        json.dump([self.word2idx, self.idx2word], open(path, 'w'))
        print('dictionary dumped to {}'.format(path))

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from {}'.format(path))
        word2idx, idx2word = json.load(open(path, 'r'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id': question['question_id'],
        'image_id': question['image_id'],
        'image': img,
        'question': question['question'],
        'answer': answer}
    return entry


def _load_dataset(cache_path, name, img_id2val):
    """ Load entries. img_id2val: dict {img_id -> val} ,
        val can be used to retrieve image or features.
    """
    train, val, test = False, False, False
    if name == 'train':
        train = True
    elif name == 'val':
        val = True
    else:
        test = True
    question_path = utils.path_for(
        train=train, val=val, test=test, question=True)
    questions = json.load(open(question_path, 'r'))
    if not config.cp_data:
        questions = questions['questions']
    questions = sorted(questions, key=lambda x: x['question_id'])
    if test:  # will be ignored anyway
        answers = [
            {'image_id': 0, 'question_id': 0, 'question_type': '',
             'labels': [], 'scores': []}
            for _ in range(len(questions))]
    else:
        answer_path = os.path.join(cache_path, '{}_target.json'.format(name))
        print('{}_target.json'.format(name))
        answers = json.load(open(answer_path, 'r'))
        answers = sorted(answers, key=lambda x: x['question_id'])
        utils.assert_eq(len(questions), len(answers))

    entries = []
    for question, answer in zip(questions, answers):
        if not test:
            utils.assert_eq(question['question_id'], answer['question_id'])
            utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        entries.append(_create_entry(img_id2val[img_id], question, answer))
    return entries


def _load_margin(cache_path, name, entries):
    """ Load answer margin per question type.
    """
    print('{}_margin.json'.format(name))
    mask_path = os.path.join(cache_path, '{}_margin.json'.format(name))
    qt_dict = json.load(open(mask_path, 'r'))
    #print(qt_dict.keys())
    for qt in qt_dict:
        ans_num_dict = utils.json_keys2int(qt_dict[qt])
        ans = torch.tensor(list(ans_num_dict.keys()), dtype=torch.int64)
        portion = torch.tensor(list(ans_num_dict.values()), dtype=torch.float32)
        qt_dict[qt] = (ans, portion)

    mask_path = os.path.join(cache_path, '{}_freq.json'.format(name))
    qt_dict_freq = json.load(open(mask_path, 'r'))

    for qt in qt_dict_freq:
        ans_num_dict = utils.json_keys2int(qt_dict_freq[qt])
        ans = torch.tensor(list(ans_num_dict.keys()), dtype=torch.int64)
        portion = torch.tensor(list(ans_num_dict.values()), dtype=torch.float32)
        qt_dict_freq[qt] = (ans, portion)

    return qt_dict, qt_dict_freq

    # for entry in entries:
    #     ans_entry = entry['answer']
    #     qt = ans_entry['question_type']
    #     ansrs = ans_entry['labels']
    #     ans_num_dict = utils.json_keys2int(qt_dict[qt])
    #     ans_margin = []
    #     for ans in ansrs:
    #         ans_margin.append(ans_num_dict.get(ans, 0.0))
    #     entry['answer']['margin'] = ans_margin
    # return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val', 'test']
        self.dictionary = dictionary
        
        # loading answer-label
        self.ans2label = json.load(open(os.path.join(
            config.cache_root, 'trainval_ans2label.json'), 'r'))

        self.label2ans = json.load(open(os.path.join(
            config.cache_root, 'trainval_label2ans.json'), 'r'))
        self.num_ans_candidates = len(self.ans2label)

        # loading image features
        image_split = 'test' if name == 'test' else 'trainval'
        self.img_id2idx = json.load(open(os.path.join(
            config.ids_path, '{}36_imgid2idx.json'.format(
                image_split)), 'r'), object_hook=utils.json_keys2int)
        self.h5_path = os.path.join(config.rcnn_path, '{}36.h5'.format(image_split))
        if config.in_memory:
            print('loading image features from h5 file')
            with h5py.File(self.h5_path, 'r') as hf:
                self.features = np.array(hf.get('image_features'))
                self.spatials = np.array(hf.get('spatial_features'))

        self.entries = _load_dataset(config.cache_root, name, self.img_id2idx)
        self.margins, self.freq = _load_margin(config.cache_root, name, self.entries)

        self.tokenize()
        self.tensorize()
        self.v_dim = config.output_features
        self.s_dim = config.num_fixed_boxes

    def tokenize(self, max_length=config.max_question_len):
        """ Tokenizes the questions.
            This will add q_token in each entry of the dataset.
            -1 represent nil, and should be treated as padding_idx in embedding.
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        if config.in_memory:
            self.features = torch.from_numpy(self.features)
            self.spatials = torch.from_numpy(self.spatials)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def load_image(self, image_id):
        """ Load one image feature. """
        if not hasattr(self, 'image_feat'):
            self.image_feat = h5py.File(self.h5_path, 'r')
        features = self.image_feat['image_features'][image_id]
        spatials = self.image_feat['spatial_features'][image_id]
        return torch.from_numpy(features), torch.from_numpy(spatials)

    def __getitem__(self, index):
        entry = self.entries[index]
        if config.in_memory:
            features = self.features[entry['image']]
            spatials = self.spatials[entry['image']]
        else:
            features, spatials = self.load_image(entry['image'])

        question_id = entry['question_id']
        question = entry['q_token']
        answer = entry['answer']
        q_type = answer['question_type']
        #qtype = answer['qtype']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)

        margin_label, margin_score = self.margins[q_type]
        freq_label, freq_score = self.freq[q_type]

        betas = [0]
        torch.set_printoptions(profile="full")
        idx = 0
        eff = 1 - torch.float_power(betas[idx], freq_score)
        per0 = (1 - betas[idx]) / eff
        per0 = per0 / torch.sum(per0) * freq_score.shape[0]
        per0 = per0.float()

        target_margin = torch.zeros(self.num_ans_candidates)
        freq_margin0 = torch.zeros(self.num_ans_candidates)

        if labels is not None:
            target.scatter_(0, labels, scores)
            target_margin.scatter_(0, margin_label, margin_score)
            freq_margin0.scatter_(0, freq_label, per0)
        bias = entry['bias'] if 'bias' in entry else 0
        return features, question, target, target_margin, bias, question_id, freq_margin0, q_type

    def __len__(self):
        return len(self.entries)
