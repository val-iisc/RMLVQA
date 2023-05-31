import os
import sys
import json
sys.path.append(os.getcwd())

import numpy as np
from scipy.stats import entropy
from collections import Counter, defaultdict

import utils.utils as utils
import utils.config as config


def get_score(occurences):
    """ Average over all 10 choose 9 sets. """
    score_soft = occurences * 0.3
    score = score_soft if score_soft < 1.0 else 1.0
    return score


def filter_answers(answers_dset, min_occurence):
    """ Filtering answers whose frequency is less than min_occurence. """
    occurence = {}
    for ans_entry in answers_dset:
        answers = ans_entry['answers']
        gtruth = ans_entry['multiple_choice_answer']
        gtruth = utils.preprocess_answer(gtruth)
        if gtruth not in occurence:
            occurence[gtruth] = set()
        occurence[gtruth].add(ans_entry['question_id'])
    for answer in list(occurence.keys()):
        if len(occurence[answer]) < min_occurence:
            occurence.pop(answer)

    print('Num of answers that appear >= {} times: {}'.format(
                                min_occurence, len(occurence)))
    return occurence


def create_ans2label(occurence, name, cache_root):
    """ Map answers to label. """
    label, label2ans, ans2label = 0, [], {}
    for answer in occurence:
        label2ans.append(answer)
        ans2label[answer] = label
        label += 1

    utils.create_dir(cache_root)

    cache_file = os.path.join(cache_root, name+'_ans2label.json')
    json.dump(ans2label, open(cache_file, 'w'))
    cache_file = os.path.join(cache_root, name+'_label2ans.json')
    json.dump(label2ans, open(cache_file, 'w'))
    return ans2label


def compute_target(answers_dset, ans2label, name, cache_root):
    """ Augment answers_dset with soft score as label. """
    target = []
    for ans_entry in answers_dset:
        answers = ans_entry['answers']
        answer_count = {}
        for answer in answers:
            answer_ = answer['answer']
            answer_count[answer_] = answer_count.get(answer_, 0) + 1

        labels, scores = [], []
        for answer in answer_count:
            if answer not in ans2label:
                continue
            labels.append(ans2label[answer])
            score = get_score(answer_count[answer])
            scores.append(score)

        target.append({
            'question_type': ans_entry['question_type'],
            'question_id': ans_entry['question_id'],
            'image_id': ans_entry['image_id'],
            'labels': labels,
            'scores': scores,
            'answer_type': ans_entry['answer_type']
        })

    utils.create_dir(cache_root)
    cache_file = os.path.join(cache_root, name+'_target.json')
    json.dump(target, open(cache_file, 'w'))


def extract_type(answers_dset, name, ans2label, cache_root):
    """ Extract answer distribution for each question type. """
    qt_dict = defaultdict(list)
    for ans_entry in answers_dset:
        qt = ans_entry['question_type']
        ans_idxs = []
        for ans in ans_entry['answers']:
            ans = utils.preprocess_answer(ans['answer'])
            ans_idx = ans2label.get(ans, None)
            if ans_idx:
                ans_idxs.append(ans_idx)
        qt_dict[qt].extend(ans_idxs) # counting later

    number = 0
    # count answers for each question type
    for qt in qt_dict:
        ans_num_dict = Counter(qt_dict[qt])
        ans_num_dict = {k: v
            for k, v in ans_num_dict.items() if v >= 50}
        total_num = sum(ans_num_dict.values())
        for ans, ans_num in ans_num_dict.items():
            ans_num_dict[ans] = float(ans_num) / total_num

        values = np.array(list(ans_num_dict.values()), dtype=np.float32)
        if entropy(values + 1e-6, base=2) >= config.entropy:
            qt_dict[qt] = {k: 0.0 for k in ans_num_dict}
            number += 1
        else:
            qt_dict[qt] = ans_num_dict
    cache_file = os.path.join(cache_root, name + '_margin.json')
    json.dump(qt_dict, open(cache_file, 'w'))
    qt_dict = defaultdict(list)
    for ans_entry in answers_dset:
        qt = ans_entry['question_type']
        ans_idxs = []
        for ans in ans_entry['answers']:
            ans = utils.preprocess_answer(ans['answer'])
            ans_idx = ans2label.get(ans, None)
            if ans_idx:
                ans_idxs.append(ans_idx)
        qt_dict[qt].extend(ans_idxs)  # counting later


    for qt in qt_dict:
        ans_num_dict = Counter(qt_dict[qt])
        # ans_num_dict = {k: v
        #     for k, v in ans_num_dict.items() if v >= 50}

        qt_dict[qt] = ans_num_dict
    cache_file = os.path.join(cache_root, name + '_freq.json')
    json.dump(qt_dict, open(cache_file, 'w'))




if __name__ == '__main__':
    train_answers = utils.get_file(train=True, answer=True)
    val_answers = utils.get_file(val=True, answer=True)
    if not config.cp_data:
        train_answers = train_answers['annotations']
        val_answers = val_answers['annotations']

    answers = train_answers + val_answers
    print("filtering answers less than minimum occurrence...")
    occurence = filter_answers(answers, config.min_occurence)
    print("create answers to integer labels...")
    ans2label = create_ans2label(occurence, 'trainval', config.cache_root)

    print("converting target for train and val answers...")
    compute_target(train_answers, ans2label, 'train', config.cache_root)
    compute_target(val_answers, ans2label, 'val', config.cache_root)

    print("extracting answer margin for each question type...")
    extract_type(train_answers, 'train', ans2label, config.cache_root)
    extract_type(val_answers, 'val', ans2label, config.cache_root)
