import os
import json
import errno
import numpy as np
from collections import defaultdict, Counter

import utils.config as config
from vqa_eval.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval


def path_for(train=False, val=False, test=False, question=False, answer=False):
    assert train + val + test == 1
    assert question + answer == 1

    if train:
        split = 'train' if config.cp_data else 'train2014'
    elif val:
        split = 'test' if config.cp_data else 'val2014'
    else:
        split = config.test_split

    if question:
        fmt = '{0}_{1}_{2}_questions.json'
    else:
        if test:
            # will be ignored anyway
            split = 'val2014'
        fmt = '{0}_{1}_{2}_annotations.json' if \
                config.cp_data else '{1}_{2}_annotations.json'

    if config.version == 'v2' and not config.cp_data:
        fmt = 'v2_' + fmt
    if config.cp_data:
        s = fmt.format(config.task, config.version, split)
    else:
        s = fmt.format(config.task, config.image_dataset, split)
    print(s)
    return os.path.join(config.main_path, s)


def get_file(train=False, val=False, test=False, question=False, answer=False):
    """ Get the correct question or answer file."""
    _file = path_for(train=train, val=val, test=test,
                            question=question, answer=answer)
    with open(_file, 'r') as fd:
        _object = json.load(fd)
    return _object


def preprocess_answer(answer):
    """ Mimicing the answer pre-processing with evaluation server. """
    dummy_vqa = lambda: None
    dummy_vqa.getQuesIds = lambda: None
    vqa_eval = VQAEval(dummy_vqa, None)

    answer = vqa_eval.processDigitArticle(
            vqa_eval.processPunctuation(answer))
    answer = answer.replace(',', '')
    return answer


def assert_eq(real, expected):
    assert real == expected, '{} (true) vs {} (expected)'.format(real, expected)


def assert_array_eq(real, expected):
    EPS = 1e-7
    assert (np.abs(real-expected) < EPS).all(), \
        '{} (true) vs {} (expected)'.format(real, expected)


def json_keys2int(x):
    return {int(k): v for k, v in x.items()}


def load_folder(folder, suffix):
    imgs = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(suffix):
            imgs.append(os.path.join(folder, f))
    return imgs


def load_imageid(folder):
    images = load_folder(folder, 'jpg')
    img_ids = set()
    for img in images:
        img_id = int(img.split('/')[-1].split('.')[0].split('_')[-1])
        img_ids.add(img_id)
    return img_ids


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def append_bias(train_dset, eval_dset, answer_voc_size):
    """
        Compute the bias:
        The bias here is just the expected score for each answer/question type
    """
    # question_type -> answer -> total score
    question_type_to_probs = defaultdict(Counter)
    # question_type -> num_occurances
    question_type_to_count = Counter()
    for ex in train_dset.entries:
        ans = ex["answer"]
        q_type = ans["question_type"]
        question_type_to_count[q_type] += 1
        if ans["labels"] is not None:
            for label, score in zip(ans["labels"], ans["scores"]):
                question_type_to_probs[q_type][label] += score

    question_type_to_prob_array = {}
    for q_type, count in question_type_to_count.items():
        prob_array = np.zeros(answer_voc_size, np.float32)
        for label, total_score in question_type_to_probs[q_type].items():
            prob_array[label] += total_score
        prob_array /= count
        question_type_to_prob_array[q_type] = prob_array

    # Now add a `bias` field to each example
    for ds in [train_dset, eval_dset]:
        for ex in ds.entries:
            q_type = ex["answer"]["question_type"]
            ex["bias"] = question_type_to_prob_array[q_type]


class Tracker:
    """ Keep track of results over time, while having access to
        monitors to display information about them.
    """
    def __init__(self):
        self.data = {}

    def track(self, name, *monitors):
        """ Track a set of results with given monitors under some name (e.g. 'val_acc').
            When appending to the returned list storage, use the monitors
            to retrieve useful information.
        """
        l = Tracker.ListStorage(monitors)
        self.data.setdefault(name, []).append(l)
        return l

    def to_dict(self):
        # turn list storages into regular lists
        return {k: list(map(list, v)) for k, v in self.data.items()}


    class ListStorage:
        """ Storage of data points that updates the given monitors """
        def __init__(self, monitors=[]):
            self.data = []
            self.monitors = monitors
            for monitor in self.monitors:
                setattr(self, monitor.name, monitor)

        def append(self, item):
            for monitor in self.monitors:
                monitor.update(item)
            self.data.append(item)

        def __iter__(self):
            return iter(self.data)

    class MeanMonitor:
        """ Take the mean over the given values """
        name = 'mean'

        def __init__(self):
            self.n = 0
            self.total = 0

        def update(self, value):
            self.total += value
            self.n += 1

        @property
        def value(self):
            return self.total / self.n

    class MovingMeanMonitor:
        """ Take an exponentially moving mean over the given values """
        name = 'mean'

        def __init__(self, momentum=0.9):
            self.momentum = momentum
            self.first = True
            self.value = None

        def update(self, value):
            if self.first:
                self.value = value
                self.first = False
            else:
                m = self.momentum
                self.value = m * self.value + (1 - m) * value
