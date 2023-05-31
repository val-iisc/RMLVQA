import torch
from torch import nn
from torch.nn import functional as F

import utils.config as config


def convert_sigmoid_logits_to_binary_logprobs(logits):
    """Computes log(sigmoid(logits)), log(1-sigmoid(logits))."""
    log_prob = -F.softplus(-logits)
    log_one_minus_prob = -logits + log_prob
    return log_prob, log_one_minus_prob


def cross_entropy_loss(logits, labels, **kwargs):
    """ Modified cross entropy loss. """
    if config.use_cos:
        logits = config.scale * (logits - (1 - kwargs['ldam']))
        # logits = config.scale * (logits - 0.9)
        # logits = config.scale * logits
    f = kwargs['per']
    nll = F.log_softmax(logits, dim=-1)
    loss = -nll * labels
    loss = loss * f
    return loss.sum(dim=-1).mean()


def cross_entropy_loss_arc(logits, labels, **kwargs):
    """ Modified cross entropy loss. """
    f = kwargs['per']
    nll = F.log_softmax(logits, dim=-1)
    loss = -nll * labels
    loss = loss * f

    return loss.sum(dim=-1).mean()

class Plain(nn.Module):
    def forward(self, logits, labels, **kwargs):
        if config.loss_type == 'ce':
            loss = cross_entropy_loss(logits, labels, **kwargs)
        elif config.loss_type == 'ce_margin':
            loss = cross_entropy_loss_arc(logits, labels, **kwargs)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss *= labels.size(1)
        return loss
