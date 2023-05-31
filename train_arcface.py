import pickle
import json
import torch
import torch.nn as nn
from tqdm import tqdm
import utils.config as config
from torch.nn import functional as F
import numpy as np
from tensorboardX import SummaryWriter

writer_tsne = SummaryWriter('runs/tsne')

def compute_supcon_loss(feats, qtype):
    tau = 1.0
    if isinstance(qtype, tuple):
      i = 0
      dic = {}
      for item in qtype:
          if item not in dic:
              dic[item] = i
              i = i + 1
      tau = 1.0
      qtype = torch.tensor([dic[item] for item in qtype]).cuda()
    feats_filt = F.normalize(feats, dim=1)
    targets_r = qtype.reshape(-1, 1)
    targets_c = qtype.reshape(1, -1)
    mask = targets_r == targets_c
    mask = mask.int().cuda()
    feats_sim = torch.exp(torch.matmul(feats_filt, feats_filt.T) / tau)
    negatives = feats_sim*(1.0 - mask)
    negative_sum = torch.sum(negatives)
    positives = torch.log(feats_sim/negative_sum)*mask
    positive_sum = torch.sum(positives)
    positive_sum = positive_sum/torch.sum(mask)

    sup_con_loss = -1*torch.mean(positive_sum)
    return sup_con_loss

def compute_acc(logits, labels):
    pred = torch.argmax(logits, dim = 1)
    pred = pred.detach().cpu().numpy()
    score = (pred == np.array(labels))
    tot_correct = score.sum()
    return tot_correct


def compute_score_with_logits(logits, labels):
    _, log_index = logits.max(dim=1, keepdim=True)
    scores = labels.gather(dim=1, index=log_index)
    return scores
    
def compute_loss(output, labels):

    #Function for calculating loss
    
    ce_loss = nn.CrossEntropyLoss(reduction='mean')(output, labels.squeeze(-1).long())
    
    return ce_loss


def saved_for_eval(dataloader, results, question_ids, answer_preds):
    """ Save as a format accepted by the evaluation server. """
    _, answer_ids = answer_preds.max(dim=1)
    answers = [dataloader.dataset.label2ans[i] for i in answer_ids]
    for q, a in zip(question_ids, answers):
        entry = {
            'question_id': q.item(),
            'answer': a,
        }
        results.append(entry)
    return results


def train(model, m_model, optim, train_loader, loss_fn, tracker, writer, tb_count, epoch, args):

    loader = tqdm(train_loader, ncols=0)
    loss_trk = tracker.track('loss', tracker.MovingMeanMonitor(momentum=0.99))
    acc_trk = tracker.track('acc', tracker.MovingMeanMonitor(momentum=0.99))

    for v, q, a, mg, bias, q_id, f1, type in loader:
        v = v.cuda()
        q = q.cuda()
        a = a.cuda()
        mg = mg.cuda()
        bias = bias.cuda()
        hidden_, ce_logits = model(v, q)
        hidden, pred = m_model(hidden_, ce_logits, mg, epoch, a)
        f1 = f1.cuda()
        dict_args = {'margin': mg, 'bias': bias, 'hidden': hidden, 'epoch': epoch, 'per': f1}
        gt = torch.argmax(a, 1)
        
        #If bias-injection or learnable margins is enabled.
        if config.learnable_margins or config.bias_inject:
            #Use cross entropy loss to train the bias-injecting module
            ce_loss = - F.log_softmax(ce_logits, dim=-1) * a
            ce_loss = ce_loss * f1
            loss = ce_loss.sum(dim=-1).mean() + loss_fn(hidden, a, **dict_args)
        else:
            loss = loss_fn(hidden, a, **dict_args)
        
        #Add the supcon loss, as mentioned in Section 3 of main paper.
        if config.supcon:
            loss = loss + compute_supcon_loss(hidden_, gt)
        writer.add_scalars('data/losses', {
        }, tb_count)
        tb_count += 1

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optim.step()
        optim.zero_grad()
        
        #Ensemble the logit heads, as mentioned in Section 3 of the main paper, if bias-injection is enabled
        if config.bias_inject or config.learnable_margins:
          ce_logits = F.normalize(ce_logits)
          pred_l = F.normalize(pred)
          pred = (ce_logits + pred_l) / 2
        batch_score = compute_score_with_logits(pred, a.data)

        fmt = '{:.4f}'.format
        loss_trk.append(loss.item())
        acc_trk.append(batch_score.mean())
        loader.set_postfix(loss=fmt(loss_trk.mean.value),
                           acc=fmt(acc_trk.mean.value))

    return tb_count


#Evaluation code
def evaluate(model, m_model, dataloader, epoch=0, write=False):
    score = 0
    results = []  # saving for evaluation
    type_score = 0
    for v, q, a, mg, _, q_id, _, qtype in tqdm(dataloader, ncols=0, leave=True):
        v = v.cuda()
        q = q.cuda()
        mg = mg.cuda()
        a = a.cuda()
        hidden, ce_logits = model(v, q)
        hidden, pred = m_model(hidden, ce_logits, mg, epoch, a)
        
        #Ensemble the logit heads
        if config.learnable_margins or config.bias_inject:
          ce_logits = F.softmax(F.normalize(ce_logits) / config.temp, 1)
          pred_l = F.softmax(F.normalize(pred), 1)
          pred = config.alpha * pred_l + (1-config.alpha) * ce_logits
        if write:
            results = saved_for_eval(dataloader, results, q_id, pred)
        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score  
        
    print(score, len(dataloader.dataset))
    score = score / len(dataloader.dataset)
    
    if write:
        print("saving prediction results to disk...")
        result_file = 'vqa_{}_{}_{}_{}_results.json'.format(
            config.task, config.test_split, config.version, epoch)
        with open(result_file, 'w') as fd:
            json.dump(results, fd)
    print(score)
    return score
