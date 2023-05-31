import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
import math
import utils.config as config
from modules.fc import FCNet
from modules.classifier import SimpleClassifier
from modules.attention import Attention, NewAttention
from modules.language_model import WordEmbedding, QuestionEmbedding
from collections import Counter


class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, fusion, num_hid, num_class):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.weight = SimpleClassifier(num_hid, num_hid * 2, num_class, 0.5)
        #self.qweight = SimpleClassifier(num_hid, num_hid * 2, 65, 0.5)
        # self.weight = nn.Parameter(torch.FloatTensor(num_class, num_hid))
        # nn.init.xavier_normal_(self.weight)

    def forward(self, v, q):
        """
        Forward=
        v: [batch, num_objs, obj_dim]
        q: [batch_size, seq_length]
        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb, _ = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)

        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr #Final multimodal features, denoted as x in the main paper. This is the UpDn model.

        #This is the bias injecting component, as shown in subsection 3.4 of the main paper
        ce_logits = self.weight(joint_repr)
        #q_logits = self.qweight(joint_repr)
        
        return joint_repr, ce_logits


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=config.scale, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.std = 0.1
        self.temp = config.temp

    def forward(self, input, learned_mg, m, epoch, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if self.training is False:
            return None, cosine
        
        #Set beta (Subsecion 3.3 in main paper
        beta_factor = epoch // 15
        beta = 1.0 - (beta_factor * 0.1)

        #Calculate the learnable instance-level margins, Subsection 3.3 in main paper
        learned_mg = torch.where(m > 1e-12, learned_mg.double(), -1000.0).float()
        margin = F.softmax(learned_mg / self.temp, dim=1)
        
        # Perform randomization as mentioned in Section 3 of main paper
        if config.randomization:
            m = torch.normal(mean=m, std=self.std)
            
        #Combine the margins, as in Subsection 3.3 of main paper.
        if config.learnable_margins:
            m[label != 0] = beta * m[label != 0] + (1 - beta) * margin[label != 0]
        m = 1 - m

        #Compute the AdaArc angular margins and the corresponding logits
        self.cos_m = torch.cos(m)
        self.sin_m = torch.sin(m)
        self.th = torch.cos(math.pi - m)
        self.mm = torch.sin(math.pi - m) * m
        # --------------------------- cos(theta) & phi(theta) ---------------------------

        # cosine = input
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        output = phi * self.s
        return output, cosine


def l2_norm(input, dim=-1):
    norm = torch.norm(input, dim=dim, keepdim=True)
    output = torch.div(input, norm)
    return output


def build_baseline(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    fusion = FCNet([num_hid, num_hid*2], dropout=0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net,
                     fusion, num_hid, dataset.num_ans_candidates)


def build_baseline_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    fusion = FCNet([num_hid, num_hid*2], dropout=0.5)
    basemodel = BaseModel(w_emb, q_emb, v_att, q_net, v_net,
                     fusion, num_hid, dataset.num_ans_candidates)
    margin_model = ArcMarginProduct(num_hid, dataset.num_ans_candidates)
    return basemodel, margin_model
