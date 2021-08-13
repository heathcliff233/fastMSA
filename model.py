import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import Dataset, DataLoader

class MyEncoder(nn.Module):
    def __init__(self, bert, proj_dim):
        super(MyEncoder, self).__init__()
        self.bert = bert 
        self.num_layers = bert.num_layers
        repr_layers = -1
        self.repr_layers = (repr_layers + self.num_layers + 1) % (self.num_layers + 1)
        self.recast = nn.Linear(768, proj_dim) if proj_dim != 0 else None

    def forward_once(self, x):
        x = self.bert(x, repr_layers=[self.repr_layers])['representations'][self.repr_layers]
        x[:,1:] *= 0
        x = x.sum(dim=1)
        #x = x[:,0]
        if self.recast :
            x = self.recast(x)
        return x

    def forward(self, batch):
        seq1, seq2 = batch
        qebd = self.forward_once(seq1)
        cebd = self.forward_once(seq2)
        return qebd, cebd

    def get_loss(self, ebd):
        qebd, cebd = ebd 
        sim_mx = dot_product_scores(qebd, cebd)
        label = torch.arange(sim_mx.shape[0], dtype=torch.long)
        sm_score = F.log_softmax(sim_mx, dim=1)
        loss = F.nll_loss(
            sm_score,
            label.to(sm_score.device),
            reduction="mean"
        )
        return loss
    
    def get_acc(self, ebd):
        qebd, cebd = ebd 
        sim_mx = dot_product_scores(qebd, cebd)
        label = torch.arange(sim_mx.shape[0], dtype=torch.long)
        sm_score = F.log_softmax(sim_mx, dim=1)
        max_score, max_idxs = torch.max(sm_score, 1)
        correct_predictions_count = (
            max_idxs == label.to(sm_score.device)
        ).sum()
        return correct_predictions_count, sim_mx.shape[0]


def dot_product_scores(q_vectors, ctx_vectors):
    """
    calculates q->ctx scores for every row in ctx_vector
    :param q_vector:
    :param ctx_vector:
    :return:
    """
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r
