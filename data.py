import torch
from torch.utils.data import Dataset
from typing import Sequence, Tuple, List, Union
import linecache
import re
import os
import random
import subprocess
from Bio import SeqIO

class EbdDataset(Dataset):
    def __init__(self, path: List[str]):
        #self.records = []
        #for i in path:
        #    self.records.extend(list(SeqIO.parse(i, "fasta")))
        self.records = list(SeqIO.parse(path, "fasta"))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rec = self.records[index]
        #return rec.id, re.sub('[(a-z)(\-)]', '', rec.seq.__str__())
        return rec.id, re.sub('[(\-)]', '', rec.seq.__str__())

class QueryDataset(Dataset):
    def __init__(self, path: List[str]):
        self.records = []
        for i in path:
            self.records.extend(list(SeqIO.parse(i, "fasta")))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rec = self.records[index]
        #return rec.id, re.sub('[(a-z)(\-)]', '', rec.seq.__str__())
        return rec.id, re.sub('[(\-)]', '', rec.seq.__str__())

class  MyDataset(Dataset):
    def __init__(self, names: List[str], lines: List[int]):
        self.names = names
        self.lines = lines

    def get_pair(self, path: str, lines: int) -> Tuple[str, str]:
        lines = lines//2
        idx2 = random.randint(0, lines-1)
        #seq1 = re.sub('[(a-z)(\-)]', '', linecache.getline(path, 2))
        #seq2 = re.sub('[(a-z)(\-)]', '', linecache.getline(path, 2*idx2 + 2))
        seq1 = re.sub('[(\-)]', '', linecache.getline(path, 2))
        seq2 = re.sub('[(\-)]', '', linecache.getline(path, 2*idx2 + 2))

        return seq1, seq2

    def __getitem__(self, index: int) -> Tuple[str, str]:
        seq1, seq2 = self.get_pair(self.names[index], self.lines[index])
        return seq1, seq2

    def __len__(self):
        return len(self.names)

    def get_batch_indices(self, batch_size: int) -> List[List[int]] :
        batches = []
        buf = []
        iters = len(self.names) // batch_size

        for i in range(iters):
            buf = random.sample(range(len(self.names)), batch_size)
            batches.append(buf)

        return batches

from torch.utils.data.distributed import DistributedSampler

class DistributedProxySampler(DistributedSampler):
    """Sampler that restricts data loading to a subset of input sampler indices.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Input sampler is assumed to be of constant size.

    Arguments:
        sampler: Input data sampler.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, sampler, num_replicas=None, rank=None):        
        super(DistributedProxySampler, self).__init__(sampler, num_replicas=num_replicas, rank=rank, shuffle=False)
        self.sampler = sampler

    def __iter__(self):
        # deterministically shuffle based on epoch
        torch.manual_seed(self.epoch)
        indices = list(self.sampler)

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        if len(indices) != self.total_size:
            raise RuntimeError("{} vs {}".format(len(indices), self.total_size))

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError("{} vs {}".format(len(indices), self.num_samples))

        return iter(indices)

from typing import Sequence, Tuple, List, Union
class BatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        limit_size = 500
        batch_size = len(raw_batch)
        max_len = max(max(len(seq1),len(seq2)) for seq1, seq2 in raw_batch)
        max_len = min(limit_size, max_len)
        tokens1 = torch.empty(
            (
                batch_size,
                max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens2 = torch.empty(
            (
                batch_size,
                max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens1.fill_(self.alphabet.padding_idx)
        tokens2.fill_(self.alphabet.padding_idx)

        for i, (seq_str1, seq_str2) in enumerate(raw_batch):
            if self.alphabet.prepend_bos:
                tokens1[i, 0] = self.alphabet.cls_idx
                tokens2[i, 0] = self.alphabet.cls_idx
            seq1 = torch.tensor([self.alphabet.get_idx(s) for s in seq_str1[:limit_size]], dtype=torch.int64)
            seq2 = torch.tensor([self.alphabet.get_idx(s) for s in seq_str2[:limit_size]], dtype=torch.int64)
            tokens1[
                i,
                int(self.alphabet.prepend_bos) : min(len(seq_str1), max_len) + int(self.alphabet.prepend_bos),
            ] = seq1
            tokens2[
                i,
                int(self.alphabet.prepend_bos) : min(len(seq_str2), max_len) + int(self.alphabet.prepend_bos),
            ] = seq2
            if self.alphabet.append_eos:
                tokens1[i, min(len(seq_str1), max_len) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx
                tokens2[i, min(len(seq_str2), max_len) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx
        return tokens1, tokens2

class SingleConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        limit_size = 500
        batch_size = len(raw_batch)
        max_len = max(len(seq) for id, seq in raw_batch)
        max_len = min(limit_size, max_len)
        ids = []
        tokens = torch.empty(
            (
                batch_size,
                max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)

        for i, (id, seq_str) in enumerate(raw_batch):
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq1 = torch.tensor([self.alphabet.get_idx(s) for s in seq_str[:limit_size]], dtype=torch.int64)
            ids.append(id)
            tokens[
                i,
                int(self.alphabet.prepend_bos) : min(len(seq_str), max_len) + int(self.alphabet.prepend_bos),
            ] = seq1
            if self.alphabet.append_eos:
                tokens[i, min(len(seq_str), max_len) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx
        return ids, tokens

