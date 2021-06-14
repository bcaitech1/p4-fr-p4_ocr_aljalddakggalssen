import torch
import numpy as  np
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class BeamInfo:
    score: torch.Tensor
    seq: torch.Tensor
    features: list
    pos_enc_cache: torch.Tensor
    has_finished: torch.Tensor
    just_finished: torch.Tensor

    # add 할때 eos이거나 부모의 현재 batch has_finished True면
    # score를 마이너스 무한대로 주기 -> 그리고 해당 batch has_finished True로
    # has_finished도 select_top_k 구현해야됨

    # has_finished인 애들 따로 관리?

class BeamSearcher:
    def __init__(self, k, st_id, eos_id, pad_id, layer_nums, batch_size, max_seq, device,
        use_tube=False):
        self.k = k
        self.st_id = st_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.layer_nums = layer_nums
        self.device = device
        self.batch_size = batch_size
        self.max_seq = max_seq
        self.use_tube = use_tube
        self.reset()

    def reset(self):
        self.cur_max_idx = 1

        self.cache = [
            BeamInfo(
                torch.zeros(self.batch_size).to(self.device),
                torch.LongTensor(self.batch_size).view(-1, 1).fill_(self.st_id).to(self.device), 
                [None] * self.layer_nums,
                None, 
                torch.zeros(self.batch_size, dtype=torch.bool).to(self.device),
                torch.zeros(self.batch_size, dtype=torch.bool).to(self.device),
            )
        ]
        self.candidates = []
        self.fin = [[] for _ in range(self.batch_size)]

    def get(self, idx):
        cur = self.cache[idx]
        return cur

    def add(self, par_idx, features, generated, pos_enc_cache=None):
        # generated [b, c]
        probs = torch.softmax(generated, -1)
        probs = torch.log(probs)
        
        values, indices = torch.topk(probs, self.k)

        par_info = self.cache[par_idx]

        for i in range(self.k):
            seq = torch.cat([par_info.seq, indices[:, i].unsqueeze(-1)], dim=-1)
            len_seq = len(seq)
            score = par_info.score * (len_seq-1)/len_seq + values[:, i]/len_seq
            features = features

            just_finished = (seq[:, -1] == self.eos_id) & (~par_info.has_finished)
            has_finished = par_info.has_finished | (seq[:, -1] == self.eos_id)
            old_finished = has_finished & (~just_finished)
            score = score.masked_fill(old_finished, float('-inf'))

            cur_info = BeamInfo(score, seq, features, pos_enc_cache, has_finished, just_finished)
            self.candidates.append(cur_info)

    def _select_top_k(self, arr, k):
        scores = [ele.score for ele in arr]
        stacked_scores = torch.stack(scores, dim=1)
        val, idx = torch.topk(stacked_scores, k, dim=1)
        new_scores = val.T

        seqs = [ele.seq for ele in arr]
        stacked_seqs = torch.stack(seqs, dim=1)
        b, candid, s = stacked_seqs.shape
        bidx = idx.unsqueeze(-1).expand(-1, -1, s)
        new_seqs = torch.gather(stacked_seqs, 1, bidx)
        new_seqs = [new_seqs[:, i] for i in range(candid)]

        features = [ele.features for ele in arr]
        blsd = [torch.stack(x, 1) for x in features] # Batch, Layers, Seq, Dim
        bclsd = torch.stack(blsd, 1) # Batch, Candidates, Layers, Seq, Dim
        b, candid, layers, s, d = bclsd.shape
        bidx = idx.reshape(b, k, 1, 1, 1).expand(-1, -1, layers, s, d)
        new_features = torch.gather(bclsd, 1, bidx) # Batch, Candidates, Layers, Seq, Dim
        new_features = [[new_features[:, j, i] for i in range(layers)] for j in range(k)]

        if self.use_tube:
            pos_encoded = [ele.pos_enc_cache for ele in arr]
            cat_encoded = torch.stack(pos_encoded, dim=1)
            b, candid, s, d = cat_encoded.shape
            bidx = idx.reshape(b, k, 1, 1).expand(-1, -1, s, d)
            new_encoded = torch.gather(cat_encoded, 1, bidx)
            new_encoded = [new_encoded[:, i] for i in range(k)]
        else:
            new_encoded = [None] * k

        has_finished = [ele.has_finished for ele in arr]
        stacked_has_finished = torch.stack(has_finished, dim=1)
        new_has_finised = torch.gather(stacked_has_finished, 1, idx)
        new_has_finised = new_has_finised.T

        just_finished = [ele.just_finished for ele in arr]
        stacked_just_finished = torch.stack(just_finished, dim=1)
        new_just_finished = torch.gather(stacked_just_finished, 1, idx)
        new_just_finished = new_just_finished.T

        return [
            BeamInfo(
                new_scores[i],
                new_seqs[i],
                new_features[i],
                new_encoded[i],
                new_has_finised[i],
                new_just_finished[i],
            ) for i in range(k)
        ]

    def keep_k(self, last=False):
        # 더이상 할게 없으면 False
        # 아니면 True

        top_candidates = self._select_top_k(self.candidates, len(self.candidates))

        cnt = 0
        idx = 0
        len_can = len(self.candidates)

        self.cache = []
        while cnt < self.k and idx < len_can:
            candidate = top_candidates[idx]
            
            for i in range(self.batch_size):
                if candidate.just_finished[i] or last:
                    cur_score = candidate.score[i]
                    cur_seq = candidate.seq[i]

                    self.fin[i].append((cur_score.item(), cur_seq))

            if not candidate.has_finished.all():
                self.cache.append(candidate)
                cnt += 1
            idx += 1
            
        self.candidates = []
        self.cur_max_idx = cnt

        return self.cur_max_idx != 0

    def get_best(self):
        result = []
        scores = []
        for i in range(self.batch_size):
            best = sorted(self.fin[i], key=lambda x: x[0], reverse=True)
            result.append(best[0][1])
            scores.append(best[0][0])
        
        long_len = max([len(x) for x in result])
        long_len = max(self.max_seq, long_len)
        for i in range(self.batch_size):
            gap = long_len - len(result[i])
            result[i] = torch.cat((result[i], torch.Tensor([self.pad_id] * gap).to(self.device).to(torch.long)), dim=0)

        return torch.stack(result, dim=0)[:, 1:], scores