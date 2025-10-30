import torch
import numpy as np
import torch.nn as nn
from datetime import datetime

class DateEmbedding(nn.Module):
    def __init__(self, hidden_units, max_years=50):
        super().__init__()
        self.hidden_units = hidden_units

        self.year_embed = nn.Embedding(max_years + 1, hidden_units, padding_idx=0)  # 0~max_years
        self.month_embed = nn.Embedding(13, hidden_units, padding_idx=0)  # 1~12月
        self.day_embed = nn.Embedding(32, hidden_units, padding_idx=0)  # 1~31日

        self.fusion_mlp = nn.Sequential(
            nn.Linear(3 * hidden_units, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_units, hidden_units),
            nn.LayerNorm(hidden_units)
        )

        self.base_year = 2000

    def decompose_timestamp(self, ts_tensor):
        device = ts_tensor.device
        batch_size, seq_len = ts_tensor.shape

        years = torch.zeros_like(ts_tensor, dtype=torch.long, device=device)
        months = torch.zeros_like(ts_tensor, dtype=torch.long, device=device)
        days = torch.zeros_like(ts_tensor, dtype=torch.long, device=device)

        non_zero_mask = (ts_tensor != 0)

        if non_zero_mask.any():
            non_zero_ts = ts_tensor[non_zero_mask].cpu().numpy()

            dt_objects = np.array([datetime.fromtimestamp(ts) for ts in non_zero_ts])

            years_np = np.array([dt.year for dt in dt_objects])
            months_np = np.array([dt.month for dt in dt_objects])
            days_np = np.array([dt.day for dt in dt_objects])

            relative_years = np.clip(years_np - self.base_year, 0, self.year_embed.num_embeddings - 2)

            years[non_zero_mask] = torch.tensor(relative_years + 1, device=device)  # +1避开padding_idx
            months[non_zero_mask] = torch.tensor(months_np, device=device)
            days[non_zero_mask] = torch.tensor(days_np, device=device)

        return years, months, days

    def forward(self, ts_seq):
        # 分解时间戳为四个分量
        year_idx, month_idx, day_idx = self.decompose_timestamp(ts_seq)

        year_emb = self.year_embed(year_idx)
        month_emb = self.month_embed(month_idx)
        day_emb = self.day_embed(day_idx)

        combined = torch.cat([year_emb, month_emb, day_emb], dim=-1)

        return self.fusion_mlp(combined)

class SimplifiedMoETimeEncoding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_experts = args.num_experts
        self.hidden_units = args.hidden_units

        self.date_embedding = DateEmbedding(args.hidden_units)
        
        self.cross_attn_fusion = nn.MultiheadAttention(
            embed_dim=args.hidden_units,
            num_heads=2,  
            dropout=args.dropout_rate,
            batch_first=True
        )

        self.gate_generator = nn.Sequential(
            nn.Linear(args.hidden_units, args.hidden_units),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(args.hidden_units, args.num_experts),
            nn.Softmax(dim=-1)
        )

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(args.hidden_units, args.hidden_units),
                nn.ReLU(),
                nn.LayerNorm(args.hidden_units),
                nn.Dropout(args.dropout_rate),
                nn.Linear(args.hidden_units, args.hidden_units),
                nn.Sigmoid()
            ) for _ in range(self.num_experts)
        ])

        self.residual = nn.Linear(args.hidden_units, args.hidden_units)

    def forward(self, time_emb, domain_embs, ts_seqs):
        # 获取日期嵌入
        date_embs = self.date_embedding(ts_seqs)
        
        fused_feat, _ = self.cross_attn_fusion(
            query=domain_embs,
            key=date_embs,
            value=date_embs,
            need_weights=False
        )
        
        gate_weights = self.gate_generator(fused_feat)
        
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(time_emb)
            expert_outputs.append(expert_out)

        expert_stack = torch.stack(expert_outputs, dim=3)
        moe_output = torch.sum(expert_stack * gate_weights.unsqueeze(2), dim=-1)

        return self.residual(time_emb) + moe_output
        
class RelativeTimeEmbedding(torch.nn.Module):
    def __init__(self, max_interval_pos, embed_dim):
        super().__init__()
        self.embed = torch.nn.Embedding(max_interval_pos, embed_dim)
        self.max_interval_pos = max_interval_pos

    def forward(self, time_intervals):
        time_intervals = time_intervals.clone()
        time_intervals[time_intervals == 0] = 1e-9  # 避免log(0)

        log_intervals = torch.log(time_intervals.float())
        scaled_log = 100 * log_intervals  
        pos_indices = torch.floor(scaled_log).long()
        pos_indices = torch.clamp(pos_indices, max=self.max_interval_pos - 1)
       
        batch_size, seq_len = time_intervals.size()
        embeddings = self.embed(pos_indices.view(-1)).view(batch_size, seq_len, -1)
        return embeddings


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) 
        outputs = outputs + inputs
        return outputs

class ExcelRec(torch.nn.Module):
    def __init__(self, opt, user_num, item_num, args):
        super().__init__()
        self.opt = opt
        self.user_num = user_num
        self.item_num = item_num
        self.args = args
        self.dev = args.device

        tensorA_PCA = torch.load('moive_pca64.pt').to(self.dev)
        tensorB_PCA = torch.load('book_pca64.pt').to(self.dev)

        tensorA = torch.load('movie_embeddings.pt').to(self.dev)
        tensorB = torch.load('book_embeddings.pt').to(self.dev)


        padding = torch.zeros(1, tensorB.shape[1], device=args.device)

        llm_item_emb = torch.cat((padding, tensorA, tensorB), dim=0)
        self.llm_item_emb = torch.nn.Embedding.from_pretrained(llm_item_emb, freeze=True)

        padding_pca = torch.zeros(1, tensorA_PCA.shape[1], device=args.device)
        llm_emb_A = torch.cat([
            padding_pca,
            tensorA_PCA,
            torch.zeros(opt["itemnum"] - opt["source_item_num"], args.hidden_units, device=args.device)
        ])
        self.llm_item_emb_A = torch.nn.Embedding.from_pretrained(llm_emb_A, freeze=False)

        llm_emb_B = torch.cat([
            padding_pca,
            torch.zeros(opt["source_item_num"], args.hidden_units, device=args.device),
            tensorB_PCA
        ])
        self.llm_item_emb_B = torch.nn.Embedding.from_pretrained(llm_emb_B, freeze=False)

        self.pos_embA = torch.nn.Embedding(args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.pos_embB = torch.nn.Embedding(args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.cross_posemb = torch.nn.Embedding(args.maxlen + 1, args.hidden_units, padding_idx=0)

        self.adapter = nn.Sequential(
            nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1]/2)),
            nn.ReLU(),
            nn.Linear(int(llm_item_emb.shape[1]/2), args.hidden_units)
        )

        self.item_index = torch.arange(0, self.opt["itemnum"], 1)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.item_index = self.item_index.to(self.dev)

        self.loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')

        
        self.layer_norm = nn.LayerNorm(args.hidden_units)

        self.time_embeddingA = RelativeTimeEmbedding(
            max_interval_pos=1000,
            embed_dim=args.hidden_units
        )
        self.time_embeddingB = RelativeTimeEmbedding(
            max_interval_pos=1000,
            embed_dim=args.hidden_units
        )
        self.time_embeddingAB = RelativeTimeEmbedding(
            max_interval_pos=1000,
            embed_dim=args.hidden_units
        )

        self.domain_embedding = torch.nn.Embedding(
            num_embeddings=3,
            embedding_dim=args.hidden_units,
            padding_idx=0
        )

        self.moe_time = SimplifiedMoETimeEncoding(args)

        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.delta1 = nn.Parameter(torch.tensor(0.1))
        self.delta2 = nn.Parameter(torch.tensor(0.1))

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            self.pos_sigmoid = torch.nn.Sigmoid()
            self.neg_sigmoid = torch.nn.Sigmoid()

    def id_log2feats(self, seqs, log_seqs, ts_seq, domain, domain_seqs=None):  
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        poss *= (log_seqs != 0)
                
        rel_intervals = self.compute_relative_intervals(ts_seq)

        if domain == 'AB':
            seqs = seqs + self.cross_posemb(torch.LongTensor(poss).to(self.dev))
            time_emb = self.time_embeddingAB(rel_intervals)
            time_emb = self.moe_time(time_emb, domain_seqs, ts_seq)
            seqs = seqs + time_emb
        elif domain == 'A':
            seqs = seqs + self.pos_embA(torch.LongTensor(poss).to(self.dev))
            time_emb = self.time_embeddingA(rel_intervals)
            seqs = seqs + time_emb
        else:
            seqs = seqs + self.pos_embB(torch.LongTensor(poss).to(self.dev))
            time_emb = self.time_embeddingB(rel_intervals)
            seqs = seqs + time_emb

        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                                      attn_mask=attention_mask)
            
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)
        return log_feats

    def compute_relative_intervals(self, ts_seq):
        """计算相对时间间隔"""
        device = ts_seq.device
        arr = ts_seq.clone()
        mask = (arr != 0).float()

        non_zero = (arr != 0).int()
        start_indices = torch.argmax(non_zero, dim=1)

        shifted = torch.zeros_like(arr)
        shifted[:, 1:] = arr[:, :-1]

        cols = torch.arange(arr.size(1), device=device)
        diff_mask = (cols > start_indices.unsqueeze(1))

        res = torch.where(diff_mask, arr - shifted, torch.zeros_like(arr))
        return res

    def safe_pow(self, x, exponent, eps=1e-5):
        x = torch.clamp(x, min=eps, max=1 - eps)  
        return torch.pow(x, exponent)

    def forward(self, user_id, seq, pos, neg, domain_seq, ts_seq, target_domain, seqA, posA, negA, domain_seqA, ts_seqA, seqB, posB, negB, domain_seqB, ts_seqB, cross_act, src_act, tgt_act):
        seqA_tensor = torch.LongTensor(seqA).to(self.dev)
        seqB_tensor = torch.LongTensor(seqB).to(self.dev)
        seq_tensor = torch.LongTensor(seq).to(self.dev)
        cross_act_tensor = torch.FloatTensor(cross_act).to(self.dev)
        src_act_tensor = torch.FloatTensor(src_act).to(self.dev)
        tgt_act_tensor = torch.FloatTensor(tgt_act).to(self.dev)

        ts_seqA_tensor = torch.LongTensor(ts_seqA).to(self.dev)
        ts_seqB_tensor = torch.LongTensor(ts_seqB).to(self.dev)
        ts_seq_tensor = torch.LongTensor(ts_seq).to(self.dev)

        domain_seq_tensor = torch.LongTensor(domain_seq).to(self.dev)
        domain_ids = torch.zeros_like(domain_seq_tensor)  # 默认padding
        domain_ids[domain_seq_tensor == -1] = 1  # 源域
        domain_ids[domain_seq_tensor == 1] = 2  # 目标域
        domain_embs = self.domain_embedding(domain_ids)


        seqA_emb = self.llm_item_emb_A(seqA_tensor)
        seqA_emb *= self.llm_item_emb_A.embedding_dim ** 0.5

       
        feats_A = self.id_log2feats(seqA_emb, seqA, ts_seqA_tensor, "A", domain_seqs=None)
        
        domain_modA = self.safe_pow(src_act_tensor, self.delta1).unsqueeze(-1)
        
        feats_A *= domain_modA
        featsA_last = feats_A[:, -1, :]
        posA_tensor = torch.LongTensor(posA).to(self.dev)
        posA_emb = self.llm_item_emb_A(posA_tensor)
        negA_tensor = torch.LongTensor(negA).to(self.dev)
        negA_emb = self.llm_item_emb_A(negA_tensor)

        seqB_emb = self.llm_item_emb_B(seqB_tensor)
        seqB_emb *= self.llm_item_emb_B.embedding_dim ** 0.5
        feats_B = self.id_log2feats(seqB_emb, seqB, ts_seqB_tensor, "B", domain_seqs=None)
       
        domain_modB = self.safe_pow(tgt_act_tensor, self.delta2).unsqueeze(-1)
        
        feats_B *= domain_modB
       
        posB_tensor = torch.LongTensor(posB).to(self.dev)
        posB_emb = self.llm_item_emb_B(posB_tensor)
        negB_tensor = torch.LongTensor(negB).to(self.dev)
        negB_emb = self.llm_item_emb_B(negB_tensor)

        seq_emb = self.llm_item_emb(seq_tensor)
        seq_emb = self.adapter(seq_emb)
        seq_emb *= seq_emb.size(-1) ** 0.5

        feats = self.id_log2feats(seq_emb, seq, ts_seq_tensor, "AB", domain_embs)
        cross_mod = self.safe_pow(cross_act_tensor, self.gamma).unsqueeze(-1)

        feats *= cross_mod

        pos_tensor = torch.LongTensor(pos).to(self.dev)
        pos_emb = self.llm_item_emb(pos_tensor)
        pos_emb = self.adapter(pos_emb)
        neg_tensor = torch.LongTensor(neg).to(self.dev)
        neg_emb = self.llm_item_emb(neg_tensor)
        neg_emb = self.adapter(neg_emb)

        pos_logits = (feats * pos_emb).sum(dim=-1)
        neg_logits = (feats * neg_emb).sum(dim=-1)
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.dev), torch.zeros(neg_logits.shape,
                                                                                            device=self.dev)
        indices = (pos != 0)  
        pos_loss, neg_loss = self.loss_func(pos_logits[indices], pos_labels[indices]), self.loss_func(
            neg_logits[indices], neg_labels[indices])
        loss = pos_loss + neg_loss

        pos_logitsA = (feats_A * posA_emb).sum(dim=-1)
        neg_logitsA = (feats_A * negA_emb).sum(dim=-1)

        domain_maskA = (domain_seqA == 1)
        valid_pos_maskA = (posA > 0)  
        fusion_maskA = valid_pos_maskA & domain_maskA  

        pos_logitsA[fusion_maskA] += pos_logits[fusion_maskA]
        neg_logitsA[fusion_maskA] += neg_logits[fusion_maskA]
        pos_labelsA, neg_labelsA = torch.ones(pos_logitsA.shape, device=self.dev), torch.zeros(neg_logitsA.shape,
                                                                                               device=self.dev)
        indicesA = (posA != 0)
        pos_lossA, neg_lossA = self.loss_func(pos_logitsA[indicesA], pos_labelsA[indicesA]), self.loss_func(
            neg_logitsA[indicesA], neg_labelsA[indicesA])
        lossA = pos_lossA + neg_lossA


        pos_logitsB = (feats_B * posB_emb).sum(dim=-1)
        neg_logitsB = (feats_B * negB_emb).sum(dim=-1)

        domain_maskB = (domain_seqB == 1)
        valid_pos_maskB = (posB > 0)  
        fusion_maskB = valid_pos_maskB & domain_maskB  

        pos_logitsB[fusion_maskB] += pos_logits[fusion_maskB]
        neg_logitsB[fusion_maskB] += neg_logits[fusion_maskB]
        
        pos_labelsB, neg_labelsB = torch.ones(pos_logitsB.shape, device=self.dev), torch.zeros(neg_logitsB.shape,
                                                                                               device=self.dev)
        indicesB = (posB != 0)  
        pos_lossB, neg_lossB = self.loss_func(pos_logitsB[indicesB], pos_labelsB[indicesB]), self.loss_func(
            neg_logitsB[indicesB], neg_labelsB[indicesB])
        lossB = pos_lossB + neg_lossB

        loss = lossA.mean() + lossB.mean()

        return loss

    def predict(self, uid, seq, domain_seq, time_seq, item_indice, item_indicesA, item_indicesB,  seqA, time_seqA, seqB, time_seqB, target_domain):
        seqA_tensor = torch.LongTensor(seqA).to(self.dev)
        seqB_tensor = torch.LongTensor(seqB).to(self.dev)
        seq_tensor = torch.LongTensor(seq).to(self.dev)

        ts_seqA_tensor = torch.LongTensor(time_seqA).to(self.dev)
        ts_seqB_tensor = torch.LongTensor(time_seqB).to(self.dev)
        ts_seq_tensor = torch.LongTensor(time_seq).to(self.dev)

        domain_seq_tensor = torch.LongTensor(domain_seq).to(self.dev)
        domain_ids = torch.zeros_like(domain_seq_tensor)  # 默认padding
        domain_ids[domain_seq_tensor == -1] = 1  # 源域
        domain_ids[domain_seq_tensor == 1] = 2  # 目标域
        domain_embs = self.domain_embedding(domain_ids)

        seq_emb = self.llm_item_emb(seq_tensor)
        seq_emb = self.adapter(seq_emb)
        seq_emb = seq_emb * (seq_emb.size(-1) ** 0.5)
       
        feats = self.id_log2feats(seq_emb, seq, ts_seq_tensor, 'AB', domain_embs)
        feats_last = feats[:, -1, :]
        item_embs = self.llm_item_emb(torch.LongTensor(item_indice).to(self.dev))
        item_embs = self.adapter(item_embs)
        logits = (item_embs * feats_last.unsqueeze(1)).sum(-1)

        seqA_emb = self.llm_item_emb_A(seqA_tensor)
        seqA_emb *= self.llm_item_emb_A.embedding_dim ** 0.5
        feats_A = self.id_log2feats(seqA_emb, seqA, ts_seqA_tensor, 'A')
        featsA_last = feats_A[:, -1, :]

        seqB_emb = self.llm_item_emb_B(seqB_tensor)
        seqB_emb *= self.llm_item_emb_B.embedding_dim ** 0.5
        feats_B = self.id_log2feats(seqB_emb, seqB, ts_seqB_tensor, 'B')
        featsB_last = feats_B[:, -1, :]

        item_embs_A = self.llm_item_emb_A(torch.LongTensor(item_indicesA).to(self.dev))
        
        logits_A = (item_embs_A * featsA_last.unsqueeze(1)).sum(-1)

        item_embs_B = self.llm_item_emb_B(torch.LongTensor(item_indicesB).to(self.dev))
        
        logits_B = (item_embs_B * featsB_last.unsqueeze(1)).sum(-1)

        target_domain = torch.LongTensor(target_domain).to(self.dev)

        logits[target_domain == -1] += logits_A[target_domain == -1]
        logits[target_domain == 1] += logits_B[target_domain == 1]

        return logits
