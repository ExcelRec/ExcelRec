import numpy as np
import torch
import torch.nn.functional as F

class RelativeTimeEmbedding(torch.nn.Module):
    def __init__(self, max_interval_pos, embed_dim):
        super().__init__()
        self.embed = torch.nn.Embedding(max_interval_pos, embed_dim)
        self.max_interval_pos = max_interval_pos

    def forward(self, time_intervals):
        # 处理时间间隔为0的情况
        time_intervals = time_intervals.clone()
        time_intervals[time_intervals == 0] = 1e-9  # 避免log(0)
        
        # 对数缩放 + 离散化
        log_intervals = torch.log(time_intervals.float())
        scaled_log = 100 * log_intervals  # μ=100（论文参数）
        pos_indices = torch.floor(scaled_log).long()
        pos_indices = torch.clamp(pos_indices, max=self.max_interval_pos - 1)
        # 获取嵌入向量
        batch_size, seq_len = time_intervals.size()
        embeddings = self.embed(pos_indices.view(-1)).view(batch_size, seq_len, -1)
        return embeddings

class MoEAdapter(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts):
        super(MoEAdapter, self).__init__()
        self.num_experts = num_experts
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, output_dim)
            ) for _ in range(num_experts)
        ])
        self.gating_network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, num_experts)
        )


    def forward(self, x):
        # Gating network decides which experts to use for each input
        gating_weights = F.softmax(self.gating_network(x), dim=-1)  # (batch_size, num_experts)

        # Compute outputs from all experts and weigh them based on gating_weights
        expert_outputs = torch.stack([expert(x) for expert in self.experts],
                                     dim=2)  # (batch_size, num_experts, hidden_dim)

        # Weighted sum of expert outputs
        weighted_output = torch.sum(expert_outputs * gating_weights.unsqueeze(-1), dim=2)  # (batch_size, hidden_dim)

        return weighted_output


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
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs = outputs + inputs
        return outputs


class SASRec(torch.nn.Module):
    def __init__(self, opt, user_num, item_num, args):
        super(SASRec, self).__init__()
        self.opt = opt
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.time_embedding = RelativeTimeEmbedding(
            max_interval_pos=1000,
            embed_dim=args.hidden_units
        )
        tensor1 = torch.load('movie_embeddings.pt')
        tensor2 = torch.load('book_embeddings.pt')
        padding = torch.zeros(1, tensor2.shape[1], device=args.device)
        # 使用 torch.cat 在第0维上合并这两个张量
        llm_item_emb = torch.cat((padding, tensor1, tensor2), dim=0)
        self.text_emb = llm_item_emb
        self.llm_item_emb = torch.nn.Embedding.from_pretrained(llm_item_emb, freeze=True)
        self.moe_adapter_cross_seqs = MoEAdapter(input_dim=llm_item_emb.shape[1], hidden_dim=int(llm_item_emb.shape[1] / 2), output_dim=args.hidden_units, num_experts=2)
        
        self.item_emb = torch.nn.Embedding(self.opt["itemnum"], args.hidden_units, padding_idx=0)
        self.item_emb_X = torch.nn.Embedding(self.opt["itemnum"], args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        self.cross_posemb = torch.nn.Embedding(args.mix_maxlen+1, args.hidden_units, padding_idx=0)
        self.item_index = torch.arange(0, self.opt["itemnum"], 1)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.item_index = self.item_index.to(self.dev)

        # 新增交叉注意力层
        self.cross_attn1 = torch.nn.MultiheadAttention(args.hidden_units, args.num_heads, dropout=args.dropout_rate)
        self.cross_attn2 = torch.nn.MultiheadAttention(args.hidden_units, args.num_heads, dropout=args.dropout_rate)
        
        # 新增层归一化
        self.cross_attn_layernorm1 = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.cross_attn_layernorm2 = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.temperature = 1.0

        #self.stability_proj = torch.nn.Linear(args.hidden_units, args.hidden_units)
        self.stability_proj = torch.nn.Sequential(
            torch.nn.Linear(args.hidden_units, args.hidden_units),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(args.hidden_units)
        )
        self.temporal_moe = TemporalMoE(hidden_dim=args.hidden_units, num_experts=args.num_experts)
        
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
        
    def id_log2feats(self, seqs, log_seqs): # TODO: fp64 and int64 as default in python, trim?
        
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        poss *= (log_seqs != 0)
        
        seqs = seqs + self.pos_emb(torch.LongTensor(poss).to(self.dev))
            
        seqs = self.emb_dropout(seqs)
        
        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                            attn_mask=attention_mask)
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)
        return log_feats
        
        
    def compute_stability(self, seq_emb):
        # seq_emb: (B, L, D)
        if seq_emb.size(1) == 0:
            return torch.zeros(seq_emb.size(0), self.hidden_dim).to(seq_emb.device)
        
        variance = torch.var(seq_emb, dim=1)  # (B, D)
        return self.stability_proj(variance)   # (B, hidden_dim)
    def compute_relative_intervals(self, arr):
        res = arr.copy()
        mask = arr != 0
        start_indices = np.argmax(mask, axis=1)
        
        shifted = np.zeros_like(arr)
        shifted[:, 1:] = arr[:, :-1]  # 正确获取前序元素
        
        # 构建差分掩码
        cols = np.arange(arr.shape[1])
        diff_mask = cols > start_indices[:, None]
        
        # 处理首个非零元素
        rows = np.arange(arr.shape[0])
        res[rows, start_indices] = 0
        
        # 执行向量化差值计算
        res = np.where(diff_mask, arr - shifted, res)
        return torch.from_numpy(res).to(self.dev)
        
    def forward(self, user_ids, cross_seqs, log_seqs, pos_seqs, neg_seqs, pos_pop, neg_pop, pos_pop_single, neg_pop_single, ts_croseq, ts_seq, args):
        ts_croseq_intervals = self.compute_relative_intervals(ts_croseq)
        croseq_embed = self.time_embedding(ts_croseq_intervals)
        time_embed_agg = croseq_embed.mean(dim=1)  # [B, embed_dim]

        ts_seq_intervals = self.compute_relative_intervals(ts_seq)
        seq_embed = self.time_embedding(ts_seq_intervals)
        seq_time_embed_agg = seq_embed.mean(dim=1)  # [B, embed_dim]
        
        log_seqs_tensor = torch.LongTensor(log_seqs).to(self.dev)
        seqs = self.item_emb_X(log_seqs_tensor)
        seqs *= self.item_emb_X.embedding_dim ** 0.5
        log_feats = self.id_log2feats(seqs, log_seqs) 
        log_feats_last = log_feats[:, -1, :]

        cross_seqs_tensor = torch.LongTensor(cross_seqs).to(self.dev)
        mul_seqs = self.item_emb(cross_seqs_tensor)
        mul_seqs *= self.item_emb.embedding_dim ** 0.5
        mul_id_feats = self.id_log2feats(mul_seqs, cross_seqs)
        mul_id_last = mul_id_feats[:, -1, :]
        
        mul_llm_seqs = self.llm_item_emb(cross_seqs_tensor)
        mul_llm_seqs = self.moe_adapter_cross_seqs(mul_llm_seqs)
        mul_llm_seqs *= self.llm_item_emb.embedding_dim ** 0.5
        stability_vec = self.compute_stability(mul_llm_seqs)  #[128,64]
        
        mul_llm_feats = self.id_log2feats(mul_llm_seqs, cross_seqs)
        mul_llm_l_feats = mul_llm_feats[:, -1, :]

        mul_llm = self.temporal_moe(mul_llm_l_feats, stability_vec, time_embed_agg, seq_time_embed_agg)
        
        # 第一个融合：mul_id_last作为Q，mul_llm作为K/V
        q1 = mul_id_last.unsqueeze(1)  # (batch_size, 1, hidden_units)
        # 添加时间步维度到mul_llm
        mul_llm_expanded = mul_llm.unsqueeze(0)  # (1, batch_size, hidden_units)
        attn_output1, _ = self.cross_attn1(
            query=q1.transpose(0, 1),  # (1, batch_size, hidden_units)
            key=mul_llm_expanded,      # (1, batch_size, hidden_units)
            value=mul_llm_expanded     # (1, batch_size, hidden_units)
        )
        attn_output1 = attn_output1.transpose(0, 1).squeeze(1)  # (batch_size, hidden_units)
        attn_output1 = self.cross_attn_layernorm1(attn_output1 + mul_id_last)
        
        # 第二个融合：log_feats_last作为Q，mul_llm作为K/V
        q2 = log_feats_last.unsqueeze(1)  # (batch_size, 1, hidden_units)
        # 添加时间步维度到mul_llm
        mul_llm_expanded = mul_llm.unsqueeze(0)  # (1, batch_size, hidden_units)
        attn_output2, _ = self.cross_attn2(
            query=q2.transpose(0, 1),  # (1, batch_size, hidden_units)
            key=mul_llm_expanded,      # (1, batch_size, hidden_units)
            value=mul_llm_expanded     # (1, batch_size, hidden_units)
        )
        attn_output2 = attn_output2.transpose(0, 1).squeeze(1)  # (batch_size, hidden_units)
        attn_output2 = self.cross_attn_layernorm2(attn_output2 + log_feats_last


        def info_nce_loss(features_s, features_t, temperature=0.5):
            features = torch.cat([features_s, features_t], dim=0)
            similarity = torch.matmul(features, features.T) / temperature
            labels = torch.cat([torch.arange(len(features_s)).to(device=self.dev), len(features_s)+torch.arange(len(features_t)).to(device=self.dev)])
            loss = F.cross_entropy(similarity, labels)
            return loss
        loss_s = info_nce_loss(attn_output1, log_feats_last.detach())
        loss_c = info_nce_loss(attn_output2, mul_id_last.detach())
        
        fuse_feats_last = log_feats_last + mul_id_last + attn_output1 + attn_output2
        
        pos_pop_tensor = torch.tensor(pos_pop).to(self.dev)
        pos_single_tensor = torch.tensor(pos_pop_single).to(self.dev)
        feats = fuse_feats_last * pos_pop_tensor.unsqueeze(-1) * pos_single_tensor.unsqueeze(-1)
        
        pos_embs = self.item_emb_X(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb_X(torch.LongTensor(neg_seqs).to(self.dev))
        pos_logits = torch.sum(pos_embs * feats.unsqueeze(1), -1)
        neg_logits = torch.sum(neg_embs * feats.unsqueeze(1), -1) # [batch,seq_len]

        pos_logits = self.pos_sigmoid(pos_logits)
        neg_logits = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits, loss_s, loss_c # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, cross_seqs, time_seqs, time_croseqs, item_indices, args): # for inference
        ts_croseq_intervals = self.compute_relative_intervals(time_croseqs)
        croseq_embed = self.time_embedding(ts_croseq_intervals)
        time_embed_agg = croseq_embed.mean(dim=1)  # [B, embed_dim]

        ts_seq_intervals = self.compute_relative_intervals(time_seqs)
        seq_embed = self.time_embedding(ts_seq_intervals)
        time_seq_embed_agg = seq_embed.mean(dim=1)  # [B, embed_dim]
        
        log_seqs_tensor = torch.LongTensor(log_seqs).to(self.dev)
        seqs = self.item_emb_X(log_seqs_tensor)
        seqs *= self.item_emb_X.embedding_dim ** 0.5
        log_feats = self.id_log2feats(seqs, log_seqs) # user_ids hasn't been used yet
        log_feats_last = log_feats[:, -1, :]

        cross_seqs_tensor = torch.LongTensor(cross_seqs).to(self.dev)
        mul_seqs = self.item_emb(cross_seqs_tensor)
        mul_seqs *= self.item_emb.embedding_dim ** 0.5
        mul_id_feats = self.id_log2feats(mul_seqs, cross_seqs)
        mul_id_last = mul_id_feats[:, -1, :]

        mul_llm_seqs = self.llm_item_emb(cross_seqs_tensor)
        mul_llm_seqs = self.moe_adapter_cross_seqs(mul_llm_seqs)
        mul_llm_seqs *= self.llm_item_emb.embedding_dim ** 0.5
        stability_vec = self.compute_stability(mul_llm_seqs)
        
        mul_llm_feats = self.id_log2feats(mul_llm_seqs, cross_seqs)
        mul_llm_l_feats = mul_llm_feats[:, -1, :]
        # mul_llm = self.temporal_moe(mul_llm_l_feats, stability_vec)
        mul_llm = self.temporal_moe(mul_llm_l_feats, stability_vec, time_embed_agg, time_seq_embed_agg)
        # 第一个融合：mul_id_last作为Q，mul_llm作为K/V
        q1 = mul_id_last.unsqueeze(1)  # (batch_size, 1, hidden_units)
        # 添加时间步维度到mul_llm
        mul_llm_expanded = mul_llm.unsqueeze(0)  # (1, batch_size, hidden_units)
        attn_output1, _ = self.cross_attn1(
            query=q1.transpose(0, 1),  # (1, batch_size, hidden_units)
            key=mul_llm_expanded,      # (1, batch_size, hidden_units)
            value=mul_llm_expanded     # (1, batch_size, hidden_units)
        )
        attn_output1 = attn_output1.transpose(0, 1).squeeze(1)  # (batch_size, hidden_units)
        attn_output1 = self.cross_attn_layernorm1(attn_output1 + mul_id_last)
        
        # 第二个融合：log_feats_last作为Q，mul_llm作为K/V
        q2 = log_feats_last.unsqueeze(1)  # (batch_size, 1, hidden_units)
        # 添加时间步维度到mul_llm
        mul_llm_expanded = mul_llm.unsqueeze(0)  # (1, batch_size, hidden_units)
        attn_output2, _ = self.cross_attn2(
            query=q2.transpose(0, 1),  # (1, batch_size, hidden_units)
            key=mul_llm_expanded,      # (1, batch_size, hidden_units)
            value=mul_llm_expanded     # (1, batch_size, hidden_units)
        )
        attn_output2 = attn_output2.transpose(0, 1).squeeze(1)  # (batch_size, hidden_units)
        attn_output2 = self.cross_attn_layernorm2(attn_output2 + log_feats_last)
       
        final_feat = log_feats_last + mul_id_last + attn_output1 + attn_output2
        item_embs = self.item_emb_X(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        preds = self.pos_sigmoid(logits) # rank same item list for different users

        return preds # logits # (U, I)
