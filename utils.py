import os
from collections import defaultdict
from multiprocessing import Process, Queue
import numpy as np
import torch
from tqdm import tqdm


def evaluate(num_batch, model, dataLoader, args, epoch = 0, topk = 5, test = False):
    all_pred_rank = []
    all_target_domain = []
    if test:
        print("***************Running Test********************")
        model_path = os.path.join(args.output_dir, 'pytorch_model.bin')
        checkpoint = torch.load(model_path, map_location=args.device)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(args.device)

    else:
        print("********************Evaluating Model*******************")
        print("********************Epoch:", epoch)

    with torch.no_grad():
        for _ in tqdm(range(num_batch)):
            batch_data = dataLoader.next_batch()
            (u, seq, pos, neg, domain_seq, ts_seq, target_domain, seqA, posA, negA, domain_seqA, ts_seqA,
             seqB, posB, negB, domain_seqB, ts_seqB, item_idx, item_idxA, item_idxB, _, _, _) = batch_data

            if isinstance(batch_data, tuple) and len(batch_data) >= 7:
                target_domain = batch_data[6]

            u = np.array(u)
            seq = np.array(seq)
            pos = np.array(pos)
            neg = np.array(neg)
            domain_seq = np.array(domain_seq)
            ts_seq = np.array(ts_seq)
            seqA = np.array(seqA)
            posA = np.array(posA)
            negA = np.array(negA)
            domain_seqA = np.array(domain_seqA)
            ts_seqA = np.array(ts_seqA)
            seqB = np.array(seqB)
            posB = np.array(posB)
            negB = np.array(negB)
            domain_seqB = np.array(domain_seqB)
            ts_seqB = np.array(ts_seqB)
            target_domain = np.array(target_domain)
            item_idx = np.array(item_idx)
            item_idxA = np.array(item_idxA)
            item_idxB = np.array(item_idxB)

            item_idx = torch.LongTensor(item_idx).to(args.device)

            item_idxA = torch.LongTensor(item_idxA).to(args.device)

            item_idxB = torch.LongTensor(item_idxB).to(args.device)

            item_idx_np = item_idx.cpu().numpy() if item_idx.is_cuda else item_idx.numpy()
            item_idxA_np = item_idxA.cpu().numpy() if item_idxA.is_cuda else item_idxA.numpy()
            item_idxB_np = item_idxB.cpu().numpy() if item_idxB.is_cuda else item_idxB.numpy()

            pre_logits = -model.predict(
                u, seq, ts_seq, domain_seq,
                item_idx_np, item_idxA_np, item_idxB_np,
                seqA, ts_seqA, seqB, ts_seqB, target_domain
            )
            per_pred_rank = pre_logits.argsort(dim=1).argsort(dim=1)[:, 0]
            all_pred_rank.append(per_pred_rank.cpu())

            target_domain_tensor = torch.LongTensor(target_domain).to(args.device)
            all_target_domain.append(target_domain_tensor.cpu())

        pred_rank = torch.cat(all_pred_rank).numpy()
        target_domain = torch.cat(all_target_domain).numpy()
        res_dict = metric_report(pred_rank, topk)

        pred_rank_A = pred_rank[target_domain == -1]
        pred_rank_B = pred_rank[target_domain == 1]

        res_dict_A = metric_domain_report(pred_rank_A, topk, domain="A")
        res_dict_B = metric_domain_report(pred_rank_B, topk, domain="B")

        res_dict = {**res_dict, **res_dict_A, **res_dict_B}
        return res_dict


def metric_report(pre_rank, topk):
    NDCG, HT = 0, 0
    for rank in pre_rank:
        if rank < topk:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return {
        'NDCG@{}'.format(topk): NDCG / len(pre_rank),
        'HT@{}'.format(topk): HT / len(pre_rank)
    }

def metric_domain_report(pred_rank_A, topk, domain):
    NDCG, HT = 0, 0
    for rank in pred_rank_A:
        if rank < topk:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return {
        'NDCG@{}_{}'.format(topk, domain): NDCG / len(pred_rank_A),
        'HT@{}_{}'.format(topk, domain): HT / len(pred_rank_A)
    }


