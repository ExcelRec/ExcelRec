import copy
import os
import time
import torch
import argparse
import numpy as np
from ExcelRec import ExcelRec
from utils import *



def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='movie_book')
parser.add_argument('--train_dir', default='default')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.005, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=64, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=20, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--num_experts', default=4, type=int)
parser.add_argument('--dropout_rate', default=0.3, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--leakey', type=float, default=0.1)
parser.add_argument('--output_dir', default='excelRec_moviebook', type=str)
parser.add_argument("--early_stop", default=5, type=int)
args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

result_file = os.path.join(args.output_dir, 'test_results.txt')
with open(result_file, 'w') as f:
    f.write("Epoch\tOverall_NDCG\tOverall_HR\tDomainA_NDCG\tDomainA_HR\tDomainB_NDCG\tDomainB_HR\n")

opt = vars(args)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    set_seed(42)
    early_stop_counter = 0
    opt["source_item_num"] = 37020
    opt["target_item_num"] = 73722
   
    opt["itemnum"] = opt["source_item_num"] + opt["target_item_num"] + 1

    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, usernum, itemnum, last_indices] = dataset

    trainLoader = WarpSampler(user_train, last_indices, usernum, itemnum, opt, batch_size=args.batch_size,
                              maxlen=args.maxlen, n_workers=1)

    validLoader = WarpSampler(user_valid, last_indices, usernum, itemnum, opt, batch_size=args.batch_size,
                              maxlen=args.maxlen, n_workers=1, eval_model=True)

    testLoader = WarpSampler(user_test, last_indices, usernum, itemnum, opt, batch_size=args.batch_size,
                              maxlen=args.maxlen, n_workers=1, eval_model=True)

    num_batch = (len(user_train) - 1) // args.batch_size + 1
    eval_batch = (len(eval_dataset) - 1) // args.batch_size + 1
    test_batch = (len(test_dataset) - 1) // args.batch_size + 1
    cc = 0.0
    model = ExcelRec(opt, usernum, itemnum, args).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass  
    model.train()

    epoch_start_idx = 1
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr , weight_decay=1e-5)

    best_ndcg = 0.0
    best_model_state = None

    topk = 10

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for step in range(num_batch):

            (u, seq, pos, neg, domain_seq, ts_seq, target_domain, seqA, posA, negA, domain_seqA, ts_seqA,
             seqB, posB, negB, domain_seqB, ts_seqB, _, _, _, cross_act, src_act, tgt_act) = trainLoader.next_batch()

            (u, seq, pos, neg, domain_seq, ts_seq, seqA, posA, negA, domain_seqA, ts_seqA, seqB, posB, negB, domain_seqB, ts_seqB, cross_act, src_act, tgt_act) = np.array(u), np.array(seq), np.array(pos), np.array(neg), np.array(
                domain_seq), np.array(ts_seq), np.array(seqA), np.array(posA), np.array(negA), np.array(domain_seqA), np.array(ts_seqA),np.array(seqB), np.array(posB), np.array(negB), np.array(domain_seqB), np.array(ts_seqB), np.array(cross_act), np.array(src_act), np.array(tgt_act)

            loss = model(u, seq, pos, neg, domain_seq, ts_seq, target_domain, seqA, posA, negA, domain_seqA, ts_seqA, seqB, posB, negB, domain_seqB, ts_seqB, cross_act, src_act, tgt_act)

            adam_optimizer.zero_grad()

            loss.backward()
            adam_optimizer.step()
            epoch_loss += loss.item()

        if epoch % 5 == 0:
            model.eval()
            print('Evaluating', end='')
            valid_results = evaluate(eval_batch, model, validLoader, args, epoch, topk)
            print(f'epoch:{epoch}')
            print(f'[Valid Overall] NDCG@10: {valid_results["NDCG@10"]:.4f}, HR@10: {valid_results["HT@10"]:.4f}')
            print(f'[Valid DomainA] NDCG@10: {valid_results["NDCG@10_A"]:.4f}, HR@10: {valid_results["HT@10_A"]:.4f}')
            print(f'[Valid DomainB] NDCG@10: {valid_results["NDCG@10_B"]:.4f}, HR@10: {valid_results["HT@10_B"]:.4f}')
            
            current_ndcg = valid_results["NDCG@10"]
            if current_ndcg > best_ndcg:
                best_ndcg = current_ndcg
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save({'state_dict': best_model_state},
                           os.path.join(args.output_dir, 'pytorch_model.bin'))
                early_stop_counter = 0  # 重置计数器
            else:
                early_stop_counter += 1
                if early_stop_counter >= args.early_stop:
                    print(f'Early stopping at epoch {epoch}, best NDCG@10: {best_ndcg:.4f} at epoch {best_epoch}')
                    break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()

    epoch = 0
    test_results  = evaluate(test_batch, model, testLoader, args, epoch, topk, test=True)

    print("\n=========== Final Test Results ===========")
    print(f'[Test Overall] NDCG@10: {test_results["NDCG@10"]:.4f}, HT@10: {test_results["HT@10"]:.4f}')
    print(f'[Test DomainA] NDCG@10: {test_results["NOCG@10_A"]:.4f}, HT@10_A: {test_results["HT@10_A"]:.4f}')
    print(f'[Test DomainB] NDCG@10: {test_results["NOCG@10_B"]:.4f}, HT@10_B: {test_results["HT@10_B"]:.4f}')

    # 写入结果文件
    with open(result_file, 'a') as f:
        f.write(f"Final\t{test_results['NDCG@10']:.4f}\t{test_results['HT@10']:.4f}\t")
        f.write(f"{test_results['NDCG@10_A']:.4f}\t{test_results['HT@10_A']:.4f}\t")
        f.write(f"{test_results['NDCG@10_B']:.4f}\t{test_results['HT@10_B']:.4f}\n")

