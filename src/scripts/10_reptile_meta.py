
import os
import json
import time
import math
import hashlib
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Config & Setup ---

# Quick (unsafe) workaround to avoid the libiomp5md.dll crash.
# Use this only to continue working in the notebook quickly.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
print("Set KMP_DUPLICATE_LIB_OK=TRUE")

# Paths
# Script is in src/scripts, so root is ../..
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / 'data' / 'processed'
META_DIR = DATA_DIR / 'meta_vocab'
META_DIR.mkdir(parents=True, exist_ok=True)
TASKS_OUT = META_DIR / 'tasks_reduced_hashed_top200k.pt'
TASKS_CSV = META_DIR / 'tasks_summary_hashed_top200k.csv'

# Hash vocab size
K = 200_000
PAD_IDX = 0
HASH_MOD = K

# Task builder params
MIN_PAIRS_PER_TASK = 50   # keep tasks with >= this many pairs
MAX_TASKS = 300           # reduce number of tasks for quicker experiments (tunable)
MAX_PREFIX_LEN = 20

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE, 'K=', K)


# --- Helper Functions ---

def token_to_hash_id(token: str, K=HASH_MOD):
    # deterministic MD5 hash mapping, returns 1..K (0 reserved for PAD)
    if token is None or token == '':
        return PAD_IDX
    # normalize token to str
    s = str(token)
    h = hashlib.md5(s.encode('utf-8')).hexdigest()
    idx = (int(h, 16) % K) + 1
    return idx

class SASRecSmall(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, max_len=20, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.item_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=2048, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
    def forward(self, x):
        B, L = x.size()
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        seq = self.item_emb(x) + self.pos_emb(pos_ids)
        seq = self.encoder(seq)
        last = seq[:, -1, :]
        logits = self.out(last)
        return logits, last

def process_file_to_task(path: Path, max_prefix_len=MAX_PREFIX_LEN):
    df = pd.read_parquet(path)
    # expect columns 'prefix' and 'target' (prefix as space-separated tokens or list)
    # tolerate both formats
    P_list = []
    T_list = []
    for _, r in df.iterrows():
        pref = r.get('prefix', '')
        # if prefix already stored as list-like, handle; else treat as string
        if isinstance(pref, (list, tuple)):
            tokens = [str(x) for x in pref if x is not None and x != '']
        else:
            # assume space-separated token ids or ASINs; handle empty string
            tokens = [t for t in str(pref).split() if t != '']
        # map tokens via hashing
        ids = [token_to_hash_id(t) for t in tokens]
        if len(ids) > max_prefix_len:
            ids = ids[-max_prefix_len:]
        # remove leading PADs if they were created by empty tokens; but keep lengths for nonzeros
        if len(ids) == 0:
            padded = [PAD_IDX] * max_prefix_len
            nonzero_len = 0
        else:
            padded = [PAD_IDX] * (max_prefix_len - len(ids)) + ids
            nonzero_len = sum(1 for x in ids if x != PAD_IDX)
        # if target missing, skip
        target = r.get('target', None)
        if target is None:
            continue
        # map target to hashed id (string target will be hashed)
        tid = token_to_hash_id(target)
        P_list.append(padded)
        T_list.append(int(tid))
    if len(P_list) < MIN_PAIRS_PER_TASK:
        return None
    P_t = torch.LongTensor(P_list)
    T_t = torch.LongTensor(T_list)
    # compute example nonzero length stats quickly
    nonzero_example_len = (P_t != PAD_IDX).sum(dim=1).clamp(max=MAX_PREFIX_LEN)
    return {'name': path.name, 'P': P_t, 'T': T_t, 'n_pairs': P_t.size(0),
            'median_nonzero_len': int(nonzero_example_len.median().item()),
            'frac_nonzero_gt0': float((nonzero_example_len>0).float().mean().item())}

# helper to get minibatches from a task
def task_sampler_from_task(tdict, support_batch=64):
    P = tdict['P']
    T = tdict['T']
    N = P.size(0)
    idxs = np.arange(N)
    def gen(batch_size=support_batch):
        np.random.shuffle(idxs)
        for i in range(0, N, batch_size):
            sel = idxs[i:i+batch_size]
            yield P[sel], T[sel]
    return gen

# utility: copy model parameters (state_dict)
def clone_state_dict(state):
    return {k: v.clone().detach() for k,v in state.items()}


def main():
    # --- Build Tasks ---
    
    # Found prefix-target files (update pattern if different)
    prefix_glob = list((DATA_DIR).glob("*prefix*target*.parquet"))  # adjust if your naming differs
    print("Found candidate prefix-target files:", len(prefix_glob))

    tasks = []    # each task: {'name': name, 'P': LongTensor (N, L), 'T': LongTensor (N)}
    
    # iterate
    count = 0
    for p in tqdm(prefix_glob, desc="Scanning files"):
        t = process_file_to_task(p)
        if t is None:
            continue
        tasks.append(t)
        count += 1
        if MAX_TASKS and count >= MAX_TASKS:
            break

    print("Built tasks:", len(tasks))
    # Save tasks in compact format: store P and T as tensors (could be large)
    torch.save(tasks, TASKS_OUT)
    # Also write CSV summary
    rows = [{'name': t['name'], 'pairs': t['n_pairs'],
             'median_nonzero_len': t['median_nonzero_len'],
             'frac_nonzero_gt0': t['frac_nonzero_gt0']} for t in tasks]
    pd.DataFrame(rows).to_csv(TASKS_CSV, index=False)
    print("Saved tasks:", TASKS_OUT, "summary:", TASKS_CSV)

    # --- Sanity Checks ---
    tasks = torch.load(TASKS_OUT)
    print("Total tasks loaded:", len(tasks))
    # show top 10 by pairs
    sorted_tasks = sorted(tasks, key=lambda x: x['n_pairs'], reverse=True)
    for t in sorted_tasks[:10]:
        print(t['name'], "pairs=", t['n_pairs'], "median_nonzero_len=", t['median_nonzero_len'],
              "frac_nonzero_gt0=", f"{t['frac_nonzero_gt0']:.3f}")
    # distribution of frac_nonzero_gt0
    fracs = [t['frac_nonzero_gt0'] for t in tasks]
    if fracs:
        print("frac_nonzero_gt0 median:", np.median(fracs), "mean:", np.mean(fracs))

    # --- Meta Model Setup ---
    
    # Create meta-model with hashed vocab
    META_VOCAB = K + 1  # 0..K
    meta_model = SASRecSmall(vocab_size=META_VOCAB, embed_dim=64, max_len=MAX_PREFIX_LEN).to(DEVICE)
    print("Meta-model created. Vocab:", META_VOCAB)

    # --- Reptile Training Loop ---
    
    # hyperparams (tune)
    META_ITERS = 500            # number of meta-iterations
    TASK_BATCH = 4              # number of tasks sampled per meta-iteration
    INNER_STEPS = 5             # SGD steps per task (support)
    SUPPORT_BATCH = 64          # batch size for support updates
    INNER_LR = 1e-3
    META_STEP = 0.1             # step size to move meta weights toward adapted weights
    
    tasks = torch.load(TASKS_OUT)
    print("Starting Reptile meta-training (tasks:", len(tasks), ")")
    meta_state = meta_model.state_dict()

    for it in range(META_ITERS):
        sampled = np.random.choice(len(tasks), size=min(TASK_BATCH, len(tasks)), replace=False)
        meta_state_before = clone_state_dict(meta_state)
        adapted_states = []
        for tid in sampled:
            tinfo = tasks[tid]
            # build a small copy model
            local_model = SASRecSmall(vocab_size=META_VOCAB, embed_dim=64, max_len=MAX_PREFIX_LEN).to(DEVICE)
            local_model.load_state_dict(meta_state)  # start from meta
            local_opt = torch.optim.AdamW(local_model.parameters(), lr=INNER_LR, weight_decay=1e-6)
            # inner-loop: iterate INNER_STEPS over support batches
            gen = task_sampler_from_task(tinfo, support_batch=SUPPORT_BATCH)()
            step = 0
            try:
                while step < INNER_STEPS:
                    Xb, yb = next(gen)
                    Xb = Xb.to(DEVICE)
                    yb = yb.to(DEVICE)
                    local_model.train()
                    _, final = local_model(Xb)
                    # sampled softmax loss
                    V = local_model.item_emb.weight.size(0)
                    pos_scores = (final * local_model.item_emb.weight[yb]).sum(dim=1)
                    neg_idx = torch.randint(0, V, (Xb.size(0), 32), device=DEVICE)
                    neg_w = local_model.item_emb.weight[neg_idx]
                    neg_scores = (neg_w * final.unsqueeze(1)).sum(dim=2)
                    logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
                    labels = torch.zeros(Xb.size(0), dtype=torch.long, device=DEVICE)
                    loss = F.cross_entropy(logits, labels)
                    local_opt.zero_grad(); loss.backward(); local_opt.step()
                    step += 1
            except StopIteration:
                pass
            adapted_states.append(clone_state_dict(local_model.state_dict()))
            # free local model
            del local_model, local_opt

        # meta-update: move meta_state toward average adapted_state
        avg_state = {}
        for k in meta_state:
            stacked = torch.stack([s[k].to('cpu') for s in adapted_states], dim=0)
            avg = torch.mean(stacked, dim=0)
            avg_state[k] = avg.to(meta_state[k].device)
        # apply reptile update: meta = meta + eps * (avg - meta)
        for k in meta_state:
            meta_state[k] = meta_state[k] + META_STEP * (avg_state[k].to(meta_state[k].device) - meta_state[k])

        # every N iterations optionally evaluate quick validation
        if (it+1) % 50 == 0 or it == 0:
            # compute a very cheap diagnostic: random task few-shot adapt and eval on its heldout pairs
            idx = np.random.randint(len(tasks))
            tdiag = tasks[idx]
            # split task into support/query
            N = tdiag['P'].size(0)
            qn = max(1, int(0.2 * N))
            perm = np.random.permutation(N)
            sup_idx = perm[:-qn]; qry_idx = perm[-qn:]
            # adapt from meta_state for a few steps
            tmp_model = SASRecSmall(vocab_size=META_VOCAB, embed_dim=64, max_len=MAX_PREFIX_LEN).to(DEVICE)
            tmp_model.load_state_dict(meta_state)
            tmp_opt = torch.optim.AdamW(tmp_model.parameters(), lr=INNER_LR)
            # support steps
            for s in range(5):
                sel = sup_idx[s::5][:SUPPORT_BATCH] if len(sup_idx)>0 else sup_idx
                if len(sel)==0: break
                Xb = tdiag['P'][sel].to(DEVICE)
                yb = tdiag['T'][sel].to(DEVICE)
                _, final = tmp_model(Xb)
                pos_scores = (final * tmp_model.item_emb.weight[yb]).sum(dim=1)
                neg_idx = torch.randint(0, tmp_model.item_emb.weight.size(0), (Xb.size(0), 32), device=DEVICE)
                neg_w = tmp_model.item_emb.weight[neg_idx]
                neg_scores = (neg_w * final.unsqueeze(1)).sum(dim=2)
                logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
                loss = F.cross_entropy(logits, torch.zeros(Xb.size(0), dtype=torch.long, device=DEVICE))
                tmp_opt.zero_grad(); loss.backward(); tmp_opt.step()
            # evaluate on query
            hits = 0; total = 0
            for qi in qry_idx:
                Xq = tdiag['P'][qi].unsqueeze(0).to(DEVICE)
                tq = int(tdiag['T'][qi].item())
                _, final = tmp_model(Xq)
                scores = torch.matmul(final, tmp_model.item_emb.weight.t())
                topk = scores.topk(20, dim=1).indices.squeeze(0).cpu().numpy()
                total += 1
                if tq in topk: hits += 1
            quick_recall = hits / total if total>0 else 0.0
            print(f"[Reptile] iter {it+1}/{META_ITERS} quick_recall@20={quick_recall:.4f}")

    # After meta loop - save meta_state as meta initialization
    meta_model.load_state_dict(meta_state)
    torch.save({'meta_state': meta_state}, META_DIR / 'reptile_meta_state_top200k.pt')
    print("Saved meta init to:", META_DIR / 'reptile_meta_state_top200k.pt')


    # --- Evaluatoin on MARS ---
    
    # load MARS shard (built earlier), map using same hashing
    MARS_SHARD_FILE = DATA_DIR / 'mars_shards' / 'mars_shard_full.pt'
    if not MARS_SHARD_FILE.exists():
        print("Likely missing MARS shard, skipping definitive MARS eval if not present.")
        # We try to proceed if we can rebuild it or if the file exists. 
        # The notebook had a "raise" here, but for a script we might want to be graceful or strict.
        # Strict matching original logic:
        # raise FileNotFoundError("Please ensure MARS shard exists (built in 07_transfer_to_mars).")
    
    if MARS_SHARD_FILE.exists():
        mp = torch.load(MARS_SHARD_FILE)
        # P_all = mp['prefix']   # NOTE: these were constructed earlier with original item2id mapping; we re-hash tokens here
        
        MARS_PAIRS = DATA_DIR / 'mars_prefix_target.parquet'
        if not MARS_PAIRS.exists():
             print("Please create mars_prefix_target.parquet first.")
        else:
            df_mars_pairs = pd.read_parquet(MARS_PAIRS)
            # build hashed MARS tensors
            P_list = []
            T_list = []
            for _, r in df_mars_pairs.iterrows():
                pref = r['prefix'] if isinstance(r['prefix'], str) else ''
                tokens = [t for t in str(pref).split() if t!='']
                ids = [token_to_hash_id(t) for t in tokens]
                if len(ids) > MAX_PREFIX_LEN: ids = ids[-MAX_PREFIX_LEN:]
                padded = [PAD_IDX]*(MAX_PREFIX_LEN - len(ids)) + ids
                P_list.append(padded)
                T_list.append(token_to_hash_id(r['target']))
            P_H = torch.LongTensor(P_list)
            T_H = torch.LongTensor(T_list)
            print("Built hashed MARS pairs:", P_H.size(0))

            # Split train/val/test
            n = P_H.size(0)
            test_n = max(1, int(0.1*n))
            val_n = max(1, int(0.1*n))
            train_n = n - val_n - test_n
            train_P, train_T = P_H[:train_n].to(DEVICE), T_H[:train_n].to(DEVICE)
            val_P, val_T = P_H[train_n:train_n+val_n].to(DEVICE), T_H[train_n:train_n+val_n].to(DEVICE)
            test_P, test_T = P_H[train_n+val_n:].to(DEVICE), T_H[train_n+val_n:].to(DEVICE)
            print("MARS splits: train", train_P.size(0), "val", val_P.size(0), "test", test_P.size(0))

            # Load meta init
            meta_ck = torch.load(META_DIR / 'reptile_meta_state_top200k.pt', map_location=DEVICE)
            meta_state = meta_ck['meta_state']
            
            # Few-shot fine-tune (support small K shots) — try different K_shots
            def adapt_and_eval(K_shots=50, adapt_steps=10, lr=1e-4):
                # sample K_shots from train
                idxs = np.random.choice(train_P.size(0), size=min(K_shots, train_P.size(0)), replace=False)
                Xs = train_P[idxs]
                ys = train_T[idxs]
                model = SASRecSmall(vocab_size=META_VOCAB, embed_dim=64, max_len=MAX_PREFIX_LEN).to(DEVICE)
                model.load_state_dict(meta_state)
                opt = torch.optim.AdamW(model.parameters(), lr=lr)
                for s in range(adapt_steps):
                    model.train()
                    _, final = model(Xs)
                    pos_scores = (final * model.item_emb.weight[ys]).sum(dim=1)
                    neg_idx = torch.randint(0, model.item_emb.weight.size(0), (Xs.size(0), 32), device=DEVICE)
                    neg_w = model.item_emb.weight[neg_idx]
                    neg_scores = (neg_w * final.unsqueeze(1)).sum(dim=2)
                    logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
                    loss = F.cross_entropy(logits, torch.zeros(Xs.size(0), dtype=torch.long, device=DEVICE))
                    opt.zero_grad(); loss.backward(); opt.step()
                # evaluate on test
                model.eval()
                hits = 0; rr_sum = 0.0; total = 0
                with torch.no_grad():
                    for i in range(test_P.size(0)):
                        Xq = test_P[i].unsqueeze(0)
                        target = int(test_T[i].item())
                        _, final = model(Xq)
                        scores = torch.matmul(final, model.item_emb.weight.t())
                        topk = scores.topk(20, dim=1).indices.squeeze(0).cpu().numpy()
                        total += 1
                        if target in topk:
                            hits += 1
                            rank = int((topk == target).nonzero()[0]) + 1
                            rr_sum += 1.0 / rank
                recall = hits/total if total>0 else 0.0
                mrr = rr_sum/total if total>0 else 0.0
                return recall, mrr

            for K_shots in [10, 50, 100, 200]:
                r, m = adapt_and_eval(K_shots=K_shots, adapt_steps=10, lr=1e-4)
                print(f"Few-shot K={K_shots} -> Recall@20={r:.4f}, MRR={m:.4f}")

    # --- Save Info ---
    torch.save({'meta_state': meta_state, 'K': K, 'pad': PAD_IDX}, META_DIR / 'meta_info_top200k.pt')
    print("Saved meta info.")

if __name__ == "__main__":
    main()
