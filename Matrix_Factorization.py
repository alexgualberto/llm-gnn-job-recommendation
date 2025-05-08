import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, precision_score, average_precision_score
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit, ToUndirected, NormalizeFeatures
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.sampler import NegativeSampling
import scipy.stats as stats
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(name)s — %(levelname)s — %(message)s',
    handlers=[logging.FileHandler("mf_training.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration for matrix factorization training."""
    EDGE_INDEX_PATH       = 'interactions.csv'
    MODEL_SAVE_PATH       = 'mf_checkpoint.pth'
    EMBEDDING_DIM         = 256
    LEARNING_RATE         = 1e-3
    WEIGHT_DECAY          = 1e-5
    BATCH_SIZE            = 256
    TEST_BATCH_SIZE       = BATCH_SIZE * 3
    NUM_EPOCHS            = 100
    PATIENCE              = 5
    VAL_RATIO             = 0.1
    TEST_RATIO            = 0.1
    DISJOINT_TRAIN_RATIO  = 0.8
    NEG_SAMPLING_RATIO    = 1
    NUM_NEIGHBORS         = [40, 20]
    TOP_K                 = [5, 10, 20, 50, 100]
    DEVICE                = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MFModel(nn.Module):
    """Matrix factorization model with embeddings and biases."""
    def __init__(self, num_users, num_items, emb_dim, dropout=0.2):
        super().__init__()
        self.user_emb  = nn.Embedding(num_users, emb_dim)
        self.item_emb  = nn.Embedding(num_items, emb_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.dropout   = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, edge_index):
        user_idx, item_idx = edge_index
        u = self.dropout(self.user_emb(user_idx))
        i = self.dropout(self.item_emb(item_idx))
        b_u = self.user_bias(user_idx).squeeze()
        b_i = self.item_bias(item_idx).squeeze()
        return (u * i).sum(dim=1) + b_u + b_i

def check_data_files_exist(cfg: Config):
    """Ensure the interaction file exists."""
    if not Path(cfg.EDGE_INDEX_PATH).exists():
        raise FileNotFoundError(f"Missing file: {cfg.EDGE_INDEX_PATH}")

def load_data(cfg: Config):
    """Load interactions and return counts and edge index."""
    df = pd.read_csv(cfg.EDGE_INDEX_PATH)
    num_users = int(df.user_id.max()) + 1
    num_items = int(df.job_id.max()) + 1
    edge_index = torch.stack([
        torch.tensor(df.user_id.values, dtype=torch.long),
        torch.tensor(df.job_id.values, dtype=torch.long)
    ], dim=0)
    logger.info(f"Users: {num_users}, Items: {num_items}, Interactions: {edge_index.size(1)}")
    return num_users, num_items, edge_index

def build_hetero(num_users, num_items, edge_index: torch.Tensor) -> HeteroData:
    """Create HeteroData for link prediction."""
    data = HeteroData()
    data['user'].x = torch.ones((num_users, 1))
    data['job'].x  = torch.ones((num_items, 1))
    data['user', 'rates', 'job'].edge_index = edge_index
    data = ToUndirected()(data)
    return NormalizeFeatures()(data)

def split_and_load(data: HeteroData, cfg: Config):
    """Split data and create link neighbor loaders."""
    splitter = RandomLinkSplit(
        num_val=cfg.VAL_RATIO,
        num_test=cfg.TEST_RATIO,
        disjoint_train_ratio=cfg.DISJOINT_TRAIN_RATIO,
        neg_sampling_ratio=cfg.NEG_SAMPLING_RATIO,
        add_negative_train_samples=False,
        edge_types=('user','rates','job'),
        rev_edge_types=('job','rev_rates','user'),
        is_undirected=True
    )
    train_data, val_data, test_data = splitter(data)
    def make_loader(split_data, shuffle):
        return LinkNeighborLoader(
            data=split_data,
            num_neighbors=cfg.NUM_NEIGHBORS,
            neg_sampling=NegativeSampling(mode='binary') if shuffle else None,
            edge_label_index=(('user','rates','job'), split_data['user','rates','job'].edge_label_index),
            edge_label=split_data['user','rates','job'].edge_label,
            batch_size=cfg.BATCH_SIZE if shuffle else cfg.TEST_BATCH_SIZE,
            shuffle=shuffle
        )
    return make_loader(train_data, True), make_loader(val_data, False), make_loader(test_data, False)

@torch.no_grad()
def train_epoch(model: MFModel, loader: LinkNeighborLoader, optimizer, device):
    """Train model for one epoch."""
    model.train()
    total_loss = total = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch['user','rates','job'].edge_label_index)
        loss = F.binary_cross_entropy_with_logits(logits, batch['user','rates','job'].edge_label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * logits.size(0)
        total += logits.size(0)
    return total_loss / total

@torch.no_grad()
def eval_epoch(model: MFModel, loader: LinkNeighborLoader, device):
    """Evaluate model and compute metrics."""
    model.eval()
    total_loss = total = 0
    preds, gts = [], []
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = batch.to(device)
        logits = model(batch['user','rates','job'].edge_label_index)
        loss = F.binary_cross_entropy_with_logits(logits, batch['user','rates','job'].edge_label)
        total_loss += loss.item() * logits.size(0)
        total += logits.size(0)
        preds.append(torch.sigmoid(logits).cpu())
        gts.append(batch['user','rates','job'].edge_label.cpu())
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(gts).numpy()
    bin_pred = (y_pred >= 0.6).astype(int)
    return total_loss/total, {
        'auc': roc_auc_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, bin_pred),
        'recall': recall_score(y_true, bin_pred, zero_division=0),
        'f1': f1_score(y_true, bin_pred, zero_division=0),
        'precision': precision_score(y_true, bin_pred, zero_division=0),
        'avg_precision': average_precision_score(y_true, y_pred)
    }

@torch.no_grad()
def calc_precision_recall_at_k(model: MFModel, loader: LinkNeighborLoader, k_list, device):
    """Compute Precision@k and Recall@k metrics."""
    model.eval()
    user_preds, user_pos = defaultdict(list), defaultdict(set)
    for batch in tqdm(loader, desc="Collecting preds", leave=False):
        batch = batch.to(device)
        idx = batch['user','rates','job'].edge_label_index
        labels = batch['user','rates','job'].edge_label
        scores = torch.sigmoid(model(idx)).cpu()
        for i in range(idx.size(1)):
            u, j = idx[0,i].item(), idx[1,i].item()
            if labels[i].item() > 0.5:
                user_pos[u].add(j)
            user_preds[u].append((j, scores[i].item()))
    metrics = {}
    for k in k_list:
        precs, recs = [], []
        for u, preds in user_preds.items():
            if not user_pos[u]:
                continue
            preds.sort(key=lambda x: x[1], reverse=True)
            topk = {j for j,_ in preds[:k]}
            hits = len(topk & user_pos[u])
            precs.append(hits/k)
            recs.append(hits/len(user_pos[u]))
        metrics[f'precision@{k}'] = np.mean(precs) if precs else 0.0
        metrics[f'recall@{k}']    = np.mean(recs) if recs else 0.0
    return metrics

def main():
    """Run multi-seed training, evaluation, and summarize results."""
    seeds = list(range(40,50))
    all_results = defaultdict(list)
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        logger.info(f"Seed {seed}")
        cfg = Config()
        check_data_files_exist(cfg)
        num_users, num_items, edge_index = load_data(cfg)
        data = build_hetero(num_users, num_items, edge_index)
        train_loader, val_loader, test_loader = split_and_load(data, cfg)
        model = MFModel(num_users, num_items, cfg.EMBEDDING_DIM).to(cfg.DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)
        best_loss, wait = float('inf'), 0
        for epoch in range(1, cfg.NUM_EPOCHS+1):
            train_loss = train_epoch(model, train_loader, optimizer, cfg.DEVICE)
            val_loss, _ = eval_epoch(model, val_loader, cfg.DEVICE)
            scheduler.step(val_loss)
            if val_loss < best_loss:
                best_loss, wait = val_loss, 0
                torch.save(model.state_dict(), cfg.MODEL_SAVE_PATH)
            else:
                wait += 1
                if wait >= cfg.PATIENCE:
                    break
        model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=cfg.DEVICE))
        _, cls_metrics = eval_epoch(model, test_loader, cfg.DEVICE)
        rk_metrics = calc_precision_recall_at_k(model, test_loader, cfg.TOP_K, cfg.DEVICE)
        for k, v in {**cls_metrics, **rk_metrics}.items():
            all_results[k].append(v)
        logger.info(" | ".join(f"{m}={v:.4f}" for m, v in {**cls_metrics, **rk_metrics}.items()))
        del model, optimizer, scheduler, data, train_loader, val_loader, test_loader
        gc.collect(); torch.cuda.empty_cache()
    print("\n=== Summary ===")
    for metric, vals in all_results.items():
        mean = np.mean(vals)
        ci = stats.norm.ppf(0.975) * np.std(vals, ddof=1) / np.sqrt(len(vals))
        print(f"{metric}: {mean:.4f} ± {ci:.4f}")

if __name__ == "__main__":
    main()
