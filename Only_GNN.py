import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
import random
import logging
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, precision_score
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.transforms import RandomLinkSplit, ToUndirected, NormalizeFeatures
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.sampler import NegativeSampling
import scipy.stats as stats
import gc

# Configure logging
torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("onlygnn_hybrid_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration for GNN-based link prediction."""
    EDGE_INDEX_PATH      = 'interactions.csv'
    MODEL_SAVE_PATH      = 'best_trained_gnn_model.pth'
    HIDDEN_CHANNELS      = 256
    EMBEDDING_DIM        = 256
    LEARNING_RATE        = 1e-4
    BATCH_SIZE           = 256
    TEST_BATCH_SIZE      = BATCH_SIZE * 3
    NUM_EPOCHS           = 300
    PATIENCE             = 5
    VAL_RATIO            = 0.1
    TEST_RATIO           = 0.1
    DISJOINT_TRAIN_RATIO = 0.8
    NEG_SAMPLING_RATIO   = 1
    NUM_NEIGHBORS        = [40, 20]
    SCHEDULER_FACTOR     = 0.5
    SCHEDULER_PATIENCE   = 2
    SCHEDULER_THRESHOLD  = 0.01
    SCHEDULER_MIN_LR     = 1e-6
    TOP_K_VALUES         = [5, 10, 20, 50, 100]
    DEVICE               = torch_device

class GNN(torch.nn.Module):
    """Two-layer GraphSAGE for node embedding."""
    def __init__(self, hidden_channels, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv(-1, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(-1, hidden_channels, aggr='mean')
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        return self.conv2(x, edge_index)

class Classifier(torch.nn.Module):
    """Dot-product link predictor."""
    def forward(self, x_user: Tensor, x_job: Tensor, edge_label_index: Tensor) -> Tensor:
        u = x_user[edge_label_index[0]]
        j = x_job[edge_label_index[1]]
        return torch.sum(u * j, dim=1)

class Model(torch.nn.Module):
    """Heterogeneous GNN with user/job embeddings and link prediction."""
    def __init__(self, hidden_channels, embedding_dim, num_users, num_jobs, dropout=0.2):
        super().__init__()
        self.user_embed = torch.nn.Embedding(num_users, embedding_dim)
        self.job_embed  = torch.nn.Embedding(num_jobs, embedding_dim)
        self.user_lin   = torch.nn.Linear(embedding_dim, hidden_channels)
        self.job_lin    = torch.nn.Linear(embedding_dim, hidden_channels)
        self.user_bn    = torch.nn.BatchNorm1d(hidden_channels)
        self.job_bn     = torch.nn.BatchNorm1d(hidden_channels)
        self.dropout    = torch.nn.Dropout(dropout)
        self.gnn        = to_hetero(GNN(hidden_channels, dropout), metadata=(['user','job'], [('user','rates','job'),('job','rev_rates','user')]))
        self.clf        = Classifier()
        torch.nn.init.xavier_uniform_(self.user_embed.weight)
        torch.nn.init.xavier_uniform_(self.job_embed.weight)

    def forward(self, data: HeteroData) -> Tensor:
        u_id, j_id = data['user'].node_id, data['job'].node_id
        u_x = self.user_embed(u_id)
        j_x = self.job_embed(j_id)
        u_x = F.relu(self.user_bn(self.user_lin(u_x)))
        j_x = F.relu(self.job_bn(self.job_lin(j_x)))
        u_x, j_x = self.dropout(u_x), self.dropout(j_x)
        x_dict = self.gnn({'user': u_x, 'job': j_x}, data.edge_index_dict)
        return self.clf(x_dict['user'], x_dict['job'], data['user','rates','job'].edge_label_index)

def set_seed(seed: int):
    """Ensure reproducibility across modules."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(config: Config) -> pd.DataFrame:
    """Read interaction dataframe."""
    return pd.read_csv(config.EDGE_INDEX_PATH)

def create_hetero_data(df: pd.DataFrame) -> HeteroData:
    """Build HeteroData from interactions."""
    data = HeteroData()
    data['user'].node_id = torch.unique(torch.tensor(df.user_id.values))
    data['job'].node_id  = torch.unique(torch.tensor(df.job_id.values))
    data['user','rates','job'].edge_index = torch.stack([torch.tensor(df.user_id.values), torch.tensor(df.job_id.values)], dim=0)
    data = ToUndirected()(data)
    return NormalizeFeatures()(data)

def split_data(data: HeteroData, config: Config):
    """Split edges into train/val/test."""
    splitter = RandomLinkSplit(
        num_val=config.VAL_RATIO,
        num_test=config.TEST_RATIO,
        disjoint_train_ratio=config.DISJOINT_TRAIN_RATIO,
        neg_sampling_ratio=config.NEG_SAMPLING_RATIO,
        add_negative_train_samples=False,
        edge_types=('user','rates','job'),
        rev_edge_types=('job','rev_rates','user'),
        is_undirected=True
    )
    return splitter(data)

def create_data_loaders(train, val, test, config: Config):
    """Instantiate LinkNeighborLoaders for each split."""
    return (
        LinkNeighborLoader(train, num_neighbors=config.NUM_NEIGHBORS, neg_sampling=NegativeSampling('binary'), edge_label_index=(('user','rates','job'), train['user','rates','job'].edge_label_index), edge_label=train['user','rates','job'].edge_label, batch_size=config.BATCH_SIZE, shuffle=True),
        LinkNeighborLoader(val,   num_neighbors=config.NUM_NEIGHBORS, edge_label_index=(('user','rates','job'), val['user','rates','job'].edge_label_index),   edge_label=val['user','rates','job'].edge_label,   batch_size=config.TEST_BATCH_SIZE, shuffle=False),
        LinkNeighborLoader(test,  num_neighbors=config.NUM_NEIGHBORS, edge_label_index=(('user','rates','job'), test['user','rates','job'].edge_label_index),  edge_label=test['user','rates','job'].edge_label,  batch_size=config.TEST_BATCH_SIZE, shuffle=False)
    )

def train_epoch(model, loader, optimizer, device):
    """Train for one epoch with gradient clipping."""
    model.train()
    total_loss = total = 0
    for batch in tqdm(loader, desc='Training', leave=False):
        optimizer.zero_grad()
        batch = batch.to(device)
        pred = model(batch)
        label = batch['user','rates','job'].edge_label
        loss = F.binary_cross_entropy_with_logits(pred, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * pred.numel()
        total += pred.numel()
    return total_loss / total

@torch.no_grad()
def validate(model, loader, device):
    """Validate on held-out edges."""
    model.eval()
    total_loss = total = 0
    for batch in tqdm(loader, desc='Validating', leave=False):
        batch = batch.to(device)
        pred = model(batch)
        loss = F.binary_cross_entropy_with_logits(pred, batch['user','rates','job'].edge_label)
        total_loss += loss.item() * pred.numel()
        total += pred.numel()
    return total_loss / total

@torch.no_grad()
def evaluate(model, loader, device):
    """Compute AUC, accuracy, precision, recall, and F1."""
    model.eval()
    preds, truths = [], []
    for batch in tqdm(loader, desc='Testing', leave=False):
        batch = batch.to(device)
        prob = torch.sigmoid(model(batch)).cpu()
        preds.append(prob)
        truths.append(batch['user','rates','job'].edge_label.cpu())
    preds = torch.cat(preds).numpy()
    truths = torch.cat(truths).numpy()
    bin_pred = (preds > 0.5).astype(int)
    return {
        'auc': roc_auc_score(truths, preds),
        'accuracy': accuracy_score(truths, bin_pred),
        'precision': precision_score(truths, bin_pred),
        'recall': recall_score(truths, bin_pred),
        'f1': f1_score(truths, bin_pred)
    }

@torch.no_grad()
def calculate_precision_recall_at_k(model, loader, k_values, device):
    """Compute Precision@k and Recall@k per user."""
    model.eval()
    user_preds, user_pos = defaultdict(list), defaultdict(set)
    for batch in tqdm(loader, desc='Calculating P@K', leave=False):
        batch = batch.to(device)
        eidx = batch['user','rates','job'].edge_label_index
        labels = batch['user','rates','job'].edge_label
        scores = torch.sigmoid(model(batch)).cpu()
        for i in range(eidx.size(1)):
            u, j = eidx[0,i].item(), eidx[1,i].item()
            if labels[i].item() > 0.5:
                user_pos[u].add(j)
            user_preds[u].append((j, scores[i].item()))
    metrics = {}
    for k in k_values:
        prec_list, rec_list = [], []
        for u, preds in user_preds.items():
            if not user_pos[u]:
                continue
            jobs = [x for x,_ in sorted(preds, key=lambda x: x[1], reverse=True)]
            hits = len(set(jobs[:k]) & user_pos[u])
            prec_list.append(hits / k)
            rec_list.append(hits / len(user_pos[u]))
        metrics[f'precision@{k}'] = np.mean(prec_list) if prec_list else 0.0
        metrics[f'recall@{k}']    = np.mean(rec_list)  if rec_list  else 0.0
    return metrics

def train_model(model, train_loader, val_loader, optimizer, scheduler, config):
    """Train with early stopping and LR scheduler."""
    best_state, best_val, no_imp = None, float('inf'), 0
    for epoch in tqdm(range(1, config.NUM_EPOCHS+1), desc='Epochs'):
        train_loss = train_epoch(model, train_loader, optimizer, config.DEVICE)
        val_loss   = validate(model, val_loader, config.DEVICE)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']
        logger.info(f'Epoch {epoch:03d}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={lr:.6f}')
        if val_loss < best_val:
            best_state, best_val, no_imp = {k:v.cpu() for k,v in model.state_dict().items()}, val_loss, 0
        else:
            no_imp += 1
            if no_imp >= config.PATIENCE or lr <= config.SCHEDULER_MIN_LR:
                break
    return best_state, best_val

from tqdm import tqdm as main_tqdm

def main():
    """Run multi-seed evaluation and output summary."""
    seeds = list(range(40,50))
    all_results = defaultdict(list)
    for seed in main_tqdm(seeds, desc='Seeds'):
        set_seed(seed)
        logger.info(f'Seed {seed}')
        config = Config()
        df = load_data(config)
        data = create_hetero_data(df)
        train_data, val_data, test_data = split_data(data, config)
        loaders = create_data_loaders(train_data, val_data, test_data, config)
        model = Model(config.HIDDEN_CHANNELS, config.EMBEDDING_DIM,
                      num_users=data['user'].node_id.size(0),
                      num_jobs=data['job'].node_id.size(0)).to(config.DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE, threshold=config.SCHEDULER_THRESHOLD,
            min_lr=config.SCHEDULER_MIN_LR)
        best_state, _ = train_model(model, *loaders, optimizer, scheduler, config)
        if best_state:
            model.load_state_dict(best_state)
        cls_metrics = evaluate(model, loaders[2], config.DEVICE)
        rk_metrics  = calculate_precision_recall_at_k(model, loaders[2], config.TOP_K_VALUES, config.DEVICE)
        for k, v in {**cls_metrics, **rk_metrics}.items():
            all_results[k].append(v)
        logger.info('Results: ' + ', '.join(f'{m}={val:.4f}' for m,val in {**cls_metrics, **rk_metrics}.items()))
        del model, optimizer, scheduler, *loaders
        gc.collect(); torch.cuda.empty_cache()
    print('\n=== Summary 40-49 ===')
    for metric, vals in all_results.items():
        mean = np.mean(vals)
        ci = stats.norm.ppf(0.975) * np.std(vals, ddof=1) / np.sqrt(len(vals))
        print(f'{metric}: {mean:.4f} ± {ci:.4f}')

if __name__ == '__main__':
    main()
