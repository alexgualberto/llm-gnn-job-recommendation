import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
import logging
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, precision_score
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
from torch_geometric.loader import LinkNeighborLoader
from collections import defaultdict
import scipy.stats as stats
import gc

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pure_embeddings_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration for embeddings evaluation."""
    USER_EMBEDDINGS_PATH = 'user_embeddings.npy'
    JOB_EMBEDDINGS_PATH = 'job_embeddings.npy'
    EDGE_INDEX_PATH = 'interactions.csv'
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    DISJOINT_TRAIN_RATIO = 0.8
    NEG_SAMPLING_RATIO = 1
    NUM_NEIGHBORS = [40, 20]
    BATCH_SIZE = 256
    TEST_BATCH_SIZE = BATCH_SIZE * 3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def calculate_precision_recall_at_k(model, loader, k_values=None, device=None):
    """Compute Precision@k and Recall@k for each user."""
    if k_values is None:
        k_values = [5, 10, 20, 50, 100]
    device = device or Config.DEVICE
    model.eval()
    user_preds = defaultdict(list)
    user_pos = defaultdict(set)

    for batch in tqdm(loader, desc="Collecting predictions"):
        batch = batch.to(device)
        eidx = batch['user', 'rates', 'job'].edge_label_index
        labels = batch['user', 'rates', 'job'].edge_label
        scores = torch.sigmoid(model(batch)).cpu()
        for i in range(eidx.size(1)):
            u = eidx[0, i].item()
            j = eidx[1, i].item()
            s = scores[i].item()
            l = labels[i].item()
            user_preds[u].append((j, s))
            if l > 0.5:
                user_pos[u].add(j)

    precision, recall = {k: [] for k in k_values}, {k: [] for k in k_values}
    for u, preds in user_preds.items():
        if not user_pos[u]:
            continue
        preds.sort(key=lambda x: x[1], reverse=True)
        recs = [job for job, _ in preds]
        for k in k_values:
            topk = set(recs[:k])
            hits = len(topk & user_pos[u])
            precision[k].append(hits / k)
            recall[k].append(hits / len(user_pos[u]))

    metrics = {}
    for k in k_values:
        metrics[f'precision@{k}'] = np.mean(precision[k]) if precision[k] else 0.0
        metrics[f'recall@{k}'] = np.mean(recall[k]) if recall[k] else 0.0
    logger.info("Computed Precision@k and Recall@k.")
    return metrics

@torch.no_grad()
def evaluate_all(model, loader, device=None):
    """Compute AUC, accuracy, precision, recall, and F1."""
    device = device or Config.DEVICE
    model.eval()
    preds, truths = [], []

    for batch in tqdm(loader, desc="Testing embeddings"):
        batch = batch.to(device)
        out = torch.sigmoid(model(batch)).cpu()
        preds.append(out)
        truths.append(batch['user', 'rates', 'job'].edge_label.cpu())

    y_pred = torch.cat(preds).numpy().ravel()
    y_true = torch.cat(truths).numpy().ravel().astype(int)

    assert set(np.unique(y_true)).issubset({0, 1}), "Ground-truth labels must be binary."

    return {
        'auc': roc_auc_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, (y_pred > 0.5).astype(int)),
        'precision': precision_score(y_true, (y_pred > 0.5).astype(int)),
        'recall': recall_score(y_true, (y_pred > 0.5).astype(int)),
        'f1': f1_score(y_true, (y_pred > 0.5).astype(int))
    }

class Classifier(torch.nn.Module):
    """Dot-product classifier for link prediction."""
    def forward(self, x_user: Tensor, x_job: Tensor, edge_label_index: Tensor) -> Tensor:
        u = x_user[edge_label_index[0]]
        v = x_job[edge_label_index[1]]
        return torch.sum(u * v, dim=1)

class Model(torch.nn.Module):
    """Pure L2-normalized dot-product between user and job embeddings."""
    def __init__(self):
        super().__init__()
        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        usr = F.normalize(data['user'].x, dim=1)
        job = F.normalize(data['job'].x, dim=1)
        return self.classifier(usr, job, data['user', 'rates', 'job'].edge_label_index)

def load_data(config: Config):
    """Load embeddings and interaction indices."""
    users = np.load(config.USER_EMBEDDINGS_PATH)
    jobs = np.load(config.JOB_EMBEDDINGS_PATH)
    df = pd.read_csv(config.EDGE_INDEX_PATH)
    edge_index = torch.stack([
        torch.tensor(df.user_id.values, dtype=torch.long),
        torch.tensor(df.job_id.values, dtype=torch.long)
    ], dim=0)
    return users, jobs, edge_index

def create_hetero_data(users: np.ndarray, jobs: np.ndarray, edge_index: Tensor) -> HeteroData:
    """Build HeteroData for link prediction."""
    data = HeteroData()
    data['user'].x = torch.tensor(users, dtype=torch.float32)
    data['job'].x = torch.tensor(jobs, dtype=torch.float32)
    data['user', 'rates', 'job'].edge_index = edge_index
    return ToUndirected()(data)

def split_and_loader(data: HeteroData, config: Config) -> LinkNeighborLoader:
    """Create LinkNeighborLoader for test split."""
    splitter = RandomLinkSplit(
        num_val=config.VAL_RATIO,
        num_test=config.TEST_RATIO,
        disjoint_train_ratio=config.DISJOINT_TRAIN_RATIO,
        neg_sampling_ratio=config.NEG_SAMPLING_RATIO,
        add_negative_train_samples=False,
        edge_types=('user', 'rates', 'job'),
        rev_edge_types=('job', 'rev_rates', 'user'),
        is_undirected=True
    )
    _, _, test_data = splitter(data)
    return LinkNeighborLoader(
        data=test_data,
        num_neighbors=config.NUM_NEIGHBORS,
        edge_label_index=(('user', 'rates', 'job'), test_data['user', 'rates', 'job'].edge_label_index),
        edge_label=test_data['user', 'rates', 'job'].edge_label,
        batch_size=config.TEST_BATCH_SIZE,
        shuffle=False
    )

def main():
    """Run evaluation across multiple random seeds."""
    seeds = list(range(40, 50))
    all_results = defaultdict(list)

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        logger.info(f"Starting seed {seed}")

        cfg = Config()
        users, jobs, eidx = load_data(cfg)
        data = create_hetero_data(users, jobs, eidx)
        test_loader = split_and_loader(data, cfg)
        model = Model().to(cfg.DEVICE)

        cls_metrics = evaluate_all(model, test_loader, cfg.DEVICE)
        rk_metrics = calculate_precision_recall_at_k(model, test_loader, device=cfg.DEVICE)

        logger.info(
            " | ".join(f"{k}={v:.4f}" for k, v in {**cls_metrics, **rk_metrics}.items())
        )

        for metric, value in {**cls_metrics, **rk_metrics}.items():
            all_results[metric].append(value)

        del model, data, test_loader, users, jobs, eidx
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n=== Summary over {len(seeds)} seeds ===")
    for metric, vals in all_results.items():
        mean = np.mean(vals)
        ci = stats.norm.ppf(0.975) * np.std(vals, ddof=1) / np.sqrt(len(vals))
        print(f"{metric}: {mean:.4f} ± {ci:.4f}")

if __name__ == '__main__':
    main()
