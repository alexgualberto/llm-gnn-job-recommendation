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
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.transforms import RandomLinkSplit, ToUndirected, NormalizeFeatures
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.sampler import NegativeSampling
import time
from collections import defaultdict
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hybrid_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration for model training and evaluation."""
    USER_EMBEDDINGS_PATH = 'user_embeddings.npy'
    JOB_EMBEDDINGS_PATH = 'job_embeddings.npy'
    EDGE_INDEX_PATH = 'interactions.csv'
    MODEL_SAVE_PATH = 'best_trained_gnn_model.pth'

    HIDDEN_CHANNELS = 256
    LEARNING_RATE = 1e-4

    BATCH_SIZE = 256
    TEST_BATCH_SIZE = BATCH_SIZE * 3
    NUM_EPOCHS = 300
    PATIENCE = 5

    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    DISJOINT_TRAIN_RATIO = 0.8
    NEG_SAMPLING_RATIO = 100

    NUM_NEIGHBORS = [40, 20]

    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 2
    SCHEDULER_THRESHOLD = 0.01
    SCHEDULER_MIN_LR = 1e-6

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GNN(torch.nn.Module):
    """Two-layer GraphSAGE model with mean aggregation."""
    def __init__(self, hidden_channels, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv(-1, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(-1, hidden_channels, aggr='mean')
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

class Classifier(torch.nn.Module):
    """Dot-product classifier for link prediction."""
    def forward(self, x_user: Tensor, x_job: Tensor, edge_label_index: Tensor) -> Tensor:
        user_feats = x_user[edge_label_index[0]]
        job_feats = x_job[edge_label_index[1]]
        return torch.sum(user_feats * job_feats, dim=1)

class Model(torch.nn.Module):
    """Full GNN-based link prediction model."""
    def __init__(self, hidden_channels, user_input_dim, job_input_dim, dropout=0.2):
        super().__init__()
        self.job_lin = torch.nn.Linear(job_input_dim, hidden_channels)
        self.user_lin = torch.nn.Linear(user_input_dim, hidden_channels)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.user_bn = torch.nn.BatchNorm1d(hidden_channels)
        self.job_bn = torch.nn.BatchNorm1d(hidden_channels)

        self.gnn = GNN(hidden_channels, dropout)
        metadata = (['user', 'job'], [('user', 'rates', 'job'), ('job', 'rev_rates', 'user')])
        self.gnn = to_hetero(self.gnn, metadata=metadata)
        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        u = self.user_lin(data['user'].x)
        u = self.user_bn(u)
        u = F.relu(u)
        u = self.dropout(u)

        j = self.job_lin(data['job'].x)
        j = self.job_bn(j)
        j = F.relu(j)
        j = self.dropout(j)

        x_dict = {'user': u, 'job': j}
        x_dict = self.gnn(x_dict, data.edge_index_dict)

        return self.classifier(
            x_dict['user'],
            x_dict['job'],
            data['user', 'rates', 'job'].edge_label_index
        )

# Utility functions

def check_data_files_exist(config):
    paths = [config.USER_EMBEDDINGS_PATH, config.JOB_EMBEDDINGS_PATH, config.EDGE_INDEX_PATH]
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")
    return True

def load_data(config):
    logger.info("Loading data...")
    users = np.load(config.USER_EMBEDDINGS_PATH)
    jobs = np.load(config.JOB_EMBEDDINGS_PATH)
    df = pd.read_csv(config.EDGE_INDEX_PATH)
    edge_index = torch.stack([
        torch.tensor(df['user_id'].values, dtype=torch.long),
        torch.tensor(df['job_id'].values, dtype=torch.long)
    ], dim=0)
    return users, jobs, edge_index

def create_hetero_data(user_emb, job_emb, edge_index):
    logger.info("Building HeteroData...")
    data = HeteroData()
    data['user'].x = torch.tensor(user_emb, dtype=torch.float32)
    data['job'].x = torch.tensor(job_emb, dtype=torch.float32)
    data['user', 'rates', 'job'].edge_index = edge_index
    data = ToUndirected()(data)
    data = NormalizeFeatures()(data)
    logger.info(f"Data metadata: {data.metadata()}")
    return data

# Training and evaluation loops

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = total = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        batch = batch.to(device)
        pred = model(batch)
        label = batch['user', 'rates', 'job'].edge_label
        loss = F.binary_cross_entropy_with_logits(pred, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * pred.numel()
        total += pred.numel()
    return total_loss / total

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = total = 0
    for batch in tqdm(loader, desc="Validating", leave=False):
        batch = batch.to(device)
        pred = model(batch)
        label = batch['user', 'rates', 'job'].edge_label
        loss = F.binary_cross_entropy_with_logits(pred, label)
        total_loss += loss.item() * pred.numel()
        total += pred.numel()
    return total_loss / total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, truths = [], []
    for batch in tqdm(loader, desc="Testing", leave=False):
        batch = batch.to(device)
        probs = torch.sigmoid(model(batch)).cpu()
        preds.append(probs)
        truths.append(batch['user', 'rates', 'job'].edge_label.cpu())
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(truths).numpy().astype(int)
    return {
        'auc': roc_auc_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred > 0.6),
        'precision': precision_score(y_true, y_pred > 0.6),
        'recall': recall_score(y_true, y_pred > 0.6),
        'f1': f1_score(y_true, y_pred > 0.6)
    }

@torch.no_grad()
def precision_recall_at_k(model, loader, ks, device):
    model.eval()
    user_preds, user_pos = defaultdict(list), defaultdict(set)
    for batch in tqdm(loader, desc="Collecting predictions", leave=False):
        batch = batch.to(device)
        idx = batch['user', 'rates', 'job'].edge_label_index
        labels = batch['user', 'rates', 'job'].edge_label
        scores = torch.sigmoid(model(batch)).cpu()
        for i in range(idx.size(1)):
            u, j = idx[0, i].item(), idx[1, i].item()
            s, l = scores[i].item(), labels[i].item()
            user_preds[u].append((j, s))
            if l > 0.6:
                user_pos[u].add(j)
    res = {}
    for k in ks:
        ps, rs = [], []
        for u, preds in user_preds.items():
            if not user_pos[u]: continue
            top = [j for j, _ in sorted(preds, key=lambda x: x[1], reverse=True)[:k]]
            hits = len(set(top) & user_pos[u])
            ps.append(hits / k)
            rs.append(hits / len(user_pos[u]))
        res[f'precision@{k}'] = np.mean(ps) if ps else 0.0
        res[f'recall@{k}'] = np.mean(rs) if rs else 0.0
    return res

# Data split and loaders

def split_data(data, config):
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

def create_loaders(train, val, test, config):
    return (
        LinkNeighborLoader(
            data=train,
            num_neighbors=config.NUM_NEIGHBORS,
            neg_sampling=NegativeSampling('binary'),
            edge_label_index=(('user','rates','job'), train['user','rates','job'].edge_label_index),
            edge_label=train['user','rates','job'].edge_label,
            batch_size=config.BATCH_SIZE,
            shuffle=True
        ),
        LinkNeighborLoader(
            data=val,
            num_neighbors=config.NUM_NEIGHBORS,
            edge_label_index=(('user','rates','job'), val['user','rates','job'].edge_label_index),
            edge_label=val['user','rates','job'].edge_label,
            batch_size=config.TEST_BATCH_SIZE
        ),
        LinkNeighborLoader(
            data=test,
            num_neighbors=config.NUM_NEIGHBORS,
            edge_label_index=(('user','rates','job'), test['user','rates','job'].edge_label_index),
            edge_label=test['user','rates','job'].edge_label,
            batch_size=config.TEST_BATCH_SIZE
        )
    )

# Main pipeline

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    config = Config()

    print("--- Job Recommendation with GNN ---")
    load_pretrained = input("Load existing model? (y/n): ").lower() in ('y','yes')
    if load_pretrained:
        path = input(f"Model path [{config.MODEL_SAVE_PATH}]: ")
        if path: config.MODEL_SAVE_PATH = path

    print(f"Using device: {config.DEVICE}")
    if not input("Proceed? (y/n): ").lower() in ('y','yes'):
        print("Canceled.")
        return

    start = time.time()
    check_data_files_exist(config)
    users, jobs, edge_idx = load_data(config)
    data = create_hetero_data(users, jobs, edge_idx)

    model = Model(config.HIDDEN_CHANNELS, data['user'].x.shape[1], data['job'].x.shape[1]).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE, threshold=config.SCHEDULER_THRESHOLD,
        min_lr=config.SCHEDULER_MIN_LR
    )

    if load_pretrained:
        from pathlib import Path
        if Path(config.MODEL_SAVE_PATH).exists():
            ckpt = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
            model.load_state_dict(ckpt['model_state_dict'])
            logger.info("Model loaded.")
        else:
            logger.warning("Pretrained model not found, training a new one.")

    if not load_pretrained:
        train_d, val_d, test_d = split_data(data, config)
        train_loader, val_loader, test_loader = create_loaders(train_d, val_d, test_d, config)

        best_state, best_val = None, float('inf')
        patience = config.PATIENCE
        for epoch in range(1, config.NUM_EPOCHS+1):
            tr_loss = train_epoch(model, train_loader, optimizer, config.DEVICE)
            val_loss = validate(model, val_loader, config.DEVICE)
            scheduler.step(val_loss)
            lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch}, train_loss={tr_loss:.4f}, val_loss={val_loss:.4f}, lr={lr:.6f}")
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k:v.cpu() for k,v in model.state_dict().items()}
                epoch_patience = 0
            else:
                epoch_patience += 1
            if epoch_patience >= patience or lr <= config.SCHEDULER_MIN_LR:
                logger.info("Stopping early.")
                break

        if best_state:
            model.load_state_dict(best_state)
            torch.save({
                'model_state_dict': best_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val,
                'date_saved': pd.Timestamp.now().isoformat()
            }, config.MODEL_SAVE_PATH)
            logger.info(f"Model saved to {config.MODEL_SAVE_PATH}")

    # Final evaluation
    train_d, val_d, test_d = split_data(data, config)
    _, _, test_loader = create_loaders(train_d, val_d, test_d, config)
    eval_metrics = evaluate(model, test_loader, config.DEVICE)
    rank_metrics = precision_recall_at_k(model, test_loader, [5,10,20,50,100], config.DEVICE)

    logger.info(f"Evaluation: {eval_metrics}")
    logger.info(f"Ranking metrics: {rank_metrics}")

    elapsed = time.time() - start
    logger.info(f"Total time: {elapsed/60:.2f} minutes")

if __name__ == '__main__':
    main()
