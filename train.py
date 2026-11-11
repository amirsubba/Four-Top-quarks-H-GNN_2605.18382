import os
import numpy as np
import uproot
import joblib

import torch
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

torch.set_num_threads(8)
torch.set_num_interop_threads(8)

print(f"PyTorch threads: {torch.get_num_threads()}")
print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'Not set')}")

from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, Softmax
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_scatter import scatter_mean, scatter_add

from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc)

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from HyPER.models import MPNNs, EdgeLoss, CombinedLoss

import warnings
warnings.filterwarnings("ignore")

if torch.backends.mps.is_available():
  device = torch.device("mps")
else:
  device = torch.device("cpu")

print("Using device:", device)

os.makedirs("results", exist_ok=True)

jet_features = ['Jet.PT', 'Jet.Phi', 'Jet.Eta', 'Jet.Mass', 'Jet.NCharged', 'Jet.EhadOverEem', 'Jet.BTag', 'Jet.Flavor', 
                'Jet.T', 'Jet.DeltaEta','Jet.DeltaPhi', 'Jet.ChargedEnergyFraction', 'Jet.NeutralEnergyFraction']

lep_features = ['Electron.PT', 'Electron.Eta', 'Electron.Phi', 'Muon.PT', 'Muon.Eta', 'Muon.Phi']
met_features = ['MissingET.MET', 'MissingET.Phi']
all_features = jet_features + lep_features + met_features

root_dir = '/path/to/data'

data_files = {
    'tttt': f'{root_dir}/tttt_small_train.root',
    'ttVV': f'{root_dir}/ttvv_bkg_lep.root',
    'ttH':  f'{root_dir}/tth_bkg_lep.root',
    'ttZ':  f'{root_dir}/ttz_bkg_lep.root',
    'ttW':  f'{root_dir}/ttw_bkg_lep.root',
}

class_names = ['tttt', 'ttVV', 'ttH', 'ttZ', 'ttW']
num_class = len(class_names)
class_to_idx = {name: idx for idx, name in enumerate(class_names)}
sig_class = 0
bkg_class = list(range(1, num_class))

# == HYPERPARAMETERS ==
batch_size = 128
DelR_thresh_edge = 0.1
DelR_thresh_hyper = 1.5
node_feat_dim = 13
tree_name = 'Delphes'

def delta_R(eta1, phi1, eta2, phi2):
  d_eta = eta1 - eta2
  d_phi = (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi
  return np.sqrt(d_eta**2 + d_phi**2)

def norm_feat(x):
  if x.shape[0] == 0:
    return x
  else:
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-08)

def pad_feat_vector(vec, target_dim=node_feat_dim):
  flat = []
  for v in vec:
    if hasattr(v, '__len__'):
      if len(v) == 0:
        flat.append(0.0)
      else:
        flat.append(float(v[0]))
    else:
      flat.append(float(v))
  
  arr = np.array(flat, dtype=np.float32)
  if arr.size >= target_dim:
    return arr[:target_dim]
  else:
    pad = np.zeros(target_dim - arr.size, dtype=np.float32)
    return np.concatenate([arr, pad])

def convert_to_hyper_data(x_nodes, eta_nodes, phi_nodes, label, DelR_thresh_edge=0.1, DelR_thresh_hyper=0.8):
  N = x_nodes.shape[0]
  if N == 0:
    return None

  edge_index = []
  edge_attr = []
  edge_target = []
  
  for ii in range(N):
    for jj in range(N):
      if ii >= jj:
        continue
      DR = delta_R(eta_nodes[ii], phi_nodes[ii], eta_nodes[jj], phi_nodes[jj])
      
      # Only create edges if DR > 0.1 (required for graph structure)
      if DR > DelR_thresh_edge:
        edge_index.append([ii, jj])
        edge_index.append([jj, ii])
        edge_attr.append([DR])
        edge_attr.append([DR])
        
        phys_thresh = 0.4
        target = 0 if DR < phys_thresh else 1
        edge_target.append([target])
        edge_target.append([target])
  
  # If no edges found, drop the event (graph has no structure)
  if not edge_index:
    return None
  
  edge_index = torch.tensor(edge_index, dtype=torch.long).T
  edge_attr_s = torch.tensor(edge_attr, dtype=torch.float)
  edge_attr_t = torch.tensor(edge_target, dtype=torch.float)

  # Always create hyperedges, even if they have only single nodes
  hyperedge_list = []
  
  for kk in range(N):
    neighbors = [kk]
    for ll in range(N):
      if kk != ll:
        DR = delta_R(eta_nodes[kk], phi_nodes[kk], eta_nodes[ll], phi_nodes[ll])
        if DR < DelR_thresh_hyper:
          neighbors.append(ll)
    hyperedge_list.append(neighbors)
  
  # Always create hyperedge tensors
  node_indices = np.concatenate(hyperedge_list).astype(np.int64)
  hyperedge_indices = np.repeat(np.arange(len(hyperedge_list)),
                                [len(h) for h in hyperedge_list]).astype(np.int64)
  hyperedge_index = torch.from_numpy(np.stack([node_indices, hyperedge_indices])).long()

  hyperedge_to_graph = torch.zeros(len(hyperedge_list), dtype=torch.long)
  u_s = torch.tensor([[x_nodes[:, 0].sum(), x_nodes[:, 1].mean()]], dtype=torch.float)

  data = Data(
    x_s=torch.tensor(x_nodes, dtype=torch.float),
    u_s=u_s,
    edge_index=edge_index,
    edge_attr_s=edge_attr_s,
    edge_attr_s_batch=torch.zeros(edge_index.shape[1], dtype=torch.long),
    edge_attr_t=edge_attr_t.view(-1, 1),
    edge_index_h=hyperedge_index,
    edge_index_h_batch=hyperedge_to_graph,
    x_t=torch.tensor([label], dtype=torch.long),
    batch=torch.zeros(N, dtype=torch.long)
  )
  
  return data

def load_data_hyper_format(file_path, label):
  if not os.path.exists(file_path):
    print(f"Warning: File {file_path} not found. Skipping...")
    return []
  
  file = uproot.open(file_path)
  tree = file[tree_name]
  events = tree.arrays(all_features, library='np')
  data_list = []
  num_events = len(events[jet_features[0]])
  
  for i in range(num_events): 
    jets = []
    for j in range(len(events[jet_features[0]][i])):
      vec = [float(events[feat][i][j]) if hasattr(events[feat][i][j], '__len__') == False
             else float(events[feat][i][j][0]) if len(events[feat][i][j]) > 0 else 0.0
             for feat in jet_features]
      jets.append(pad_feat_vector(vec))
    x_jets = np.vstack(jets) if jets else np.zeros((0, node_feat_dim))
    
    electrons = []
    for j in range(len(events['Electron.PT'][i])):
      vec = [float(events['Electron.PT'][i][j]),
             float(events['Electron.Phi'][i][j]),
             float(events['Electron.Eta'][i][j])]
      electrons.append(pad_feat_vector(vec))
    x_elec = np.vstack(electrons) if electrons else np.zeros((0, node_feat_dim))
    
    muons = []
    for j in range(len(events['Muon.PT'][i])):
      vec = [float(events['Muon.PT'][i][j]),
             float(events['Muon.Phi'][i][j]),
             float(events['Muon.Eta'][i][j])]
      muons.append(pad_feat_vector(vec))
    x_muon = np.vstack(muons) if muons else np.zeros((0, node_feat_dim))
    
    x_leptons = np.vstack([x_elec, x_muon]) if (x_elec.size + x_muon.size) > 0 else np.zeros((0, node_feat_dim))
    
    met_vec = [float(events['MissingET.MET'][i]), float(events['MissingET.Phi'][i])]
    x_met = pad_feat_vector(met_vec).reshape(1, -1)
    
    x_all = np.vstack([x_jets, x_leptons, x_met])
    x_all = norm_feat(x_all)
    
    eta_all = np.concatenate([
      x_jets[:, 2] if x_jets.shape[0] > 0 else np.array([]),
      x_leptons[:, 2] if x_leptons.shape[0] > 0 else np.array([])
    ])
    phi_all = np.concatenate([
      x_jets[:, 1] if x_jets.shape[0] > 0 else np.array([]),
      x_leptons[:, 1] if x_leptons.shape[0] > 0 else np.array([])
    ])
    
    if eta_all.size > 0:
      data = convert_to_hyper_data(
        np.vstack([x_jets, x_leptons]), 
        eta_all, 
        phi_all, 
        label, 
        DelR_thresh_edge, 
        DelR_thresh_hyper
      )
      if data is not None:
        data.x_s = torch.cat([data.x_s, torch.tensor(x_met, dtype=torch.float)], dim=0)
        data.batch = torch.cat([data.batch, torch.zeros(1, dtype=torch.long)], dim=0)
        data_list.append(data)
  
  return data_list

print("Loading data...")
dataset = []
class_sample_counts = {}

signal_data = load_data_hyper_format(data_files['tttt'], sig_class)
dataset.extend(signal_data)
class_sample_counts['tttt'] = len(signal_data)
print(f"Loaded {len(signal_data)} Signal samples")

for i, (bkg_name, file_path) in enumerate([item for item in data_files.items() if item[0] != 'tttt']):
  bkg_label = i + 1
  bkg_data = load_data_hyper_format(file_path, bkg_label)
  dataset.extend(bkg_data)
  class_sample_counts[class_names[bkg_label]] = len(bkg_data)
  print(f"Loaded {len(bkg_data)} {class_names[bkg_label]} samples")

print(f"Total dataset size: {len(dataset)}")
print("Class distribution:", class_sample_counts)

np.random.shuffle(dataset)
split = int(0.8 * len(dataset))
train_data, test_data = dataset[:split], dataset[split:]

train_labels = [d.x_t.item() for d in train_data]
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = torch.FloatTensor(class_weights)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

logger = TensorBoardLogger("lightning_logs", name="hypergraph_signal_vs_bkgs")

print(f"Class weights: {class_weights}")

class SignalVsBkgHyperedgeModel(torch.nn.Module):
    def __init__(self, node_in_channels, global_in_channels, message_feats, dropout, num_class):
        super().__init__()
        self.num_classes = num_class

        self.node_transform = torch.nn.Linear(node_in_channels, message_feats)
        self.global_transform = torch.nn.Linear(global_in_channels, message_feats)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(message_feats * 2, message_feats),
            torch.nn.BatchNorm1d(message_feats),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(message_feats, message_feats // 2),
            torch.nn.BatchNorm1d(message_feats // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(message_feats // 2, num_class)
        )

    def forward(self, x, u, batch, edge_index_h):
        x_t = self.node_transform(x)
        u_t = self.global_transform(u)

        node_idx, hyper_idx = edge_index_h
        num_hyperedges = hyper_idx.max().item() + 1
        
        counts = torch.bincount(hyper_idx, minlength=num_hyperedges)
        
        hyperedge_features = torch.zeros(num_hyperedges, x_t.size(1), device=x.device)
        hyperedge_features = hyperedge_features.scatter_add(
            0,
            hyper_idx.unsqueeze(1).expand(-1, x_t.size(1)),
            x_t[node_idx]
        )
        
        counts = counts.float().unsqueeze(1).clamp(min=1)
        hyperedge_features = hyperedge_features / counts
        
        first_nodes = torch.zeros(num_hyperedges, dtype=torch.long, device=x.device)
        for h in range(num_hyperedges):
            mask = (hyper_idx == h)
            if mask.any():
                first_nodes[h] = node_idx[mask][0]  
        
        hyperedge_graph = batch[first_nodes]
        combined = torch.cat([hyperedge_features, u_t[hyperedge_graph]], dim=1)
        logits = self.classifier(combined)
        
        return logits, hyperedge_graph 

class SignalVsBkgHyPERModel(LightningModule):
    def __init__(self, num_classes=num_class, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.signal_class = sig_class
        self.background_classes = bkg_class

        self.model = MPNNs(
            node_in_channels=node_feat_dim,
            edge_in_channels=1,
            global_in_channels=2,
            node_out_channels=32,
            edge_out_channels=1,
            global_out_channels=32,
            message_feats=32,
            dropout=0.01,
            activation=torch.nn.ReLU(),
            p_out='edge'
        )

        self.hyper = SignalVsBkgHyperedgeModel(
            node_in_channels=32,
            global_in_channels=32,
            message_feats=32,
            dropout=0.01,
            num_class=num_classes
        )

        self.criterion_edge = BCEWithLogitsLoss(reduction='none')
        self.criterion_hyper = CrossEntropyLoss(weight=class_weights)
        self.alpha = 0.5
        self.softmax = Softmax(dim=1)

        self.y_true, self.y_pred, self.y_prob = [], [], []
        self.train_losses, self.val_losses = [], []

    def forward(self, data):
        x, edge_attr_prime, u = self.model(
            data.x_s,
            data.edge_index,
            data.edge_attr_s,
            data.u_s,
            data.batch
        )
        
        logits_hyper, hyperedge_to_graph = self.hyper(x, u, data.batch, data.edge_index_h)
        
        num_graphs = data.u_s.shape[0]
        
        assert logits_hyper.shape[0] == hyperedge_to_graph.shape[0], \
            f"Mismatch: logits_hyper shape {logits_hyper.shape}, hyperedge_to_graph shape {hyperedge_to_graph.shape}"
        
        logits_graph = scatter_mean(
            logits_hyper, 
            hyperedge_to_graph, 
            dim=0, 
            dim_size=num_graphs
        )
        
        return logits_graph, edge_attr_prime

    def training_step(self, batch, batch_idx):
        logits, edge_attr_prime = self.forward(batch)
        
        if logits.shape[0] != batch.x_t.shape[0]:
            if logits.shape[0] < batch.x_t.shape[0]:
                pad = torch.zeros(batch.x_t.shape[0] - logits.shape[0], logits.shape[1], device=logits.device)
                logits = torch.cat([logits, pad], dim=0)
            else:
                logits = logits[:batch.x_t.shape[0]]

        loss_hyper = self.criterion_hyper(logits, batch.x_t.long())
        loss_edge = EdgeLoss(
            edge_attr_prime.view(-1),
            batch.edge_attr_t.view(-1),
            batch.edge_attr_s_batch,
            self.criterion_edge,
            reduction='mean'
        )
        loss = CombinedLoss(loss_edge, loss_hyper, self.alpha, 'mean')
        
        self.train_losses.append(loss.detach().cpu().item())
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, edge_attr_prime = self.forward(batch)
        
        if logits.shape[0] != batch.x_t.shape[0]:
            if logits.shape[0] < batch.x_t.shape[0]:
                pad = torch.zeros(batch.x_t.shape[0] - logits.shape[0], logits.shape[1], device=logits.device)
                logits = torch.cat([logits, pad], dim=0)
            else:
                logits = logits[:batch.x_t.shape[0]]
        
        probs = self.softmax(logits)
        preds = torch.argmax(logits, dim=1)
        
        self.y_true.extend(batch.x_t.detach().cpu().numpy())
        self.y_pred.extend(preds.detach().cpu().numpy())
        self.y_prob.extend(probs.detach().cpu().numpy())
        
        acc = (preds == batch.x_t.long()).float().mean()
        self.log('val_step_acc', acc, prog_bar=True)
        
        val_loss = self.criterion_hyper(logits, batch.x_t.long())
        self.val_losses.append(val_loss.detach().cpu().item())
        return val_loss

    def on_validation_epoch_end(self):
        if len(self.y_true) == 0:
            return
            
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        y_prob = np.array(self.y_prob)
        
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        y_true_bin = (y_true == self.signal_class).astype(int)
        signal_prob = y_prob[:, self.signal_class]
        
        if len(np.unique(y_true_bin)) > 1:
            signal_auc = roc_auc_score(y_true_bin, signal_prob)
        else:
            signal_auc = 0.5
            
        signal_f1 = f1_score(y_true_bin, (y_pred == self.signal_class).astype(int), zero_division=0)
        
        self.log('val_signal_auc', signal_auc, prog_bar=True)
        self.log('val_signal_f1', signal_f1)
        self.log('val_accuracy', acc)
        
        self.y_true.clear()
        self.y_pred.clear()
        self.y_prob.clear()
        self.train_losses.clear()
        self.val_losses.clear()

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.8,
            patience=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_signal_auc"
            }
        }

model = SignalVsBkgHyPERModel(num_classes=num_class, class_weights=class_weights)

from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    dirpath='results/checkpoints/4ClassdR0d1',
    filename='hgnn_epoch_{epoch:02d}_auc_{val_signal_auc:.4f}',
    save_top_k=3,
    monitor='val_signal_auc',
    mode='max',
    save_weights_only=True
)

trainer = Trainer(
    max_epochs=100,
    accelerator='auto',
    logger=logger,
    gradient_clip_val=1.0,
    callbacks=[checkpoint_callback],
    log_every_n_steps=10
)

print("\nStarting training...")
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
print("\nTraining complete!")

trainer.save_checkpoint('results/checkpoints/4ClassdR0d1/hgnn_tttt_dR_0d1_full_model.ckpt')
print("Model saved to results/checkpoints/4ClassdR0d1/hgnn_tttt_dR_0d1_full_model.ckpt")
