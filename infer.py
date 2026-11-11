import os
import time
import resource
import multiprocessing as mp
import numpy as np
import pandas as pd
import uproot
import torch

from torch_geometric.data import Data, DataLoader
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, Softmax
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_scatter import scatter_mean, scatter_add
from pytorch_lightning import LightningModule
from HyPER.models import MPNNs, EdgeLoss, CombinedLoss

import warnings
warnings.filterwarnings("ignore")

soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
try:
  resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
except Exception:
  pass

if torch.backends.mps.is_available():
  device = torch.device("mps")
elif torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
print("Using device:", device)

jet_features = ['Jet.PT', 'Jet.Phi', 'Jet.Eta', 'Jet.Mass', 'Jet.NCharged', 'Jet.EhadOverEem', 'Jet.BTag', 'Jet.Flavor',
                'Jet.T', 'Jet.DeltaEta', 'Jet.DeltaPhi', 'Jet.ChargedEnergyFraction', 'Jet.NeutralEnergyFraction']
lep_features = ['Electron.PT', 'Electron.Eta', 'Electron.Phi', 'Muon.PT', 'Muon.Eta', 'Muon.Phi']
met_features = ['MissingET.MET', 'MissingET.Phi']
all_features = jet_features + lep_features + met_features

class_names = ['tttt', 'ttVV', 'ttH', 'ttZ', 'ttW']
num_class = len(class_names)
class_to_idx = {name: idx for idx, name in enumerate(class_names)}
sig_class = 0
bkg_class = list(range(1, 5))

batch_size = 1
DelR_Edge = 0.1
DelR_Hedge = 1.5
node_feat_dim = 13
tree_name = 'Delphes'


def delta_R(eta1, phi1, eta2, phi2):
  d_eta = eta1 - eta2
  d_phi = (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi
  return np.sqrt(d_eta**2 + d_phi**2)


def compute_rt(jets, leptons, met_pt, met_phi):
  all_objects = jets + leptons
  n_obj = len(all_objects)
  if n_obj < 2:
    return float(met_pt), float(met_pt)

  pts = np.empty(n_obj, dtype=np.float64)
  etas = np.empty(n_obj, dtype=np.float64)
  phis = np.empty(n_obj, dtype=np.float64)
  for k in range(n_obj):
    obj = all_objects[k]
    pts[k] = obj['PT']
    etas[k] = obj['Eta']
    phis[k] = obj['Phi']

  d_eta = etas[:, None] - etas[None, :]
  d_phi = (phis[:, None] - phis[None, :] + np.pi) % (2 * np.pi) - np.pi
  dR = np.sqrt(d_eta**2 + d_phi**2)

  mask = (pts[:, None] > 0) & (pts[None, :] > 0) & (dR > 0)
  np.fill_diagonal(mask, False)

  prod = pts[:, None] * pts[None, :] * dR
  obs1 = float(np.sum(prod * mask))
  obs2 = float(np.sum(np.sqrt(prod) * mask))

  return float(np.sqrt(max(0.0, obs1)) + met_pt), float(obs2 + met_pt)


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
    num_hyperedges = int(hyper_idx.max().item()) + 1
    counts = torch.bincount(hyper_idx, minlength=num_hyperedges)

    hyperedge_features = torch.zeros(num_hyperedges, x_t.size(1), device=x.device)
    hyperedge_features = hyperedge_features.scatter_add(
      0, hyper_idx.unsqueeze(1).expand(-1, x_t.size(1)), x_t[node_idx]
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
      node_in_channels=node_feat_dim, edge_in_channels=1, global_in_channels=2,
      node_out_channels=32, edge_out_channels=1, global_out_channels=32,
      message_feats=32, dropout=0.01, activation=torch.nn.ReLU(), p_out='edge'
    )
    self.hyper = SignalVsBkgHyperedgeModel(
      node_in_channels=32, global_in_channels=32, message_feats=32,
      dropout=0.01, num_class=num_classes
    )

    self.criterion_edge = BCEWithLogitsLoss(reduction='none')
    self.criterion_hyper = CrossEntropyLoss(weight=class_weights)
    self.alpha = 0.5
    self.softmax = Softmax(dim=1)
    self.y_true, self.y_pred, self.y_prob = [], [], []
    self.train_losses, self.val_losses = [], []

  def forward(self, data):
    x, edge_attr_prime, u = self.model(
      data.x_s, data.edge_index, data.edge_attr_s, data.u_s, data.batch
    )
    logits_hyper, hyperedge_to_graph = self.hyper(x, u, data.batch, data.edge_index_h)
    num_graphs = data.u_s.shape[0]
    logits_graph = scatter_mean(logits_hyper, hyperedge_to_graph, dim=0, dim_size=num_graphs)
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
    loss_edge = EdgeLoss(edge_attr_prime.view(-1), batch.edge_attr_t.view(-1),
                         batch.edge_attr_s_batch, self.criterion_edge, reduction='mean')
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
    val_loss = self.criterion_hyper(logits, batch.x_t.long())
    self.val_losses.append(val_loss.detach().cpu().item())
    return val_loss

  def configure_optimizers(self):
    optimizer = Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=10)
    return {"optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_signal_auc"}}


_WORKER_EVENTS = None
_WORKER_LABEL = None


def _init_worker(file_path, label):
  torch.set_num_threads(1)
  global _WORKER_EVENTS, _WORKER_LABEL
  file = uproot.open(file_path)
  _WORKER_EVENTS = file[tree_name].arrays(all_features, library='np')
  _WORKER_LABEL = label


def build_jet_matrix(events, i):
  n = len(events['Jet.PT'][i])
  if n == 0:
    return np.zeros((0, node_feat_dim), dtype=np.float32)
  out = np.zeros((n, len(jet_features)), dtype=np.float32)
  for j, feat in enumerate(jet_features):
    col = events[feat][i]
    for k in range(n):
      v = col[k]
      if hasattr(v, '__len__'):
        out[k, j] = float(v[0]) if len(v) > 0 else 0.0
      else:
        out[k, j] = float(v)
  if out.shape[1] < node_feat_dim:
    out = np.pad(out, ((0, 0), (0, node_feat_dim - out.shape[1])))
  elif out.shape[1] > node_feat_dim:
    out = out[:, :node_feat_dim]
  return out


def build_lepton_matrix(events, i):
  n_e = len(events['Electron.PT'][i])
  n_m = len(events['Muon.PT'][i])
  n = n_e + n_m
  if n == 0:
    return np.zeros((0, node_feat_dim), dtype=np.float32)
  out = np.zeros((n, node_feat_dim), dtype=np.float32)
  for j in range(n_e):
    out[j, 0] = float(events['Electron.PT'][i][j])
    out[j, 1] = float(events['Electron.Phi'][i][j])
    out[j, 2] = float(events['Electron.Eta'][i][j])
  for j in range(n_m):
    out[n_e + j, 0] = float(events['Muon.PT'][i][j])
    out[n_e + j, 1] = float(events['Muon.Phi'][i][j])
    out[n_e + j, 2] = float(events['Muon.Eta'][i][j])
  return out


def build_raw_jets(events, i):
  n = len(events['Jet.PT'][i])
  raw = []
  for j in range(n):
    bv = events['Jet.BTag'][i][j]
    if hasattr(bv, '__len__'):
      btag = float(bv[0]) if len(bv) > 0 else 0.0
    else:
      btag = float(bv)
    raw.append({
      'PT': float(events['Jet.PT'][i][j]),
      'Eta': float(events['Jet.Eta'][i][j]),
      'Phi': float(events['Jet.Phi'][i][j]),
      'Mass': float(events['Jet.Mass'][i][j]),
      'BTag': btag,
    })
  return raw


def build_raw_leptons(events, i):
  raw = []
  for j in range(len(events['Electron.PT'][i])):
    raw.append({'PT': float(events['Electron.PT'][i][j]),
                'Eta': float(events['Electron.Eta'][i][j]),
                'Phi': float(events['Electron.Phi'][i][j]),
                'Mass': 0.000511, 'Type': 'electron'})
  for j in range(len(events['Muon.PT'][i])):
    raw.append({'PT': float(events['Muon.PT'][i][j]),
                'Eta': float(events['Muon.Eta'][i][j]),
                'Phi': float(events['Muon.Phi'][i][j]),
                'Mass': 0.10566, 'Type': 'muon'})
  return raw


def build_graph_arrays(events, i, label):
  raw_jets = build_raw_jets(events, i)
  raw_leptons = build_raw_leptons(events, i)

  met_pt = float(events['MissingET.MET'][i])
  met_phi = float(events['MissingET.Phi'][i])
  rt1, rt2 = compute_rt(raw_jets, raw_leptons, met_pt, met_phi)

  x_jets = build_jet_matrix(events, i)
  x_leptons = build_lepton_matrix(events, i)

  eta_j = x_jets[:, 2] if x_jets.shape[0] > 0 else np.array([], dtype=np.float32)
  phi_j = x_jets[:, 1] if x_jets.shape[0] > 0 else np.array([], dtype=np.float32)
  eta_l = x_leptons[:, 2] if x_leptons.shape[0] > 0 else np.array([], dtype=np.float32)
  phi_l = x_leptons[:, 1] if x_leptons.shape[0] > 0 else np.array([], dtype=np.float32)

  eta_all = np.concatenate([eta_j, eta_l])
  phi_all = np.concatenate([phi_j, phi_l])
  if eta_all.size == 0:
    return None

  x_all = np.vstack([x_jets, x_leptons])
  N = x_all.shape[0]
  if N == 0:
    return None

  eta64 = np.asarray(eta_all, dtype=np.float64)
  phi64 = np.asarray(phi_all, dtype=np.float64)
  d_eta = eta64[:, None] - eta64[None, :]
  d_phi = (phi64[:, None] - phi64[None, :] + np.pi) % (2 * np.pi) - np.pi
  DR = np.sqrt(d_eta**2 + d_phi**2)

  iu = np.triu_indices(N, k=1)
  dr_upper = DR[iu]
  keep = dr_upper > DelR_Edge
  ii = iu[0][keep]
  jj = iu[1][keep]
  dr_kept = dr_upper[keep]
  if ii.size == 0:
    return None

  edge_index_np = np.stack([np.concatenate([ii, jj]),
                            np.concatenate([jj, ii])], axis=0).astype(np.int64)
  edge_attr_np = np.concatenate([dr_kept, dr_kept]).reshape(-1, 1).astype(np.float32)
  targets = (dr_kept >= 0.4).astype(np.float32)
  edge_target_np = np.concatenate([targets, targets]).reshape(-1, 1).astype(np.float32)

  hyperedge_list = []
  for kk in range(N):
    m = DR[kk] < DelR_Hedge
    m[kk] = True
    nb = np.nonzero(m)[0].tolist()
    if len(nb) > 1:
      hyperedge_list.append(nb)
  if not hyperedge_list:
    return None

  node_indices = np.concatenate(hyperedge_list).astype(np.int64)
  hyperedge_indices = np.repeat(np.arange(len(hyperedge_list)),
                                [len(h) for h in hyperedge_list]).astype(np.int64)
  hyperedge_index_np = np.stack([node_indices, hyperedge_indices]).astype(np.int64)

  x_met = np.zeros((1, node_feat_dim), dtype=np.float32)
  x_met[0, 0] = met_pt
  x_met[0, 1] = met_phi

  x_s_np = np.vstack([x_all, x_met]).astype(np.float32)
  u_s_np = np.array([[x_all[:, 0].sum(), x_all[:, 1].mean()]], dtype=np.float32)

  return {
    'x_s': x_s_np,
    'u_s': u_s_np,
    'edge_index': edge_index_np,
    'edge_attr_s': edge_attr_np,
    'edge_attr_t': edge_target_np,
    'edge_index_h': hyperedge_index_np,
    'label': int(label),
    'rt1': float(rt1),
    'rt2': float(rt2),
    'met_pt': float(met_pt),
    'met_phi': float(met_phi),
    'event_index': int(i),
    'num_nodes': int(x_s_np.shape[0]),
    'num_hyperedges': int(len(hyperedge_list)),
  }


def process_event(i):
  return build_graph_arrays(_WORKER_EVENTS, i, _WORKER_LABEL)


def arrays_to_data(arr):
  N = arr['num_nodes']
  num_h = arr['num_hyperedges']

  edge_index = torch.from_numpy(arr['edge_index']).long()
  edge_attr_s = torch.from_numpy(arr['edge_attr_s']).float()
  edge_attr_t = torch.from_numpy(arr['edge_attr_t']).float()
  edge_index_h = torch.from_numpy(arr['edge_index_h']).long()

  data = Data(
    x_s=torch.from_numpy(arr['x_s']),
    u_s=torch.from_numpy(arr['u_s']),
    edge_index=edge_index,
    edge_attr_s=edge_attr_s,
    edge_attr_s_batch=torch.zeros(edge_index.shape[1], dtype=torch.long),
    edge_attr_t=edge_attr_t,
    edge_index_h=edge_index_h,
    edge_index_h_batch=torch.zeros(num_h, dtype=torch.long),
    x_t=torch.tensor([arr['label']], dtype=torch.long),
    batch=torch.zeros(N, dtype=torch.long)
  )
  data.rt_observables = torch.tensor([[arr['rt1'], arr['rt2']]], dtype=torch.float32)
  data.met_pt = arr['met_pt']
  data.met_phi = arr['met_phi']
  data.event_index = arr['event_index']
  return data


def get_num_events(file_path):
  file = uproot.open(file_path)
  return len(file[tree_name]['Jet.PT'].array(library='np'))


def iter_events_from_file(file_path, label, n_procs=4, chunk_size=2000, max_events=None):
  num_events = get_num_events(file_path)
  if max_events:
    num_events = min(num_events, max_events)

  print(f"  total events in file: {num_events}")

  t0 = time.time()
  ctx = mp.get_context("spawn")
  with ctx.Pool(processes=n_procs, initializer=_init_worker, initargs=(file_path, label)) as pool:
    start = 0
    while start < num_events:
      end = min(start + chunk_size, num_events)
      results = pool.map(process_event, range(start, end), chunksize=8)
      for arr in results:
        if arr is not None:
          yield arr
      start = end
      if start % 20000 < chunk_size:
        print(f"  processed {start}/{num_events} [{time.time() - t0:.1f}s]")


def run_inference_and_save(model, file_path, output_csv, label=sig_class, threshold=0.0,
                            n_procs=4, chunk_size=2000, max_events=None,
                            save_every=50000):
  print(f"\nProcessing: {file_path}")
  print(f"Output: {output_csv}")

  tagged = []
  total_processed = 0
  total_kept = 0

  writer_header = True
  if os.path.exists(output_csv):
    os.remove(output_csv)

  def flush(force=False):
    nonlocal tagged, writer_header
    if not tagged:
      return
    df = pd.DataFrame(tagged)
    df.to_csv(output_csv, mode='a', header=writer_header, index=False, float_format='%.6f')
    writer_header = False
    tagged = []

  model.eval()
  with torch.inference_mode():
    for arr in iter_events_from_file(file_path, label, n_procs=n_procs,
                                     chunk_size=chunk_size, max_events=max_events):
      data = arrays_to_data(arr)
      batch = next(iter(DataLoader([data], batch_size=1, shuffle=False))).to(device)

      logits, _ = model(batch)
      probs = torch.softmax(logits, dim=1)
      signal_prob = float(probs[0, sig_class].cpu().item())

      total_processed += 1
      if signal_prob >= threshold:
        rt1 = float(arr['rt1'])
        rt2 = float(arr['rt2'])
        tagged.append({'amir': rt1, 'sanmay': rt2, 'prob_tttt': signal_prob})
        total_kept += 1

      if len(tagged) >= save_every:
        flush()

      if total_processed % 20000 == 0:
        print(f"  inferred {total_processed} events, tagged {total_kept}")

  flush(force=True)

  print(f"Done. Processed {total_processed}, tagged {total_kept}")
  return total_processed, total_kept


samples = {
  'tttt': {'files': ['run_01.root']},
  'ttvv': {'files': ['run_01.root']},
  'tth': {'files': ['run_01.root']},
  'ttz': {'files': ['run_01.root']},
  'ttw': {'files': ['run_01.root']},
  'ctt1': {'files': ['run_01.root', 'run_02.root']},
  'cqq1': {'files': ['run_01.root', 'run_02.root']},
  'cqt1': {'files': ['run_01.root', 'run_02.root']},
  'cqt8': {'files': ['run_01.root', 'run_02.root']},
  'reouh': {'files': ['run_01.root', 'run_02.root', 'run_03.root', 'run_04.root']},
  'imouh': {'files': ['run_01.root', 'run_02.root', 'run_03.root', 'run_04.root']},
  'ctt1cqq1': {'files': ['run_01.root']},
  'ctt1cqt1': {'files': ['run_01.root']},
  'ctt1cqt8': {'files': ['run_01.root']},
  'cqq1cqt1': {'files': ['run_01.root']},
  'cqq1cqt8': {'files': ['run_01.root']},
  'cqt1cqt8': {'files': ['run_01.root']},
  'reim': {'files': ['run_01.root']}
}

data_base_path = "/path/to/data/data"
output_base_path = "/path/where/you/want/to/save/data"
thresholds = [0.0]
N_PROCS = 4
CHUNK = 2000
SAVE_EVERY = 50000


def main(checkpoint_path):
  if not os.path.exists(checkpoint_path):
    print(f"ERROR: Checkpoint not found: {checkpoint_path}")
    return

  print(f"Loading model from {checkpoint_path}")
  model = SignalVsBkgHyPERModel.load_from_checkpoint(
    checkpoint_path, num_classes=num_class, class_weights=None, strict=False
  )
  model.eval()
  model = model.to(device)
  print(f"Model loaded on {device}")

  summary = []

  for sample_name, sample_info in samples.items():
    for root_file in sample_info['files']:
      run_number = root_file.replace('run_', '').replace('.root', '')
      input_file = os.path.join(data_base_path, "signal", sample_name, root_file)
      if not os.path.exists(input_file):
        print(f"WARNING: missing {input_file}")
        continue

      output_dir = os.path.join(output_base_path, sample_name)
      os.makedirs(output_dir, exist_ok=True)

      for thresh in thresholds:
        output_file = os.path.join(output_dir, f"run_{run_number}_thresh_{thresh}.csv")
        print("\n" + "=" * 70)
        print(f"Sample: {sample_name} | File: {root_file} | Threshold: {thresh}")
        print(f"Output: {output_file}")
        print("=" * 70)

        total, kept = run_inference_and_save(
          model=model,
          file_path=input_file,
          output_csv=output_file,
          label=sig_class,
          threshold=thresh,
          n_procs=N_PROCS,
          chunk_size=CHUNK,
          max_events=None,
          save_every=SAVE_EVERY
        )
        summary.append({
          'sample': sample_name,
          'file': root_file,
          'threshold': thresh,
          'total': total,
          'tagged': kept
        })

  print("\n" + "=" * 70)
  print("SUMMARY")
  print("=" * 70)
  for row in summary:
    rate = row['tagged'] / row['total'] * 100 if row['total'] > 0 else 0.0
    print(f"{row['sample']:<12} {row['file']:<14} thr={row['threshold']}: "
          f"{row['tagged']}/{row['total']} ({rate:.2f}%)")


if __name__ == "__main__":
  checkpoint_path = '/path/to/checkpoint/hgnn_tttt_Edge_0d1_Hedge_1d5.ckpt'
  main(checkpoint_path)
