import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import uniform_filter1d
from scipy.signal import detrend

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             accuracy_score, recall_score, precision_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)
from matplotlib.patches import Patch

from loader.patients import PatientsCSVLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SimpleAttention(nn.Module):
    def __init__(self, window_size, hidden_dim=128, n_patient_features=10):
        super().__init__()
        self.embed = nn.Linear(window_size, hidden_dim)
        self.context = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.patient_embed = nn.Sequential(
            nn.Linear(n_patient_features, hidden_dim),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        # 3 classes: 0=Asymptomatic(<5), 1=Mild-Moderate(5-29), 2=Severe(>=30)
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3),
        )
        self.regression_head = nn.Linear(hidden_dim, 1)
        self.attention_weights = None

    def forward(self, x, patient_features):
        embedded = self.embed(x)
        scores = self.context(embedded)
        weights = F.softmax(scores, dim=1)
        self.attention_weights = weights.detach()
        weighted_sum = (embedded * weights).sum(dim=1)

        patient_embed = self.patient_embed(patient_features)
        fused = torch.cat([weighted_sum, patient_embed], dim=1)
        shared_repr = self.shared(fused)

        classification = self.classification_head(shared_repr)   # raw logits
        regression = self.regression_head(shared_repr)
        return classification, regression


def ahi_to_class(ahi):
    """0=Asymptomatic(<5), 1=Mild-Moderate(5-29), 2=Severe(>=30)"""
    if ahi < 5:
        return 0
    elif ahi < 30:
        return 1
    else:
        return 2


# ── Data loading ──────────────────────────────────────────────────────────────
samples = []
for patient_id in range(1, 41):
    for night_id in range(1, 3):
        samples.append((patient_id, night_id))
samples.remove((8, 1))
samples.remove((14, 2))

df = PatientsCSVLoader.load_dataframe('C:/Users/picul/Videos/Applied/Dataset_Full/patients.csv')

patient_features, y_regression, y_classification = [], [], []

for (p_id, n_id) in samples:
    row = df[(df['user_id'] == p_id) & (df['night_id'] == n_id)]

    attacks = row.iloc[0]['AHI']
    attacks = 0.0 if (pd.isna(attacks) or attacks == '' or attacks == 'NaN') \
              else float(str(attacks).replace(',', '.'))

    odi = row.iloc[0]['ODI']
    odi = 0 if (pd.isna(odi) or odi == '' or odi == 'NaN') else float(odi)

    age    = row.iloc[0]['age']
    sex    = 1 if row.iloc[0]['sex'] == 'M' else 0
    height = row.iloc[0]['height']
    weight = row.iloc[0]['weight']
    pulse  = row.iloc[0]['pulse']
    bp     = row.iloc[0]['BPsys/BPdia']

    patient_features.append([age, sex, height, weight, pulse,
                              float(bp.split('/')[0]), odi])
    y_regression.append(attacks)
    y_classification.append(ahi_to_class(attacks))

patient_features = np.array(patient_features)
y_regression     = np.array(y_regression,     dtype=np.float32)
y_classification = np.array(y_classification, dtype=np.int64)

print("Class distribution:", np.bincount(y_classification),
      "  [Asympt / Mild-Mod / Severe]")

X = np.load("C:/Users/picul/Videos/Applied/Dataset_Full/Embeddings/windowed_power_embeddings.npy")
print("X:", X.shape, "  y_reg:", y_regression.shape,
      "  y_cls:", y_classification.shape, "  pf:", patient_features.shape)
shift = np.load("C:/Users/picul/Videos/Applied/Dataset_Full/Embeddings/spectral_centroids.npy")
ratio = np.load("C:/Users/picul/Videos/Applied/Dataset_Full/Embeddings/power_ratios_2.npy")
print("Shift:", shift.shape, " ratio: ", ratio.shape)

total_power = X.sum(axis=-1)  # (N, num_windows)

print("Output shape:", total_power.shape)  # (N, num_windows)

# ── Split ─────────────────────────────────────────────────────────────────────
(X_train, X_test,
 y_reg_train, y_reg_test,
 y_cls_train, y_cls_test,
 pf_train, pf_test,
 ratio_train, ratio_test, total_train, total_test,
 shift_train, shift_test) = train_test_split(
    X, y_regression, y_classification, patient_features, ratio, total_power, shift,
    test_size=0.2, random_state=1
)

# ── To tensors ────────────────────────────────────────────────────────────────
X_train     = torch.FloatTensor(X_train)
X_test      = torch.FloatTensor(X_test)
y_reg_train = torch.FloatTensor(y_reg_train).reshape(-1, 1)
y_reg_test  = torch.FloatTensor(y_reg_test).reshape(-1, 1)
y_cls_train = torch.LongTensor(y_cls_train)
y_cls_test  = torch.LongTensor(y_cls_test)
pf_train    = torch.FloatTensor(pf_train)
pf_test     = torch.FloatTensor(pf_test)

# ── Normalise (compute stats on train, apply to both) ─────────────────────────
X_mean, X_std = X_train.mean(), X_train.std()
X_train = (X_train - X_mean) / X_std
X_test  = (X_test  - X_mean) / X_std

y_reg_mean, y_reg_std = y_reg_train.mean(), y_reg_train.std()
y_reg_train = (y_reg_train - y_reg_mean) / y_reg_std   # in-place replacement
y_reg_test  = (y_reg_test  - y_reg_mean) / y_reg_std   # same variable, now normalised

pf_mean, pf_std = pf_train.mean(dim=0), pf_train.std(dim=0)
pf_train = (pf_train - pf_mean) / pf_std
pf_test  = (pf_test  - pf_mean) / pf_std

# ── Move everything to device in one place ────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

X_train     = X_train.to(device)
y_cls_train = y_cls_train.to(device)
y_reg_train = y_reg_train.to(device)
pf_train    = pf_train.to(device)

X_test     = X_test.to(device)
y_cls_test = y_cls_test.to(device)
y_reg_test = y_reg_test.to(device)      # ← was never moved before; this was the bug
pf_test    = pf_test.to(device)

dataset = TensorDataset(X_train, y_cls_train, y_reg_train, pf_train)
loader  = DataLoader(dataset, batch_size=24, shuffle=True)

# ── Model ─────────────────────────────────────────────────────────────────────
window_size = 30
model = SimpleAttention(window_size, n_patient_features=7, hidden_dim=16)
print(f"Parameters: {count_parameters(model)}")
model = model.to(device)

# Inverse-frequency class weights
class_counts = np.bincount(y_cls_train.cpu().numpy(), minlength=3).astype(np.float32)
class_weights = (1.0 / (class_counts + 1e-6))
class_weights = (class_weights / class_weights.sum() * 3)
class_weights = torch.FloatTensor(class_weights).to(device)

criterion_class = nn.CrossEntropyLoss(weight=class_weights)
criterion_reg   = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
# Cosine annealing: decays smoothly to lr_min over 600 epochs, no cliff drops
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=600, eta_min=1e-5)

# ── Training loop ─────────────────────────────────────────────────────────────
losses_train, losses_test = [], []

for epoch in range(600):
    model.train()
    epoch_loss, batch_count = 0.0, 0

    for x_b, y_cls_b, y_reg_b, pf_b in loader:
        optimizer.zero_grad()

        pred_cls, pred_reg = model(x_b, pf_b)
        loss_cls = criterion_class(pred_cls, y_cls_b)

        sick_mask = y_cls_b > 0          # non-asymptomatic in this batch
        loss_reg  = criterion_reg(pred_reg[sick_mask], y_reg_b[sick_mask]) \
                    if sick_mask.sum() > 0 else torch.tensor(0.0, device=device)

        loss = loss_cls + loss_reg
        loss.backward()
        optimizer.step()

        epoch_loss  += loss.item()
        batch_count += 1

    scheduler.step()
    avg_train = epoch_loss / batch_count
    losses_train.append(avg_train)

    model.eval()
    with torch.no_grad():
        pred_cls_test, pred_reg_test = model(X_test, pf_test)

        loss_cls_test = criterion_class(pred_cls_test, y_cls_test)

        sick_mask_test = y_cls_test > 0
        loss_reg_test  = criterion_reg(pred_reg_test[sick_mask_test],
                                       y_reg_test[sick_mask_test]) \
                         if sick_mask_test.sum() > 0 else torch.tensor(0.0, device=device)

        avg_test = (loss_cls_test + loss_reg_test).item()
        losses_test.append(avg_test)

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:4d}: train {avg_train:.4f}  test {avg_test:.4f}  "
              f"lr {scheduler.get_last_lr()[0]:.2e}")

# ── Loss curve ────────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(losses_train, label='Train', color='#2196F3', linewidth=1.5)
ax.plot(losses_test,  label='Test',  color='#F44336', linewidth=1.5)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('Train / Test Loss'); ax.legend()
plt.tight_layout(); plt.show()

# ── Final predictions ─────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    pred_cls_test, pred_reg_test = model(X_test, pf_test)
    saliency = model.attention_weights.cpu()

y_cls_np      = y_cls_test.cpu().numpy()
y_reg_np      = y_reg_test.cpu().numpy()
pred_cls_np   = pred_cls_test.cpu().numpy()
pred_reg_np   = pred_reg_test.cpu().numpy()

pred_cls_labels = np.argmax(pred_cls_np, axis=1)

# De-normalise regression
pred_reg_denorm = pred_reg_np * y_reg_std.item() + y_reg_mean.item()
y_reg_denorm    = y_reg_np    * y_reg_std.item() + y_reg_mean.item()

class_names = ['Asymptomatic\n(<5)', 'Mild-Moderate\n(5–29)', 'Severe\n(≥30)']
colors      = ['#4CAF50', '#FF9800', '#F44336']

# ── Full-dataset predictions for confusion matrix ─────────────────────────────
X_full  = torch.FloatTensor((np.array(list(X)) - X_mean.item()) / X_std.item()).to(device)
pf_full = torch.FloatTensor((patient_features   - pf_mean.cpu().numpy()) / pf_std.cpu().numpy()).to(device)

# Normalise using train stats (already computed above)
X_all  = (torch.FloatTensor(X)                - X_mean)  / X_std
pf_all = (torch.FloatTensor(patient_features) - pf_mean) / pf_std
X_all  = X_all.to(device)
pf_all = pf_all.to(device)

model.eval()
with torch.no_grad():
    pred_cls_all, pred_reg_all = model(X_all, pf_all)

pred_cls_all_labels = np.argmax(pred_cls_all.cpu().numpy(), axis=1)
y_cls_all = y_classification  # original numpy array, never split

cm_full = confusion_matrix(y_cls_all, pred_cls_all_labels)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(cm_full, display_labels=class_names).plot(ax=ax, colorbar=True, cmap='Blues')
ax.set_title('Confusion Matrix — Full Dataset (train+test)', fontsize=13, fontweight='600', pad=12)
plt.tight_layout()
plt.show()

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_cls_np, pred_cls_labels)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, colorbar=True, cmap='Blues')
ax.set_title('Confusion Matrix — AHI Severity', fontsize=13, fontweight='600', pad=12)
plt.tight_layout(); plt.show()

# ── Per-class probability distributions ──────────────────────────────────────
probs = F.softmax(torch.FloatTensor(pred_cls_np), dim=1).numpy()
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for c, (ax, cname, col) in enumerate(zip(axes, class_names, colors)):
    for tc in range(3):
        mask = y_cls_np == tc
        ax.hist(probs[mask, c], bins=10, alpha=0.6,
                label=f'True {class_names[tc].split(chr(10))[0]}', density=True)
    ax.set_title(f'P(class = {cname.split(chr(10))[0]})', color=col, fontweight='600')
    ax.set_xlabel('Predicted probability'); ax.set_ylabel('Density')
    ax.legend(fontsize=7)
plt.suptitle('Predicted Class Probability Distributions', fontsize=12, fontweight='600')
plt.tight_layout(); plt.show()

# ── Regression scatter (full dataset) ────────────────────────────────────────
with torch.no_grad():
    _, pred_reg_all = model(X_all, pf_all)

pred_reg_all_denorm = pred_reg_all.cpu().numpy() * y_reg_std.item() + y_reg_mean.item()
y_reg_all_denorm = y_regression  # already in original AHI scale, no de-norm needed
pred_cls_all_colors = [colors[c] for c in pred_cls_all_labels]

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_reg_all_denorm.flatten(), pred_reg_all_denorm.flatten(),
           c=pred_cls_all_colors, alpha=0.7, edgecolors='k', linewidths=0.4, s=60)
lims = [min(y_reg_all_denorm.min(), pred_reg_all_denorm.min()) - 2,
        max(y_reg_all_denorm.max(), pred_reg_all_denorm.max()) + 2]
ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.5, label='Perfect')
ax.set_xlabel('True AHI'); ax.set_ylabel('Predicted AHI')
ax.set_title('True vs Predicted AHI — Full Dataset', fontsize=12, fontweight='600')
legend_els = [Patch(facecolor=colors[i], label=class_names[i].replace('\n', ' '))
              for i in range(3)]
legend_els.append(plt.Line2D([0], [0], linestyle='--', color='k', label='Perfect'))
ax.legend(handles=legend_els, fontsize=8)
plt.tight_layout(); plt.show()

# ── Attention saliency for non-asymptomatic samples ───────────────────────────
saliency_np = saliency.squeeze(-1).numpy()   # (batch, num_windows)
for i in range(len(y_cls_np)):
    if y_cls_np[i] == 0:
        continue

    shift_i = shift_test[i]
    ratio_i = ratio_test[i]
    total_i = total_test[i]

    sal = saliency_np[i]
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    sal = uniform_filter1d(sal, size=2)

    fig, ax = plt.subplots(figsize=(12, 4))

    shift_norm = (shift_i - shift_i.min()) / (shift_i.max() - shift_i.min())
    shift_norm = shift_norm.mean(axis=0)
    shift_detrended = detrend(shift_norm)

    ratio_norm =  (ratio_i - ratio_i.min()) / (ratio_i.max() - ratio_i.min())
    ratio_norm = ratio_norm.mean(axis=0)
    ratio_detrended = detrend(ratio_norm)


    total_norm =  (total_i - total_i.min()) / (total_i.max() - total_i.min())
    total_detrended = detrend(total_norm)

    # Plot
    # ax.plot(shift_detrended, linewidth=1, color='#fba22e', label='Spectral Centroid', alpha=0.85)
    ax.plot(total_detrended, linewidth=2, color='#e2a22e', label='Total power', alpha=0.85)

    ax.plot(sal, linewidth=2, color='#A23B72', alpha=0.85, label='Attention Saliency')
    true_lbl = class_names[y_cls_np[i]].replace('\n', ' ')
    pred_lbl = class_names[pred_cls_labels[i]].replace('\n', ' ')
    ax.set_title(
        f'Sample {i} | True: {true_lbl} (AHI={y_reg_denorm[i,0]:.1f})  '
        f'Pred: {pred_lbl} (AHI={pred_reg_denorm[i,0]:.1f})',
        fontsize=11, fontweight='600')
    ax.set_xlabel('Time Window'); ax.set_ylabel('Normalised Saliency')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(); plt.tight_layout(); plt.show()

# ── Metrics ───────────────────────────────────────────────────────────────────
mse  = mean_squared_error(y_reg_denorm, pred_reg_denorm)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_reg_denorm, pred_reg_denorm)

acc       = accuracy_score(y_cls_np, pred_cls_labels)
rec_mac   = recall_score(y_cls_np, pred_cls_labels, average='macro',    zero_division=0)
prec_mac  = precision_score(y_cls_np, pred_cls_labels, average='macro', zero_division=0)
f1_mac    = f1_score(y_cls_np, pred_cls_labels, average='macro',        zero_division=0)
f1_each   = f1_score(y_cls_np, pred_cls_labels, average=None,           zero_division=0)

print(f"\n{'='*60}")
print(f"TEST SET PERFORMANCE")
print(f"{'='*60}")
print(f"Regression (AHI — all samples):")
print(f"  MSE:  {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"\nClassification (3-class severity):")
print(f"  Accuracy:          {acc:.4f}")
print(f"  Recall    (macro): {rec_mac:.4f}")
print(f"  Precision (macro): {prec_mac:.4f}")
print(f"  F1        (macro): {f1_mac:.4f}")
for c, name in enumerate(class_names):
    print(f"  F1 {name.split(chr(10))[0]:22s}: {f1_each[c]:.4f}")
print(f"{'='*60}\n")