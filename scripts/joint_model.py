import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import spearmanr
from scipy.signal import detrend
from scipy.ndimage import uniform_filter1d

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from loader.patients import PatientsCSVLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def count_parameters(model):
    """Count total number of trainable parameters."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total


class TransformerAttention(nn.Module):
    def __init__(self, window_size, hidden_dim=128, n_patient_features=10, n_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        # Embed windows to hidden dimension
        self.embed = nn.Linear(window_size, hidden_dim)

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=0.3,
            batch_first=True
        )

        # Feed-forward network for transformer
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Layer norms
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Patient embedding
        self.patient_embed = nn.Sequential(
            nn.Linear(n_patient_features, hidden_dim),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Fusion and shared representation
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )

        # Task heads
        self.classification_head = nn.Linear(hidden_dim, 1)
        self.regression_head = nn.Linear(hidden_dim, 1)

        self.attention_weights = None

    def forward(self, x, patient_features):
        # x shape: (batch, num_windows, window_size)
        embedded = self.embed(x)  # (batch, num_windows, hidden_dim)

        # Self-attention with residual and layer norm
        attn_out, attn_weights = self.attention(embedded, embedded, embedded)
        self.attention_weights = attn_weights.detach()
        embedded = self.ln1(embedded + attn_out)

        # Feed-forward with residual and layer norm
        ff_out = self.ff(embedded)
        embedded = self.ln2(embedded + ff_out)

        # Pool: mean over windows
        pooled = embedded.mean(dim=1)  # (batch, hidden_dim)

        # Patient embedding
        patient_embed = self.patient_embed(patient_features)

        # Fusion
        fused = torch.cat([pooled, patient_embed], dim=1)
        shared_repr = self.shared(fused)

        # Task heads
        classification = torch.sigmoid(self.classification_head(shared_repr))
        regression = self.regression_head(shared_repr)

        return classification, regression

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

        # self.classification_head = nn.Linear(hidden_dim, 1)
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
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

        classification = torch.sigmoid(self.classification_head(shared_repr))
        regression = self.regression_head(shared_repr)

        return classification, regression


class DualBranchRegression(nn.Module):
    def __init__(self, embedding_dim, n_patient_features=7):
        super().__init__()

        # EEG embedding branch
        self.eeg_branch = nn.Sequential(
            nn.Linear(embedding_dim, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 5),
            nn.LeakyReLU(),
        )

        # Patient features branch
        self.patient_branch = nn.Sequential(
            nn.Linear(n_patient_features, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 5),
            nn.LeakyReLU(),
        )

        # Fusion and output
        self.fusion = nn.Sequential(
            nn.Linear(10, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, eeg_embedding, patient_features):
        eeg_out = self.eeg_branch(eeg_embedding)
        patient_out = self.patient_branch(patient_features)
        fused = torch.cat([eeg_out, patient_out], dim=1)
        output = self.fusion(fused)
        return output



# Filling the samples with the couples of patient and night id
samples = []
for patient_id in range(1, 40 + 1):
    for night_id in range(1, 2+1):
        samples.append((patient_id, night_id))

# Removing problematic samples (Totally not understandable)
samples.remove((8, 1))
samples.remove((14, 2))

df = PatientsCSVLoader.load_dataframe('C:/Users/picul/Videos/Applied/Dataset_Full/patients.csv')
# shape: (n_epochs, n_channels * 5)

# Extract physiological features
patient_features = []
y_regression = []
y_classification = []

for (p_id, n_id) in samples:
    row = df[(df['user_id'] == p_id) & (df['night_id'] == n_id)]

    attacks = row.iloc[0]['AHI']
    if pd.isna(attacks) or attacks == '' or attacks == 'NaN':
        attacks = 0
    else:
        attacks = float(attacks.replace(',', '.'))

    odi = row.iloc[0]['ODI']
    if pd.isna(odi) or odi == '' or odi == 'NaN':
        odi = 0

    age = row.iloc[0]['age']
    sex = 1 if row.iloc[0]['sex'] == 'M' else 0
    height = row.iloc[0]['height']
    weight = row.iloc[0]['weight']
    pulse = row.iloc[0]['pulse']
    bp = row.iloc[0]['BPsys/BPdia']

    patient_features.append([age, sex, height, weight, pulse, float(bp.split('/')[0]), odi])
    y_regression.append(attacks)
    y_classification.append(1 if attacks > 0 else 0)

patient_features = np.array(patient_features)
y_regression = np.array(y_regression)
y_classification = np.array(y_classification)

X = np.load("C:/Users/picul/Videos/Applied/Dataset_Full/Embeddings/windowed_power_embeddings.npy")
shift = np.load("C:/Users/picul/Videos/Applied/Dataset_Full/Embeddings/spectral_centroids.npy")
# shift = np.load("C:/Users/picul/Videos/Applied/Dataset_Full/Embeddings/spectral_centroids.npy")

print(X.shape)

print(y_regression.shape)
print(y_classification.shape)
print(patient_features.shape)
print(shift.shape)

X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test, pf_train, pf_test, shift_train, shift_test = train_test_split(
    X, y_regression, y_classification, patient_features, shift, test_size=0.2, random_state=1
)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_reg_train = torch.FloatTensor(y_reg_train).reshape(-1, 1)
y_reg_test = torch.FloatTensor(y_reg_test).reshape(-1, 1)
y_class_train = torch.FloatTensor(y_class_train).reshape(-1, 1)
y_class_test = torch.FloatTensor(y_class_test).reshape(-1, 1)
pf_train = torch.FloatTensor(pf_train)
pf_test = torch.FloatTensor(pf_test)

X_train_mean = X_train.mean()
X_train_std = X_train.std()
X_train = (X_train - X_train_mean) / X_train_std
X_test = (X_test - X_train_mean) / X_train_std

y_reg_train_mean = y_reg_train.mean()
y_reg_train_std = y_reg_train.std()
y_reg_train = (y_reg_train - y_reg_train_mean) / y_reg_train_std
y_reg_test = (y_reg_test - y_reg_train_mean) / y_reg_train_std

pf_train_mean = pf_train.mean(dim=0)
pf_train_std = pf_train.std(dim=0)
pf_train = (pf_train - pf_train_mean) / pf_train_std
pf_test = (pf_test - pf_train_mean) / pf_train_std

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train = X_train.to(device)
y_class_train = y_class_train.to(device)
y_reg_train = y_reg_train.to(device)
pf_train = pf_train.to(device)

X_test = X_test.to(device)
y_class_test = y_class_test.to(device)
y_reg_test = y_reg_test.to(device)
pf_test = pf_test.to(device)

dataset = TensorDataset(X_train, y_class_train, y_reg_train, pf_train)
loader = DataLoader(dataset, batch_size=24, shuffle=True)

window_size = 30

#
#
#
#
#
#
model = SimpleAttention(window_size, n_patient_features=7, hidden_dim=16)
print(count_parameters(model))

# model = TransformerAttention(window_size, n_patient_features=7, hidden_dim=16)
# print(count_parameters(model))
model = model.to(device)  # <-- ADD THIS LINE

criterion_class = nn.BCELoss()
criterion_reg = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00018, weight_decay=0.5e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.15)

import torch

# Check if CUDA is available
print(torch.cuda.is_available())

# Check current device
print(torch.cuda.current_device())

# Get device name
print(torch.cuda.get_device_name(0))

# More detailed info
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"Current device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

losses_train = []
losses_test = []
for epoch in range(600):
    epoch_loss = 0
    batch_count = 0

    for x_batch, y_class_batch, y_reg_batch, patient_features_batch in loader:
        optimizer.zero_grad()

        y_pred_class, y_pred_reg = model(x_batch, patient_features_batch)

        loss_class = criterion_class(y_pred_class, y_class_batch)

        sick_mask = y_class_batch == 1
        if sick_mask.sum() > 0:
            loss_reg = criterion_reg(y_pred_reg[sick_mask], y_reg_batch[sick_mask])
        else:
            loss_reg = 0

        loss = loss_class + loss_reg

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batch_count += 1

    avg_train_loss = epoch_loss / batch_count
    losses_train.append(avg_train_loss)

    with torch.no_grad():
        y_pred_class_test, y_pred_reg_test = model(X_test.to(device), pf_test.to(device))

        test_loss_class = criterion_class(y_pred_class_test, y_class_test)

        sick_mask_test = y_class_test == 1
        if sick_mask_test.sum() > 0:
            test_loss_reg = criterion_reg(y_pred_reg_test[sick_mask_test], y_reg_test[sick_mask_test])
        else:
            test_loss_reg = 0

        avg_test_loss = test_loss_class + test_loss_reg
        losses_test.append(avg_test_loss.item())

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}: train {avg_train_loss:.4f}, test {avg_test_loss:.4f}")

plt.plot(losses_train, label="Train")
plt.plot(losses_test, label="Test")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

with torch.no_grad():
    y_pred_class_test, y_pred_reg_test = model(X_test, pf_test)
    saliency = model.attention_weights.cpu()
num_windows = saliency.size(1)
y_class_test_np = y_class_test.cpu().numpy()
y_reg_test_np = y_reg_test.cpu().numpy()
y_pred_class_np = y_pred_class_test.cpu().numpy()
y_pred_reg_np = y_pred_reg_test.cpu().numpy()

y_pred_reg_denorm = y_pred_reg_np * y_reg_train_std.numpy() + y_reg_train_mean.numpy()
y_reg_test_denorm = y_reg_test_np * y_reg_train_std.numpy() + y_reg_train_mean.numpy()



# Reshape: (batch*num_heads, num_windows, num_windows) -> (batch, num_heads, num_windows, num_windows)
batch_size = len(X_test)
n_heads = model.n_heads
print("SIZE: ", saliency.size())


# Average across rows to get saliency per window
saliency = saliency.mean(dim=1).cpu().numpy()  # (batch, num_windows)


for i in range(len(y_class_test_np)):
    is_sick = y_class_test_np[i, 0] > 0.5
    pred_sick = y_pred_class_np[i, 0] > 0.5


    shift_i = shift_test[i]
    print(f"Sample {i}: Actual={'Sick' if is_sick else 'Healthy'}, Predicted={'Sick' if pred_sick else 'Healthy'}")

    if is_sick:

        actual_attacksd = y_reg_test_denorm[i, 0]
        pred_attacksd = y_pred_reg_denorm[i, 0]
        actual_attacks = y_reg_test_np[i, 0]
        pred_attacks = y_pred_reg_np[i, 0]
        print(f"  Actual attacks: {actual_attacks:.4f}, Predicted attacks: {pred_attacks:.4f}")
        print(f"  Actual attacks: {actual_attacksd:.4f}, Predicted attacks: {pred_attacksd:.4f}")

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(12, 5))

        # Normalize and smooth
        shift_norm = (shift_i - shift_i.min()) / (shift_i.max() - shift_i.min())
        saliency_norm = (saliency[i] - saliency[i].min()) / (saliency[i].max() - saliency[i].min())
        saliency_norm = uniform_filter1d(saliency_norm, size=2)

        shift_norm = shift_norm.mean(axis=0)
        shift_detrended = detrend(shift_norm)

        correlation, p_value = spearmanr(shift_detrended, saliency_norm)

        # Plot
        ax.plot(shift_detrended, linewidth=2.5, color='#aba92e', label='Power Spectral Centroid', alpha=0.85)
        ax.plot(saliency_norm, linewidth=2.5, color='#A23B72', label='Neural Network Saliency', alpha=0.85)

        # Styling
        ax.set_xlabel('Time Window', fontsize=11, fontweight='normal')
        ax.set_ylabel('Normalized Activity', fontsize=11, fontweight='normal')
        ax.set_title(
            f'Sample {i}: Spectral Centroid vs Saliency | AHI: {actual_attacks:.2f} > {pred_attacks:.2f})',
            fontsize=12, fontweight='600', pad=15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(loc='upper left', framealpha=0.95, frameon=True, edgecolor='#cccccc')

        plt.tight_layout()
        plt.show()

        # print(f"Correlation: {correlation:.4f}, p-value: {p_value:.2e}")


from sklearn.metrics import mean_squared_error, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

# MSE on denormalized regression predictions
mse = mean_squared_error(y_reg_test_denorm, y_pred_reg_denorm)
rmse = np.sqrt(mse)

# Binary classification metrics
y_class_test_binary = (y_class_test_np > 0.5).astype(int).flatten()
y_pred_class_binary = (y_pred_class_np > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_class_test_binary, y_pred_class_binary)
recall = recall_score(y_class_test_binary, y_pred_class_binary, zero_division=0)
precision = precision_score(y_class_test_binary, y_pred_class_binary, zero_division=0)
f1 = f1_score(y_class_test_binary, y_pred_class_binary, zero_division=0)
auc = roc_auc_score(y_class_test_binary, y_pred_class_np.flatten())

print(f"\n{'='*60}")
print(f"TEST SET PERFORMANCE")
print(f"{'='*60}")
print(f"Regression (AHI):")
print(f"  MSE:  {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"\nClassification (Apnea Yes/No):")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  ROC-AUC:   {auc:.4f}")
print(f"{'='*60}\n")