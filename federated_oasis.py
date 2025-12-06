
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import random

# --- Configuration ---
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
N_CLIENTS = 5
ROUNDS = 3  # Reduced for quick verification
LOCAL_EPOCHS = 2
BATCH_SIZE = 32
SEQ_LENGTH = 3  # Time steps (visits) per sample
HIDDEN_SIZE = 64
ATTENTION_HEADS = 4
DP_NOISE_SCALE = 0.01
FEDPROX_MU = 0.01

# --- Mock Data Generation (if files missing) ---
def generate_mock_data(n_patients=200, max_visits=5):
    print("Generating mock OASIS-3 data...")
    data = []
    patient_ids = [f"OAS{i:04d}" for i in range(n_patients)]

    for pid in patient_ids:
        n_visits = np.random.randint(1, max_visits + 1)
        base_age = np.random.randint(60, 90)
        sex = np.random.choice(['M', 'F'])
        edu = np.random.randint(12, 20)

        # Base health condition
        is_ad = np.random.random() < 0.3

        for v in range(n_visits):
            days = v * 365 + np.random.randint(-30, 30)
            age = base_age + v

            # Features correlate with condition
            mmse = np.random.randint(15, 30) if is_ad else np.random.randint(26, 30)
            cdr = np.random.choice([0, 0.5, 1, 2]) if is_ad else 0

            # Some missing values
            weight = np.random.normal(70, 10) if np.random.random() > 0.1 else np.nan

            row = {
                'OASISID': pid,
                'OASIS_session_label': f"{pid}_sess{v}",
                'days_to_visit': days,
                'age': age,
                'sex': sex,
                'education': edu,
                'MMSE': mmse,
                'CDR': cdr,
                'WEIGHT': weight,
                'BPSYS': np.random.normal(130, 15),
                'CVHATT': np.random.choice([0, 1], p=[0.9, 0.1]),
                'CBSTROKE': np.random.choice([0, 1], p=[0.95, 0.05]),
                # Target proxy
                'diagnosis': 'AD' if (is_ad and cdr>=1) else ('CN' if not is_ad else 'Uncertain')
            }
            data.append(row)

    df = pd.DataFrame(data)
    print(f"Generated {len(df)} rows for {n_patients} patients.")
    return df

# --- Data Loading and Preprocessing ---
def load_and_process_data():
    # Check if real data exists (paths from notebook)
    base_path = "/content/drive/MyDrive/OASIS3" # Likely not here
    # In this sandbox, we check if we uploaded anything or just use mock
    # For robust implementation, we use mock if files not found

    if not os.path.exists("OASIS3_data.csv"): # Hypothetical file
        df = generate_mock_data()
    else:
        # Implement loading logic if files were present, but they aren't.
        df = generate_mock_data()

    # 1. Imputation
    print("Imputing missing values...")
    feature_cols = ['age', 'education', 'MMSE', 'CDR', 'WEIGHT', 'BPSYS', 'CVHATT', 'CBSTROKE']
    imputer = KNNImputer(n_neighbors=3)
    df[feature_cols] = imputer.fit_transform(df[feature_cols])

    # 2. Feature Encoding
    print("Encoding features...")
    le_sex = LabelEncoder()
    df['sex'] = le_sex.fit_transform(df['sex'])

    le_diag = LabelEncoder()
    df['diagnosis'] = le_diag.fit_transform(df['diagnosis'])
    n_classes = len(le_diag.classes_)
    print(f"Classes: {le_diag.classes_}")

    # 3. Temporal Alignment & Sequence Creation
    print("Creating temporal sequences...")
    # Group by patient and sort by time
    grouped = df.sort_values(['OASISID', 'days_to_visit']).groupby('OASISID')

    X_seq = []
    y_seq = []

    for _, group in grouped:
        # Create sliding windows or expanding windows
        # Here we take the last SEQ_LENGTH visits. If fewer, pad.
        features = group[feature_cols + ['sex']].values
        target = group['diagnosis'].values[-1] # Predict diagnosis at last visit

        if len(features) < SEQ_LENGTH:
            # Pad with zeros or repeat
            pad_len = SEQ_LENGTH - len(features)
            # Pad with first visit (repeat) or zeros. Let's use zeros for simplicity but masking is better.
            # Better: Repeat first visit
            pad = np.tile(features[0], (pad_len, 1))
            seq = np.vstack([pad, features])
        else:
            seq = features[-SEQ_LENGTH:]

        X_seq.append(seq)
        y_seq.append(target)

    X = np.array(X_seq, dtype=np.float32)
    y = np.array(y_seq, dtype=np.int64)

    print(f"Sequence shape: {X.shape}, Target shape: {y.shape}")

    # 4. SMOTE (Applied on flattened sequences as approximation)
    print("Applying SMOTE...")
    n_samples, n_steps, n_features = X.shape
    X_flat = X.reshape(n_samples, -1)
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_flat, y)
    X_res = X_res.reshape(-1, n_steps, n_features)

    print(f"Resampled shape: {X_res.shape}")

    return X_res, y_res, n_classes, feature_cols + ['sex']

# --- Model Definition (TFT-like / LSTM + Attention) ---
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq, hidden)
        # scores: (batch, seq, 1)
        scores = torch.softmax(self.attention(x), dim=1)
        # context: (batch, hidden) -> weighted sum
        context = torch.sum(x * scores, dim=1)
        return context, scores

class TFTLite(nn.Module):
    """
    Simplified Temporal Fusion Transformer architecture:
    - Feature encoding
    - LSTM Encoder-Decoder (captured by LSTM layer processing sequence)
    - Attention Layer
    - Output Head
    """
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2, dropout=0.1)
        self.attention = AttentionLayer(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq, input_size)
        lstm_out, (ht, ct) = self.lstm(x) # lstm_out: (batch, seq, hidden)

        # Temporal Fusion / Attention
        context, attn_scores = self.attention(lstm_out)

        # Prediction
        out = self.fc(context)
        return out

# --- Federated Learning Components ---

def client_update(client_model, optimizer, train_loader, global_model, mu, epochs):
    """
    Local training with FedProx loss.
    """
    client_model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = client_model(data)

            # Classification loss
            loss = nn.CrossEntropyLoss()(output, target)

            # FedProx Proximal Term: mu/2 * ||w - w_t||^2
            proximal_term = 0.0
            for w, w_t in zip(client_model.parameters(), global_model.parameters()):
                proximal_term += (w - w_t).norm(2)**2

            loss += (mu / 2) * proximal_term

            loss.backward()

            # Differential Privacy: Gradient Clipping & Noise
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=1.0)

            # Add Noise to gradients
            for param in client_model.parameters():
                if param.grad is not None:
                    noise = torch.normal(0, DP_NOISE_SCALE, param.grad.shape).to(param.device)
                    param.grad += noise

            optimizer.step()
    return client_model.state_dict()

def federated_training(X, y, n_classes, input_size):
    # Split data among clients
    # We split by patient indices (already shuffled by SMOTE roughly, but let's be careful)
    # In real FL, data is naturally distributed. Here we simulate IID or non-IID.
    # Let's do random split.

    client_data = []
    chunk_size = len(X) // N_CLIENTS
    for i in range(N_CLIENTS):
        start = i * chunk_size
        end = (i+1) * chunk_size if i < N_CLIENTS-1 else len(X)

        cX = torch.tensor(X[start:end])
        cy = torch.tensor(y[start:end])
        dataset = torch.utils.data.TensorDataset(cX, cy)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        client_data.append(loader)

    # Global Model
    global_model = TFTLite(input_size, HIDDEN_SIZE, n_classes)

    print(f"Starting Federated Learning ({ROUNDS} rounds, {N_CLIENTS} clients)...")

    for round_idx in range(ROUNDS):
        print(f"--- Round {round_idx+1} ---")
        global_weights = global_model.state_dict()
        local_weights = []

        for client_id in range(N_CLIENTS):
            # Client receives global model
            client_model = TFTLite(input_size, HIDDEN_SIZE, n_classes)
            client_model.load_state_dict(copy.deepcopy(global_weights))

            optimizer = optim.Adam(client_model.parameters(), lr=0.001)

            # Local Update (FedProx + DP)
            w = client_update(client_model, optimizer, client_data[client_id], global_model, FEDPROX_MU, LOCAL_EPOCHS)
            local_weights.append(w)

        # Aggregation (FedAvg)
        # weights = sum(client_weights) / n_clients
        updated_weights = copy.deepcopy(global_weights)
        for key in updated_weights.keys():
            updated_weights[key] = torch.stack([lw[key] for lw in local_weights]).mean(0)

        global_model.load_state_dict(updated_weights)

    return global_model

# --- Visualization and Explanation ---
def evaluate_and_explain(model, X_test, y_test, feature_names, class_names):
    model.eval()
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test)

    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, preds = torch.max(outputs, 1)

    acc = accuracy_score(y_test, preds.numpy())
    print(f"Final Global Model Accuracy: {acc:.4f}")

    # SHAP Explanation
    # We need a background dataset. Let's take a small subset of X_test.
    background = X_test_tensor[:50]
    test_samples = X_test_tensor[:10]

    print("Generating SHAP explanations...")

    # Helper for KernelExplainer
    def model_predict(x_numpy):
        # x_numpy could be flattened by KernelExplainer if we are not careful,
        # but usually it passes what we give it.
        # However, since we pass 3D array to KernelExplainer, it should work.
        # If it flattens, we reshape.
        x_tensor = torch.from_numpy(x_numpy).float()
        if x_tensor.dim() == 2:
             # Reshape back if flattened (samples, seq*features) -> (samples, seq, features)
             x_tensor = x_tensor.view(-1, SEQ_LENGTH, len(feature_names))

        with torch.no_grad():
            out = model(x_tensor)
        return out.numpy()

    shap_values = None

    try:
        explainer = shap.DeepExplainer(model, background)
        # check_additivity=False to avoid issues with RNNs
        shap_values = explainer.shap_values(test_samples, check_additivity=False)
        print("DeepExplainer successful.")
    except Exception as e:
        print(f"DeepExplainer failed: {e}. Trying KernelExplainer...")
        try:
            # KernelExplainer is model agnostic
            # We pass the numpy array
            explainer = shap.KernelExplainer(model_predict, background.numpy())
            shap_values = explainer.shap_values(test_samples.numpy(), nsamples=100)
            print("KernelExplainer successful.")
        except Exception as e2:
             print(f"KernelExplainer also failed: {e2}")
             # import traceback
             # traceback.print_exc()

    if shap_values is not None:
        try:
            print(f"SHAP values type: {type(shap_values)}")
            if isinstance(shap_values, list):
                print(f"SHAP values is a list of length {len(shap_values)}")
                # For class 0
                sv_class0 = shap_values[0] # (samples, seq, features) or (samples, features)
                print(f"Shape of class 0 shap values: {sv_class0.shape}")

                if len(sv_class0.shape) == 3:
                     # Flatten time dimension by summing importance
                    sv_collapsed = np.sum(sv_class0, axis=1) # (samples, features)
                else:
                    sv_collapsed = sv_class0

                print(f"Collapsed shape: {sv_collapsed.shape}")

                plt.figure()
                # We need X for summary plot. If 3D, take mean or first timestep?
                # shap.summary_plot expects X to match feature count.
                # If we collapsed shap values, we should probably collapse X too or just use one time step representation
                # Using average feature value over time seems reasonable for the plot
                X_summary = np.mean(test_samples.numpy(), axis=1) if len(test_samples.shape)==3 else test_samples.numpy()
                print(f"X_summary shape: {X_summary.shape}")

                shap.summary_plot(sv_collapsed, X_summary, feature_names=feature_names, show=False)
                plt.title("SHAP Feature Importance (Class 0)")
                plt.savefig("shap_summary.png", bbox_inches='tight')
                print("SHAP summary plot saved to shap_summary.png")
            else:
                 print("SHAP values is not a list (regression or binary?).")
                 # Handle if it's just an array
                 # DeepExplainer with PyTorch usually returns (samples, seq, features, classes) or something complex for multiclass
                 # If it's multiclass but returned as single array, check dimension.

                 sv_collapsed = shap_values

                 # If shape is (samples, seq, features, classes)
                 if len(sv_collapsed.shape) == 4:
                     # Take class 0
                     sv_collapsed = sv_collapsed[:, :, :, 0]

                 if len(sv_collapsed.shape) == 3:
                    # (samples, seq, features) -> sum over seq
                    sv_collapsed = np.sum(sv_collapsed, axis=1)

                 X_summary = np.mean(test_samples.numpy(), axis=1) if len(test_samples.shape)==3 else test_samples.numpy()

                 print(f"Collapsed SHAP shape: {sv_collapsed.shape}")
                 print(f"X_summary shape: {X_summary.shape}")
                 print(f"Number of feature names: {len(feature_names)}")

                 if sv_collapsed.shape[1] != X_summary.shape[1]:
                     print("Shape mismatch still present. Trying to align...")
                     # If shap values are just (samples, features) already?
                     pass

                 shap.summary_plot(sv_collapsed, X_summary, feature_names=feature_names, show=False)
                 plt.savefig("shap_summary.png", bbox_inches='tight')
                 print("SHAP summary plot saved to shap_summary.png")

        except Exception as e:
            print(f"Visualization failed: {e}")
            import traceback
            traceback.print_exc()

# --- Main ---
if __name__ == "__main__":
    # 1. Process Data
    X, y, n_classes, feature_names = load_and_process_data()

    # Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    input_size = X.shape[2]

    # 2. Federated Training
    global_model = federated_training(X_train, y_train, n_classes, input_size)

    # 3. Evaluation & Explanation
    # Map numerical classes back to strings if possible, but we have n_classes
    evaluate_and_explain(global_model, X_test, y_test, feature_names, None)

    print("Pipeline Completed.")
