import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import os
from config import (
    DATA_FILE, WINDOW_SIZE, HIDDEN_DIM, LATENT_DIM, 
    EPOCHS, BATCH_SIZE, LEARNING_RATE, PATIENCE,
    MODEL_PATH, SCALER_PATH, THRESHOLDS_PATH, FEATURE_NAMES_PATH,
    FEATURES
)

class DiskDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = torch.FloatTensor(data)
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        return self.data[idx : idx + self.window_size]

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, window_size):
        super(LSTMAutoencoder, self).__init__()
        
        # Encoder
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
        self.window_size = window_size

    def forward(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        latent = torch.relu(self.encoder_fc(h_n[-1]))
        
        decoder_input = torch.relu(self.decoder_fc(latent))
        decoder_input = decoder_input.repeat(self.window_size, 1, 1).transpose(0, 1)
        
        out, _ = self.decoder_lstm(decoder_input)
        return self.output_layer(out)

def train_model():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Run collector first.")
        return

    # 1. Load and Preprocess Data
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} samples.")
    
    # Select features that are present in the dataframe
    available_features = [f for f in FEATURES if f in df.columns]
    print(f"Using features: {available_features}")
    data = df[available_features].values
    
    # Save the ACTUAL features used for inference
    with open(FEATURE_NAMES_PATH, "w") as f:
        f.write("\n".join(available_features))
    
    # Scale data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Chronological split (80/20)
    split = int(len(scaled_data) * 0.8)
    train_data_raw, val_data_raw = scaled_data[:split], scaled_data[split:]
    
    # 2. Prepare DataLoaders
    train_dataset = DiskDataset(train_data_raw, WINDOW_SIZE)
    val_dataset = DiskDataset(val_data_raw, WINDOW_SIZE)
    
    # CRITICAL: shuffle=False for time-series
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Model Setup
    input_dim = len(available_features)
    model = LSTMAutoencoder(input_dim, HIDDEN_DIM, LATENT_DIM, WINDOW_SIZE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Training Loop with Early Stopping
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples...")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                output = model(batch)
                loss = criterion(output, batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
        # Early Stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE and (epoch + 1) >= MIN_EPOCHS:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
    # 5. Compute Thresholds from Training Distribution
    print("Computing anomaly thresholds...")
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    errors = []
    with torch.no_grad():
        # Using full training set for thresholding
        full_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
        for batch in full_loader:
            output = model(batch)
            # Calculate MSE per window
            mse_per_window = torch.mean((output - batch)**2, dim=[1, 2])
            errors.extend(mse_per_window.cpu().numpy())
    
    errors = np.array(errors)
    threshold_warn = np.percentile(errors, 95)
    threshold_crit = np.percentile(errors, 99)
    
    np.save(THRESHOLDS_PATH, np.array([threshold_warn, threshold_crit]))
    print(f"Thresholds saved: WARN={threshold_warn:.6f}, CRIT={threshold_crit:.6f}")
    
    # 6. Save Scaler Params
    np.save(SCALER_PATH, np.array([scaler.mean_, scaler.scale_]))
    print("Model and scaler saved.")

if __name__ == "__main__":
    train_model()
