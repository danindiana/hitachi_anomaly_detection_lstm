import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import os

# Configuration
DATA_FILE = "disk_features.csv"
WINDOW_SIZE = 10  # Number of past seconds to look at
HIDDEN_DIM = 32
LATENT_DIM = 8
EPOCHS = 20
BATCH_SIZE = 16

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
        # x shape: (batch, window_size, input_dim)
        _, (h_n, _) = self.encoder_lstm(x)
        # Use the last hidden state
        latent = torch.relu(self.encoder_fc(h_n[-1]))
        
        # Prepare for decoder
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
    
    # Select features (excluding non-numeric/constant time columns)
    features = ["read_kb_s", "write_kb_s", "avg_latency_ms", "io_utilization_pct", "temperature"]
    data = df[features].values
    
    # Scale data (Standardization is crucial for LSTMs)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # 2. Prepare DataLoaders
    dataset = DiskDataset(scaled_data, WINDOW_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. Model Setup
    input_dim = len(features)
    model = LSTMAutoencoder(input_dim, HIDDEN_DIM, LATENT_DIM, WINDOW_SIZE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Starting training with {len(dataset)} samples...")
    
    # 4. Training Loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(dataloader):.4f}")
            
    # 5. Save Model and Scaler (for inference)
    torch.save(model.state_dict(), "disk_model.pth")
    # Save the scaler mean/std to a file or pickle for real-time use
    np.save("scaler_params.npy", [scaler.mean_, scaler.scale_])
    print("Model and scaler saved.")

if __name__ == "__main__":
    train_model()
