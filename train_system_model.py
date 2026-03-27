import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import os

# Configuration
DATA_FILE = "system_wide_features.csv"
WINDOW_SIZE = 15  # 15 seconds of history
HIDDEN_DIM = 64
LATENT_DIM = 16
EPOCHS = 30
BATCH_SIZE = 32

class SystemDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = torch.FloatTensor(data)
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        return self.data[idx : idx + self.window_size]

class SystemAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, window_size):
        super(SystemAutoencoder, self).__init__()
        # Encoder
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2)
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=2)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        self.window_size = window_size

    def forward(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        latent = torch.relu(self.encoder_fc(h_n[-1]))
        decoder_input = torch.relu(self.decoder_fc(latent))
        decoder_input = decoder_input.repeat(self.window_size, 1, 1).transpose(0, 1)
        out, _ = self.decoder_lstm(decoder_input)
        return self.output_layer(out)

def train():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    df = pd.read_csv(DATA_FILE)
    # Exclude non-numeric columns
    features = [c for c in df.columns if c not in ["timestamp", "unix_time"]]
    data = df[features].values
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    dataset = SystemDataset(scaled_data, WINDOW_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    input_dim = len(features)
    model = SystemAutoencoder(input_dim, HIDDEN_DIM, LATENT_DIM, WINDOW_SIZE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Training System-Wide Model with {len(dataset)} samples and {input_dim} features...")
    
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
            
    torch.save(model.state_dict(), "system_model.pth")
    np.save("system_scaler_params.npy", [scaler.mean_, scaler.scale_])
    # Save feature names for inference
    with open("feature_names.txt", "w") as f:
        f.write("\n".join(features))
    print("System model and scaler saved.")

if __name__ == "__main__":
    train()
