import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from sklearn.preprocessing import RobustScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Autoencoder")

# Check for GPU availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MarketAutoencoder(nn.Module):
    """
    Denoising Autoencoder for market feature compression.

    Architecture: input_dim -> 16 -> 8 -> 4 (latent) -> 8 -> 16 -> input_dim
    """

    def __init__(self, input_dim, latent_dim=4):
        super(MarketAutoencoder, self).__init__()

        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),

            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2),

            nn.Linear(8, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.LeakyReLU(0.2),

            nn.Linear(8, 16),
            nn.LeakyReLU(0.2),

            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


def train_autoencoder(
    feature_matrix,
    epochs: int = 2500,
    lr: float = 0.002,
    noise_factor: float = 0.1,
    early_stop_patience: int = 100,
    early_stop_delta: float = 1e-6,
    latent_dim: int = 4
):
    """
    Train a denoising autoencoder for feature compression.

    Args:
        feature_matrix: DataFrame of features (coins x features)
        epochs: Maximum training epochs
        lr: Learning rate
        noise_factor: Noise multiplier for denoising
        early_stop_patience: Epochs to wait before early stopping
        early_stop_delta: Minimum improvement for early stopping
        latent_dim: Dimension of latent space

    Returns:
        Tuple of (trained model, latent space numpy array)
    """
    # Scale data
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(feature_matrix.values)

    data_tensor = torch.tensor(scaled_data, dtype=torch.float32).to(DEVICE)
    input_dim = data_tensor.shape[1]

    model = MarketAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    logger.info(f"Training Denoising Autoencoder on {DEVICE}")
    logger.info(f"Architecture: {input_dim}->16->8->{latent_dim}")

    # Early stopping setup
    best_loss = float('inf')
    patience_counter = 0

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Add noise for denoising
        noise = torch.randn_like(data_tensor) * noise_factor
        noisy_input = data_tensor + noise

        reconstructed, latent = model(noisy_input)

        # Loss against clean original
        loss = criterion(reconstructed, data_tensor)

        loss.backward()
        optimizer.step()

        current_loss = loss.item()

        # Logging
        if epoch % 500 == 0:
            logger.info(f"Epoch {epoch}: Loss = {current_loss:.6f}")

        # Early stopping check
        if current_loss < best_loss - early_stop_delta:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch} (best loss: {best_loss:.6f})")
            break

    # Get final latent space
    model.eval()
    with torch.no_grad():
        _, final_latent = model(data_tensor)

    # Move back to CPU for numpy conversion
    final_latent_np = final_latent.cpu().numpy()

    logger.info(f"Training complete. Final loss: {best_loss:.6f}")
    return model, final_latent_np