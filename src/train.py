from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.logger_config import setup_logger

logger = setup_logger("train")


class LSTMForecaster(nn.Module):
    """
    Arquitetura de Rede Neural para Previsão de Séries Temporais (Nike NKE).
    Combina camadas Bidirecionais para captura de contexto global e
    camadas Lineares para regressão do retorno financeiro.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size_1: int = 64,
        hidden_size_2: int = 32,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_size = input_size

        # Camada 1: Bidirecional - captura padrões em ambas as direções da série
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        # Camada 2: Unidirecional - consolida o output da camada anterior
        self.lstm2 = nn.LSTM(
            input_size=hidden_size_1 * 2,  # *2 devido à bidirecionalidade da camada 1
            hidden_size=hidden_size_2,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout2 = nn.Dropout(dropout)

        # Cabeça de Regressão
        self.fc1 = nn.Linear(hidden_size_2, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        # Pegamos apenas o último estado da sequência (Last Time Step)
        x = x[:, -1, :]
        x = self.dropout2(x)

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.output(x)
        return x.squeeze(-1)


@dataclass
class TrainingHistory:
    history: Dict[str, List[float]]


class EarlyStopping:
    """Interrompe o treino se a perda de validação parar de melhorar."""

    def __init__(self, patience: int = 12, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state_dict = None
        self.should_stop = False

    def step(self, val_loss: float, model: nn.Module):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Salva o melhor estado do modelo
            self.best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def get_device() -> torch.device:
    """Detecta se há GPU disponível, caso contrário usa CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_lstm_model(input_shape: Tuple[int, int]) -> LSTMForecaster:
    _, n_features = input_shape
    return LSTMForecaster(input_size=n_features)


def create_dataloader(X, y, batch_size: int, shuffle: bool) -> DataLoader:
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_model(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def predict(
    model: nn.Module, X, batch_size: int = 256, device: torch.device | None = None
) -> np.ndarray:
    """Realiza inferência (usado tanto no treino quanto na FastAPI)."""
    device = device or get_device()
    model = model.to(device)
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataloader = DataLoader(X_tensor, batch_size=batch_size, shuffle=False)

    outputs = []
    with torch.no_grad():
        for X_batch in dataloader:
            X_batch = X_batch.to(device)
            batch_preds = model(X_batch).detach().cpu().numpy()
            outputs.append(batch_preds)

    return (
        np.concatenate(outputs).astype(np.float32)
        if outputs
        else np.array([], dtype=np.float32)
    )


def train_model(
    model: nn.Module,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs: int = 80,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    patience: int = 12,
) -> Tuple[nn.Module, TrainingHistory]:
    device = get_device()
    model = model.to(device)

    train_loader = create_dataloader(X_train, y_train, batch_size, shuffle=False)
    val_loader = create_dataloader(X_val, y_val, batch_size, shuffle=False)

    # HuberLoss é mais robusto a outliers financeiros que MSE pura
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Redutor de LR ajuda a encontrar o mínimo global em séries ruidosas
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6
    )
    early_stopper = EarlyStopping(patience=patience)

    history = {"loss": [], "val_loss": []}
    logger.info("Iniciando treinamento em: %s", device)

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = evaluate_model(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                "Epoch %s/%s | Train Loss: %.6f | Val Loss: %.6f",
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
            )

        early_stopper.step(val_loss, model)
        if early_stopper.should_stop:
            logger.info("Early stopping acionado na epoch %s", epoch + 1)
            break

    if early_stopper.best_state_dict:
        model.load_state_dict(early_stopper.best_state_dict)

    return model, TrainingHistory(history=history)
