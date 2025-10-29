import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class LogisticRegression:
    
    def __init__(self, input_dim: int, learning_rate: float = 0.01, max_epochs: int = 1000,
                 tolerance: float = 1e-6):
      
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        
        # Model parameters
        self.weights = nn.Parameter(torch.randn(input_dim, 1, dtype=torch.float32, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float32, requires_grad=True))
        
        # Optimizer
        self.optimizer = optim.SGD([self.weights, self.bias], lr=self.learning_rate)
        self.criterion = nn.BCELoss()  # Binary cross-entropy
        
        # Training history
        self.loss_history = []
        self.fitted = False
    
    def sigmoid(self, z: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + torch.exp(-z))
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:

        return self.sigmoid(X @ self.weights + self.bias)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y.reshape(-1,1), dtype=torch.float32)
        
        prev_loss = float('inf')
        for epoch in range(self.max_epochs):
            self.optimizer.zero_grad()
            
            y_pred = self.forward(X_tensor)
            loss = self.criterion(y_pred, y_tensor)
            
            loss.backward()
            self.optimizer.step()
            
            current_loss = loss.item()
            self.loss_history.append(current_loss)
            
            if abs(prev_loss - current_loss) < self.tolerance:
                print(f"Converged after {epoch+1} epochs")
                break
            prev_loss = current_loss
        
        self.fitted = True
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
 
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            probs = self.forward(X_tensor)
        return probs.numpy().flatten()
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
     
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict:
       
        y_pred = self.predict(X, threshold)
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred),
            'confusion_matrix': confusion_matrix(y, y_pred)
        }
    
    def get_parameters(self) -> dict:
     
        if not self.fitted:
            raise ValueError("Model must be fitted before accessing parameters")
        return {
            'weights': self.weights.detach().numpy(),
            'bias': float(self.bias.detach().numpy())
        }
