import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mlflow
from loguru import logger

class Entrenador:
    """
    Gestiona el entreno completo
    """
    def __init__(
            self,
            modelo: nn.Module,
            lr: float = 0.001,
            batch_size: int = 32,
            paciencia: int = 10
    ):
        self.modelo = modelo
        self.lr = lr
        self.batch_size = batch_size
        self.paciencia = paciencia

        self.optimizador = torch.optim.Adam(
            modelo.parameters(), lr=lr)
        self.criterio = nn.MSELoss()
        # Historial de pérdidas
        self.historial_train = []
        self.historial_val = []

    def entrenar(
            self,
            X: np.ndarray,
            epochs: int = 50,
            proporcion_val: float = 0.1,
            nombre_experimento: str = "autoencoder-ltsm"
    ) -> None:
        """
        Ejecuta el loop de entrenamiento completo.
        """
        # Preparación de datos 
        X_train, X_val = self.dividir_datos(X, proporcion_val)
        loader_train = self.crear_loader(X_train, shuffle=True)
        loader_val = self.crear_loader(X_val, shuffle=False)

        logger.info(
            f"Entrenamiento: {len(X_train)} secuencias | "
            f"Validación: {len(X_val)} secuencias | "
            f"Batch size: {self.batch_size}"
        )

        # MLflow: registramos hiperparámetros 
        mlflow.set_experiment(nombre_experimento)
        with mlflow.start_run():
            mlflow.log_params({
                "lr": self.lr,
                "batch_size": self.batch_size,
                "epochs": epochs,
                "paciencia": self.paciencia,
                "proporcion_val": proporcion_val,
            })

            mejor_perdida_val = float("inf")
            epochs_sin_mejora = 0
            for epoch in range(1, epochs + 1):
                # Forward + Backward en train 
                perdida_train = self.epoch_train(loader_train)
                # Solo forward en validación (sin actualizar pesos) 
                perdida_val = self.epoch_val(loader_val)

                self.historial_train.append(perdida_train)
                self.historial_val.append(perdida_val)
                # Registramos métricas en MLflow 
                mlflow.log_metrics(
                    {"perdida_train": perdida_train, "perdida_val": perdida_val},
                    step=epoch
                )

                if epoch % 10 == 0:
                    logger.info(
                        f"Epoch {epoch}/{epochs} | "
                        f"Train: {perdida_train:.6f} | "
                        f"Val: {perdida_val:.6f}"
                    )

                #  Early stopping 
                if perdida_val < mejor_perdida_val:
                    mejor_perdida_val = perdida_val
                    epochs_sin_mejora = 0
                    # Guardamos el mejor modelo hasta ahora
                    torch.save(self.modelo.state_dict(), "mejor_modelo.pt")
                else:
                    epochs_sin_mejora += 1

                if epochs_sin_mejora >= self.paciencia:
                    logger.info(
                        f"Early stopping en epoch {epoch}. "
                        f"Mejor pérdida val: {mejor_perdida_val:.6f}"
                    )
                    break

            # Cargamos el modelo al final
            self.modelo.load_state_dict(torch.load("mejor_modelo.pt"))
            mlflow.log_metric("mejor_perdida_val", mejor_perdida_val)
            logger.info("Entrenamiento completado.")

    def epoch_train(self, loader: DataLoader) -> float:
        """
        Ejecuta un epoch completo de entrenamiento y devuelve la pérdida media.
        """
        self.modelo.train()
        perdida_total = 0.0

        for (batch,) in loader:
            reconstruccion = self.modelo(batch)
            perdida = self.criterio(reconstruccion, batch)

            self.optimizador.zero_grad()
            perdida.backward()
            self.optimizador.step()

            perdida_total += perdida.item()

        return perdida_total / len(loader)

    def epoch_val(self, loader: DataLoader) -> float:
        """
        Ejecuta un epoch de validación sin actualizar pesos.
        """
        self.modelo.eval()
        perdida_total = 0.0

        with torch.no_grad():
            for (batch,) in loader:
                reconstruccion = self.modelo(batch)
                perdida = self.criterio(reconstruccion, batch)
                perdida_total += perdida.item()

        return perdida_total / len(loader)

    def dividir_datos(
        self, X: np.ndarray, proporcion_val: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Divide los datos en train y validación respetando el orden temporal.
        """
        n_val = int(len(X) * proporcion_val)
        return X[:-n_val], X[-n_val:]

    def crear_loader(self, X: np.ndarray, shuffle: bool) -> DataLoader:
        """Convierte un array numpy en un DataLoader de PyTorch."""
        tensor = torch.FloatTensor(X)
        dataset = TensorDataset(tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)