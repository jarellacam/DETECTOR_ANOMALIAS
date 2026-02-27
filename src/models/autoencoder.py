import torch
import torch.nn as nn

class CodificadorLSTM(nn.Module):
    """
    Procesa los días de la entrada uno por uno (ej: 30).
    En cada día actualiza su estado oculto teniendo en cuenta lo que vio antes (hidden[-1]).
    """
    def __init__(self, n_features: int, dim_latente: int, num_capas: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=dim_latente,
            num_layers=num_capas,
            batch_first=True   
        )

    def forward(self, x):
        # x shape: (batch, seq_len, n_features)
        _, (estado_oculto, _) = self.lstm(x)
        # Nos quedamos con la última capa
        return estado_oculto[-1]


class DecodificadorLSTM(nn.Module):
    """
    Recibe el vector del codificador e intenta reconstruir los días originales.
    Como el vector recibido no tiene dimensión temporal, añadimos dicha dimension con
    unsqueeze(1).repeat(1,30,1), repitiendo como se ve el valor n veces (en este caso 30) 
    para que LSTM tenga algo que procesar en cada paso   
    """
    def __init__(self, dim_latente: int, n_features: int, seq_len: int, num_capas: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(
            input_size=dim_latente,
            hidden_size=n_features,
            num_layers=num_capas,
            batch_first=True
        )

    def forward(self, x):
        # unsqueeze(1) añade la dimensión temporal: (batch, 1, dim_latente)
        # repeat(1, seq_len, 1) la repite seq_len veces: (batch, seq_len, dim_latente)
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        # La LSTM reconstruye la secuencia
        salida, _ = self.lstm(x)
        # salida shape: (batch, seq_len, n_features)
        return salida


class AutoencoderLSTM(nn.Module):
    """
    Detección de anomalías en series temporales.
    Como ha reconstruido los dstos, no sabrá replicar del todo bien las anomalías, 
    por lo que el error será alto en dichos valores.
    """
    def __init__(
        self,
        n_features: int,
        dim_latente: int,
        seq_len: int,
        num_capas: int = 1
    ):
        super().__init__()
        self.codificador = CodificadorLSTM(n_features, dim_latente, num_capas)
        self.decodificador = DecodificadorLSTM(dim_latente, n_features, seq_len, num_capas)

    def forward(self, x):
        # Comprimimos
        representacion = self.codificador(x)
        # Reconstruimos
        reconstruccion = self.decodificador(representacion)
        return reconstruccion