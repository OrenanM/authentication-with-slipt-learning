# create_data_flexible.py
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# -----------------------------
# Transformações
# -----------------------------
def transform_none(windows: np.ndarray) -> np.ndarray:
    """
    Mantém as janelas no domínio do tempo.
    Entrada: (N, L)  -> Saída: (N, 1, L)   [canal para Conv1d]
    """
    X = windows.astype(np.float32)
    X = np.expand_dims(X, axis=1)  # (N, 1, L)
    return X

def transform_fft(windows: np.ndarray) -> np.ndarray:
    """
    Aplica rFFT em cada janela (espectro de magnitude).
    Entrada: (N, L) -> Saída: (N, 1, F) onde F = L//2 + 1
    """
    # rfft em última dimensão
    spec = np.fft.rfft(windows, axis=1)          # (N, F) se axis=1? cuidado: windows=(N, L)
    # windows: (N, L) -> rfft sobre axis=1 retorna (N, F)
    mag = np.abs(spec).astype(np.float32)
    X = np.expand_dims(mag, axis=1)              # (N, 1, F)
    return X

TRANSFORMS = {
    "none": transform_none,
    "fft": transform_fft,
}

# -----------------------------
# Carregamento com janelas fixas
# -----------------------------
def load_data_from_csv(dataset_directory: str, size_window: int = 64, col_name: str = ' II'):
    """
    Lê CSVs, remove NaN, corta para múltiplo de size_window e faz reshape (N_janelas, size_window).
    Extrai label do nome do arquivo via csv_file[6:8] (ajuste se necessário).
    Retorna listas: data_list (cada item (N_i, size_window)), target_list (cada item (N_i,))
    """
    csv_files = sorted(os.listdir(dataset_directory))
    data_list, target_list = [], []

    for n, csv_file in enumerate(csv_files):
        if not csv_file.lower().endswith('.csv'):
            continue
        full_path = os.path.join(dataset_directory, csv_file)
        df = pd.read_csv(full_path)

        sig = df[col_name].to_numpy(dtype=float)
        sig = sig[np.isfinite(sig)]

        usable_len = (len(sig) // size_window) * size_window
        if usable_len == 0:
            continue

        sig = sig[:usable_len]
        windows = sig.reshape(-1, size_window)  # (N, L)

        # Extração da label
        try:
            label_int = int(csv_file[6:8])  # ajuste se seu padrão for outro
        except ValueError:
            label_int = 0

        labels = np.full((windows.shape[0],), label_int, dtype=int)

        data_list.append(windows.astype(np.float32))
        target_list.append(labels)
        if n >= 9:
            break

    if not data_list:
        raise RuntimeError("Nenhum CSV válido encontrado ou séries curtas demais.")

    return data_list, target_list

# -----------------------------
# Salvamento (treino/ teste)
# -----------------------------
def save_data(
    data_windows_list,
    targets_list,
    save_path: str,
    size_dataset: int = None,
    transform: str = "none",
    num_clients: int = 53,
    stratify_per_file: bool = True
):
    """
    data_windows_list: list de arrays (N_i, L)
    targets_list:      list de arrays (N_i,)
    transform: 'none' -> (N,1,L) ; 'fft' -> (N,1,F)
    - Split 80/20 por arquivo
    - Salva treino por label (um .npz por label encontrado em cada arquivo)
    - Junta teste de todos os arquivos, embaralha e reparte entre num_clients
    """
    os.makedirs(os.path.join(save_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "test"), exist_ok=True)

    if transform not in TRANSFORMS:
        raise ValueError(f"Transform '{transform}' inválida. Use {list(TRANSFORMS.keys())}.")

    apply_transform = TRANSFORMS[transform]

    X_test_total, y_test_total = [], []

    for X_i, y_i in zip(data_windows_list, targets_list):
        # Decide se dá para estratificar por arquivo (só faz sentido se houver mais de uma classe no arquivo)
        stratify_arg = y_i if (stratify_per_file and len(np.unique(y_i)) > 1) else None

        X_train, X_test, y_train, y_test = train_test_split(
            X_i, y_i, test_size=0.2, random_state=42, stratify=stratify_arg
        )
        
        # Verifica se o tamanho do dataset foi especificado
        if size_dataset is None:
            size_dataset = X_train.shape[0]

        # Reduz o número de windows com base no tamanho do dataset e um fator de ajuste
        adjustment_factor = round(125 / 64)  
        X_train = X_train[:size_dataset * adjustment_factor]
        y_train = y_train[:size_dataset * adjustment_factor]  
    
        
        # Aplica a transformação escolhida
        X_train_t = apply_transform(X_train)  # (N,1,L) ou (N,1,F)
        X_test_t  = apply_transform(X_test)

        # Salva treino por rótulo (assumindo arquivo homogêneo; se não for, salva por rótulo distinto)
        unique_labels = np.unique(y_train)
        for lab in unique_labels:
            idx = (y_train == lab)
            if not np.any(idx):
                continue
            np.savez(
                os.path.join(save_path, "train", f"{int(lab)}.npz"),
                data=X_train_t[idx].astype(np.float32),
                target=y_train[idx].astype(int),
            )
        # acumula teste global
        X_test_total.append(X_test_t.astype(np.float32))
        y_test_total.append(y_test.astype(int))

    if len(X_test_total) == 0:
        raise RuntimeError("Nenhum conjunto de teste gerado.")

    X_test_total = np.concatenate(X_test_total, axis=0)
    y_test_total = np.concatenate(y_test_total, axis=0)

    # Embaralha globalmente
    X_test_total, y_test_total = shuffle(X_test_total, y_test_total, random_state=42)

    # Reparte entre clientes
    total = len(X_test_total)
    client_size = total // num_clients
    if client_size == 0:
        raise RuntimeError(
            f"Teste insuficiente ({total}) para {num_clients} clientes. "
            f"Reduza num_clients ou gere mais dados."
        )

    for i in range(num_clients):
        start = i * client_size
        end = (i + 1) * client_size if i < num_clients - 1 else total
        np.savez(
            os.path.join(save_path, "test", f"{i+1}.npz"),
            data=X_test_total[start:end].astype(np.float32),
            target=y_test_total[start:end].astype(int),
        )

# -----------------------------
# Main / CLI
# -----------------------------
def main(dataset_directory: str, save_path: str, size_window: int = 64, size_dataset = None,
         col_name: str = ' II', transform: str = "none", num_clients: int = 53):
    
    data_list, target_list = load_data_from_csv(dataset_directory, size_window=size_window, col_name=col_name)
    save_data(
        data_list,
        target_list,
        size_dataset=size_dataset,
        save_path=save_path,
        transform=transform,
        num_clients=num_clients,
        stratify_per_file=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gerar dataset por janelas com ou sem transformação (FFT).")
    parser.add_argument("--dataset_directory", type=str, default='bidmc_csv/', help="Pasta com CSVs.")
    parser.add_argument("--save_path", type=str, default='ECG', help="Pasta de saída (serão criadas subpastas train/ e test/).")
    parser.add_argument("--size_window", type=int, default=80, help="Tamanho da janela (L).")
    parser.add_argument("--col_name", type=str, default=" II", help="Nome da coluna do ECG no CSV.")
    parser.add_argument("--transform", type=str, choices=["none", "fft"], default="none", help="Transformação por janela.")
    parser.add_argument("--num_clients", type=int, default=10, help="Número de clientes para particionar o teste.")
    parser.add_argument("--size_dataset", type=int, default=None, help='Numero de dados de cada cliente')
    args = parser.parse_args()

    main(
        dataset_directory=args.dataset_directory,
        save_path=args.save_path,
        size_window=args.size_window,
        size_dataset=args.size_dataset,
        col_name=args.col_name,
        transform=args.transform,
        num_clients=args.num_clients
    )
