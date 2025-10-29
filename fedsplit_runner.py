"""fedsplit_runner.py

Executa servidor gRPC e clientes para FedAvg / Split Learning em um mesmo
script, garantindo que **todos** iniciem com a **mesma** inicialização de pesos.

⚠️ Importante: a lógica de treino/inferência permanece a mesma. Foram
adicionados: integração com o evaluate() que retorna TP/TN/FP/FN, cálculo de
métricas agregadas por round (acc, precision, recall, f1) e persistência em CSV.
Agora também é possível nomear a simulação via --run-name (ex.: "simulation 1").
"""

from __future__ import annotations

# ===========================
# Imports
# ===========================
import argparse
import copy
import threading
import time
from concurrent import futures
from pathlib import Path
from datetime import datetime
import csv
import re

import grpc
import torch

import proto.fedsplit_communication_pb2_grpc as pb2_grpc
from client import Client
from server import AggregateServicer, ReturnServerModel, TrainServerServicer
from utils.models import FedAvgCNN


# ===========================
# Utilidades
# ===========================

def resolve_device(choice: str) -> str:
    """Resolve o dispositivo a ser utilizado.

    Parâmetros
    ----------
    choice : str
        "auto", "cpu" ou "cuda".

    Retorno
    -------
    str
        Dispositivo escolhido ("cpu" ou "cuda").
    """
    if choice == "auto":
        try:
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return choice  # "cpu" ou "cuda"


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _slugify(name: str) -> str:
    """Converte um nome livre (ex.: 'Simulation 1') para algo seguro de usar em arquivos."""
    name = name.strip().lower()
    # troca separadores por hífen
    name = re.sub(r"[ \t\n\r/\\]+", "-", name)
    # remove qualquer coisa não [a-z0-9-_]
    name = re.sub(r"[^a-z0-9\-_]", "", name)
    # compacta hifens
    name = re.sub(r"-{2,}", "-", name)
    return name or "run"


def _metrics_csv_path(args: argparse.Namespace) -> Path:
    """Define o caminho do CSV de métricas, respeitando --metrics-csv e --run-name."""
    if args.metrics_csv:
        p = Path(args.metrics_csv)
    else:
        base = "metrics"
        if args.run_name:
            base = f"{base}_{_slugify(args.run_name)}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        p = Path("results") / f"{base}_{timestamp}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _append_metrics_row(csv_path: Path, header: list[str], row: list):
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)


# ===========================
# Servidor
# ===========================

def build_server(
    head_model: torch.nn.Module | None,
    split: bool,
    num_clients: int,
    port: int,
    max_message_mb: int,
    server_workers: int,
) -> grpc.Server:
    """Constroi e devolve o servidor gRPC.

    - A agregação federada (AggregateServicer) é sempre registrada.
    - Se ``split=True``, também registra os serviços de treino (TrainServer) e
      de retorno do modelo (ReturnServerModel), usando **a mesma head** que
      inicializa os clientes (deepcopy já feito antes de registrar).
    """
    device = resolve_device("auto")
    max_len = max_message_mb * 1024 * 1024

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=server_workers),
        options=[
            ("grpc.max_send_message_length", max_len),
            ("grpc.max_receive_message_length", max_len),
        ],
    )

    # Agregação federada (sempre disponível)
    aggregate_server = AggregateServicer(num_clients=num_clients)
    pb2_grpc.add_AggregateServicer_to_server(aggregate_server, server)

    # Split Learning (opcional)
    if split:
        if head_model is None:
            raise ValueError("head_model não pode ser None quando split=True.")
        # Garante que a head do servidor começa com os MESMOS pesos dos clientes
        head_for_server = copy.deepcopy(head_model).to(device)

        train_server = TrainServerServicer(head_for_server, num_clients=num_clients)
        get_model_server = ReturnServerModel(head_for_server)

        pb2_grpc.add_TrainServerServicer_to_server(train_server, server)
        pb2_grpc.add_ReturnServerModelServicer_to_server(get_model_server, server)

    server.add_insecure_port(f"[::]:{port}")
    return server


def run_server(head_model: torch.nn.Module | None, args: argparse.Namespace) -> None:
    """Inicializa e bloqueia na execução do servidor até término."""
    server = build_server(
        head_model=head_model,
        split=args.split,
        num_clients=args.num_clients,
        port=args.port,
        max_message_mb=args.max_message_mb,
        server_workers=args.server_workers,
    )
    server.start()
    print(f"[SERVER] Started on port {args.port} | split={args.split}")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("[SERVER] Shutting down...")


# ===========================
# Clientes
# ===========================

def run_clients(
    num_clients: int,
    init_full_model: torch.nn.Module | None,
    init_base: torch.nn.Module | None,
    init_head: torch.nn.Module | None,
    args: argparse.Namespace,
) -> None:
    """Cria e executa N clientes com **cópias idênticas** dos mesmos pesos.

    - ``split=False``: cada cliente recebe um ``deepcopy`` de ``init_full_model``.
    - ``split=True`` : cada cliente recebe ``deepcopy`` de ``init_base`` e
      ``init_head``.
    """
    device = resolve_device(args.device)
    print(f"[CLIENTS] device={device} | split={args.split}")

    # Barreira global para sincronizar todos por batch
    barrier = threading.Barrier(args.num_clients)

    # Criação dos clientes
    clients = []
    for cid in range(1, args.num_clients + 1):
        if args.split:
            if init_base is None or init_head is None:
                raise ValueError(
                    "init_base/init_head não podem ser None quando split=True."
                )
            local_model = None  # não usado em split
            base_model = copy.deepcopy(init_base)
            head_model = copy.deepcopy(init_head)
        else:
            if init_full_model is None:
                raise ValueError(
                    "init_full_model não pode ser None quando split=False."
                )
            local_model = copy.deepcopy(init_full_model)
            base_model = None
            head_model = None

        client = Client(
            id=cid,
            local_model=local_model,          # usado quando split_train=False
            head_model=head_model,            # usado quando split_train=True
            base_model=base_model,            # usado quando split_train=True
            device=device,
            barrier=barrier,
            split_train=args.split,           # importante: casa com o servidor
            batch_size=args.batch_size,
            learning_rate=args.lr,
            local_epochs=args.local_epochs,
            num_classes=num_clients
        )
        clients.append(client)

    # Mesmo número de *batches* por época (evita deadlock de barreira)
    steps_per_epoch = (
        args.steps_per_epoch
        if args.steps_per_epoch is not None
        else min(len(c.train_loader) for c in clients)
    )
    for c in clients:
        c.steps_per_epoch = steps_per_epoch

    # Cabeçalho e path do CSV de métricas
    csv_path = _metrics_csv_path(args)
    header = [
        "timestamp","run_name","round","acc","precision","recall","f1",
        "tp","tn","fp","fn","total","correct",
        "num_clients","lr","batch_size","local_epochs","steps_per_epoch",
        "split","device","seed"
    ]

    # Loop de rounds (o train de cada cliente já faz: barreira + aggregate por batch)
    for rnd in range(args.rounds):
        print(f"[CLIENTS] Round {rnd} | steps_per_epoch={steps_per_epoch}")

        threads = [threading.Thread(target=c.train) for c in clients]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Inicializa variáveis globais
        total_tp = 0
        total_tn = 0
        total_fp = 0
        total_fn = 0
        total = 0
        total_loss = 0.0

        # Loop pelos clientes
        for c in clients:
            corr, t, tp, tn, fp, fn, loss = c.evaluate()

            # Soma os valores para cada classe
            total_tp = tp
            total_tn = tn
            total_fp = fp
            total_fn = fn

            total += t
            total_loss += loss

        # Métricas globais
        mean_loss = total_loss / total
        correct = total_tp + total_tn
        acc = _safe_div(correct, total)
        precision = _safe_div(total_tp, total_tp + total_fp)
        recall = _safe_div(total_tp, total_tp + total_fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)

        # Log amigável no console
        print(
            f"[CLIENTS] round={rnd} | acc={acc:.4f} "
            f"prec={precision:.4f} rec={recall:.4f} f1={f1:.4f} "
            f"tp={total_tp} tn={total_tn} fp={total_fp} fn={total_fn} total={total} loss={mean_loss:.6f}"
            f"run='{args.run_name or ''}'"
        )

        # Persistência em CSV
        _append_metrics_row(
            csv_path,
            header,
            [
                datetime.now().isoformat(timespec="seconds"),
                args.run_name or "",
                rnd,
                f"{acc:.8f}",
                f"{precision:.8f}",
                f"{recall:.8f}",
                f"{f1:.8f}",
                total_tp, total_tn, total_fp, total_fn, total, correct,
                args.num_clients, args.lr, args.batch_size, args.local_epochs, steps_per_epoch,
                int(args.split), resolve_device(args.device), args.seed,
            ],
        )


def _safe_div(a, b):
    return a / b if b != 0 else 0


# ===========================
# Execução combinada (ALL)
# ===========================

def run_all(args: argparse.Namespace) -> None:
    """Sobe servidor e executa clientes no mesmo processo."""
    torch.manual_seed(args.seed)
    init_model = FedAvgCNN(num_classes=args.num_clients + 1)

    # Quebra em base/head para split; mantém full para modo não-split
    init_base = copy.deepcopy(init_model.conv)
    init_head = copy.deepcopy(init_model.fc)

    # Servidor em *background* (usa a MESMA head inicial)
    server_thread = threading.Thread(
        target=run_server, args=(init_head, args), daemon=True
    )
    server_thread.start()

    # Espera a porta abrir (dica: para produção, implemente *healthcheck*)
    time.sleep(args.server_startup_wait)

    # Executa clientes com deepcopies da mesma inicialização
    if args.split:
        run_clients(args.num_clients, None, init_base, init_head, args)
    else:
        run_clients(args.num_clients, init_model, None, None, args)


# ===========================
# CLI
# ===========================

def parse_args() -> argparse.Namespace:
    """Define e interpreta os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        prog="fedsplit_runner",
        description=(
            "Execute servidor gRPC e clientes FedAvg / Split Learning no mesmo "
            "script, com inicialização idêntica."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--server", action="store_true", help="Rodar somente o servidor")
    mode.add_argument("--clients", action="store_true", help="Rodar somente os clientes")
    # Se não for especificado, o modo ALL é assumido no main().

    parser.add_argument(
        "--split",
        action="store_true",
        help=(
            "Ativa Split Learning (server expõe Train/ReturnModel; clientes "
            "usam split_train=True)"
        ),
    )
    parser.add_argument("--num-clients", type=int, default=53, help="Número de clientes")
    parser.add_argument(
        "--rounds", type=int, default=500, help="Número de rounds (ciclos de treino/avaliação)"
    )
    parser.add_argument(
        "--local-epochs", type=int, default=1, help="Épocas locais por round (dentro de Client.train)"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Tamanho do batch")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate do otimizador")
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help=("Força nº de batches por epoch (se None, usa o mínimo comum entre clientes)"),
    )

    parser.add_argument("--port", type=int, default=50051, help="Porta gRPC do servidor")
    parser.add_argument(
        "--server-workers", type=int, default=10, help="Máx workers do servidor gRPC"
    )
    parser.add_argument(
        "--max-message-mb", type=int, default=50, help="Limite de mensagem gRPC (MB)"
    )
    parser.add_argument(
        "--server-startup-wait",
        type=float,
        default=1.0,
        help="Aguardar (s) após subir o servidor no modo combinado (ALL)",
    )

    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto", help="Dispositivo dos clientes"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Semente aleatória para inicialização idêntica"
    )
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default=None,
        help=(
            "Caminho do CSV para salvar métricas por round. "
            "Se não informado, será criado em ./results/metrics_[run-name]_YYYYmmdd_HHMMSS.csv"
        ),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Nome legível da simulação (ex.: 'simulation 1'). "
            "Será usado no nome do arquivo e gravado no CSV."
        ),
    )

    return parser.parse_args()


# ===========================
# Main
# ===========================

def main() -> None:
    """Ponto de entrada do script via CLI."""
    args = parse_args()

    # Observação: se sua classe Client tem host/porta fixos (localhost:50051),
    # mantenha --port 50051 ou adapte Client para aceitar host/port no __init__.

    if args.server:
        # Apenas servidor — ainda garantimos inicialização determinística da head
        torch.manual_seed(args.seed)
        tmp_model = FedAvgCNN(num_classes=args.num_clients + 1)
        init_head = copy.deepcopy(tmp_model.fc)
        run_server(init_head, args)

    elif args.clients:
        # Apenas clientes — criamos a MESMA inicialização local para todos
        torch.manual_seed(args.seed)
        init_model = FedAvgCNN(num_classes=args.num_clients + 1)
        if args.split:
            run_clients(args.num_clients, None, copy.deepcopy(init_model.conv), copy.deepcopy(init_model.fc), args)
        else:
            run_clients(args.num_clients, init_model, None, None, args)

    else:
        # Modo combinado (sobe servidor e roda clientes)
        run_all(args)


if __name__ == "__main__":
    main()
