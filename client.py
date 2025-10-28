import grpc

import torch
import torch.nn as nn

import proto.fedsplit_communication_pb2 as pb2
import proto.fedsplit_communication_pb2_grpc as pb2_grpc
from proto.serialize_utils import *
from data.utils import load_data

import copy
import threading

from utils.models import FedAvgCNN


class Client(object):
    def __init__(self, id, local_model=None, head_model=None, base_model=None,
                 device='cpu', batch_size=8, learning_rate=0.00005, local_epochs=1,
                 split_train=True, barrier=None, steps_per_epoch=None):
        self.id = id
        self.host = 'localhost'
        self.server_port = 50051
        self.split_train = split_train
        MAX_LENGTH = 50 * 1024 * 1024
        self.channel = grpc.insecure_channel(
            f'{self.host}:{self.server_port}',
            options=[('grpc.max_send_message_length', MAX_LENGTH),
                     ('grpc.max_receive_message_length', MAX_LENGTH)]
        )
        if self.split_train:
            self.train_server_stub = pb2_grpc.TrainServerStub(self.channel)
            self.get_head_model = pb2_grpc.ReturnServerModelStub(self.channel)
        self.aggregate_stub = pb2_grpc.AggregateStub(self.channel)

        # modelos
        self.device = device
        
        if self.split_train:
            self.head_model = head_model.to(self.device)
            self.base_model = base_model.to(self.device)
        else: 
            self.local_model = local_model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(
            (self.base_model if self.split_train else self.local_model).parameters(),
            lr=learning_rate
        )
        self.local_epochs = local_epochs

        self.train_loader, self.test_loader = load_data(self.id, batch_size=batch_size)
        self.barrier = barrier
        self.steps_per_epoch = steps_per_epoch  # pode ser None

    def aggregate(self, batch_weight, timeout_seconds: float = 300.0):
        # escolhe qual modelo enviar
        model = self.base_model if self.split_train else self.local_model

        # envie o PESO do batch, não do dataset inteiro
        send_models = serialize_model(self.id, int(batch_weight), model.parameters())

        aggregate_model = self.aggregate_stub.AggregateModels(send_models, timeout=timeout_seconds)
        _, _, global_tensors = deserialize_model(aggregate_model, device=self.device)

        set_paramaters(model, global_tensors)

    def train(self):
        for epoch in range(1, self.local_epochs + 1):
            for step, (X, y) in enumerate(self.train_loader):
                if self.steps_per_epoch is not None and step >= self.steps_per_epoch:
                    break

                X, y = X.to(self.device), y.to(self.device)

                if self.split_train:
                    embeddings = self.base_model(X)
                    ser = serialize_tensor(embeddings)
                    send_embeddings = pb2.Embedding(
                        id=self.id,
                        embedding=ser,
                        shape=list(embeddings.shape),  # garanta lista
                        evaluate=False
                    )
                    gradients = self.train_server_stub.GetTrainResponse(send_embeddings)
                    recv_grad = deserialize_gradient(gradients).to(self.device)
                    embeddings.backward(gradient=recv_grad)
                else:
                    logits = self.local_model(X)
                    loss = self.criterion(logits, y.long())
                    loss.backward()

                self.optim.step()
                self.optim.zero_grad()

                # --- SINCRONIZAÇÃO POR BATCH (dupla barreira) ---
                try:
                    # 1) todos terminaram o backward/step do batch k
                    self.barrier.wait()
                    # 2) todos agregam e aplicam o global do batch k
                    self.aggregate(batch_weight=y.size(0))
                    # 3) todos confirmam que aplicaram o global antes do próximo batch
                    self.barrier.wait()
                except threading.BrokenBarrierError:
                    # trate reset/saída conforme sua estratégia
                    return

    def evaluate(self):
        # Se estiver em split learning, puxa a cabeça atual do servidor
        if self.split_train:
            head_msg = self.get_head_model.GetServerModel(pb2.Empty())
            _, _, head_tensors = deserialize_model(head_msg)
            set_paramaters(self.head_model, head_tensors)
            # garante que o sequencial está coerente
            self.local_model = nn.Sequential(self.base_model, self.head_model).to(self.device)

        self.local_model.eval()  # propaga para base/head

        total, correct, total_loss = 0, 0, 0.0
        tp = tn = fp = fn = 0

        with torch.no_grad():
            for X, y in self.test_loader:   # <- precisa ser (X, y)
                X, y = X.to(self.device), y.to(self.device)

                logits = self.local_model(X)

                # Se a saída for [N, C], usa argmax para pegar a classe com maior probabilidade
                pred = torch.argmax(logits, dim=1)

                # Calcular loss
                loss = self.criterion(y, logits)
                total_loss += loss.item()

                # Acumula acertos e total
                correct += (pred == y).sum().item()
                total   += y.numel()

                # Atualiza a matriz de confusão para métricas
                for t, p in zip(y.view(-1), pred.view(-1)):
                    tp += (t == p)  # Verdadeiro positivo
                    fp += (t != p) & (p != 0)  # Falso positivo
                    fn += (t != p) & (p == 0)  # Falso negativo
                    tn += (t != p) & (p == 0)  # Verdadeiro negativo

        # Converte para inteiros Python
        correct_i = int(correct)
        total_i   = int(total)
        tp_i      = int(tp)
        tn_i      = int(tn)
        fp_i      = int(fp)
        fn_i      = int(fn)

        return correct_i, total_i, tp_i, tn_i, fp_i, fn_i, total_loss


if __name__ == "__main__":
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_clients = 53
    n_rounds = 100

    torch.manual_seed(42)
    model_CNN = FedAvgCNN(num_classes=num_clients)

    # cria clientes
    clients = []
    barrier = threading.Barrier(num_clients)
    for id_client in range(0, num_clients):
        model = copy.deepcopy(model_CNN)
        client = Client(
            id=id_client,
            local_model=model,              # usado quando split_train=False
            head_model=model.fc,            # usado quando split_train=True
            base_model=model.conv,          # usado quando split_train=True
            device=device,
            barrier=barrier,
            split_train=False                # ou False, conforme seu caso
        )
        clients.append(client)

    # GARANTA mesmo nº de batches por epoch para evitar deadlock
    steps_per_epoch = min(len(c.train_loader) for c in clients)
    for c in clients:
        c.steps_per_epoch = steps_per_epoch

    for round in range(n_rounds):
        print(f'Round {round}')

        # treino (agrega por batch dentro de client.train)
        threads_train = [threading.Thread(target=c.train) for c in clients]
        for t in threads_train: t.start()
        for t in threads_train: t.join()

        # (removido) NÃO agregue novamente aqui — já foi por batch

        # avaliação
        correct = total = 0
        total_loss = 0.0
        for c in clients:
            c_, tl_, t_ = c.evaluate()
            correct += c_
            total_loss += tl_
            total += t_
        print(f'acc: {correct/total:.4f}, loss: {total_loss/total:.4f}')