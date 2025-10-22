import grpc
from concurrent import futures
import proto.fedsplit_communication_pb2 as pb2
import proto.fedsplit_communication_pb2_grpc as pb2_grpc
from proto.serialize_utils import *

import torch
import torch.nn as nn

from typing import List

import threading

from utils.models import FedAvgCNN

class ReturnServerModel(pb2_grpc.ReturnServerModelServicer):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def GetServerModel(self, request, context):
        global_model = [param for param in self.model.parameters()]
        send_models = serialize_model(
            0, 0, global_model)
        return send_models
    
class AggregateServicer(pb2_grpc.AggregateServicer):

    def __init__(self, num_clients: int = 10, device: torch.device | None = None):
        super().__init__()
        self.num_clients = int(num_clients)
        self.device = device or torch.device("cpu")

        # o modelo global é uma lista de tensores (um por parâmetro)
        self.global_model: List[torch.Tensor] = []

        self.uploaded_ids: List[int] = []
        self.uploaded_weights: List[int] = []
        self.uploaded_models: List[List[torch.Tensor]] = []

        self.lock = threading.Lock()

    def AggregateModels(self, request, context):
        with self.lock, torch.no_grad():
            
            client_id, weight, grads = deserialize_model(request)

            self.uploaded_ids.append(client_id)
            self.uploaded_weights.append(weight)
            self.uploaded_models.append(grads)

            if len(self.uploaded_models) >= self.num_clients:
                self.aggregate_round()
                # limpar buffers da rodada
                self.uploaded_ids.clear()
                self.uploaded_weights.clear()
                self.uploaded_models.clear()

            response = serialize_model(client_id, weight, self.global_model)
            return response

    def aggregate_round(self) -> None:
        #print('aggregate')
        total_weight = sum(self.uploaded_weights)
        if total_weight <= 0:
            raise ValueError("Soma de pesos inválida na rodada.")

        # inicializa o acumulador na primeira vez, copiando o “shape” do primeiro cliente
        if not self.global_model:
            self.global_model = [torch.zeros_like(t, device=self.device)
                                 for t in self.uploaded_models[0]]
        else:
            # zera acumulador
            for p in self.global_model:
                p.zero_()

        # soma ponderada normalizada (FedAvg)
        for w, client_tensors in zip(self.uploaded_weights, self.uploaded_models):
            scale = float(w) / float(total_weight)
            for p, c in zip(self.global_model, client_tensors):
                p.add_(c, alpha=scale)  # <- modifica de fato o acumulador
    
class TrainServerServicer(pb2_grpc.TrainServerServicer):
    def __init__(self, model, num_clients=1, learning_rate=0.00005, device=None):
        super().__init__()
        self.device = device
        self.num_clients = num_clients
        self.model = model.to(self.device)

        # Define função de erro e otimizador do cliente
        self.criterion = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Lock para garantir que o servidor só atenda um cliente por vez
        self.lock = threading.Lock()

        # Fila para armazenar os clientes que aguardam o acesso
        self.queue = []

    def GetTrainResponse(self, received_embedding, context):
        # Tenta adquirir o lock. Se não conseguir, coloca o cliente na fila
        with self.lock:
            # Processa a requisição do cliente enquanto o lock é mantido
            self.model.train()
            #print(f'client {received_embedding.id} traning...')

            # recebe os embeddings
            embedding_in = deserialize_embedding(received_embedding).to(self.device)

            # define as labels do cliente
            shape_embedding = list(received_embedding.shape)
            labels = torch.tensor([received_embedding.id]*shape_embedding[0], dtype=torch.long, device=self.device)

            # faz a predição e calcula os gradientes
            output = self.model(embedding_in)
            error = self.criterion(output, labels)
            error.backward()

            # "caminha" com o gradiente
            self.optim.step()
            self.optim.zero_grad()

            # envia os gradientes para o cliente
            send_grad_server = serialize_tensor(embedding_in.grad)
            grad_error = pb2.Gradients(grad=send_grad_server, shape=shape_embedding)

            return grad_error

def serve(split=False, num_clients=53):
    torch.manual_seed(42)
    model = FedAvgCNN(num_classes=num_clients)
    MAX_LENGTH = 50 * 1024 * 1024
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                options=[('grpc.max_send_message_length', MAX_LENGTH),    
                        ('grpc.max_receive_message_length', MAX_LENGTH)])
    train_server = TrainServerServicer(model.fc, num_clients=num_clients)
    get_model_server = ReturnServerModel(model.fc)
    aggregate_server = AggregateServicer(num_clients=num_clients)

    if split: 
        pb2_grpc.add_TrainServerServicer_to_server(train_server, server)
        pb2_grpc.add_ReturnServerModelServicer_to_server(get_model_server, server)
    pb2_grpc.add_AggregateServicer_to_server(aggregate_server, server)

    server.add_insecure_port('[::]:50051')
    server.start()

    print("Server started at 50051")

    server.wait_for_termination()

if __name__ == '__main__':
    serve()
