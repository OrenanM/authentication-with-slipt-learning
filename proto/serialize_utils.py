import torch
import proto.fedsplit_communication_pb2 as pb2

# ============================================================
# 🧩 Funções utilitárias para serialização e desserialização
# ============================================================

def serialize_tensor(tensor: torch.Tensor) -> list[float]:
    """
    Converte um tensor PyTorch em uma lista plana (1D) de floats.
    Isso permite que ele seja transmitido via gRPC em mensagens Protobuf.
    """
    return tensor.view(-1).tolist()


def deserialize_gradient(gradient: pb2.Models) -> torch.Tensor:
    """
    Reconstrói um tensor de gradiente recebido de uma mensagem pb2.Models.
    Supõe que o campo .grad e .shape estão presentes.
    """
    shape_tensor = list(gradient.shape)
    tensor_gradient = torch.tensor(gradient.grad, dtype=torch.float32)
    return tensor_gradient.view(shape_tensor)


def deserialize_embedding(embed: pb2.Models) -> torch.Tensor:
    """
    Reconstrói um tensor de embeddings a partir de uma mensagem pb2.Models,
    definindo requires_grad=True para permitir backpropagation posterior.
    """
    shape_tensor = list(embed.shape)
    tensor_embedding = torch.tensor(embed.embedding, dtype=torch.float32)
    return tensor_embedding.view(shape_tensor).requires_grad_(True)


def serialize_model(client_id: int, weight: int, model_params) -> pb2.Models:
    """
    Serializa os parâmetros do modelo local em uma mensagem pb2.Models.

    Args:
        client_id: ID do cliente participante.
        weight: Peso do cliente (número de amostras ou batches).
        model_params: Iterável de tensores (por exemplo, model.parameters()).
    """
    gradients_msg = []
    for p in model_params:
        flat = serialize_tensor(p)
        gradients_msg.append(pb2.Gradients(grad=flat, shape=list(p.shape)))

    return pb2.Models(id=client_id, data_sample=weight, gradients=gradients_msg)


def deserialize_model(model: pb2.Models, device: str ='cpu') -> tuple[int, int, list[torch.Tensor]]:
    """
    Reconstrói os tensores de parâmetros de uma mensagem pb2.Models.

    Returns:
        client_id (int): ID do cliente.
        weight (int): peso do cliente (número de amostras).
        client_tensors (list[Tensor]): lista de tensores reconstruídos.
    """
    client_id = int(model.id)
    weight = int(model.data_sample)

    client_tensors = []
    for g in model.gradients:
        shape = list(g.shape)
        t = torch.tensor(g.grad, dtype=torch.float32, device=device).view(*shape)
        client_tensors.append(t)

    return client_id, weight, client_tensors


def set_paramaters(model: torch.nn.Module, global_tensors: list[torch.Tensor]) -> None:
    """
    Copia os parâmetros globais (recebidos do servidor) para o modelo local.

    Args:
        model: Modelo local do cliente (torch.nn.Module).
        global_tensors: Lista de tensores agregados pelo servidor.
    """
    with torch.no_grad():
        for p, new_p in zip(model.parameters(), global_tensors):
            p.data.copy_(new_p.data)
