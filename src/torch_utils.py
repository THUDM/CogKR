import torch
import torch.nn


# pad a tensor at given dimension
def pad_tensor(tensor: torch.Tensor, length, value=0, dim=0) -> torch.Tensor:
    return torch.cat(
        (tensor, tensor.new_full((*tensor.size()[:dim], length - tensor.size(dim), *tensor.size()[dim + 1:]), value)),
        dim=dim)


# transform the list of list to tensor with possible padding
def list2tensor(data_list: list, padding_idx, dtype=torch.long, device=torch.device("cpu")):
    max_len = max(map(len, data_list))
    max_len = max(max_len, 1)
    data_tensor = torch.stack(
        tuple(pad_tensor(torch.tensor(data, dtype=dtype), max_len, padding_idx, 0) for data in data_list)).to(
        device)
    return data_tensor


def load_embed_state_dict(path, model):
    kg_state_dict = torch.load(path, map_location='cpu')['state_dict']
    state_dict = {}
    if model == 'complex':
        state_dict['entity_embeddings.weight'] = torch.cat(
            (kg_state_dict['kg.entity_embeddings.weight'], kg_state_dict['kg.entity_img_embeddings.weight']), dim=-1)
        state_dict['relation_embeddings.weight'] = torch.cat(
            (kg_state_dict['kg.relation_embeddings.weight'], kg_state_dict['kg.relation_img_embeddings.weight']),
            dim=-1)
    elif model == 'conve' or model == 'distmult':
        for param_name in ['kg.entity_embeddings.weight', 'kg.relation_embeddings.weight']:
            state_dict[param_name.split('.', 1)[1]] = kg_state_dict[param_name]
    return state_dict


def load_embedding(model, entity_path, relation_path):
    def load_from_file(path):
        embeds = []
        with open(path) as file:
            for line in file:
                line = line.strip().split()
                embeds.append(list(map(float, line)))
        return embeds

    entity_embeds = load_from_file(entity_path)
    entity_embeds = torch.tensor(entity_embeds)
    model.entity_embeddings.weight.data[:len(entity_embeds)] = entity_embeds
    relation_embeds = load_from_file(relation_path)
    relation_embeds = torch.tensor(relation_embeds)
    model.relation_embeddings.weight.data[:len(relation_embeds)] = relation_embeds
