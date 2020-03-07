import torch
import torch.nn as nn
import torch.distributions
import torch_scatter
import grapher
from collections import deque
from torch_utils import list2tensor
import networkx
from networkx.algorithms.shortest_paths.generic import shortest_path_length
import math
import numpy as np
import io
import random
import pdb


class Summary(nn.Module):
    def __init__(self, hidden_size, graph: grapher.KG, entity_embeddings: nn.Embedding,
                 relation_embeddings: nn.Embedding):
        nn.Module.__init__(self)
        self.graph = graph
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings
        self.embed_size = entity_embeddings.embedding_dim
        self.hidden_size = hidden_size
        self.neighbor_layer = nn.Linear(2 * self.embed_size, self.hidden_size)
        self.transform_layer = nn.Linear(self.embed_size, self.hidden_size)
        self.relation_layer = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.relation_activation = nn.ReLU()
        self.predict_layer = nn.Linear(self.hidden_size, relation_embeddings.num_embeddings - 1)
        self.node_activation = nn.ReLU()

    def forward(self, support_pairs, predict=False, evaluate=False):
        batch_size = len(support_pairs)
        # create the aggregate data
        entities, neighbor_entities, neighbor_relations, offsets = [], [], [], []
        for batch_id, pair in enumerate(support_pairs):
            for entity in pair:
                entities.append(entity)
                edges = list(self.graph.my_edges(entity, batch_id=batch_id if not evaluate else None))
                offsets.append(len(neighbor_entities))
                neighbor_entities.extend(map(lambda x: x[1], edges))
                neighbor_relations.extend(map(lambda x: x[2], edges))
        # transform to torch.Tensor
        entities = torch.tensor(entities, dtype=torch.long, device=self.entity_embeddings.weight.device)
        neighbor_entities = torch.tensor(neighbor_entities, dtype=torch.long,
                                         device=self.entity_embeddings.weight.device)
        neighbor_relations = torch.tensor(neighbor_relations, dtype=torch.long,
                                          device=self.relation_embeddings.weight.device)
        offsets = torch.tensor(offsets, dtype=torch.long, device=self.entity_embeddings.weight.device)
        # retrieve entity embeddings and transform
        entity_embeddings = self.entity_embeddings(entities)
        entity_embeddings = self.transform_layer(entity_embeddings)
        # retrieve neighbor embeddings and aggregate
        neighbor_entity_embeds = nn.functional.embedding_bag(neighbor_entities, self.entity_embeddings.weight, offsets,
                                                             sparse=self.entity_embeddings.sparse)
        neighbor_relation_embeds = nn.functional.embedding_bag(neighbor_relations, self.relation_embeddings.weight,
                                                               offsets, sparse=self.relation_embeddings.sparse)
        # concatenate aggregate results and transform
        neighbor_embeddings = torch.cat((neighbor_entity_embeds, neighbor_relation_embeds), dim=-1)
        neighbor_embeddings = self.neighbor_layer(neighbor_embeddings)
        node_embeddings = self.node_activation(entity_embeddings + neighbor_embeddings)
        node_embeddings = node_embeddings.view(batch_size, -1)
        pair_embeddings = self.relation_activation(self.relation_layer(node_embeddings))
        if predict:
            scores = self.predict_layer(pair_embeddings)
            return scores
        else:
            return pair_embeddings


class CogGraph:
    def __init__(self, graph: grapher.KG, entity_dict: dict, relation_dict: dict, entity_pad: int, relation_pad: int,
                 max_nodes: int, max_neighbors: int, topk: int, device):
        self.graph = graph
        # the padding id for entities and relations
        self.entity_pad = entity_pad
        self.relation_pad = relation_pad
        self.node_pad = max_nodes
        self.node_pos_pad = max_nodes + 1
        self.entity_dict = entity_dict
        self.entity_num = len(entity_dict) + 1
        self.id2entity = sorted(self.entity_dict.keys(), key=self.entity_dict.get)
        self.relation_dict = relation_dict
        self.id2relation = sorted(self.relation_dict.keys(), key=self.relation_dict.get)
        self.max_nodes = max_nodes
        self.max_neighbors = max_neighbors
        self.device = device
        self.debug = False
        self.topk = topk

    def init(self, start_entities: list, other_correct_answers, evaluate=False):
        self.evaluate = evaluate
        self.batch_size = len(start_entities)
        # self.other_correct_answers = list2tensor(other_correct_answers, padding_idx=self.entity_pad, dtype=torch.long, device=self.device)
        self.other_correct_answers = [np.array(list(answer_set)) for answer_set in other_correct_answers]
        batch_index = torch.arange(0, self.batch_size, dtype=torch.long)
        # each line is the head entity and relation type
        self.neighbor_matrix = torch.zeros(self.batch_size, self.max_nodes + 2, self.max_neighbors, 2, dtype=torch.long)
        # padding the neighbors
        self.neighbor_matrix[:, :, :, 0] = self.node_pad
        self.neighbor_matrix[:, :, :, 1] = self.relation_pad
        # neighbor number of each node
        self.neighbor_nums = torch.zeros(self.batch_size, self.max_nodes + 2, dtype=torch.long)
        self.stop_states = [False for _ in range(self.batch_size)]
        self.frontier_queues = [deque([start_entity]) for start_entity in start_entities]
        self.node_lists = [[start_entity] for start_entity in start_entities]
        # self.antecedents = [[set()] for _ in range(self.batch_size)]
        self.entity2node = [{start_entity: 0} for start_entity in start_entities]
        self.entity_translate = torch.full((self.batch_size, self.entity_num), fill_value=self.node_pad,
                                           dtype=torch.long)
        self.entity_translate[batch_index, start_entities] = 0
        self.current_nodes = [{0} for _ in range(self.batch_size)]
        if self.debug:
            self.debug_outputs = [io.StringIO() for _ in range(self.batch_size)]

    def to_networkx(self):
        graphs = []
        for batch_id in range(self.batch_size):
            graph = networkx.MultiDiGraph()
            node_list = self.node_lists[batch_id]
            for node_id in range(len(self.node_lists[batch_id])):
                neighbor_num = self.neighbor_nums[batch_id, node_id]
                for neighbor_node, neighbor_relation in self.neighbor_matrix[batch_id, node_id, :neighbor_num].tolist():
                    graph.add_edge(self.id2entity[node_list[neighbor_node]], self.id2entity[node_list[node_id]],
                                   self.id2relation[neighbor_relation])
            graphs.append(graph)
        return graphs

    def step(self, currents, last_step=False):
        """
        :return current: current entities (batch_size, rollout_num)
                         current nodes (batch_size, rollout_num)
                         current masks (batch_size, rollout_num)
                candidates: node id (batch_size, rollout_num, max_neighbors, )
                            entity id (batch_size, rollout_num, max_neighbors, )
                            relation id (batch_size, rollout_num, max_neighbors, )
                max_neighbors can be dynamic
        """
        batch_index = torch.arange(0, self.batch_size)
        current_entities, current_masks = currents
        current_nodes = self.entity_translate[batch_index.unsqueeze(-1), current_entities]
        # TODO change KG.quick_edges
        candidates, candidate_masks = self.graph.quick_edges(current_entities.cpu())
        candidate_entities, candidate_relations = candidates.select(-1, 0), candidates.select(-1, 1)
        candidate_nodes = self.entity_translate[
            batch_index.unsqueeze(-1).unsqueeze(-1), candidate_entities]
        self.states = (
        (current_nodes, current_entities, current_masks), (candidate_nodes, candidate_entities, candidate_relations))
        if last_step:
            for batch_id in range(self.batch_size):
                other_masks = np.isin(candidate_entities[batch_id].numpy(), self.other_correct_answers[batch_id],
                                      invert=True)
                candidate_masks[batch_id] &= torch.from_numpy(other_masks).byte()
        return (current_nodes.to(self.device), current_entities.to(self.device), current_masks.to(self.device)), (
            candidate_nodes.to(self.device), candidate_entities.to(self.device), candidate_relations.to(self.device),
            candidate_masks.to(self.device))

    def update(self, actions: torch.LongTensor, action_nums: torch.LongTensor):
        """
        :param actions: batch_size, topk, each contains the next hop nodes of a current entity
        :param action_nums: batch_size
        :return: something to help update the hidden representations
        """
        current_nodes, current_entities, current_masks = self.states[0]
        batch_index = torch.arange(self.batch_size, device=self.device)
        candidate_nodes, candidate_entities, candidate_relations = self.states[1]
        candidate_num = candidate_nodes.size(2)
        rollout_ids, candidate_ids = (actions / candidate_num), actions.fmod(candidate_num)
        aggregate_entities = []
        attend_entities = []
        action_nums = action_nums.tolist()
        node_id_batch, neighbor_id_batch = [], []
        new_entity_data = [[], [], []]
        neighbor_nums = self.neighbor_nums.tolist()
        for batch_id, nexthops in enumerate(actions.tolist()):
            attend_entity, aggregate_entity, node_ids, neighbor_ids = set(), set(), [], []
            for topn in range(len(nexthops)):
                rollout_id, candidate_id = rollout_ids[batch_id, topn].item(), candidate_ids[batch_id, topn].item()
                head = current_nodes[batch_id, rollout_id].item()
                if head == self.node_pad or topn >= action_nums[batch_id]:
                    node_ids.append(self.node_pos_pad)
                    neighbor_ids.append(0)
                    continue
                entity = candidate_entities[batch_id, rollout_id, candidate_id].item()
                if entity not in self.entity2node[batch_id]:
                    self.entity2node[batch_id][entity] = len(self.node_lists[batch_id])
                    new_entity_data[0].append(batch_id)
                    new_entity_data[1].append(entity)
                    new_entity_data[2].append(len(self.node_lists[batch_id]))
                    self.node_lists[batch_id].append(entity)
                    # record the antecedents of new node
                    # self.antecedents[batch_id].append(
                    #     {self.node_lists[batch_id][head]} | self.antecedents[batch_id][head])
                    if self.debug:
                        self.debug_outputs[batch_id].write("New node added ")
                # TODO should we revisit old nodes?
                node_id = self.entity2node[batch_id][entity]
                attend_entity.add(entity)
                # If self-loop, skip
                if node_id == head:
                    node_ids.append(self.node_pos_pad)
                    neighbor_ids.append(0)
                    continue
                # TODO rewrite the last edge if overflow
                if neighbor_nums[batch_id][node_id] >= self.max_neighbors:
                    node_ids.append(self.node_pos_pad)
                    neighbor_ids.append(0)
                    continue
                neighbor_num = neighbor_nums[batch_id][node_id]
                if self.debug:
                    relation = candidate_relations[batch_id, topn].item()
                    self.debug_outputs[batch_id].write(
                        "Node: {}, Entity:{}, Relation: {}\n".format(node_id, entity, relation))
                aggregate_entity.add(entity)
                node_ids.append(node_id)
                neighbor_ids.append(neighbor_num)
                neighbor_nums[batch_id][node_id] += 1
                # update the antecedent information
                # self.antecedents[batch_id][node_id].add(self.node_lists[batch_id][head])
                # self.antecedents[batch_id][node_id].update(self.antecedents[batch_id][head])
            attend_entities.append(list(attend_entity))
            aggregate_entities.append(list(aggregate_entity))
            node_id_batch.append(node_ids)
            neighbor_id_batch.append(neighbor_ids)
        # update the entity translate
        self.entity_translate[new_entity_data[0], new_entity_data[1]] = torch.tensor(new_entity_data[2],
                                                                                     dtype=torch.long)
        self.neighbor_nums = torch.tensor(neighbor_nums, dtype=torch.long)
        # set the neighbor matrix of added nodes
        # NO_OP will select the last relation, which is just the padding
        update_relations = candidate_relations[batch_index.unsqueeze(-1), rollout_ids, candidate_ids]
        update_nodes = current_nodes[batch_index.unsqueeze(-1), rollout_ids]
        self.neighbor_matrix[batch_index.unsqueeze(-1), node_id_batch, neighbor_id_batch] = torch.stack(
            (update_nodes, update_relations), dim=-1)
        aggregate_nodes = [list(map(self.entity2node[batch_id].get, aggregate_entities[batch_id])) for batch_id in
                           range(self.batch_size)]
        attend_nums = torch.tensor(list(map(len, attend_entities)), dtype=torch.long, device=self.device)
        attend_entities = list2tensor(attend_entities, padding_idx=self.entity_pad, dtype=torch.long,
                                      device=self.device)
        attend_masks = torch.arange(attend_entities.size(-1), device=self.device).unsqueeze(0) < attend_nums.unsqueeze(
            -1)
        aggregate_nodes = list2tensor(aggregate_nodes, padding_idx=self.node_pos_pad, dtype=torch.long,
                                      device=self.device)
        aggregate_nums = torch.tensor(list(map(len, aggregate_entities)), dtype=torch.long, device=self.device)
        aggregate_entities = list2tensor(aggregate_entities, dtype=torch.long, device=self.device,
                                         padding_idx=self.entity_pad)
        # (batch_size, topk)
        neighbors_num = self.neighbor_nums[batch_index.unsqueeze(-1), aggregate_nodes].to(self.device)
        # (batch_size, topk, max_neighbors) get the neighbors of aim
        max_neighbor_num = torch.max(neighbors_num)
        neighbors = self.neighbor_matrix[batch_index.unsqueeze(-1), aggregate_nodes].to(self.device)
        neighbors = neighbors[:, :, :max_neighbor_num]
        if self.debug:
            for batch_id in range(len(actions)):
                self.debug_outputs[batch_id].write("Update aims:\n")
                for i in range(aggregate_nums[batch_id].item()):
                    aggregate_entity = aggregate_nodes[batch_id, i].item()
                    self.debug_outputs[batch_id].write("Node: {} ".format(aggregate_entity))
                    self.debug_outputs[batch_id].write(
                        str(self.neighbor_matrix[batch_id, aggregate_entity,
                            :neighbors_num[batch_id, i]].tolist()) + "\n")
        return (aggregate_nodes, aggregate_entities, aggregate_nums), (neighbors, neighbors_num), (
        attend_entities, attend_masks)


class Agent(nn.Module):
    def __init__(self, entity_embeddings: nn.Embedding, relation_embeddings: nn.Embedding, max_nodes: int,
                 embed_size: int, hidden_size: int, query_size: int, use_rnn: bool):
        nn.Module.__init__(self)
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.sqrt_embed_size = math.sqrt(self.embed_size)
        self.max_nodes = max_nodes
        if query_size is None:
            query_size = hidden_size
        self.query_size = query_size
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings
        self.use_rnn = use_rnn
        if self.use_rnn:
            self.hiddenRNN = nn.GRUCell(input_size=2 * self.embed_size, hidden_size=self.hidden_size)
        else:
            print("Not use RNN")
            self.hiddenRNN = nn.GRUCell(input_size=self.embed_size, hidden_size=self.hidden_size)
            self.update_layer = nn.Linear(self.embed_size + self.hidden_size, self.hidden_size)
            self.update_activation = nn.LeakyReLU()
        self.nexthop_layer = nn.Linear(self.hidden_size + self.query_size + self.embed_size, self.hidden_size)
        self.nexthop_activation = nn.LeakyReLU()
        self.candidate_layer = nn.Linear(2 * self.embed_size + self.hidden_size, self.hidden_size)
        self.candidate_activation = nn.LeakyReLU()
        self.rank_layer = nn.Linear(self.hidden_size + self.query_size, 1)
        # this should combine with the loss function
        self.rank_activation = nn.Sequential()
        self.debug = False

    def debug_tensor(self, t: torch.Tensor, local_info):
        # print a tensor according to its batch_id
        if self.debug:
            for batch_id in range(self.batch_size):
                self.debug_outputs[batch_id].write(str(t))

    def init(self, start_entities: torch.Tensor, query_representations=None):
        self.batch_size = start_entities.size(0)
        self.node_embeddings = torch.zeros(self.batch_size, self.max_nodes + 2, self.hidden_size, dtype=torch.float,
                                           device=self.entity_embeddings.weight.device)
        if query_representations is not None:
            if query_representations.size(0) == 1:
                query_representations = query_representations.expand(self.batch_size, -1)
            self.query_representations = query_representations
        else:
            self.query_representations = torch.zeros(self.batch_size, self.embed_size, dtype=torch.float,
                                                     device=self.entity_embeddings.weight.device)
        init_embeddings = torch.zeros(self.batch_size, self.hidden_size, device=start_entities.device)
        if not self.use_rnn:
            entity_embeddings = self.entity_embeddings(start_entities)
            init_embeddings = torch.cat((init_embeddings, entity_embeddings), dim=-1)
            init_embeddings = self.update_activation(self.update_layer(init_embeddings))
        self.node_embeddings[:, 0] = init_embeddings
        if self.debug:
            self.debug_outputs = [io.StringIO() for _ in range(self.batch_size)]

    def aggregate(self, aim_data, neighbor_data):
        """
        :param aim_data: (batch_size, topk) ids of updated nodes, used to update hidden representations
                         (batch_size, topk) ids of updated entities
                         (batch_size, ) number of aims
        :param neighbor_data: (batch_size, topk, max_neighbors, 2) node and relation type
                              (batch_size, topk) number of neighbors
        :return: None
        """
        aim_nodes, aim_entities, aim_nums = aim_data
        neighbors, neighbors_num = neighbor_data
        batch_size, topk, max_neighbors = neighbors.size()[:3]
        batch_index = torch.arange(batch_size, device=aim_entities.device)
        # resize the neighbors to gather representations
        neighbor_nodes, neighbor_relations = neighbors[:, :, :, 0], neighbors[:, :, :, 1]
        # (batch_size, topk, embed_size) get the entity embeddings of aims to update
        entity_embeddings = self.entity_embeddings(aim_entities)
        # (batch_size, topk, max_neighbors, embed_size)
        # get the hidden representations and relation embeddings of neighbors
        node_embeddings = self.node_embeddings[batch_index.view(-1, 1, 1).expand_as(neighbor_nodes), neighbor_nodes]
        relation_embeddings = self.relation_embeddings(neighbor_relations)
        # resize to get the correct embedding shapes
        # node_embeddings = node_embeddings.view(batch_size, topk, max_neighbors, self.embed_size)
        # relation_embeddings = relation_embeddings.view(batch_size, topk, max_neighbors, self.embed_size)
        if self.use_rnn:
            # (batch_size, topk, max_neighbors, 2 * embed_size) concatenated neighbor embeddings
            neighbor_embeddings = torch.cat(
                (entity_embeddings.unsqueeze(2).expand_as(relation_embeddings), relation_embeddings), dim=-1)
            updated_embeddings = self.hiddenRNN(
                neighbor_embeddings.reshape(batch_size * topk * max_neighbors, 2 * self.embed_size),
                node_embeddings.reshape(batch_size * topk * max_neighbors, self.hidden_size))
            updated_embeddings = updated_embeddings.reshape(batch_size, topk, max_neighbors, self.hidden_size)
        else:
            # (batch_size, topk, max_neighbors, embed_size + hidden_size) concatenated neighbor embeddings
            neighbor_embeddings = relation_embeddings
            updated_embeddings = self.hiddenRNN(
                neighbor_embeddings.reshape(batch_size * topk * max_neighbors, self.embed_size),
                node_embeddings.reshape(batch_size * topk * max_neighbors, self.hidden_size))
            updated_embeddings = updated_embeddings.reshape(batch_size, topk, max_neighbors, self.hidden_size)
        # mask padding neighbors
        masks = torch.arange(0, max_neighbors, device=neighbor_embeddings.device).view(1, 1,
                                                                                       -1) >= neighbors_num.unsqueeze(
            -1)
        updated_embeddings = updated_embeddings.masked_fill(masks.unsqueeze(-1), 0.0)
        # avoid division by zeros here
        neighbors_num = neighbors_num.type(torch.float) + (neighbors_num == 0.0).type(torch.float)
        updated_embeddings = updated_embeddings.sum(dim=2) / neighbors_num.unsqueeze(-1)
        if not self.use_rnn:
            updated_embeddings = torch.cat((updated_embeddings, entity_embeddings), dim=-1)
            updated_embeddings = self.update_activation(self.update_layer(updated_embeddings))
        # write the updated embeddings
        self.node_embeddings[batch_index.unsqueeze(-1), aim_nodes] = updated_embeddings

    def next_hop(self, currents: tuple, candidates) -> (torch.Tensor, torch.Tensor):
        """
        :param currents: (batch_size, rollout_num) pos of current entities
        :param candidates: entity id (batch_size, rollout_num, max_neighbors)
                           node pos (batch_size, rollout_num, max_neighbors)
                           relation id (batch_size, rollout_num, max_neighbors)
                           mask (batch_size, rollout_num, max_neighbors)
        :param topk: topk actions to select
        :return: entity id (batch_size, topk), relation id (batch_size, topk) mask (batch_size, topk)
        """
        current_nodes, current_entities, current_masks = currents
        candidate_nodes, candidate_entities, candidate_relations, candidate_masks = candidates
        batch_size, rollout_num, max_neighbors = candidate_nodes.size()
        batch_index = torch.arange(batch_size, device=current_nodes.device).unsqueeze(1)
        # (batch_size, embed_size) get the hidden representations of current nodes
        current_representations = self.node_embeddings[batch_index, current_nodes]
        if self.use_rnn:
            current_embeddings = self.entity_embeddings(current_entities)
        else:
            current_embeddings = torch.zeros(batch_size, rollout_num, self.embed_size, device=current_nodes.device)
        # concatenate the hidden states with query embeddings
        current_embeddings = torch.cat(
            (current_representations, self.query_representations.unsqueeze(1).expand(batch_size, rollout_num, -1),
             current_embeddings), dim=-1)
        current_state = self.nexthop_activation(self.nexthop_layer(current_embeddings))
        # (batch_size, rollout_num, max_neighbors, hidden_size) get the node representations of candidates
        node_embeddings = self.node_embeddings[batch_index.unsqueeze(-1), candidate_nodes]
        # (batch_size, rollout_num, max_neighbors, embed_size) get the entity embeddings of candidates
        entity_embeddings = self.entity_embeddings(candidate_entities)
        # (batch_size, rollout_num, max_neighbors, embed_size) get the relation embeddings of candidates
        relation_embeddings = self.relation_embeddings(candidate_relations)
        # (batch_size, max_neighbors, 2 * embed_size + hidden_size) concatenated representations
        candidate_embeddings = torch.cat((node_embeddings, entity_embeddings, relation_embeddings), dim=-1)
        # (batch_size, max_neighbors, embed_size) transformed representations
        candidate_embeddings = self.candidate_activation(self.candidate_layer(candidate_embeddings))
        # (batch_size, rollout_num, max_neighbors) (batch_size, rollout_num, hidden_size) * (batch_size, rollout_num, max_neighbors, hidden_size)
        candidate_scores = torch.matmul(candidate_embeddings, current_state.unsqueeze(-1)).squeeze(-1)
        candidate_scores /= self.sqrt_embed_size
        candidate_scores = candidate_scores.masked_fill(~candidate_masks, value=-1e5)
        return candidate_scores

    def compute_score(self, start_embeddings, end_embeddings, query_representations):
        diff_embeddings = start_embeddings - end_embeddings
        node_embeddings = torch.cat((diff_embeddings, query_representations.expand_as(diff_embeddings)), dim=-1)
        node_scores = self.rank_layer(node_embeddings).squeeze(-1)
        node_scores = self.rank_activation(node_scores)
        return node_scores

    def rank(self):
        """
        :return:
        """
        node_embeddings = self.node_embeddings[:, :self.max_nodes]
        # node_embeddings = torch.cat(
        #     (self.node_embeddings[:, :1].expand(-1, self.max_nodes, -1), node_embeddings, self.query_representations.unsqueeze(1).expand(*node_embeddings.size()[:2], -1)), dim=2)
        node_embeddings = torch.cat(
            (node_embeddings, self.query_representations.unsqueeze(1).expand(*node_embeddings.size()[:2], -1)), dim=2)
        node_scores = self.rank_layer(node_embeddings).squeeze(-1)
        node_scores = self.rank_activation(node_scores)
        return node_scores


class CogKR(nn.Module):
    def __init__(self, graph: grapher.KG, entity_dict: dict, relation_dict: dict, max_steps: int, max_nodes: int,
                 max_neighbors: int,
                 embed_size: int, topk: int, device, hidden_size: int = None, reward_policy='direct', use_summary=True,
                 baseline_lambda=0.0, onlyS=False, update_hidden=True,
                 use_rnn=True, sparse_embed=False, id2entity=None, id2relation=None):
        nn.Module.__init__(self)
        self.graph = graph
        self.entity_dict = entity_dict
        self.relation_dict = relation_dict
        self.id2entity = id2entity
        self.id2relation = id2relation
        self.max_steps = max_steps
        self.max_nodes = max_nodes
        self.max_neighbors = max_neighbors
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.onlyS = onlyS
        self.update_hidden = update_hidden
        if hidden_size is None:
            self.hidden_size = embed_size
        self.topk = topk
        self.entity_num = len(entity_dict) + 1
        self.entity_embeddings = nn.Embedding(self.entity_num, embed_size, padding_idx=len(entity_dict),
                                              sparse=sparse_embed)
        self.relation_num = len(relation_dict) + 1
        self.relation_embeddings = nn.Embedding(self.relation_num, embed_size, padding_idx=len(relation_dict),
                                                sparse=sparse_embed)
        self.cog_graph = CogGraph(self.graph, self.entity_dict, self.relation_dict, len(self.entity_dict),
                                  len(self.relation_dict), self.max_nodes, self.max_neighbors, self.topk, device)
        self.summary = Summary(self.hidden_size, graph, self.entity_embeddings, self.relation_embeddings)
        if use_summary:
            query_size = hidden_size
        else:
            query_size = embed_size
            print("Not use summary module")
        self.agent = Agent(self.entity_embeddings, self.relation_embeddings, self.max_nodes, self.embed_size,
                           self.hidden_size, query_size=query_size, use_rnn=use_rnn)
        if self.onlyS:
            self.loss = nn.MarginRankingLoss(margin=1.0)
        else:
            self.loss = nn.CrossEntropyLoss(reduction='sum')
        self.statistician = False
        self.statistics = {'graph_size': []}
        self.reward_policy = reward_policy
        self.reward_baseline = 0.0
        self.baseline_lambda = baseline_lambda

    def clear_statistics(self):
        for value in self.statistics.values():
            value.clear()

    def find_correct_tails(self, node_lists, end_entities, only_last=False):
        assert len(node_lists) == len(end_entities)
        correct_batch, correct_nodes = [], []
        for batch_id in range(len(node_lists)):
            end_entity = end_entities[batch_id]
            found_correct = False
            if only_last:
                node_list = self.cog_graph.current_nodes[batch_id]
            else:
                node_list = range(0, len(self.cog_graph.node_lists[batch_id]))
            for node_id in node_list:
                if self.cog_graph.node_lists[batch_id][node_id] == end_entity:
                    correct_nodes.append(node_id)
                    found_correct = True
                    break
            if found_correct:
                correct_batch.append(batch_id)
        return correct_batch, correct_nodes

    def get_correct_path(self, correct_tails, return_graph=False):
        correct_batch, correct_nodes = self.find_correct_tails(self.cog_graph.node_lists, correct_tails)
        graphs = self.cog_graph.to_networkx()
        if return_graph:
            reason_list = [{} for _ in range(len(correct_tails))]
        else:
            reason_list = [[] for _ in range(len(correct_tails))]
        for batch_id, node_id in zip(correct_batch, correct_nodes):
            correct_tail = self.id2entity[self.cog_graph.node_lists[batch_id][node_id]]
            head = self.id2entity[self.cog_graph.node_lists[batch_id][0]]
            graph = graphs[batch_id]
            if return_graph:
                nodes = shortest_path_length(graph, target=correct_tail)
                neighbor_dict = {}
                for node in nodes:
                    neighbor_dict[node] = []
                    for e1, e2, r in graph.edges(node, keys=True):
                        if e2 in nodes:
                            neighbor_dict[node].append((e1, e2, r))
                reason_list[batch_id] = neighbor_dict
            else:
                paths = list(networkx.algorithms.all_simple_paths(graphs[batch_id], head, correct_tail))
                reason_paths = []
                for path in paths:
                    reason_path = [path[0]]
                    last_node = path[0]
                    for node in path[1:]:
                        relation = list(
                            map(lambda x: x[2], filter(lambda x: x[1] == node, graph.edges(last_node, keys=True))))
                        last_node = node
                        reason_path.append((node, relation))
                    reason_paths.append(reason_path)
                reason_list[batch_id] = reason_paths
        return reason_list

    def forward(self, start_entities: list, other_correct_answers: list, end_entities=None,
                support_pairs=None, relations=None, evaluate=False, candidates=None):
        batch_size = len(start_entities)
        device = self.entity_embeddings.weight.device
        batch_index = torch.arange(0, batch_size, device=device)
        if support_pairs is not None:
            # support for evaluate
            support_embeddings = self.summary(support_pairs, evaluate=evaluate)
        else:
            relations = torch.tensor(relations, device=device, dtype=torch.long)
            support_embeddings = self.relation_embeddings(relations)
        if self.onlyS:
            self.reward = 0.0
            self.graph_size = 0
            start_entities = torch.tensor(start_entities, device=device, dtype=torch.long)
            if evaluate:
                if candidates is None:
                    candidates = torch.arange(self.entity_embeddings.num_embeddings,
                                              device=device, dtype=torch.long)
                else:
                    candidates = torch.tensor(list(candidates), dtype=torch.long, device=device)
                scores = self.agent.compute_score(self.entity_embeddings(
                    start_entities.unsqueeze(1)), self.entity_embeddings(candidates.unsqueeze(0)), support_embeddings)
                rank_scores, rank_index = torch.sort(scores, descending=True, dim=-1)
                results = candidates[rank_index].tolist()
                return results, rank_scores
            else:
                end_entities = torch.tensor(end_entities, device=device, dtype=torch.long)
                start_entities, end_entities = start_entities.unsqueeze(-1), end_entities.unsqueeze(-1)
                support_embeddings = support_embeddings.unsqueeze(1)
                negative_entities = torch.randint(0, self.entity_embeddings.num_embeddings, size=(
                    batch_size, 1000), device=device)
                positive_scores = self.agent.compute_score(self.entity_embeddings(start_entities),
                                                           self.entity_embeddings(end_entities), support_embeddings)
                negative_scores = self.agent.compute_score(self.entity_embeddings(start_entities),
                                                           self.entity_embeddings(negative_entities),
                                                           support_embeddings)
                labels = torch.ones(batch_size, 1000, dtype=torch.float, device=device)
                rank_loss = self.loss(positive_scores, negative_scores, labels)
                return torch.zeros(1, dtype=torch.float, device=device), rank_loss
        self.cog_graph.init(start_entities, other_correct_answers, evaluate=evaluate)
        start_entities = torch.tensor(start_entities, device=device, dtype=torch.long)
        if end_entities is not None:
            end_entities = torch.tensor(end_entities, device=device, dtype=torch.long)
        self.agent.init(start_entities, query_representations=support_embeddings)
        # TODO: Normalize graph loss and entropy loss with time step
        if self.reward_policy == 'direct':
            attention = torch.zeros(batch_size, self.entity_num, device=device)
            attention[batch_index, start_entities] = 1
        else:
            graph_loss, entropy_loss = 0.0, 0.0
        current_entities = start_entities.unsqueeze(1)
        current_masks = torch.ones(batch_size, 1, dtype=torch.uint8, device=device)
        for step in range(self.max_steps):
            currents, candidates = self.cog_graph.step((current_entities, current_masks), step == self.max_steps - 1)
            candidate_nodes, candidate_entities, candidate_relations, candidate_masks = candidates
            final_scores = self.agent.next_hop(currents, candidates)
            final_scores = torch.softmax(final_scores, dim=-1)
            if self.reward_policy == 'direct':
                attention_scores = attention[batch_index.unsqueeze(1), current_entities]
                final_scores = final_scores * attention_scores.unsqueeze(-1)
                final_scores = final_scores.reshape((batch_size, -1))
                if step != self.max_steps - 1:
                    action_scores, actions = final_scores.topk(k=min(self.topk, final_scores.size(-1)), dim=-1)
                    action_nums = (action_scores > 1e-10).sum(dim=1)
                    action_entities = candidate_entities.reshape((batch_size, -1))[batch_index.unsqueeze(-1), actions]
                    attention = torch_scatter.scatter_add(action_scores, action_entities, dim=-1,
                                                          dim_size=self.entity_num)
                    attention /= attention.sum(dim=-1, keepdim=True)
                    aims, neighbors, currents = self.cog_graph.update(actions, action_nums)
                    current_entities, current_masks = currents
                    if self.update_hidden:
                        self.agent.aggregate(aims, neighbors)
                else:
                    attention = torch_scatter.scatter_add(final_scores, candidate_entities.reshape((batch_size, -1)),
                                                          dim=-1, dim_size=self.entity_num)
                    attention /= attention.sum(dim=-1, keepdim=True)
                    if end_entities is not None:
                        action_entities = candidate_entities.reshape((batch_size, -1))
                        final_scores = (action_entities == end_entities.unsqueeze(-1)).float()
                        action_scores, actions = final_scores.topk(k=min(self.topk, final_scores.size(-1)), dim=-1)
                        action_nums = (action_scores > 1e-10).sum(dim=1)
                        self.cog_graph.update(actions, action_nums)
            elif self.reward_policy == 'stochastic':
                if not evaluate:
                    entropy = -(final_scores * torch.log(final_scores + 1e-10)).sum(dim=-1).mean()
                    entropy_loss += entropy
                final_scores = final_scores.masked_fill(~current_masks.unsqueeze(-1), 0.0)
                final_scores = final_scores.reshape((batch_size, -1))
                m = torch.distributions.multinomial.Multinomial(total_count=self.topk, probs=final_scores)
                action_counts = m.sample()
                log_prob = m.log_prob(action_counts)
                graph_loss = graph_loss + log_prob
                actions = torch.argsort(action_counts, dim=-1, descending=True)
                action_nums = (action_counts > 0).sum(dim=1)
                actions = actions[:, :torch.max(action_nums)]
                aims, neighbors, currents = self.cog_graph.update(actions, action_nums)
                current_entities, current_masks = currents
                if self.update_hidden:
                    self.agent.aggregate(aims, neighbors)
            else:
                raise NotImplemented
        if not evaluate:
            if self.reward_policy == 'direct':
                correct_batch = attention[batch_index, end_entities] > 1e-10
                self.reward = attention[batch_index, end_entities].sum().item() / batch_size
                wrong_batch = batch_index[~correct_batch]
                correct_batch = batch_index[correct_batch]
                loss = -torch.log(attention[correct_batch, end_entities[correct_batch]] + 1e-10).sum()
                loss = loss - torch.log(1.01 - attention[wrong_batch].sum(dim=-1)).sum()
                loss = loss / batch_size
                return loss, 0.0
            elif self.reward_policy == 'stochastic':
                rewards = (current_entities == end_entities.unsqueeze(-1)).any(dim=-1).float()
                rewards /= current_masks.float().sum(dim=-1) + 1e-10
                self.reward = rewards.mean().item()
                if self.baseline_lambda > 0.0:
                    self.reward_baseline = (1 - self.baseline_lambda) * self.reward_baseline + \
                                           self.baseline_lambda * rewards.mean().item()
                    rewards -= self.reward_baseline
                graph_loss = (- rewards.detach() * graph_loss).mean()
                return graph_loss, entropy_loss
            else:
                raise NotImplemented
        else:
            if self.reward_policy == 'direct':
                # Unbiased evaluation protocol
                # Zhiqing Sun, Shikhar Vashishth, Soumya Sanyal, Partha P. Talukdar, Yiming Yang:
                # A Re-evaluation of Knowledge Graph Completion Methods. CoRR abs/1911.03903 (2019)
                # https://arxiv.org/pdf/1911.03903.pdf
                # TODO this will make Wiki-One slow
                rand_idxs = list(range(self.entity_num))
                random.shuffle(rand_idxs)
                entity_list = torch.arange(self.entity_num, device=device)[rand_idxs]
                attention = attention[:, rand_idxs]
                scores, results = attention.topk(dim=-1, k=20)
                results = entity_list[results]
                results = results.tolist()
                scores = scores.tolist()
                return results, scores
            elif self.reward_policy == 'stochastic':
                results = current_entities.tolist()
                scores = graph_loss.unsqueeze(-1).expand_as(current_entities).tolist()
                return results, scores
            else:
                raise NotImplemented
