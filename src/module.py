import torch
import torch.nn as nn
import torch.nn.functional as F
import grapher
from collections import deque
from torch_utils import list2tensor
import networkx
from networkx.algorithms.shortest_paths.generic import shortest_path_length
import math, random, io, statistics


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
        self.id2entity = sorted(self.entity_dict.keys(), key=self.entity_dict.get)
        self.relation_dict = relation_dict
        self.id2relation = sorted(self.relation_dict.keys(), key=self.relation_dict.get)
        self.max_nodes = max_nodes
        self.max_neighbors = max_neighbors
        self.device = device
        self.debug = False
        self.topk = topk

    def init(self, start_entities: list, ground_graphs=None, evaluate=False):
        self.evaluate = evaluate
        self.batch_size = len(start_entities)
        batch_index = torch.arange(0, self.batch_size, dtype=torch.long)
        # # save the ignore relations
        # self.ignore_relations = ignore_relations
        # each line is the head entity and relation type
        self.ground_graphs = ground_graphs
        self.neighbor_matrix = torch.zeros(self.batch_size, self.max_nodes + 2, self.max_neighbors, 2, dtype=torch.long)
        # padding the neighbors
        self.neighbor_matrix[:, :, :, 0] = self.node_pad
        self.neighbor_matrix[:, :, :, 1] = self.relation_pad
        # neighbor number of each node
        self.neighbor_nums = torch.zeros(self.batch_size, self.max_nodes + 2, dtype=torch.long)
        self.stop_states = [False for _ in range(self.batch_size)]
        self.frontier_queues = [deque([start_entity]) for start_entity in start_entities]
        self.node_lists = [[start_entity] for start_entity in start_entities]
        self.antecedents = [[set()] for _ in range(self.batch_size)]
        self.entity2node = [{start_entity: 0} for start_entity in start_entities]
        self.entity_translate = torch.full((self.batch_size, len(self.entity_dict) + 1), fill_value=self.node_pad,
                                           dtype=torch.long)
        self.entity_translate[batch_index, start_entities] = 0
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

    def step(self):
        """
        :return ids of current entities (batch_size, )
                candidates: node id (batch_size, max_neighbors, )
                            entity id (batch_size, max_neighbors, )
                            relation id (batch_size, max_neighbors, )
                max_neighbors can be dynamic
        """
        batch_index = torch.arange(0, self.batch_size)
        current_nodes, current_entities, current_antecedents = [], [], []
        for batch_id in range(self.batch_size):
            if len(self.node_lists[batch_id]) >= self.max_nodes or len(self.frontier_queues[batch_id]) == 0:
                current_nodes.append(self.node_pad)
                current_entities.append(self.entity_pad)
                current_antecedents.append([])
                self.stop_states[batch_id] = True
                if self.debug:
                    self.debug_outputs[batch_id].write("search process stoped\n")
            if self.stop_states[batch_id]:
                continue
            current_entity = self.frontier_queues[batch_id].popleft()
            current_node = self.entity2node[batch_id][current_entity]
            current_nodes.append(current_node)
            current_entities.append(current_entity)
            if self.evaluate:
                current_antecedents.append(self.antecedents[batch_id][current_node])
            else:
                current_antecedents.append(list(self.antecedents[batch_id][current_node]))
        current_nodes = torch.tensor(current_nodes, dtype=torch.long)
        if self.evaluate:
            candidate_nodes, candidate_entities, candidate_relations = [], [], []
            for batch_id in range(self.batch_size):
                current_entity = current_entities[batch_id]
                edges = self.graph.my_edges(current_entity, batch_id=None)
                if self.ground_graphs is not None and self.ground_graphs[batch_id]:
                    if current_entity in self.ground_graphs[batch_id]:
                        edges = filter(lambda x: x[1] in self.ground_graphs[batch_id][current_entity], edges)
                    else:
                        edges = []
                edges = filter(lambda x: x[1] not in current_antecedents[batch_id], edges)
                edges = list(edges)
                candidate_entities.append(list(map(lambda x: x[1], edges)))
                candidate_relations.append(list(map(lambda x: x[2], edges)))
                candidate_nodes.append(list(
                    map(lambda x: self.entity2node[batch_id].get(x, self.node_pad), candidate_entities[batch_id])))
            candidate_nums = torch.tensor(list(map(len, candidate_nodes)), dtype=torch.long, device=self.device)
            candidate_nodes = list2tensor(candidate_nodes, padding_idx=self.node_pad, dtype=torch.long)
            candidate_entities = list2tensor(candidate_entities, padding_idx=self.entity_pad, dtype=torch.long)
            candidate_relations = list2tensor(candidate_relations, padding_idx=self.relation_pad, dtype=torch.long)
            candidate_masks = torch.arange(0, candidate_entities.size(1), device=self.device).unsqueeze(
                0) < candidate_nums.unsqueeze(-1)
            self.states = (current_nodes, (candidate_nodes, candidate_entities, candidate_relations))
        else:
            current_entities = torch.tensor(current_entities, dtype=torch.long)
            candidates, candidate_masks = self.graph.quick_edges(current_entities)
            candidate_entities, candidate_relations = candidates[:, :, 0], candidates[:, :, 1]
            candidate_nodes = self.entity_translate[
                batch_index.unsqueeze(-1).expand_as(candidate_entities), candidate_entities]
            self.states = (current_nodes, (candidate_nodes, candidate_entities, candidate_relations))
            current_antecedents = list2tensor(current_antecedents, padding_idx=self.entity_pad, dtype=torch.long,
                                              device=self.device).unsqueeze(
                1)
            candidate_entities = candidate_entities.to(self.device)
            candidate_masks = candidate_masks.to(self.device)
            candidate_masks &= ((candidate_entities.unsqueeze(-1) == current_antecedents).sum(dim=-1) == 0)
        return current_nodes.to(self.device), (
            candidate_nodes.to(self.device), candidate_entities.to(self.device), candidate_relations.to(self.device),
            candidate_masks.to(self.device))

    def update(self, actions, action_nums):
        """
        :param actions: batch_size lists, each contains the next hop nodes of a current entity
        :return: something to help update the hidden representations
        """
        currents = self.states[0]
        batch_index = torch.arange(self.batch_size, device=self.device)
        candidate_nodes, candidate_entities, candidate_relations = self.states[1]
        aim_entities = []
        action_nums = action_nums.tolist()
        node_id_batch, neighbor_id_batch = [], []
        new_entity_data = [[], [], []]
        neighbor_nums = self.neighbor_nums.tolist()
        for batch_id, nexthops in enumerate(actions.tolist()):
            head = currents[batch_id].item()
            aim_entity, node_ids, neighbor_ids = set(), [], []
            for topn, action_id in enumerate(nexthops):
                if head == self.node_pad or topn >= action_nums[batch_id]:
                    node_ids.append(self.node_pos_pad)
                    neighbor_ids.append(0)
                    continue
                entity = candidate_entities[batch_id, action_id].item()
                if entity not in self.entity2node[batch_id]:
                    # avoid too many nodes
                    if len(self.node_lists[batch_id]) >= self.max_nodes:
                        node_ids.append(self.node_pos_pad)
                        neighbor_ids.append(0)
                        continue
                    self.entity2node[batch_id][entity] = len(self.node_lists[batch_id])
                    new_entity_data[0].append(batch_id)
                    new_entity_data[1].append(entity)
                    new_entity_data[2].append(len(self.node_lists[batch_id]))
                    # TODO should we revisit old nodes?
                    self.frontier_queues[batch_id].append(entity)
                    self.node_lists[batch_id].append(entity)
                    # record the antecedents of new node
                    self.antecedents[batch_id].append(
                        {self.node_lists[batch_id][head]} | self.antecedents[batch_id][head])
                    if self.debug:
                        self.debug_outputs[batch_id].write("New node added ")
                node_id = self.entity2node[batch_id][entity]
                neighbor_num = neighbor_nums[batch_id][node_id]
                if neighbor_num >= self.max_neighbors:
                    node_ids.append(self.node_pos_pad)
                    neighbor_ids.append(0)
                    continue
                if self.debug:
                    relation = candidate_relations[batch_id, action_id].item()
                    self.debug_outputs[batch_id].write(
                        "Node: {}, Entity:{}, Relation: {}\n".format(node_id, entity, relation))
                aim_entity.add(entity)
                node_ids.append(node_id)
                neighbor_ids.append(neighbor_num)
                neighbor_nums[batch_id][node_id] += 1
                # update the antecedent information
                self.antecedents[batch_id][node_id].add(self.node_lists[batch_id][head])
                self.antecedents[batch_id][node_id].update(self.antecedents[batch_id][head])
            aim_entities.append(list(aim_entity))
            node_id_batch.append(node_ids)
            neighbor_id_batch.append(neighbor_ids)
        # update the entity translate
        self.entity_translate[new_entity_data[0], new_entity_data[1]] = torch.tensor(new_entity_data[2],
                                                                                     dtype=torch.long)
        self.neighbor_nums = torch.tensor(neighbor_nums, dtype=torch.long)
        # set the neighbor matrix of added nodes
        update_relations = candidate_relations[batch_index.unsqueeze(-1), actions]
        self.neighbor_matrix[batch_index.unsqueeze(-1), node_id_batch, neighbor_id_batch] = torch.stack(
            (currents.unsqueeze(-1).expand_as(update_relations), update_relations), dim=-1)
        aim_nodes = [list(map(self.entity2node[batch_id].get, aim_entities[batch_id])) for batch_id in
                     range(self.batch_size)]
        aim_nodes = list2tensor(aim_nodes, padding_idx=self.node_pos_pad, dtype=torch.long, device=self.device)
        aim_nums = torch.tensor(list(map(len, aim_entities)), dtype=torch.long, device=self.device)
        aim_entities = list2tensor(aim_entities, dtype=torch.long, device=self.device, padding_idx=self.entity_pad)
        # (batch_size, topk)
        neighbors_num = self.neighbor_nums[batch_index.unsqueeze(-1).expand_as(aim_nodes), aim_nodes].to(self.device)
        # (batch_size, topk, max_neighbors) get the neighbors of aim
        neighbors = self.neighbor_matrix[batch_index.unsqueeze(-1).expand_as(aim_nodes), aim_nodes].to(self.device)
        if self.debug:
            for batch_id in range(len(actions)):
                self.debug_outputs[batch_id].write("Update aims:\n")
                for i in range(aim_nums[batch_id].item()):
                    aim_entity = aim_nodes[batch_id, i].item()
                    self.debug_outputs[batch_id].write("Node: {} ".format(aim_entity))
                    self.debug_outputs[batch_id].write(
                        str(self.neighbor_matrix[batch_id, aim_entity, :neighbors_num[batch_id, i]].tolist()) + "\n")
        return (aim_nodes, aim_entities, aim_nums), (neighbors, neighbors_num)


class Agent(nn.Module):
    def __init__(self, entity_embeddings: nn.Embedding, relation_embeddings: nn.Embedding, max_nodes: int,
                 embed_size: int, hidden_size: int, query_size: int = None):
        nn.Module.__init__(self)
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.sqrt_embed_size = math.sqrt(self.embed_size)
        # self.num_entity = num_entity
        # self.num_relation = num_relation
        self.max_nodes = max_nodes
        if query_size is None:
            query_size = hidden_size
        # self.entity_embeddings = nn.Embedding(num_entity + 1, embed_size, padding_idx=num_entity)
        # self.relation_embeddings = nn.Embedding(num_relation + 1, embed_size, padding_idx=num_relation)
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings
        self.hidden_layer = nn.Linear(embed_size, hidden_size)
        self.pass_layer = nn.Linear(embed_size + hidden_size, hidden_size)
        self.pass_activation = nn.Sequential()
        self.update_activation = nn.LeakyReLU()
        self.nexthop_layer = nn.Linear(hidden_size + query_size, hidden_size)
        self.nexthop_activation = nn.LeakyReLU()
        self.candidate_layer = nn.Linear(2 * embed_size + hidden_size, hidden_size)
        self.candidate_activation = nn.LeakyReLU()
        self.gate_layer = nn.Linear(hidden_size + query_size, 1)
        self.rank_layer = nn.Linear(hidden_size + query_size, 1)
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
        init_embeddings = self.entity_embeddings(start_entities)
        init_embeddings = self.update_activation(self.hidden_layer(init_embeddings))
        self.node_embeddings[:, 0] = init_embeddings
        if self.debug:
            self.debug_outputs = [io.StringIO() for _ in range(self.batch_size)]

    def aggregate(self, aim_data, neighbor_data):
        """
        :param aim_data: (batch_size, topk) ids of updated entities
                         (batch_size, ) number of aims
                         (batch_size, topk) ids of updated nodes, used to update hidden representations
        :param neighbor_data: (batch_size, topk, max_neighbors, 2) node and relation type
                              (batch_size, topk) number of neighbors
        :return: None
        """
        node_pos, aims, aims_num = aim_data
        neighbors, neighbors_num = neighbor_data
        batch_size, topk, max_neighbors = neighbors.size()[:3]
        batch_index = torch.arange(batch_size, device=aims.device)
        # resize the neighbors to gather representations
        neighbor_nodes, neighbor_relations = neighbors[:, :, :, 0], neighbors[:, :, :, 1]
        # (batch_size, topk, embed_size) get the entity embeddings of aims to update
        aim_embeddings = self.entity_embeddings(aims)
        # (batch_size, topk, max_neighbors, embed_size)
        # get the hidden representations and relation embeddings of neighbors
        node_embeddings = self.node_embeddings[batch_index.view(-1, 1, 1).expand_as(neighbor_nodes), neighbor_nodes]
        relation_embeddings = self.relation_embeddings(neighbor_relations)
        # resize to get the correct embedding shapes
        # node_embeddings = node_embeddings.view(batch_size, topk, max_neighbors, self.embed_size)
        # relation_embeddings = relation_embeddings.view(batch_size, topk, max_neighbors, self.embed_size)
        # (batch_size, topk, max_neighbors, 2 * embed_size) concatenated neighbor embeddings
        neighbor_embeddings = torch.cat((node_embeddings, relation_embeddings), dim=-1)
        neighbor_embeddings = self.pass_activation(self.pass_layer(neighbor_embeddings))
        # mask padding neighbors
        masks = torch.arange(0, max_neighbors, device=neighbor_embeddings.device).view(1, 1,
                                                                                       -1) >= neighbors_num.unsqueeze(
            -1)
        neighbor_embeddings = neighbor_embeddings.masked_fill(masks.unsqueeze(-1), 0.0)
        # avoid division by zeros here
        neighbors_num = neighbors_num.type(torch.float) + (neighbors_num == 0.0).type(torch.float)
        neighbor_embeddings = neighbor_embeddings.sum(dim=2) / neighbors_num.unsqueeze(-1)
        # apply layer normalization
        updated_embeddings = self.hidden_layer(aim_embeddings) + neighbor_embeddings
        updated_embeddings = self.update_activation(updated_embeddings)
        # write the updated embeddings
        self.node_embeddings[batch_index.unsqueeze(-1).expand_as(node_pos), node_pos] = updated_embeddings

    def next_hop(self, currents: torch.Tensor, candidates) -> (torch.Tensor, torch.Tensor):
        """
        :param currents: (batch_size, ) pos of current entities
        :param candidates: entity id (batch_size, max_neighbors)
                           node pos (batch_size, max_neighbors)
                           relation id (batch_size, max_neighbors)
                           num (batch_size, )
        :param topk: topk actions to select
        :return: entity id (batch_size, topk), relation id (batch_size, topk) mask (batch_size, topk)
        """
        candidate_nodes, candidate_entities, candidate_relations, candidate_masks = candidates
        batch_size, max_neighbors = candidate_nodes.size()
        batch_index = torch.arange(batch_size, device=currents.device)
        # (batch_size, embed_size) get the hidden representations of current nodes
        current_embeddings = self.node_embeddings[batch_index, currents]
        # concatenate the hidden states with query embeddings
        current_embeddings = torch.cat((current_embeddings, self.query_representations), dim=1)
        current_state = self.nexthop_activation(self.nexthop_layer(current_embeddings))
        # (batch_size, max_neighbors, embed_size) get the node representations of candidates
        node_embeddings = self.node_embeddings[batch_index.unsqueeze(-1).expand_as(candidate_nodes), candidate_nodes]
        # (batch_size, max_neighbors, embed_size) get the entity embeddings of candidates
        entity_embeddings = self.entity_embeddings(candidate_entities)
        # (batch_size, max_neighbors, embed_size) get the relation embeddings of candidates
        relation_embeddings = self.relation_embeddings(candidate_relations)
        # (batch_size, max_neighbors, 3 * embed_size) concatenated representations
        candidate_embeddings = torch.cat((node_embeddings, entity_embeddings, relation_embeddings), dim=-1)
        # (batch_size, max_neighbors, embed_size) transformed representations
        candidate_embeddings = self.candidate_activation(self.candidate_layer(candidate_embeddings))
        # (batch_size, 1) thresholds for expansion
        thresholds = self.gate_layer(current_embeddings)
        # (batch_size, max_neighbors) (batch_size, 1, 1, embed_size) * (batch_size, max_neighbors, embed_size, 1)
        candidate_scores = torch.matmul(current_state.unsqueeze(1).unsqueeze(2),
                                        candidate_embeddings.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        candidate_scores /= self.sqrt_embed_size
        candidate_scores = candidate_scores.masked_fill(~candidate_masks, value=-1e5)
        # generate the final scores with thresholds
        final_scores = torch.cat((candidate_scores, thresholds), dim=1)
        return final_scores, candidate_masks

    def rank(self):
        """
        :return:
        """
        node_embeddings = self.node_embeddings[:, :self.max_nodes]
        node_embeddings = torch.cat(
            (node_embeddings, self.query_representations.unsqueeze(1).expand(*node_embeddings.size()[:2], -1)), dim=2)
        node_scores = self.rank_layer(node_embeddings).squeeze(-1)
        node_scores = self.rank_activation(node_scores)
        return node_scores


class CogKR(nn.Module):
    def __init__(self, graph: grapher.KG, entity_dict: dict, relation_dict: dict, max_nodes: int, max_neighbors: int,
                 embed_size: int, topk: int, device, hidden_size: int = None, reward_policy='direct', use_summary=True, baseline_lambda=0.0,
                 sparse_embed=False, id2entity=None, id2relation=None):
        nn.Module.__init__(self)
        self.graph = graph
        self.entity_dict = entity_dict
        self.relation_dict = relation_dict
        self.id2entity = id2entity
        self.id2relation = id2relation
        self.max_nodes = max_nodes
        self.max_neighbors = max_neighbors
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        if hidden_size is None:
            self.hidden_size = embed_size
        self.topk = topk
        self.entity_embeddings = nn.Embedding(len(entity_dict) + 1, embed_size, padding_idx=len(entity_dict),
                                              sparse=sparse_embed)
        self.relation_embeddings = nn.Embedding(len(relation_dict) + 1, embed_size, padding_idx=len(relation_dict),
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
                           self.hidden_size, query_size=query_size)
        self.loss = nn.CrossEntropyLoss(reduction='sum')
        self.statistician = False
        self.statistics = {'graph_size': []}
        self.reward_policy = reward_policy
        self.reward_baseline = 0.0
        self.baseline_lambda = baseline_lambda

    def clear_statistics(self):
        for value in self.statistics.values():
            value.clear()

    def find_correct_tails(self, node_lists, end_entities):
        assert len(node_lists) == len(end_entities)
        correct_batch, correct_nodes = [], []
        for batch_id in range(len(node_lists)):
            end_entity = end_entities[batch_id]
            found_correct = False
            if len(self.cog_graph.node_lists[batch_id]) > 1:
                for node_id in range(len(self.cog_graph.node_lists[batch_id])):
                    if self.cog_graph.node_lists[batch_id][node_id] == end_entity:
                        correct_nodes.append(node_id)
                        found_correct = True
                        break
            if found_correct:
                correct_batch.append(batch_id)
        return correct_batch, correct_nodes

    def get_correct_path(self, relations, correct_tails, verbose=False, return_graph=False):
        correct_batch, correct_nodes = self.find_correct_tails(self.cog_graph.node_lists, correct_tails)
        graphs = self.cog_graph.to_networkx()
        if return_graph:
            reason_list = [{} for _ in range(len(correct_tails))]
        else:
            reason_list = [[] for _ in range(len(correct_tails))]
        for batch_id, node_id in zip(correct_batch, correct_nodes):
            if verbose:
                print("{}: Query relation: {}".format(batch_id, self.id2relation[relations[batch_id]]))
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

    def forward(self, start_entities: list, end_entities=None, ground_graphs=None, evaluate_graphs=None,
                support_pairs=None, relations=None, evaluate=False, stochastic=False):
        batch_size = len(start_entities)
        if support_pairs is not None:
            # support for evaluate
            if not (isinstance(support_pairs[0], list) or isinstance(support_pairs[0], tuple)):
                support_pairs = [support_pairs]
            support_embeddings = self.summary(support_pairs, evaluate=evaluate)
        else:
            relations = torch.tensor(relations, device=self.entity_embeddings.weight.device, dtype=torch.long)
            support_embeddings = self.relation_embeddings(relations)
        self.cog_graph.init(start_entities, evaluate_graphs, evaluate=evaluate)
        start_entities = torch.tensor(start_entities, device=self.entity_embeddings.weight.device, dtype=torch.long)
        self.agent.init(start_entities, query_representations=support_embeddings)
        graph_loss = 0.0
        while True:
            currents, candidates = self.cog_graph.step()
            if sum(self.cog_graph.stop_states) == batch_size:
                break
            final_scores, candidate_masks = self.agent.next_hop(currents, candidates)
            # evaluation
            if stochastic:
                # use stochastic policy to sample actions
                probs = nn.functional.softmax(final_scores, dim=1) + 1e-10
                m = torch.distributions.multinomial.Multinomial(total_count=self.topk, probs=probs)
                final_counts = m.sample()
                action_counts = final_counts[:, :-1]
                action_counts, actions = action_counts.topk(k=min(self.topk, action_counts.size(1)), dim=-1)
                action_nums = (action_counts > 0).sum(dim=1)
                if not evaluate:
                    # compute policy gradient here
                    graph_loss = m.log_prob(final_counts) + graph_loss
            elif ground_graphs is None:
                action_scores = final_scores[:, :-1]
                # select the top-k results and compare with thresholds
                topk = min(self.topk, action_scores.size(1))
                action_scores, actions = torch.topk(action_scores, topk)
                action_nums = (action_scores > final_scores[:, -1].unsqueeze(-1)).sum(dim=-1)
                if not evaluate:
                    # we can't optimize deterministic policy without ground truth
                    raise NotImplemented
            else:
                actions = []
                # utilize the ground graph
                action_labels = torch.zeros(batch_size, final_scores.size(1))
                action_labels[:, -1] = 1.0
                for batch_id in range(batch_size):
                    current_entity = self.cog_graph.node_lists[batch_id][self.cog_graph.states[0][batch_id]]
                    ground_graph = ground_graphs[batch_id]
                    action = []
                    # don't compute loss for stopped episodes
                    if self.cog_graph.stop_states[batch_id]:
                        action_labels[batch_id, -1] = 0.0
                    else:
                        if current_entity in ground_graph:
                            for action_id, candidate_entity in enumerate(self.cog_graph.states[1][1][batch_id]):
                                if candidate_entity in ground_graph[current_entity]:
                                    action.append(action_id)
                                    action_labels[batch_id, action_id] = 1.0
                                    action_labels[batch_id, -1] = 0.0
                        else:
                            for action_id, candidate_entity in enumerate(self.cog_graph.states[1][1][batch_id]):
                                if candidate_entity in ground_graph:
                                    action.append(action_id)
                                    action_labels[batch_id, action_id] = 1.0
                                    action_labels[batch_id, -1] = 0.0
                    actions.append(action)
                    if not evaluate:
                        # compute the log probability loss for multinomial distribution
                        action_labels = action_labels.to(final_scores.device)
                        # normalize the label here
                        action_normal = action_labels.sum(dim=1, keepdim=True)
                        action_normal += (action_normal == 0.0).type(torch.float)
                        action_labels /= action_normal
                        current_loss = - (action_labels * torch.nn.functional.log_softmax(final_scores, dim=-1)).sum(
                            dim=-1)
                        # print(current_loss)
                        graph_loss = current_loss.mean() + graph_loss
            aims, neighbors = self.cog_graph.update(actions, action_nums)
            self.agent.aggregate(aims, neighbors)
        if self.statistician:
            self.statistics['graph_size'] += list(map(len, self.cog_graph.node_lists))
        self.graph_size = statistics.mean(map(len, self.cog_graph.node_lists))
        node_scores = self.agent.rank()
        if not evaluate:
            correct_batch, correct_nodes = self.find_correct_tails(self.cog_graph.node_lists, end_entities)
            if len(correct_batch) > 0:
                batch_index = torch.arange(len(correct_batch), device=node_scores.device)
                correct_batch = torch.tensor(correct_batch, dtype=torch.long, device=node_scores.device)
                correct_nodes = torch.tensor(correct_nodes, dtype=torch.long, device=node_scores.device)
                node_scores = node_scores[correct_batch]
                node_nums = torch.tensor(list(map(len, self.cog_graph.node_lists)), dtype=torch.float,
                                         device=node_scores.device)[correct_batch]
                masks = torch.arange(0, self.agent.max_nodes, dtype=torch.float, device=node_scores.device).expand_as(
                    node_scores) >= node_nums.unsqueeze(-1)
                node_scores.masked_fill_(masks, -1e6)
                rank_loss = self.loss(node_scores, correct_nodes) / batch_size
            else:
                rank_loss = torch.zeros(1, device=node_scores.device)
            if stochastic:
                if self.reward_policy == 'ranking':
                    _, rank_index = torch.sort(node_scores, descending=True, dim=-1)
                    rewards = torch.zeros(batch_size, device=node_scores.device, dtype=torch.float)
                    if len(correct_batch) > 0:
                        rewards[correct_batch] = 1.0 / (rank_index[batch_index, correct_nodes] + 1).float()
                elif self.reward_policy == 'predict':
                    rewards, _ = F.softmax(node_scores, dim=-1).max(dim=-1)
                elif self.reward_policy == 'probability':
                    rewards = torch.full((batch_size,), -10, device=node_scores.device, dtype=torch.float)
                    if len(correct_batch) > 0:
                        rewards[correct_batch] = torch.log(F.softmax(node_scores.detach(), dim=-1)[batch_index, correct_nodes] + 1e-10)
                elif self.reward_policy == 'direct':
                    # directly use finding reward
                    rewards = torch.zeros(batch_size, device=node_scores.device, dtype=torch.float)
                    rewards[correct_batch] = 1.0
                else:
                    raise NotImplementedError
                if self.baseline_lambda > 0.0:
                    self.reward_baseline = (1 - self.baseline_lambda) * self.reward_baseline + self.baseline_lambda * rewards.mean().item()
                    rewards -= self.reward_baseline
                graph_loss = (- rewards * graph_loss).mean()
                self.reward = rewards.mean().item()
            return graph_loss, rank_loss
        else:
            # delete the start entities
            node_scores[:, 0] = -1e5
            _, rank_index = torch.sort(node_scores, descending=True, dim=-1)
            rank_index = rank_index.tolist()
            results = []
            for batch_id in range(batch_size):
                result = [self.cog_graph.node_lists[batch_id][node_id] for node_id in rank_index[batch_id] if
                          node_id < len(self.cog_graph.node_lists[batch_id])]
                results.append(result)
            return results
