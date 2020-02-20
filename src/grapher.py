from collections import OrderedDict, deque, defaultdict
from torch_utils import list2tensor
import torch
import networkx


def paths2graph(paths):
    neighbor_dict = defaultdict(set)
    for path in paths:
        last_node = path[0][0]
        for node, edge in path[1:]:
            neighbor_dict[last_node].add(node)
            last_node = node
            neighbor_dict.setdefault(node, set())
    return dict(neighbor_dict)


class KG:
    """methods to process KB"""

    def __init__(self, facts: list, entity_num: int, relation_num: int, build_matrix: bool, node_scores: list = None,
                 train_width=None, device=torch.device('cpu'), add_self_loop=False, self_loop_id=None):
        self.dataset = facts
        self.device = device
        self.train_width = train_width
        self.entity_num = entity_num
        self.relation_num = relation_num
        if entity_num is None:
            entity_num = max(map(lambda x: max(x[0], x[2]), facts)) + 1
        self.edge_data = [[] for _ in range(entity_num + 1)]
        for e1, rel, e2 in facts:
            self.edge_data[e1].append((e1, e2, rel))
        if node_scores is not None and (train_width is not None):
            neighbor_limit = train_width
            self.node_scores = node_scores
            for head in range(len(self.edge_data)):
                self.edge_data[head].sort(key=lambda x: self.node_scores[x[1]], reverse=True)
                self.edge_data[head] = self.edge_data[head][:neighbor_limit]
        if build_matrix:
            self.build_edge_matrix(add_self_loop=add_self_loop, self_loop_id=self_loop_id)
        self.ignore_relations = None
        self.ignore_edges = None
        self.ignore_relation_vectors = None
        self.ignore_edge_vectors = None
        self.ignore_pairs = None
        self.ignore_pair_vectors = None

    def build_edge_matrix(self, add_self_loop=False, self_loop_id=None):
        if add_self_loop:
            for entity in range(len(self.edge_data)):
                self.edge_data[entity].append((entity, entity, self_loop_id))
        self.edge_nums = torch.tensor(list(map(len, self.edge_data)), dtype=torch.long)
        edge_entities = [list(map(lambda x: x[1], edges)) for edges in self.edge_data]
        edge_relations = [list(map(lambda x: x[2], edges)) for edges in self.edge_data]
        edge_entities = list2tensor(edge_entities, padding_idx=self.entity_num, dtype=torch.int, device=self.device)
        edge_relations = list2tensor(edge_relations, padding_idx=self.relation_num, dtype=torch.int,
                                     device=self.device)
        self.edge_matrix = torch.stack((edge_entities, edge_relations), dim=2)

    def to_networkx(self, multi=True, neighbor_limit=None):
        if neighbor_limit is None:
            neighbor_limit = max(map(len, self.edge_data))
        if multi:
            graph = networkx.MultiDiGraph()
        else:
            graph = networkx.DiGraph()
        for edges in self.edge_data:
            for head, tail, relation in edges[:neighbor_limit]:
                if multi:
                    graph.add_edge(head, tail, relation)
                else:
                    graph.add_edge(head, tail)
        return graph

    def quick_edges(self, sources):
        batch_size, rollout_num = sources.size()
        # batch_size, rollout_num, max_candidate
        edges = self.edge_matrix[sources]
        # batch_size, rollout_num
        edge_nums = self.edge_nums[sources]
        max_edges = torch.max(edge_nums).item()
        edges = edges[:, :, :max_edges].long()
        # batch_size, rollout_num, max_candidate
        masks = torch.arange(0, edges.size(-2)).reshape((1, 1, -1)) < edge_nums.unsqueeze(-1)
        if self.ignore_edge_vectors is not None or self.ignore_pair_vectors is not None:
            masks = masks.reshape((batch_size * rollout_num, max_edges))
            flat_edges = edges.reshape((batch_size * rollout_num, max_edges, 2))
            if self.ignore_edge_vectors is not None:
                for ignore_edge_vector in self.ignore_edge_vectors:
                    correct_batch = (sources == ignore_edge_vector[0].unsqueeze(1)).reshape(-1)
                    ignore_data = ignore_edge_vector[1].repeat(rollout_num, 1, 1).transpose(0, 1).reshape(
                        batch_size * rollout_num, 2)
                    if correct_batch.any():
                        masks[correct_batch] &= (
                                (flat_edges[correct_batch] != ignore_data[correct_batch].unsqueeze(1)).sum(
                                    dim=2) != 0)
            if self.ignore_pair_vectors is not None:
                correct_batch = (sources == self.ignore_pair_vectors[0].unsqueeze(1)).reshape(-1)
                ignore_data = self.ignore_pair_vectors[1].repeat(rollout_num, 1).transpose(0, 1).reshape(
                    batch_size * rollout_num)
                if correct_batch.any():
                    masks[correct_batch] &= (
                                flat_edges[correct_batch].select(-1, 0) != ignore_data[correct_batch].unsqueeze(1))
            masks = masks.reshape((batch_size, rollout_num, max_edges))
        return edges, masks

    def ignore_batch(self):
        if self.ignore_edges is not None:
            ignore_edge_vectors = torch.tensor(self.ignore_edges, dtype=torch.long, device=self.device)
            edge_num = ignore_edge_vectors.size(1)
            self.ignore_edge_vectors = [(ignore_edge_vectors[:, i, 0], ignore_edge_vectors[:, i, 1:]) for i in
                                        range(edge_num)]
        else:
            self.ignore_edge_vectors = None
        if self.ignore_pairs is not None:
            self.ignore_pair_vectors = (
            torch.tensor(list(map(lambda x: x[0], self.ignore_pairs)), dtype=torch.long, device=self.device),
            torch.tensor(list(map(lambda x: x[1], self.ignore_pairs)), dtype=torch.long, device=self.device))
        else:
            self.ignore_pair_vectors = None

    def eval(self):
        self.ignore_pairs = self.ignore_edges = self.ignore_relations = self.ignore_pair_vectors = self.ignore_relation_vectors = self.ignore_edge_vectors = None

    def edge_between(self, source, target):
        for edge in self.edge_data[source]:
            if edge[1] == target:
                yield edge

    def my_edges(self, source=None, batch_id=None, ignore_relations=None, ignore_edges=None):
        """
        This method will return edges, support mask some relations and/or edges
        :param batch_id:
        :param source: will only report edges incident to the node
        :return:
        """
        if batch_id is not None:
            if self.ignore_relations is not None:
                ignore_relations = self.ignore_relations[batch_id]
            if self.ignore_edges is not None:
                ignore_edges = self.ignore_edges[batch_id]
        else:
            if isinstance(self.ignore_relations, set) or isinstance(self.ignore_relations, list):
                ignore_relations = self.ignore_relations
            if isinstance(self.ignore_edges, set) or isinstance(self.ignore_edges, list):
                ignore_edges = self.ignore_edges
        if ignore_relations is None:
            ignore_relations = []
        if ignore_edges is None:
            ignore_edges = []
        edges = self.edge_data[source]
        if ignore_relations is None:
            for edge in edges:
                if edge in ignore_edges:
                    continue
                yield edge
        else:
            for edge in edges:
                if edge[2] in ignore_relations:
                    continue
                if edge in ignore_edges:
                    continue
                yield edge

    def graph_between(self, source, target, cutoff, ignore_relations=None, ignore_edges=None):
        paths = self.paths_between(source, target, cutoff, ignore_relations=ignore_relations, ignore_edges=ignore_edges)
        return paths2graph(paths)

    def graphs_between(self, source, targets, cutoff, ignore_relations=None, ignore_edges=None):
        paths_dict = {target: [] for target in targets}
        if ignore_relations is None:
            ignore_relations = set()
        visited = OrderedDict.fromkeys([source])
        stack = [self.my_edges(source, ignore_relations=ignore_relations, ignore_edges=ignore_edges)]
        while stack:
            children = stack[-1]
            child = next(children, None)
            if child is None:
                stack.pop()
                visited.popitem()
            elif len(visited) < cutoff:
                if child[1] not in visited:
                    if child[1] in targets:
                        paths_dict[child[1]].append(list(visited.items()) + [child[1:]])
                    visited[child[1]] = child[2]
                    stack.append(self.my_edges(child[1], ignore_relations=ignore_relations, ignore_edges=ignore_edges))
            else:  # len(visited) == cutoff:
                if child[1] in targets and child[1] not in visited:
                    paths_dict[child[1]].append(list(visited.items()) + [child[1:]])
                for _, v, c in list(children):
                    if v in targets and v not in visited:
                        paths_dict[v].append(list(visited.items()) + [(v, c)])
                stack.pop()
                visited.popitem()
        return {target: paths2graph(paths) for target, paths in paths_dict.items()}

    def graphs_among(self, source, targets, cutoff, ignore_relations=None, ignore_edges=None):
        paths = []
        if ignore_relations is None:
            ignore_relations = set()
        visited = OrderedDict.fromkeys([source])
        stack = [self.my_edges(source, ignore_relations=ignore_relations, ignore_edges=ignore_edges)]
        while stack:
            children = stack[-1]
            child = next(children, None)
            if child is None:
                stack.pop()
                visited.popitem()
            elif len(visited) < cutoff:
                if child[1] not in visited:
                    if child[1] in targets:
                        paths.append(list(visited.items()) + [child[1:]])
                    visited[child[1]] = child[2]
                    stack.append(self.my_edges(child[1], ignore_relations=ignore_relations, ignore_edges=ignore_edges))
            else:  # len(visited) == cutoff:
                if child[1] in targets and child[1] not in visited:
                    paths.append(list(visited.items()) + [child[1:]])
                for _, v, c in list(children):
                    if v in targets and v not in visited:
                        paths.append(list(visited.items()) + [(v, c)])
                stack.pop()
                visited.popitem()
        return paths2graph(paths)

    def paths_between(self, source, target, cutoff=None, ignore_relations=None, ignore_edges=None, batch_id=None):
        if cutoff is None:
            cutoff = len(self.edge_data)
        visited = OrderedDict.fromkeys([source])
        stack = [self.my_edges(source, ignore_relations=ignore_relations, ignore_edges=ignore_edges, batch_id=None)]
        while stack:
            children = stack[-1]
            child = next(children, None)
            if child is None:
                stack.pop()
                visited.popitem()
            elif len(visited) < cutoff:
                if child[1] == target:
                    yield list(visited.items()) + [child[1:]]
                elif child[1] not in visited:
                    visited[child[1]] = child[2]
                    stack.append(self.my_edges(child[1], ignore_relations=ignore_relations, ignore_edges=ignore_edges,
                                               batch_id=None))
            else:  # len(visited) == cutoff:
                if child[1] == target:
                    yield list(visited.items()) + [child[1:]]
                for _, v, c in list(children):
                    if v == target:
                        yield list(visited.items()) + [(v, c)]
                stack.pop()
                visited.popitem()

    def multi_source_shortest_path_length(self, sources, cutoff=None):
        seen = {}  # level (number of hops) when seen in BFS
        level = 0  # the current level
        nextlevel = sources  # dict of nodes to check at next level
        if cutoff is None:
            cutoff = float('inf')

        while nextlevel and cutoff >= level:
            thislevel = nextlevel  # advance to next level
            nextlevel = set()  # and start a new list (fringe)
            for v in thislevel:
                if v not in seen:
                    seen[v] = level  # set the level of vertex v
                    adj = set(map(lambda x: x[1], self.my_edges(v)))
                    nextlevel.update(adj)  # add neighbors of v
            level += 1
        return seen
