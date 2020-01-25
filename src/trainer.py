import random, itertools
import numpy as np
from tqdm import tqdm
from grapher import KG
from torch.utils.data import BatchSampler, SequentialSampler, WeightedRandomSampler, RandomSampler
from utils import ListConcat, serialize
from collections import defaultdict


class Trainer:
    def __init__(self, graph: KG, train_facts, cutoff, reverse_relation: list, train_graphs: dict = None,
                 test_tasks=None, validate_tasks=None, evaluate_graphs: dict = None,
                 rel2candidate: dict = None, id2entity=None, id2relation=None, fact_dist=None,
                 weighted_sample=True, ignore_onehop=False, meta_learn=True, sample_weight=0.75, rollout_num=1, test_rollout_num=1):
        self.graph = graph
        self.cutoff = cutoff
        self.reverse_relation = reverse_relation
        self.ignore_onehop = ignore_onehop
        self.id2entity = id2entity
        self.id2relation = id2relation
        self.train_query, self.train_support = {}, {}
        self.meta_learn = meta_learn
        self.rollout_num = rollout_num
        self.test_rollout_num = test_rollout_num
        print("Ignore onehop: {}".format(ignore_onehop))
        print("Sample weight:", sample_weight)
        # Note that we **do not** add reverse relations here
        self.e1rel2_e2_test, self.e1rel2_e2_train = {}, {}
        for head, relation, tail in itertools.chain(train_facts):
            if relation not in self.train_query:
                self.train_query[relation] = []
            if relation not in self.train_support:
                self.train_support[relation] = []
            pair = (head, tail)
            if (train_graphs is None or (head, relation, tail) in train_graphs) and (
                    fact_dist is None or -1 < fact_dist[(head, relation, tail)] < 5):
                self.train_query[relation].append(pair)
            else:
                self.train_support[relation].append(pair)
            self.e1rel2_e2_train.setdefault((head, relation), set())
            self.e1rel2_e2_train[(head, relation)].add(tail)
            self.e1rel2_e2_test.setdefault((head, relation), set())
            self.e1rel2_e2_test[(head, relation)].add(tail)
        self.train_graphs = train_graphs
        self.task_ground, self.task_support = {}, {}
        # note we can't filter facts here
        if test_tasks is not None:
            test_support, test_eval = test_tasks
            self.test_relations = []
            for head, relation, tail in test_support:
                self.test_relations.append(relation)
                self.task_support[relation] = (head, tail)
                if not self.meta_learn:
                    self.train_query[relation] = [(head, tail)]
                self.e1rel2_e2_train.setdefault((head, relation), set())
                self.e1rel2_e2_train[(head, relation)].add(tail)
                self.e1rel2_e2_test.setdefault((head, relation), set())
                self.e1rel2_e2_test[(head, relation)].add(tail)
            for head, relation, tail in test_eval:
                if relation in self.train_query and relation not in self.test_relations:
                    self.test_relations.append(relation)
                    self.task_support[relation] = None
                self.e1rel2_e2_test.setdefault((head, relation), set())
                self.e1rel2_e2_test[(head, relation)].add(tail)
                self.task_ground.setdefault(relation, [])
                self.task_ground[relation].append((head, tail))
        if validate_tasks is not None:
            valid_support, valid_eval = validate_tasks
            self.validate_relations = []
            for head, relation, tail in valid_support:
                self.validate_relations.append(relation)
                self.task_support[relation] = (head, tail)
                if not self.meta_learn:
                    self.train_query[relation] = [(head, tail)]
                self.e1rel2_e2_train.setdefault((head, relation), set())
                self.e1rel2_e2_train[(head, relation)].add(tail)
                self.e1rel2_e2_test.setdefault((head, relation), set())
                self.e1rel2_e2_test[(head, relation)].add(tail)
            for head, relation, tail in valid_eval:
                if relation in self.train_query and relation not in self.validate_relations:
                    self.validate_relations.append(relation)
                    self.task_support[relation] = None
                self.e1rel2_e2_test.setdefault((head, relation), set())
                self.e1rel2_e2_test[(head, relation)].add(tail)
                self.task_ground.setdefault(relation, [])
                self.task_ground[relation].append((head, tail))
        self.rel2candidate = rel2candidate
        if self.rel2candidate is not None:
            valid_hit = self.check_rel2candidate(self.validate_relations)
            print("Validate relations in rel2candidate: {}".format(valid_hit))
            test_hit = self.check_rel2candidate(self.test_relations)
            print("Test relations in rel2candidate: {}".format(test_hit))
        self.evaluate_graphs = evaluate_graphs
        if self.evaluate_graphs is not None:
            valid_hit = self.check_evaluate_graphs(self.validate_relations)
            print("Validate facts in evaluate graphs: {}".format(valid_hit))
            test_hit = self.check_evaluate_graphs(self.test_relations)
            print("Test facts in evaluate graphs: {}".format(test_hit))
        if self.meta_learn:
            min_fact = 2
        else:
            min_fact = 0
        self.train_relations = list(
            filter(lambda x: len(self.train_query[x]) >= min_fact or x in self.test_relations or x in self.validate_relations,
                   self.train_query))
        print("Train relations: {}".format(len(self.train_relations)))
        self.pretrain_relations = list(
            filter(lambda x: len(self.train_support[x]) + len(self.train_query[x]) > 10, self.train_support))
        if weighted_sample:
            # use weighted sampler
            self.train_sampler = WeightedRandomSampler(
                weights=list(map(lambda x: len(self.train_query[x]) ** sample_weight, self.train_relations)), num_samples=1)
        else:
            # or use uniform sampler
            self.train_sampler = RandomSampler(range(len(self.train_relations)), num_samples=1, replacement=True)

    def check_rel2candidate(self, relations):
        hit = 0
        for relation in relations:
            if relation in self.rel2candidate:
                hit += 1
        return hit / len(relations)

    def check_evaluate_graphs(self, relations):
        hit, num = 0, 0
        for relation in relations:
            num += len(self.task_ground[relation])
            for head, tail in self.task_ground[relation]:
                if (head, relation, tail) in self.evaluate_graphs or (head, relation) in self.evaluate_graphs:
                    hit += 1
        return hit / num

    def predict_sample(self, batch_size):
        pairs, labels = [], []
        self.graph.eval()
        self.graph.ignore_edges = []
        for _ in range(batch_size):
            # for classification, we use uniform sample here
            relation = random.choice(self.pretrain_relations)
            pair = random.choice(ListConcat(self.train_support[relation], self.train_query[relation]))
            pairs.append(pair)
            labels.append(relation)
            self.graph.ignore_edges.append(
                ((pair[0], relation, pair[1]), (pair[1], self.reverse_relation[relation], pair[0])))
        return pairs, labels

    def sample(self, batch_size, specific_relation=None):
        self.graph.eval()
        if self.ignore_onehop:
            self.graph.ignore_pairs = []
        self.graph.ignore_edges = []
        support_pairs, relations, query_heads, query_tails, other_correct_answers, graphs = [], [], [], [], [], []
        for _ in range(batch_size):
            if specific_relation is not None:
                relation = specific_relation
            else:
                relation = self.train_relations[next(iter(self.train_sampler))]
            inv_relation = self.reverse_relation[relation]
            if self.meta_learn:
                support_pair = random.choice(ListConcat(self.train_support[relation], self.train_query[relation]))
                query_pair = random.choice(self.train_query[relation])
                while query_pair == support_pair:
                    query_pair = random.choice(self.train_query[relation])
                if self.train_graphs is None:
                    graph = None
                else:
                    graph = self.train_graphs.get((query_pair[0], relation, query_pair[1]))
            else:
                query_pair = random.choice(self.train_query[relation])
                graph = None
            if self.ignore_onehop:
                ignore_edges = ((query_pair[1], query_pair[0], inv_relation),)
            else:
                ignore_edges = ((query_pair[0], query_pair[1], relation),
                                (query_pair[1], query_pair[0], inv_relation))
            ground = self.e1rel2_e2_train[(query_pair[0], relation)] - {query_pair[1]}
            relations += [relation] * self.rollout_num
            query_heads += [query_pair[0]] * self.rollout_num
            query_tails += [query_pair[1]] * self.rollout_num
            other_correct_answers += [ground] * self.rollout_num
            graphs += [graph] * self.rollout_num
            if self.meta_learn:
                support_pairs += [support_pair] * self.rollout_num
            self.graph.ignore_edges += [ignore_edges] * self.rollout_num
            if self.ignore_onehop:
                self.graph.ignore_pairs += [(query_pair[0], query_pair[1])] * self.rollout_num
        self.graph.ignore_batch()
        return support_pairs, query_heads, query_tails, relations, other_correct_answers, graphs

    def train_evaluate(self, relation_num=None, specific_relation=None):
        if specific_relation is None:
            relations = random.sample(self.train_relations, relation_num)
        else:
            relations = [specific_relation]
        for relation in relations:
            facts = self.train_query[relation]
            support_pair = facts[0]
            evaluate_pairs = facts[1:]
            if evaluate_pairs:
                self.graph.ignore_relations = {relation, self.reverse_relation[relation]}
                yield relation, support_pair, evaluate_pairs
                self.graph.ignore_relations = None

    def evaluate(self, specific_relation=None, mode='test', use_graph=True):
        self.graph.eval()
        if specific_relation is None:
            if mode == 'test':
                evaluate_relations = self.test_relations
            elif mode == 'valid':
                evaluate_relations = self.validate_relations
            else:
                raise NotImplementedError
        else:
            evaluate_relations = [specific_relation]
        for relation in evaluate_relations:
            evaluate_facts = self.task_ground[relation]
            support_pair = self.task_support[relation]
            ground_sets = [self.e1rel2_e2_test[(head, relation)] - {tail} for head, tail in evaluate_facts]
            if use_graph and self.evaluate_graphs is not None:
                evaluate_graphs = []
                for head, tail in evaluate_facts:
                    if (head, relation, tail) in self.evaluate_graphs:
                        evaluate_graphs.append(self.evaluate_graphs[(head, relation, tail)])
                    else:
                        evaluate_graphs.append(self.evaluate_graphs.get((head, relation)))
                yield relation, support_pair, evaluate_facts, ground_sets, evaluate_graphs
            else:
                yield relation, support_pair, evaluate_facts, ground_sets

    def evaluate_generator(self, module, data_loader, batch_size=16, save_result=None, save_graph=None):
        if save_result:
            save_result = open(save_result, "w")
        if save_graph:
            self.reason_graphs = {}
        for relation_id, support_pair, evaluate_data, ground_sets, *evaluate_graphs in tqdm(data_loader):
            relation = self.id2relation[relation_id]
            if relation_id in self.rel2candidate:
                candidates = set(self.rel2candidate[relation_id])
            else:
                candidates = None
            for idx in BatchSampler(SequentialSampler(evaluate_data), batch_size=batch_size, drop_last=False):
                batch = evaluate_data[idx[0]:idx[-1] + 1]
                ground = ground_sets[idx[0]: idx[-1] + 1]
                start_entities = [data[0] for data in batch] * self.test_rollout_num
                tail_entities = [data[1] for data in batch] * self.test_rollout_num
                other_correct_answers = [item for item in ground] * self.test_rollout_num
                if len(evaluate_graphs) > 0:
                    graphs = evaluate_graphs[0][idx[0]: idx[-1] + 1]
                else:
                    graphs = None
                if self.meta_learn:
                    results, scores = module(start_entities, other_correct_answers=other_correct_answers, support_pairs=[support_pair], evaluate=True, evaluate_graphs=graphs, candidates=candidates | set(tail_entities))
                else:
                    relations = [relation_id]
                    results, scores = module(start_entities, other_correct_answers=other_correct_answers, relations=relations, evaluate=True, evaluate_graphs=graphs, candidates=candidates | set(tail_entities))
                if save_graph:
                    reason_paths = module.get_correct_path(relation_id, tail_entities, return_graph=True)
                    for batch_id in range(len(batch)):
                        self.reason_graphs["\t".join(
                            (self.id2entity[start_entities[batch_id]], relation, self.id2entity[tail_entities[batch_id]]))] = \
                            reason_paths[batch_id]
                for batch_id in range(len(batch)):
                    entities = [entity for rollout_id in range(self.test_rollout_num) for entity in results[rollout_id * len(batch) + batch_id]]
                    score = [score for rollout_id in range(self.test_rollout_num) for score in scores[rollout_id * len(batch) + batch_id]]
                    sorted_indx = np.argsort(-np.array(score))
                    seen, result = set(), []
                    for r in sorted_indx:
                        if entities[r] not in seen and entities[r] != batch[batch_id][0]:
                            result.append(entities[r])
                            seen.add(entities[r])
                    # for rollout_id in range(self.test_rollout_num):
                    #     result, score = results[rollout_id * len(batch) + batch_id], scores[rollout_id * len(batch) + batch_id]
                    #     for j in range(len(result)):
                    #         entity_scores[batch_id][result[j]] += score[j]
                    # result = list(entity_scores[batch_id])
                    # result = sorted(result, key=entity_scores[batch_id].get, reverse=True)
                    if save_result:
                        save_result.write("\t".join([self.id2entity[start_entities[batch_id]], relation,
                                                    self.id2entity[batch[batch_id][1]]] + list(
                            map(lambda x: self.id2entity[x], result))) + "\n")
                    result = list(filter(lambda x: x not in ground[batch_id] or x == batch[batch_id][1], result))
                    if relation_id in self.rel2candidate:
                        result = list(filter(lambda x: x in candidates or x == batch[batch_id][1], result))
                    yield [batch[batch_id][1]], result                    
        if save_result:
            save_result.close()
        if save_graph:
            serialize(self.reason_graphs, save_graph, in_json=True)
