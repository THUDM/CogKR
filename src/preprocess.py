from utils import unserialize, serialize, load_facts, save_facts, add_reverse_relations, build_dict, save_index, \
    translate_facts, load_index
from torch_utils import load_embed_state_dict
from grapher import KG
from tqdm import tqdm
import os, networkx, itertools, random
import torch
from networkx.algorithms.link_analysis import pagerank


class Preprocess:
    def __init__(self, root_directory):
        self.root_directory = root_directory
        self.data_directory = os.path.join(self.root_directory, "data")
        if not os.path.exists(self.data_directory):
            os.mkdir(self.data_directory)
        self.data_path_dict = {
            "facts_data": "complete_graph", "graph_data": "train_graphs", "valid_tasks": "valid_tasks",
            "test_tasks": "test_tasks", "rel2candidate": "rel2candidates", "evaluate_graphs": "evaluate_graphs",
            "pagerank": 'pagerank', "entity_dict": "entity_dict", "relation_dict": "relation_dict"
        }

    def load_index(self):
        if os.path.exists(os.path.join(self.data_directory, 'ent2id.txt')):
            self.entity_dict = load_index(os.path.join(self.data_directory, 'ent2id.txt'))
            print("Load preprocessed entity index")
        elif os.path.exists(os.path.join(self.data_directory, 'ent2ids')):
            self.entity_dict = unserialize(os.path.join(self.data_directory, "ent2ids"), form='json')
            print("Load raw entity index")
        else:
            print("Entity Index not exist")
            self.entity_dict = {}
        if os.path.exists(os.path.join(self.data_directory, 'relation2id.txt')):
            self.relation_dict = load_index(os.path.join(self.data_directory, 'relation2id.txt'))
            print("Load preprocessed relation index")
        elif os.path.exists(os.path.join(self.data_directory, 'relation2ids')):
            self.relation_dict = unserialize(os.path.join(self.data_directory, "relation2ids"), form='json')
            print("Load raw relation index")
        else:
            print("Relation Index not exist")
            self.relation_dict = {}

    def load_raw_data(self):
        self.train_facts = load_facts(os.path.join(self.data_directory, "train.txt"))
        self.test_facts = load_facts(os.path.join(self.data_directory, "test_support.txt")) + load_facts(
            os.path.join(self.data_directory, "test_eval.txt"))
        self.valid_facts = load_facts(os.path.join(self.data_directory, "valid_support.txt")) + load_facts(
            os.path.join(self.data_directory, "valid_eval.txt"))
        if os.path.exists(os.path.join(self.data_directory, "rel2candidates.json")):
            print("Load rel2candidates")
            self.rel2candidate = unserialize(os.path.join(self.data_directory, "rel2candidates.json"))
        else:
            self.rel2candidate = {}

    def transform_data(self):
        self.train_facts = add_reverse_relations(self.train_facts)
        self.raw_train_facts = self.train_facts
        self.entity_dict, self.relation_dict = build_dict(
            itertools.chain(self.train_facts, add_reverse_relations(self.test_facts),
                            add_reverse_relations(self.valid_facts)), entity_dict=self.entity_dict,
            relation_dict=self.relation_dict)
        self.id2entity = sorted(self.entity_dict.keys(), key=self.entity_dict.get)
        self.id2relation = sorted(self.relation_dict.keys(), key=self.relation_dict.get)
        self.train_facts = translate_facts(self.train_facts, entity_dict=self.entity_dict,
                                           relation_dict=self.relation_dict)
        self.valid_facts = translate_facts(self.valid_facts, entity_dict=self.entity_dict,
                                           relation_dict=self.relation_dict)
        self.test_facts = translate_facts(self.test_facts, entity_dict=self.entity_dict,
                                          relation_dict=self.relation_dict)
        if self.rel2candidate:
            self.rel2candidate = {self.relation_dict[key]: list(map(self.entity_dict.get, value)) for key, value in
                                  self.rel2candidate.items() if key in self.relation_dict}
        else:
            relations = set(map(lambda x: x[1], self.valid_facts)) | set(map(lambda x: x[1], self.test_facts))
            self.rel2candidate = {key: list(range(len(self.entity_dict))) for key in relations}

    def save_data(self, save_train=False):
        if save_train:
            save_facts(self.raw_train_facts, os.path.join(self.data_directory, "train.txt"))
        if not os.path.exists(os.path.join(self.data_directory, "rel2candidates.json")):
            rel2candidates = {key: list(map(self.id2entity.__getitem__, value)) for key, value in
                              self.rel2candidate.items()}
            train_tasks = set(map(lambda x: x[1], load_facts(os.path.join(self.data_directory, "train.txt"))))
            for task in train_tasks:
                rel2candidates[task] = self.id2entity
            serialize(rel2candidates, os.path.join(self.data_directory, "rel2candidates.json"), in_json=True)
        serialize(self.rel2candidate, os.path.join(self.data_directory, "rel2candidates"))
        save_index(self.id2entity, os.path.join(self.data_directory, "ent2id.txt"))
        save_index(self.id2relation, os.path.join(self.data_directory, "relation2id.txt"))

    def search_evaluate_graph(self, wiki=True):
        self.kg = KG(self.train_facts, entity_num=len(self.entity_dict), relation_num=len(self.relation_dict))
        rel2candidates = {key: set(value) for key, value in self.rel2candidate.items()}
        if wiki:
            evaluate_graphs = {}
            for head, relation, tail in tqdm(itertools.chain(self.valid_facts, self.test_facts)):
                candidates = rel2candidates[relation] | {tail}
                evaluate_graphs[(head, relation, tail)] = self.kg.graphs_among(head, candidates, cutoff=3)
        else:
            evaluate_pairs = {}
            for head, relation, tail in itertools.chain(self.valid_facts, self.test_facts):
                evaluate_pairs.setdefault((head, relation), set())
                evaluate_pairs[(head, relation)].add(tail)
            evaluate_graphs = {}
            for (head, relation), tail_set in tqdm(evaluate_pairs.items()):
                candidates = rel2candidates[relation] | tail_set
                evaluate_graphs[(head, relation)] = self.kg.graphs_among(head, candidates, cutoff=3)
        serialize(evaluate_graphs, os.path.join(self.data_directory, "evaluate_graphs"))

    def compute_pagerank(self):
        self.kg = KG(self.train_facts, entity_num=len(self.entity_dict), relation_num=len(self.relation_dict))
        graph = networkx.DiGraph(self.kg.to_networkx())
        print("Begin to compute pagerank")
        self.pagerank = pagerank(graph)
        self.pagerank = [self.pagerank[entity] for entity in range(len(self.pagerank))]
        print("Begin to save pagerank")
        with open(os.path.join(self.data_directory, "pagerank.txt"), "w") as output:
            for value in self.pagerank:
                output.write("{}\n".format(value))
        print("Complete save pagerank")


def preprocess_embedding(embed_path, data_path, model, output_path, output_model, entity_dict, relation_dict):
    state_dict = load_embed_state_dict(embed_path, model)

    print(state_dict.keys())
    print(list(map(lambda x: x.size(), state_dict.values())))

    def output_embedding(id_file, embeddings, output_file, entity_dict):
        id2id = {}
        with open(os.path.join(data_path, id_file)) as file:
            for idx, line in enumerate(file):
                line = line.strip().split()
                entity = line[0]
                if entity in entity_dict:
                    id2id[entity_dict[entity]] = idx
        sums = 0
        with open(output_file, "w") as file:
            for i in range(max(id2id)):
                if i in id2id:
                    idx = id2id[i]
                    embedding = embeddings[idx].tolist()
                    sums += 1
                else:
                    embedding = torch.randn(embeddings.size(1)).tolist()
                file.write("\t".join(list(map(str, embedding))) + "\n")
            print(sums / max(id2id))

    output_embedding(os.path.join(data_path, "entity2id.txt"), state_dict['entity_embeddings.weight'],
                     os.path.join(output_path, "entity2vec." + output_model), entity_dict)
    output_embedding(os.path.join(data_path, "relation2id.txt"), state_dict['relation_embeddings.weight'],
                     os.path.join(output_path, "relation2vec." + output_model), relation_dict)

if __name__ == "__main__":
    from parse_args import args

    preprocess = Preprocess(args.directory)
    preprocess.load_raw_data()
    preprocess.load_index()
    preprocess.transform_data()
    if args.process_data:
        preprocess.save_data(save_train=args.save_train)
        preprocess.compute_pagerank()
    if args.search_evaluate_graph:
        print("Search Evaluate Graph")
        preprocess.search_evaluate_graph(wiki=args.wiki)