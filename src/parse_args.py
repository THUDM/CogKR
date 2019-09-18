import argparse

parser = argparse.ArgumentParser(description='Cognitive Graph for Knowledge Graph Reasoning')
parser.add_argument('--directory', type=str, help='root directory')
parser.add_argument('--gpu', type=int, help='specify the gpu number to use')
parser.add_argument('--config', default=None)
parser.add_argument('--comment', type=str, default='init')
parser.add_argument('--log_dir', type=str, default=None)

parser.add_argument('--inference', action='store_true')
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--process_data', action='store_true')
parser.add_argument('--save_train', action='store_true')
parser.add_argument('--wiki', action='store_true')
parser.add_argument('--search_evaluate_graph', action='store_true')
parser.add_argument('--get_fact_dist', action='store_true')
parser.add_argument('--save_result', type=str, default=None)
parser.add_argument('--save_graph', type=str, default=None)
parser.add_argument('--inference_time', action='store_true')
parser.add_argument('--save_minerva', action='store_true')

parser.add_argument('--load_state', type=str, default=None, help='specify the state file to load')
parser.add_argument('--load_pretrain', type=str, default=None)
parser.add_argument('--load_embed', type=str, default=None)

parser.add_argument('--sparse_embed', action='store_true')
parser.add_argument('--relation_encode', action='store_true')

args = parser.parse_args()
