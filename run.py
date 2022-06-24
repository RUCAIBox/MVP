import argparse
import sys
from mvp.utils import get_local_time
sys.path.append('mvp/evaluator/multiwoz')

from mvp.quick_start import run_mvp
from logging import getLogger

if __name__ == '__main__':
    time = get_local_time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='MVP', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='cnndm', help='name of datasets')
    parser.add_argument('--config_files', type=str, default=None, help='config files')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    result = run_mvp(model=args.model, dataset=args.dataset, config_file_list=config_file_list, config_dict={})
