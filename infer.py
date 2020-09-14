import os
import sys
import time
import logging
import argparse
import ast
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
import paddle.fluid as fluid

# from model import TSN1
from model.Dresnet import TSNResNet # 这是修改的代码
from reader import KineticsReader
from config import parse_config, merge_configs, print_configs

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        type=str,
        default='Dresnet',
        help='name of model to train.')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/Dresnet.txt',
        help='path to config file of model')
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=False,
        help='default use gpu.')
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='weight path, None to use weights from Paddle.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='sample number in a batch for inference.')
    parser.add_argument(
        '--filelist',
        type=str,
        default=None,
        help='path to inferenece data file lists file.')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='mini-batch interval to log.')
    parser.add_argument(
        '--infer_topk',
        type=int,
        default=1,
        help='topk predictions to restore.')
    parser.add_argument(
        '--save_dir', type=str, default='./output', help='directory to store results')
    args = parser.parse_args()
    return args


def infer(args):
    # parse config
    config = parse_config(args.config)
    infer_config = merge_configs(config, 'infer', vars(args))
    print_configs(infer_config, "Infer")
    with fluid.dygraph.guard():
        # infer_model = TSN1.TSNResNet('TSN', infer_config['MODEL']['num_layers'],
        #                             infer_config['MODEL']['num_classes'],
        #                             infer_config['MODEL']['seg_num'], 0.00002)
        # 上面注释是原代码，下面是修改后的
        infer_model = TSNResNet('TSN', infer_config['MODEL']['num_layers'],
                                    infer_config['MODEL']['num_classes'],
                                    seg_num = infer_config['MODEL']['seg_num'])

        label_dic = np.load('label_dir.npy', allow_pickle=True).item()
        label_dic = {v: k for k, v in label_dic.items()}

        # get infer reader
        infer_reader = KineticsReader(args.model_name.upper(), 'infer', infer_config).create_reader()

        # if no weight files specified, exit()
        if args.weights:
            weights = args.weights
        else:
            print("model path must be specified")
            exit()
            
        para_state_dict, _ = fluid.load_dygraph(weights)
        infer_model.load_dict(para_state_dict)
        infer_model.eval()
        
        for batch_id, data in enumerate(infer_reader()):
            dy_x_data = np.array([x[0] for x in data]).astype('float32')
            y_data = [x[1] for x in data]
            
            img = fluid.dygraph.to_variable(dy_x_data)
            
            out = infer_model(img).numpy()[0]
            label_id = np.where(out==np.max(out))
            print("实际标签{}, 预测结果{}".format(y_data, label_dic[label_id[0][0]]))
            
            
            
if __name__ == "__main__":
    args = parse_args()
    # check whether the installed paddle is compiled with GPU
    logger.info(args)

    infer(args)
