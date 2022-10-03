import json
import math
import logging
import shutil
import time
import sys
import argparse

import torch
import numpy as np
from pathlib import Path

from tqdm import tqdm

from evaluation.evaluation import eval_edge_prediction
from model.EDGPAT import EDGPAT
from utils.metric import get_metric
from utils.utils import EarlyStopMonitor
from utils.data_processing import get_data, get_truth_patents, get_truth

torch.manual_seed(0)
np.random.seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='data-time-fin')
parser.add_argument('--bs', type=int, default=32, help='Batch_size')
parser.add_argument('--prefix', type=str, default='proj-field128-classall-hierarchy_sum_time',
                    help='Prefix to name the checkpoints')
parser.add_argument('--n_epoch', type=int, default=400, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=4, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=1, help='Idx for the gpu to use')
parser.add_argument('--use_interact', type=bool, default=True, help='interaction information')

parser.add_argument('--time_dim', type=int, default=64, help='Dimensions of the time embedding')
parser.add_argument('--use_time', type=bool, default=True, help='whether to use time embedding')
parser.add_argument('--time_enc', type=str, default='sin', help='Type of time encoding')

parser.add_argument('--embedding_module', type=str, default="identity", choices=[
    "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="mlp", choices=[
    "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
    "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="mean", help='Type of message '
                                                                   'aggregator')

parser.add_argument('--message_dim', type=int, default=128, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=128, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--truth', type=str, default='real-data', help='path of truth data')
parser.add_argument('--reflect5_4', type=str, default='5-4-level', help='path of from 5 to 4 level')
parser.add_argument('--reflect4_3', type=str, default='4-3-level', help='path of from 4 to 3 level')
parser.add_argument('--reflect3_2', type=str, default='3-2-level', help='path of from 3 to 2 level')
parser.add_argument('--reflect2_1', type=str, default='2-1-level', help='path of from 2 to 1 level')
parser.add_argument('--reflect4_2', type=str, default='4-2-level', help='path of from 4 to 2 level')

parser.add_argument('--n_company', type=int, default=14695, help='Number of company')
parser.add_argument('--n_field_5', type=int, default=60082, help='Number of fields')
parser.add_argument('--n_field_4', type=int, default=10049, help='Number of forth fields')
parser.add_argument('--n_field_3', type=int, default=1204, help='Number of third fields')
parser.add_argument('--n_field_2', type=int, default=124, help='Number of second fields')
parser.add_argument('--n_field_1', type=int, default=8, help='Number of first fields')
parser.add_argument('--load_pre_model', type=int, default=-1, help='load_model_path')
parser.add_argument('--alpha_loss', type=float, default=1, help='constrain the unsupervised loss item')
parser.add_argument('--use_hierarchical', type=bool, default=True, help='loss add hierarchical layers')
parser.add_argument('--predict_his', type=bool, default=True, help='the id of subfields')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE = args.bs

NUM_EPOCH = args.n_epoch
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
TRUTH = args.truth
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NUM_COM = args.n_company
NUM_Fie_5, NUM_Fie_4, NUM_Fie_3, NUM_Fie_2, NUM_Fie_1 = args.n_field_5, args.n_field_4, args.n_field_3, args.n_field_2, args.n_field_1
REF5_4, REF4_3, REF3_2, REF2_1 = args.reflect5_4, args.reflect4_3, args.reflect3_2, args.reflect2_1
alpha_loss = args.alpha_loss
TIME_DIM = args.time_dim

Use_time = args.use_time
TIME_TYPE = args.time_enc
HIERARCHY = args.use_hierarchical
use_history = args.predict_his

MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim
USE_INTERACT = args.use_interact
Path(f"./saved_models/{args.prefix}/").mkdir(parents=True, exist_ok=True)
Path(f"./saved_checkpoints/{args.prefix}/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}/{args.data}-2.pth'
get_checkpoint_path = lambda \
        epoch: f'./saved_checkpoints/{args.prefix}/{args.data}-{epoch}.pth'

MODEL_LOAD_PATH = f'./saved_checkpoints/{args.prefix}/{args.data}-{args.load_pre_model}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path(f"log/{args.prefix}/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}/{}.log'.format(args.prefix, str(args.prefix) + str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

### Extract data for training, validation and testing
reflect_data, full_data, train_data, val_data, test_data = get_data(
    DATA, [REF5_4, REF4_3, REF3_2, REF2_1], use_validation=True)

truth_data = get_truth(TRUTH)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'

device = torch.device(device_string)
if HIERARCHY:
    type_patent = ["company", "field_5", "field_4", "field_3", "field_2", "field_1"]
    num_nodes = [NUM_COM, NUM_Fie_5, NUM_Fie_4, NUM_Fie_3, NUM_Fie_2, NUM_Fie_1]
else:
    type_patent = ["company", "field_5"]
    num_nodes = [NUM_COM, NUM_Fie_5]

num_hir = dict()
for i, t in enumerate(type_patent):
    if i < 2:
        continue
    num_hir[t] = num_nodes[i]

n_nodes = {t: num_nodes[i] for i, t in enumerate(type_patent)}


def save_as_json(data: dict or list, path: str):
    """
    save data as json file with path

    :param data:
    :param path:
    :return:
    """
    with open(path, "w") as file:
        file.write(json.dumps(data))
        file.close()
        print(f'{path} writes successfully.')


for i in range(args.n_runs):
    # Initialize Model
    logger.info("===============================runs :{} start===================================".format(i))
    tgn = EDGPAT(device=device,
                 n_layers=NUM_LAYER,
                 n_nodes=n_nodes, time_dim=TIME_DIM, use_time=Use_time, time_enc_type=TIME_TYPE,
                 dropout=DROP_OUT, type=type_patent,
                 message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
                 embedding_module_type=args.embedding_module,
                 message_function=args.message_function,
                 memory_updater_type=args.memory_updater,
                 reflect=reflect_data, loss_alpha=alpha_loss, use_history=use_history,
                 num_hier=num_hir)
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
    tgn = tgn.to(device)

    # for i in tgn.modules():
    #     print(i)
    logger.info("model:{}".format(tgn))
    logger.info("model training parameters:{}".format(tgn.state_dict().keys()))

    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    logger.info('num of training instances: {}'.format(num_instance))
    logger.info('num of batches per epoch: {}'.format(num_batch))

    train_losses = []
    early_stopper = EarlyStopMonitor(max_round=args.patience)

    pre_id = 0
    if args.load_pre_model > -1:
        pre_id += args.load_pre_model
        pretrain_model = torch.load(MODEL_LOAD_PATH)
        # print('pretrain model dict {}'.format(pretrain_model))
        model_dict = tgn.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_model.items() if k in model_dict}
        print(pretrain_dict.keys())
        model_dict.update(pretrain_dict)
        tgn.load_state_dict(model_dict)
        # sys.exit()

    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(NUM_EPOCH):
            total_loss = 0
            total = 0
            print("================================Epoch: %d================================" % epoch)
            start_epoch = time.time()
            logger.info('start {} epoch'.format(epoch))
            result = {}
            for top_k in [10, 20, 30, 40]:
                result.update({
                    f'recall_{top_k}': 0,
                    f'ndcg_{top_k}': 0,
                    f'PHR_{top_k}': 0
                })

            batch_range = tqdm(range(num_batch))
            y_true, y_pred = [], []

            tgn.init_memory()
            # print(tgn.memory.state_dict())

            for batch_idx in batch_range:
                loss = 0

                if batch_idx >= num_batch:
                    continue
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(num_instance, start_idx + BATCH_SIZE)
                sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                                    train_data.destinations[start_idx:end_idx]
                label_batch = train_data.labels[start_idx: end_idx]
                timestamps_batch = train_data.realt[start_idx:end_idx]
                realt_batch = train_data.ts_all[start_idx:end_idx]

                size = len(sources_batch)

                tgn = tgn.train()

                update_mem = tgn.compute_update_memory(sources_batch, destinations_batch, timestamps_batch)

                if sum(label_batch) > 0:
                    cal_loss_idx = label_batch > 0
                    lb_idx = label_batch[cal_loss_idx]
                    sour_idx = sources_batch[cal_loss_idx]

                    his_data, real_data = get_truth_patents(truth_data, sour_idx, lb_idx,
                                                            NUM_Fie_5)  # sour_idx: company index, # lb_idx: predicted year [N*Items]

                    real_data = real_data.to(device)
                    # have next year truth
                    if real_data.shape[0] != 0:
                        com_embed, field_embed, hier_embed, raw_company_embed, raw_field_embed, raw_embed = tgn.compute_embedding(
                            update_mem)
                        # print(dst_node)
                        patent_predicted = tgn.predict_patent(com_embed[sour_idx], field_embed,
                                                              nodes=his_data, com_id=sour_idx, hier_embed=hier_embed,
                                                              raw_field_embed=raw_field_embed, raw_hier_embed=raw_embed)
                        # patent_predicted = patent_predicted[ind,:]
                        total += real_data.shape[0]
                        assert patent_predicted.shape == real_data.shape
                        loss = criterion(patent_predicted, real_data)
                        scores = get_metric(y_true=real_data, y_pred=patent_predicted)

                        for k in scores:
                            result[k] += scores[k]

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        m_loss = loss.cpu().data.numpy()
                        total_loss += loss.cpu().data.numpy()

                        batch_range.set_description(f"train_loss: {m_loss};")

                tgn.update_self_memory(update_mem)
                tgn.detach_memory()

            for key in result:
                result[key] /= total
            logger.info('total_loss:{}'.format(total_loss / num_batch))

            print("================================Val================================")
            train_memory_backup = tgn.backup_memory()
            tgn.restore_memory(train_memory_backup)

            score = eval_edge_prediction(model=tgn, data=val_data, device=device, truth=truth_data,
                                         batch_size=BATCH_SIZE, NUM_Fie=NUM_Fie_5, mode="validate")
            logger.info('validate results:{}'.format(score))

            validate_ndcg_list = []
            for key in score:
                if key.startswith("ndcg_"):
                    validate_ndcg_list.append(score[key])
            validate_ndcg = np.mean(validate_ndcg_list)
            ifstop, ifimprove = early_stopper.early_stop_check(validate_ndcg)
            if ifstop:
                logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
                logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
                best_model_path = get_checkpoint_path(early_stopper.best_epoch)
                tgn.load_state_dict(torch.load(best_model_path))
                logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
                tgn.eval()
                break
            else:
                torch.save(
                    tgn.state_dict(),
                    get_checkpoint_path(early_stopper.best_epoch))

                logger.info('Saving TGN model')
                shutil.copy(get_checkpoint_path(early_stopper.best_epoch), MODEL_SAVE_PATH)
                logger.info('TGN model saved')

                ## Test
                print("================================Test================================")
                val_memory_backup = tgn.backup_memory()
                tgn.restore_memory(val_memory_backup)
                score = eval_edge_prediction(model=tgn, data=test_data, device=device, truth=truth_data,
                                             NUM_Fie=NUM_Fie_5)
                logger.info('test results:{}'.format(score))
        best_model_param = torch.load(get_checkpoint_path(early_stopper.best_epoch))

        ## Test
        print("================================Test================================")
        model_dict = tgn.state_dict()
        pretrain_dict = {k: v for k, v in best_model_param.items() if k in model_dict}
        print(pretrain_dict.keys())
        model_dict.update(pretrain_dict)
        tgn.load_state_dict(model_dict)
        val_memory_backup = tgn.backup_memory()
        tgn.restore_memory(val_memory_backup)
        score = eval_edge_prediction(model=tgn, data=test_data, device=device, truth=truth_data, NUM_Fie=NUM_Fie_5)
        logger.info('best model test results:{}'.format(score))
