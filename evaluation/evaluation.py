import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from utils.data_processing import get_truth_patents
from utils.metric import get_metric


def eval_edge_prediction(model, data, truth, device, NUM_Fie, batch_size=200, mode="test"):
    label, prediction = [], []
    scores = []
    total = 0
    with torch.no_grad():
        model = model.eval()

        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

        for k in tqdm(range(num_test_batch)):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)

            sources_batch, destinations_batch = data.sources[s_idx: e_idx], \
                                                data.destinations[s_idx: e_idx]
            timestamps_batch = data.realt[s_idx: e_idx]
            label_batch = data.labels[s_idx: e_idx]
            realt_batch = data.ts_all[s_idx: e_idx]

            cal_loss_idx = label_batch > 0
            sour_idx = sources_batch[cal_loss_idx]
            lb_idx = label_batch[cal_loss_idx]

            update_mem= model.compute_update_memory(
                sources_batch, destinations_batch, timestamps_batch)

            if len(lb_idx) != 0:
                his_data, real_data = get_truth_patents(truth, sour_idx, lb_idx, NUM_Fie)
                # ind = torch.sum(real_data,dim=-1) != 0

                real_data = real_data.to(device)
                # print(patent_predicted.shape,real_data.shape)
                if real_data.shape[0] != 0:
                    com_embed, field_embed, hier_embed, raw_company_embed, raw_field_embed, raw_embed = model.compute_embedding(update_mem)
                    # print(dst_node)
                    patent_predicted = model.predict_patent(com_embed[sour_idx], field_embed,
                                                          nodes=his_data, com_id=sour_idx, hier_embed=hier_embed,
                                                          raw_field_embed=raw_field_embed, raw_hier_embed=raw_embed)

                    # patent_predicted = patent_predicted[ind, :]
                    assert real_data.shape == patent_predicted.shape
                    total += real_data.shape[0]
                    scores.append(get_metric(y_true=real_data,y_pred=patent_predicted))
            model.update_self_memory(update_mem)
    result = {}
    for top_k in [10, 20, 30, 40]:
        result.update({
            f'recall_{top_k}': 0,
            f'ndcg_{top_k}': 0,
            f'PHR_{top_k}': 0
        })
    for score in scores:
        for key in score:
            result[key] += score[key]

    for key in result:
        result[key] /= total
    print(result)
    return result
