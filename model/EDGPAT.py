import logging
import warnings
import random

import numpy as np
import torch
from collections import defaultdict
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
from modules.memory import Memory
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.embedding_module import get_embedding_module
from model.time_encoding import TimeEncode, Time2Vec, PeriodicTimeEncoder


class EDGPAT(torch.nn.Module):
    def __init__(self, device, n_layers=2, n_nodes=0,
                 n_heads=2, dropout=0.1, message_dimension=100,
                 memory_dimension=500, embedding_module_type="time",
                 message_function="mlp", time_dim=10, time_enc_type="cos", type=None,
                 memory_updater_type="gru", use_time=False, reflect=None, loss_alpha=1,
                 use_history=True,num_hier=None, use_interact=True):
        super(EDGPAT, self).__init__()

        self.n_layers = n_layers
        self.device = device
        self.logger = logging.getLogger(__name__)

        self.n_nodes = n_nodes
        self.type = type
        self.reflect = reflect
        self.use_history = use_history
        self.use_hierarchy = len(self.type) != 2
        self.use_interact = use_interact

        self.embedding_module_type = embedding_module_type
        # self.time_encoder = Time2Vec(activation=time_enc_type, embedding_dim=time_dim)
        self.time_encoder = PeriodicTimeEncoder(embedding_dim=time_dim)

        self.use_time = use_time
        self.loss_alpha = loss_alpha

        self.memory_dimension = memory_dimension

        self.memory = Memory(type=self.type, n_nodes=self.n_nodes,
                             memory_dimension=self.memory_dimension,
                             message_dimension=message_dimension,
                             device=device)

        self.message_aggregator = get_message_aggregator(aggregator_type="mean",
                                                         device=device)
        if len(self.type) == 2:
            if self.use_interact:
                raw_message_num = [2, 2]
            else:
                raw_message_num = [1,1]
        else:
            if self.use_interact:
                raw_message_num = [2, 3, 3, 3, 3, 2]  # company, field_l1, field_l2, field_l3
            else:
                raw_message_num = [1, 2, 3, 3, 3, 2]  # company, field_l1, field_l2, field_l3
        # raw_message_num = [2, 2]  # company, field_l1, field_l2, field_l3
        # raw_message_num = [2, 3, 3, 2]  # company, field_l1, field_l2, field_l3
        raw_message_dim = [
            i * self.memory_dimension + self.time_encoder.embedding_dim if use_time else i * self.memory_dimension
            for i in raw_message_num]
        self.message_dim_fid = [message_dimension if message_function != "identity" else t_mess_dim
                                for t_mess_dim in raw_message_dim]

        self.message_function = torch.nn.ModuleDict({
            t: get_message_function(module_type=message_function,
                                    raw_message_dimension=raw_message_dim[i],
                                    message_dimension=self.message_dim_fid[i], device=device) for i, t in
            enumerate(self.type)
        })

        self.memory_updater = get_memory_updater(module_type=memory_updater_type,
                                                 memory=self.memory,
                                                 message_dimension=message_dimension,
                                                 memory_dimension=self.memory_dimension,
                                                 device=device)

        self.embedding_module_type = embedding_module_type

        self.embedding_dimension = self.memory_dimension

        self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                     n_node_features=self.memory_dimension,
                                                     dropout=dropout,
                                                     mess_dim=self.embedding_dimension)

        self.predict_patent = PredictionLayer(dim_field=self.embedding_dimension,hier_num=num_hier,
                                              n_fields=n_nodes[type[1]], n_company=n_nodes[type[0]],
                                              reflect_data=reflect,
                                              use_hierarchy=self.use_hierarchy, type_all=self.type,
                                              use_history=self.use_history)

    def compute_update_memory(self, source_nodes, destination_nodes, edge_times):
        """
        param: source_nodes: numpy_array N
        param: destination_nodes: list [[],[],...] N * k
        param: edge_times: numpy_array N
        """
        # step 1: get last update memories (in step 2)

        # step 2: produce message
        unique_src, src_id_to_messages = self.get_company_messages(source_nodes, destination_nodes,
                                                                   edge_times)
        res = self.get_hierarchy_messages(destination_nodes, source_nodes, edge_times)

        # source_time_diffs = (source_time_delta - self.mean_time_shift_src) / self.std_time_shift_src

        # step 3: aggregate message
        unique_messages_src, unique_timestamp_src = self.aggregate_message(unique_src, src_id_to_messages,
                                                                           type=self.type[0])

        # step 4: update memory
        updated_memory_src, update_timestamp_src = self.memory_updater.get_updated_memory(unique_src,
                                                                                          unique_messages_src,
                                                                                          type=self.type[0],
                                                                                          timestamps=unique_timestamp_src)
        update_res = dict()
        update_res[self.type[0]] = [unique_src, updated_memory_src, update_timestamp_src]

        for t, item in res.items():  # todo for k,v in res.items()
            # destination_time_delta, unique_dst, dst_nodes, dst_id_to_messages#
            unique_dst, dst_id_to_messages = item  # todo delete dst_nodes
            unique_messages_dst, unique_timestamp_dst = self.aggregate_message(unique_dst, dst_id_to_messages, type=t)
            # destination_time_diffs = (destination_time_delta - self.mean_time_shift_dst) / self.std_time_shift_dst
            updated_memory_dst, update_timestamp_dst = self.memory_updater.get_updated_memory(unique_dst,
                                                                                              unique_messages_dst,
                                                                                              type=t,
                                                                                              timestamps=unique_timestamp_dst)
            update_res[t] = [unique_dst, updated_memory_dst, update_timestamp_dst]
        return update_res

    def compute_embedding(self, update_mem):
        # step 5: compute the embeddings
        # {type: nodes, mem, time, time_diff}#

        t_c, t_f = self.type[0], self.type[1]
        company_embedding = self.embedding_module.compute_embedding(memory=update_mem[t_c][1],
                                                                    nodes=update_mem[t_c][0])
        # raw_company_embed = update_mem[t_c][1]
        raw_company_embed = self.memory.get_memory(t_c,range(self.n_nodes[t_c]))
        # print(company_embedding[update_mem[t_c][0]][:25,:10])

        field_embedding = self.embedding_module.compute_embedding(memory=update_mem[t_f][1],
                                                                  nodes=update_mem[t_f][0])

        raw_field_embed = self.memory.get_memory(t_f,range(self.n_nodes[t_f]))
        hier_embed = {}
        raw_embed = {}
        for t, item in update_mem.items():
            if t != self.type[0]:
                hier_embed[t] = self.embedding_module.compute_embedding(memory=update_mem[t][1],
                                                                        nodes=update_mem[t][0])
                raw_embed[t] = self.memory.get_memory(t,range(self.n_nodes[t]))

        return company_embedding, field_embedding, hier_embed, \
                raw_company_embed, raw_field_embed, raw_embed

    def update_self_memory(self, update_mem):
        # nodes, mem, time #
        for t, item in update_mem.items():
            item = update_mem[t]
            self.memory_updater.update_memory(unique_node_ids=item[0], unique_messages=item[1],
                                              type=t, timestamps=item[2])

    def aggregate_message(self, nodes, messages, type):
        # Aggregate messages for the same nodes

        unique_raw_messages = self.message_aggregator.aggregate(nodes, messages)
        unique_messages, unique_timestamp = unique_raw_messages
        if len(nodes) > 0:
            unique_messages = self.message_function[type].compute_message(unique_messages)

        return unique_messages, unique_timestamp

    def padding_nodes(self, data_list):
        """
            :param data_list: shape (batch_users, baskets, item_embed_dim)
            :return:
            """
        warnings.filterwarnings('ignore')
        length = [len(sq) for sq in data_list]
        data_list = [torch.tensor(d) for d in data_list]

        # pad in time dimension
        data = rnn_utils.pad_sequence(data_list, batch_first=True, padding_value=0)
        S = torch.zeros(data.shape)
        for i, l in enumerate(length):
            S[i, l:] = 1

        return data, length, S == 1

    def get_company_messages(self, source_nodes, destination_nodes, edge_times, type="company"):
        """
        source_n : N
        dest_n : N * S(set)
        """
        edge_times = torch.from_numpy(edge_times).to(self.device)

        source_memory = self.memory.get_memory(type, source_nodes)

        dt, length, S = self.padding_nodes(destination_nodes)
        # key = self.memory.get_memory(self.type[1],dt)
        # q = self.memory.get_memory(self.type[1],dt)
        # v = self.memory.get_memory(self.type[1],dt)
        # attn_output, attn_output_weights = self.attn(query=q.transpose(0,1),
        #                                                  key = key.transpose(0,1),
        #                                                  value = v.transpose(0,1),
        #                                              key_padding_mask = S.to(self.device))

        # ------------------- mask mean operation --------------------#
        mask = torch.arange(dt.shape[1]) < torch.tensor(length).unsqueeze(-1)
        mask = mask.unsqueeze(-1).to(self.device)
        destination_memory = torch.sum(self.memory.get_memory(self.type[1], dt) * mask, dim=1) / mask.sum(1)

        source_time_delta = edge_times - self.memory.last_update[type][source_nodes]
        if self.use_time:
            source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1))
            # source_time_delta_encoding = self.time_encoder(edge_times.unsqueeze(dim=1))
            if self.use_interact:
                source_message = torch.cat([source_memory, destination_memory, source_time_delta_encoding],
                                       dim=1)
            else:
                source_message = torch.cat([source_memory, source_time_delta_encoding],
                                           dim=1)
        else:
            if self.use_interact:
                source_message = torch.cat([source_memory, destination_memory], dim=1)
            else:
                source_message = source_memory

        messages = defaultdict(list)
        unique_sources = np.unique(source_nodes)

        for i in range(len(source_nodes)):
            messages[source_nodes[i]].append((source_message[i].unsqueeze(1), edge_times[i]))

        return [unique_sources, messages]

    def get_hierarchy_messages(self, fields_nodes, company_nodes, edge_times):
        # fields_nodes: N_src * S(set) (x: for company i, it has x fields)
        # --------------- layer 1 ------------------ #

        com_nodes, fil_nodes, all_nodes, edge_ts = [], [], [], []

        for i, fn in enumerate(fields_nodes):  # every field has its message
            l = len(fn)
            com_nodes.append(torch.tensor([company_nodes[i]]).expand(l))
            edge_ts.append(torch.tensor([edge_times[i]]).expand(l))
            fil_nodes.append(torch.tensor(fields_nodes[i]))
        all_nodes = torch.cat(fil_nodes, dim=0)
        com_nodes = torch.cat(com_nodes, dim=0)
        edge_ts = torch.cat(edge_ts, dim=0).to(self.device)

        com_memory = self.memory.get_memory(self.type[0], com_nodes)

        result = dict()

        start, end = 1, len(self.type)  # now we have three layers

        # source_time_delta_encoding = self.time_encoder(edge_ts.unsqueeze(dim=1))

        for i in range(start, end):
            # print(self.type[i])
            field_memory = self.memory.get_memory(self.type[i], all_nodes)

            source_time_delta = edge_ts - self.memory.last_update[self.type[i]][all_nodes]
            source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1))

            if i + 1 < end:
                next_memory = self.memory.get_memory(self.type[i + 1], self.reflect[i - 1][all_nodes, 1])
            if i == start:
                if self.use_time:
                    if i + 1 == end:
                        if self.use_interact:
                            source_message = torch.cat([field_memory, com_memory, source_time_delta_encoding], dim=1)
                        else:
                            source_message = torch.cat([field_memory, source_time_delta_encoding], dim=1)
                    else:
                        if self.use_interact:
                            source_message = torch.cat([field_memory, com_memory, next_memory, source_time_delta_encoding],
                                                   dim=1)
                        else:
                            source_message = torch.cat(
                                [field_memory, next_memory, source_time_delta_encoding],
                                dim=1)
                elif i + 1 == end:
                    if self.use_interact:
                        source_message = torch.cat([field_memory, com_memory], dim=1)
                    else:
                        source_message = field_memory
                else:
                    if self.use_interact:
                        source_message = torch.cat([field_memory, com_memory, next_memory], dim=1)
                    else:
                        source_message = torch.cat([field_memory, next_memory], dim=1)
            elif i + 1 == end:
                if self.use_time:
                    source_message = torch.cat([field_memory, pre_memory, source_time_delta_encoding], dim=1)
                else:
                    source_message = torch.cat([field_memory, pre_memory], dim=1)
            else:
                if self.use_time:
                    source_message = torch.cat([field_memory, pre_memory, next_memory, source_time_delta_encoding],
                                               dim=1)
                else:
                    source_message = torch.cat([field_memory, pre_memory, next_memory], dim=1)

            pre_memory = field_memory

            messages = defaultdict(list)
            all_nodes = all_nodes.numpy()
            unique_sources = np.unique(all_nodes)

            for node in range(len(all_nodes)):
                messages[all_nodes[node]].append((source_message[node].unsqueeze(1), edge_ts[node]))

            result[self.type[i]] = (unique_sources, messages)

            if i + 1 < end:
                all_nodes = self.reflect[i - 1][all_nodes, 1]

        return result

    def init_memory(self):
        self.memory.__init_memory__()

    def detach_memory(self):
        self.memory.detach_memory()

    def backup_memory(self):
        return self.memory.backup_memory()

    def restore_memory(self, memory):
        self.memory.restore_memory(memory)

    def sampleNegNode(self, all_nodes, num=10):
        sam_num = min(len(all_nodes) - 1, num)
        nega_list = []
        all_nodes = list(all_nodes)
        for i, node in enumerate(all_nodes):
            nega_list.append(torch.tensor(random.sample(all_nodes[:i] + all_nodes[i + 1:], sam_num)))
        return torch.cat(nega_list)

    def calculate_loss(self, res):
        # dict: {type: unique_src, updated_memory_src, updated_last_update_src, source_time_diffs}#
        regularizer_pos = 0
        for i, t in enumerate(self.type):
            if i > 0 and i + 1 < len(self.type):
                unique_node_i = res[t][0]
                unique_memory_i = res[t][1]
                unique_memory_hig = res[self.type[i + 1]][1]
                reflet_i = self.reflect[i - 1][unique_node_i, 1]
                regularizer_pos += torch.norm(unique_memory_i[unique_node_i] - unique_memory_hig[reflet_i]) ** 2

        return regularizer_pos * self.loss_alpha


class PredictionLayer(torch.nn.Module):
    def __init__(self, dim_field, hier_num, n_fields=60082, n_company=14695, use_history=True, reflect_data=None,
                 use_hierarchy=False, type_all=None, dropout=0.1):
        super().__init__()
        self.company_embedding = nn.Embedding(n_company, dim_field)
        self.field_embedding = nn.Embedding(n_fields, dim_field)
        self.projection_layer = nn.Linear(dim_field, dim_field)

        self.dropout = nn.Dropout(0.1)
        self.theta = nn.Parameter(torch.rand(n_company, 1), requires_grad=True)
        self.alpha_fields = nn.Parameter(torch.rand(n_fields, 1), requires_grad=True)
        self.n_fields = n_fields
        self.softmax = torch.nn.Softmax(dim=0)
        self.fc_field = nn.Linear(dim_field, 1)
        self.fc_company = nn.Linear(dim_field, 1)
        self.type = type_all
        self.reflect = reflect_data
        self.use_hierarchy = use_hierarchy
        self.mlp_his = nn.Sequential(
            nn.Linear(dim_field, dim_field // 2),
            nn.LeakyReLU(),
            nn.Linear(dim_field // 2, dim_field),
        )
        if self.use_hierarchy:
            self.gamma = nn.Parameter(torch.rand(n_fields, 1), requires_grad=True)
            self.mlp = nn.ModuleDict(
                {t: nn.Sequential(
                    nn.Linear(dim_field, dim_field // 2),
                    nn.LeakyReLU(),
                    nn.Linear(dim_field // 2, dim_field),
                ) for t in self.type}
            )

            # self.hier_field_embed = nn.ParameterDict(
            #     {t:nn.Parameter(torch.rand((num, dim_field))) for t,num in hier_num.items()}
            # )
            #
            # self.hier_field_alpha = nn.ParameterDict(
            #     {t: nn.Parameter(torch.rand((num, 1))) for t, num in hier_num.items()}
            # )

            self.fc_field_1 = nn.Linear(
                (len(self.type)-1),1)

        else:

            self.mlp = nn.Sequential(
                nn.Linear(dim_field, dim_field // 2),
                nn.LeakyReLU(),
                nn.Linear(dim_field // 2, dim_field),
            )
        self.use_his = use_history

    def forward(self, company_emb, field_emb, nodes, com_id, hier_embed,
                raw_field_embed, raw_hier_embed):
        """
            Param: x_com: torch.Tensor, shape (company_num, company_embed_dim),
                   x_fid: torch.Tensor, shape (field_num, field_embed_dim),
                   nodes: list, [(his_n, now_n),...]
            """
        batch_embedding = []
        # theta = self.softmax(self.theta)
        theta = self.theta
        company_mem_stat = (1 - theta[com_id]) * company_emb + \
                           theta[com_id] * self.company_embedding(
            torch.tensor(com_id).to(company_emb.device))  # combine company and fields
        # user_con = user_con.unsqueeze(1).expand(-1,self.n_fields,-1)
        company_mem_stat = self.dropout(company_mem_stat)
        for i, user_node in enumerate(nodes):
            # shape (user_item_num, item_embedding_dim)
            now_node = user_node[1]
            his_node = user_node[0]
            # all_node = his_node.extend(now_node)
            now_projected_fields = field_emb[now_node]

            # 1: now_node: embed + proj; his_node: mlp + proj; other: proj
            # beta, tensor, (items_total, 1), indicator vector, appear item -> 1, not appear -> 0
            beta = company_emb.new_zeros(self.n_fields, 1)
            beta[now_node] = 1
            # alpha_fields = self.softmax(self.alpha_fields)
            alpha_fields = self.alpha_fields
            if self.use_his:
                beta[his_node] = 1
                embed = (1 - beta * alpha_fields) * self.projection_layer(
                    self.field_embedding(torch.tensor(range(self.n_fields)).to(company_emb.device)))

                # appear items: (1 - self.alpha) * origin + self.alpha * update, not appear items: origin
                embed[now_node, :] = embed[now_node, :] + alpha_fields[now_node] * now_projected_fields
                his_mlp_fields = self.mlp_his(raw_field_embed[his_node])
                # his_mlp_fields = self.mlp_his(raw_field_embed[his_node])
                embed[his_node, :] = embed[his_node, :] + alpha_fields[his_node] * his_mlp_fields
                if self.use_hierarchy:
                    now_hier, his_hier = self.hierarchy_mem(hier_embed, now_node, his_node, raw_hier_embed)
                    # print(now_hier.shape,embed.shape)
                    embed[now_node, :] = self.gamma[now_node] * embed[now_node, :] + (1-self.gamma[now_node]) * torch.sum(now_hier,dim=-1)
                    embed[his_node, :] = self.gamma[his_node] * embed[his_node, :] + (1-self.gamma[his_node]) * torch.sum(his_hier,dim=-1)
                    # embed[now_node, :] = self.fc_field_1(torch.cat([embed[now_node, :].unsqueeze(-1), now_hier],dim=-1)).squeeze()
                    # embed[his_node, :] = self.fc_field_1(torch.cat([embed[his_node, :].unsqueeze(-1), his_hier],dim=-1)).squeeze()

            else:
                embed = (1 - beta * alpha_fields) * self.projection_layer(
                    self.field_embedding(torch.tensor(range(self.n_fields)).to(company_emb.device)))

                # appear items: (1 - self.alpha) * origin + self.alpha * update, not appear items: origin
                embed[now_node, :] = embed[now_node, :] + alpha_fields[now_node] * now_projected_fields

            fields_out = self.fc_field(embed)
            company_out = self.fc_company(company_mem_stat[i])

            predict_com = fields_out.squeeze() + company_out

            batch_embedding.append(predict_com)

        output = torch.stack(batch_embedding)
        # print(output.shape)
        return output

    def hierarchy_mem(self, res, now_node, history_node, raw_mem):
        """
            :param: node n*1
            """
        hier_rep_now, hier_rep_his = [], []
        hier_node_now, hier_node_his = [], []
        unique_node = now_node
        his_uni_node = history_node
        for i, t in enumerate(res.keys()):
            if i > 0 :
                his_n = set(his_uni_node.numpy()) - set(unique_node.numpy())
                embed = raw_mem[t]
                embed[unique_node] = res[t][unique_node]

                # now_projected_fields = self.hier_field_embed[t]
                # beta = now_projected_fields.new_zeros(len(res[t]), 1)
                # beta[unique_node] = 1
                # alpha_fields = self.softmax(self.hier_field_alpha[t]) todo: softmax
                # alpha_fields = self.hier_field_alpha[t]

                if len(his_n) > 0:
                    # print(his_n)
                    his_n = list(his_n)
                    embed[his_n, :] = self.mlp[t](embed[his_n])
                    # beta[his_n] = 1

                    # embed = (1 - beta * alpha_fields) * now_projected_fields
                    # embed[unique_node, :] = embed[unique_node, :] + alpha_fields[unique_node] * unique_mem_i
                    # embed[his_n, :] = embed[his_n, :] + alpha_fields[his_n] * his_n_mem

                # else:
                    # embed = (1 - beta * alpha_fields) * now_projected_fields
                    # embed[unique_node, :] = embed[unique_node, :] + alpha_fields[unique_node] * unique_mem_i

                hier_rep_now.append(embed[unique_node].unsqueeze(-1))
                hier_node_now.append(unique_node)
                hier_rep_his.append(embed[his_uni_node].unsqueeze(-1))
                hier_node_his.append(his_uni_node)
            if i+1 == len(res.keys()):
                break
            his_reflet_i = self.reflect[i][his_uni_node, 1]
            his_uni_node = his_reflet_i
            reflet_i = self.reflect[i][unique_node, 1]
            unique_node = reflet_i

        rep_now = torch.cat(hier_rep_now, dim=-1)
        rep_his = torch.cat(hier_rep_his, dim=-1)

        return rep_now, rep_his
