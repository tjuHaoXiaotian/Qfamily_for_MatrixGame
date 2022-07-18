#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File    ：q_factorisation.py
@Author  ：Hao Xiaotian, Dai Zipeng
@Date    ：2020/12/11 10:54
@Contact : xiaotianhao@tju.edu.cn
'''

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.nn.parameter import Parameter
import torch.nn.init as init
import math


class QFactorization(nn.Module):
    def __init__(self, algo, agent_num, action_num, hidden_dim):
        super(QFactorization, self).__init__()
        self.algo = algo
        self.agent_num = agent_num
        self.action_num = action_num
        self.hidden_dim = hidden_dim

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # (1) Individual Q (not shared among agents)
        self.individual_qs = Parameter(th.randn(1, agent_num, action_num))  # (1, agent_num, action_num)

        # (2) Mixing net parameters (if have)
        if self.algo in ["vdn", "qtran_vdn"]:
            pass
        elif algo in "weighted_vdn":
            self.w1 = Parameter(th.randn([agent_num, 1], dtype=th.float32))
            self.b1 = Parameter(th.randn([1, 1], dtype=th.float32))
        elif algo in ["qmix", "qtran_qmix"]:
            # First layer, (agent_num, embedding_dim)
            self.w1 = Parameter(th.randn([agent_num, hidden_dim], dtype=th.float32))
            self.b1 = Parameter(th.randn([1, hidden_dim], dtype=th.float32))
            # Second layer, (embedding_dim, 1)
            self.w2 = Parameter(th.randn([hidden_dim, 1], dtype=th.float32))
            self.b2 = Parameter(th.randn([1, 1], dtype=th.float32))
        elif algo in ["qplex"]:
            # Lambda net. Note that since there is only a single 'state' in matrix games, the input 'state' can be omitted.
            # We use an MLP here to transfer the 'joint actions' to the 'weights' of the 'advantages' of n agents.
            self.lambda_net = nn.Sequential(
                nn.Linear(agent_num * action_num, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, agent_num),
            )
        else:
            raise NotImplementedError

        # init parameters
        self._init_parameters()

    def _init_parameters(self):
        # init individual q-values
        # init.kaiming_uniform_(self.individual_qs, a=math.sqrt(5))
        # init.uniform_(self.individual_qs, -0.1, 0.1)
        print("******************* [q_i] init q tables *******************")
        q_print = self.individual_qs[0]  # [agent_num, action_num]
        for agent_idx in range(self.agent_num):
            individual_q = q_print[agent_idx]  # [action_num]
            print("-------------- agent-{}: greedy action={} --------------".format(agent_idx,
                                                                                    individual_q.max(
                                                                                        dim=0)[1].item()))
            print(individual_q.tolist())
            print("--------------------------------------\n")

    def forward(self, batch_size, batch_action, q_joint, print_log=False):
        assert batch_action.shape[0] == batch_size
        assert batch_action.shape[1] == self.agent_num
        assert batch_action.shape[2] == 1
        # (batch_size, agent_num, action_num)
        batch_individual_qs = self.individual_qs.expand(batch_size, self.agent_num, self.action_num)

        # Get action values, (batch_size, agent_num, action_num) -> (batch_size, agent_num)
        selected_individual_qs = th.gather(batch_individual_qs, dim=2, index=batch_action).squeeze(2)

        # Calculate q_total
        if self.algo in ["vdn", "qtran_vdn"]:
            q_tot = selected_individual_qs.sum(dim=1, keepdim=True)
        elif self.algo == "weighted_vdn":
            q_tot = th.mm(selected_individual_qs, th.abs(self.w1)) + self.b1
        elif self.algo in ["qmix", "qtran_qmix"]:
            # First layer
            hidden = F.elu(th.mm(selected_individual_qs, th.abs(self.w1)) + self.b1)
            # Second layer
            q_tot = th.mm(hidden, th.abs(self.w2)) + self.b2
        elif self.algo in ["qplex"]:
            q_upper = batch_individual_qs.max(dim=2, keepdim=False)[0]
            adv = (selected_individual_qs - q_upper)  # (batch_size, agent_num)
            # (batch, agent_num * action_num)
            _onehot_actions = th.zeros_like(batch_individual_qs).scatter_(-1, batch_action, 1.0).view(batch_size, -1)
            w_lambda = th.abs(self.lambda_net(_onehot_actions))  # ensure lambda >= 0

            # (1) The simple implementation.
            v_tot = q_upper.sum(dim=1, keepdim=True)  # current maximum point
            adv_tot = (w_lambda * adv).sum(dim=1, keepdim=True)  # weighted sum of the advantage values

            # (2) The official implementation according to the Equation 52 of Appendix B.2.  (behave even worse!!!!!)
            # v_tot = selected_individual_qs.sum(dim=1, keepdim=True)  # sum of q_i
            # adv_tot = ((w_lambda - 1) * adv.detach()).sum(dim=1, keepdim=True)  # weighted sum of the advantage values

            q_tot = v_tot + adv_tot
        else:
            raise NotImplementedError

        # Calculate loss
        if self.algo in ["qtran_vdn", "qtran_qmix"]:
            # Greedy actions
            individual_greedy_action = batch_individual_qs.max(dim=2, keepdim=True)[1]
            # The sample in current batch where the actions == greedy actions
            max_point_mask = ((individual_greedy_action == batch_action).long().sum(dim=1) == self.agent_num).float()
            q_clip = th.max(q_tot, q_joint).detach()
            # The core of Qtran: ensure q_tot(*) >= q_joint(*) and q_tot(greedy_action) == q_joint(greedy_action)
            loss = th.mean(max_point_mask * ((q_tot - q_joint) ** 2) + (1 - max_point_mask) * ((q_tot - q_clip) ** 2))
        else:
            loss = th.mean((q_tot - q_joint) ** 2)

        if print_log:
            print("******************* [q_i] Learned individual q tables *******************")
            q_print = self.individual_qs[0]  # [agent_num, action_num]
            for agent_idx in range(self.agent_num):
                individual_q = q_print[agent_idx]  # [action_num]
                print("-------------- agent-{}: greedy action={} --------------".format(
                    agent_idx, individual_q.max(dim=0)[1].item())
                )
                print(individual_q.tolist())
                print("--------------------------------------\n")
        return q_tot, loss


def train(algo, agent_num=2, action_num=3, hidden_dim=16, epoch=2000):
    assert action_num ** agent_num == len(payoff_flatten_vector)
    batch_size = len(payoff_flatten_vector)
    q_joint = th.from_numpy(np.asarray(payoff_flatten_vector, dtype=np.float32)).view([batch_size, 1])
    # Get the joint action indices. [bs, agent_num], e.g, [[0, 1], [0, 2], [0, 3] ....]
    # We generate all joint actions in a batch to avoid the 'exploration' problem and simply focus on the ‘estimation’.
    action_index = np.arange(0, action_num)
    cartesian_product = np.array(np.meshgrid(*[action_index] * agent_num)).T.reshape(-1, agent_num)
    batch_action = th.from_numpy(cartesian_product).unsqueeze(-1).long()  # [bs, agent_num, 1]

    q_network = QFactorization(algo=algo, agent_num=agent_num, action_num=action_num, hidden_dim=hidden_dim)
    optimizer = th.optim.Adam(params=q_network.parameters(), lr=0.01)

    for idx in range(epoch):
        q_tot, loss = q_network.forward(batch_size, batch_action, q_joint)
        if idx % 100 == 0:
            print("Iter={}: MSE loss={}".format(idx, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    q_tot, _ = q_network.forward(batch_size, batch_action, q_joint, print_log=True)

    print("******************* Predicted Q_tot: *******************")
    q_print = q_tot.detach().tolist()
    for row in range(action_num):
        start_pos = row * pow(action_num, agent_num - 1)
        print(q_print[start_pos: start_pos + pow(action_num, agent_num - 1)])

    print("******************* True Q_joint: *******************")
    q_print = q_joint.detach().tolist()
    for row in range(action_num):
        start_pos = row * pow(action_num, agent_num - 1)
        print(q_print[start_pos: start_pos + pow(action_num, agent_num - 1)])


if __name__ == "__main__":
    algorithms = ["vdn", "weighted_vdn", "qmix", "qtran_vdn", "qtran_qmix", "qplex"]
    # %%%%%%%%%%%% Step1: choose algorithm %%%%%%%%%%%%
    algo = algorithms[4]
    print("Use {}".format(algo))

    # %%%%%%%%%%%% Step2: choose matrix (for convenience of representation, we flatten the matrix into a vector) %%%%%%%%%%%%
    # payoff_flatten_vector= [1, 0, 0, 1]  # agent_num=2, action_num=2
    # payoff_flatten_vector=[8, 3, 2, -12, -13, -14, -12, -13, -14]
    # payoff_flatten_vector = [8, -12, -12, -12, 0, 0, -12, 0, 0]
    payoff_flatten_vector = [8, -12, -12,
                             -12, 6, 0,
                             -12, 0, 6]
    # payoff_flatten_vector = [20, 0, 0, 0, 12, 12, 0, 12, 12]

    # %%%%%%%%%%%% Step3: choose other parameters, note that: action_num**agent_num = |payoff-matrix| %%%%%%%%%%%%
    seed = 1
    # seed = 2

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    train(algo=algo, agent_num=2, action_num=3, hidden_dim=16, epoch=5000)
