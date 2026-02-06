import pathlib
import sys
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
import joblib
from torch.utils.tensorboard import SummaryWriter
import warnings
import datetime
import pytz
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = str(pathlib.Path(__file__).resolve().parents[3])
sys.path.append(ROOT)
sys.path.insert(0, ".")

from model.net import *
from env.high_level_env import Testing_Env, Training_Env
from RL.util.utili import get_ada, get_epsilon, LinearDecaySchedule
from RL.util.replay_buffer import ReplayBuffer_High
from RL.util.memory import episodicmemory

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--epoch_number", type=int, default=1)
parser.add_argument("--buffer_size", type=int, default=1000000, )
parser.add_argument("--dataset", type=str, default="ETHUSDT")
parser.add_argument("--q_value_memorize_freq", type=int, default=10, )
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--eval_update_freq", type=int, default=512)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--epsilon_start", type=float, default=0.7)
parser.add_argument("--epsilon_end", type=float, default=0.3)
parser.add_argument("--decay_length", type=int, default=5)
parser.add_argument("--update_times", type=int, default=10)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--transcation_cost", type=float, default=0.2 / 1000)
parser.add_argument("--back_time_length", type=int, default=1)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--n_step", type=int, default=1)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--beta", type=int, default=5)
parser.add_argument("--exp", type=str, default="exp1")
parser.add_argument("--num_step", type=int, default=10)
parser.add_argument("--run_count", type=int, default=1)
parser.add_argument("--trend_method", type=str, default="l2")
parser.add_argument("--vol_method", type=str, default="rbf")
parser.add_argument("--liq_method", type=str, default="l2")
parser.add_argument("--look_back_window", type=int, default=15)
beijing_tz = pytz.timezone('Asia/Shanghai')


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class DQN(object):
    def __init__(self, args):
        self.seed = args.seed
        seed_torch(self.seed)
        if torch.cuda.is_available():
            self.device = torch.device(args.device)
        else:
            self.device = torch.device("cpu")
        self.base_path = os.path.join("./result", f"run_{args.run_count}")
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

        self.result_path = os.path.join(self.base_path, args.dataset,
                                        f"high_level_{args.trend_method}_{args.vol_method}_{args.liq_method}")

        self.model_path = os.path.join(self.result_path)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.train_data_path = os.path.join(ROOT, "MedalHFT",
                                            "MyData", args.dataset, "whole")
        self.val_data_path = os.path.join(ROOT, "MedalHFT",
                                          "MyData", args.dataset, "whole")
        self.test_data_path = os.path.join(ROOT, "MedalHFT",
                                           "MyData", args.dataset, "whole")
        self.dataset = args.dataset
        self.num_step = args.num_step
        if "BTC" in self.dataset:
            self.max_holding_number = 0.01
        elif "ETH" in self.dataset:
            self.max_holding_number = 0.01
        elif "BNB" in self.dataset:
            self.max_holding_number = 0.1
        elif "SOL" in self.dataset:
            self.max_holding_number = 1
        elif "LTC" in self.dataset:
            self.max_holding_number = 2
        elif "LINK" in self.dataset:
            self.max_holding_number = 10
        else:
            raise Exception("we do not support other dataset yet")
        self.epoch_number = args.epoch_number

        self.log_path = os.path.join(self.model_path, "log")
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self.log_files = {}
        metrics = ["td_error", "memory_error", "KL_loss", "total_loss", "q_eval", "q_target", "epoch_return_rate_train",
                   "epoch_final_balance_train", "epoch_required_money_train", "epoch_reward_sum_train",
                   "return_rate_eval"]

        for i, metric in enumerate(metrics):
            file_path = os.path.join(self.log_path, f"{i + 1}#{metric}.txt")
            self.log_files[metric] = file_path

            with open(file_path, 'w') as f:
                f.write(f"{metric} Log\n")
                f.write("=" * 50 + "\n")

        # self.writer = SummaryWriter(self.log_path)

        self.update_counter = 0
        self.q_value_memorize_freq = args.q_value_memorize_freq

        self.tech_indicator_list = np.load('./MyData/feature_list/single_features.npy', allow_pickle=True).tolist()
        self.tech_indicator_list_trend = np.load('./MyData/feature_list/trend_features.npy', allow_pickle=True).tolist()
        self.minute1_indicator_list = np.load('./MyData/feature_list/minute1_features.npy', allow_pickle=True).tolist()
        self.minute3_indicator_list = np.load('./MyData/feature_list/minute3_features.npy', allow_pickle=True).tolist()
        self.minute5_indicator_list = np.load('./MyData/feature_list/minute5_features.npy', allow_pickle=True).tolist()
        self.clf_list = ['trend_360', 'vol_360', 'liq_360']

        self.transcation_cost = args.transcation_cost
        self.back_time_length = args.back_time_length
        self.look_back_window = args.look_back_window
        self.n_action = 2
        self.n_state_1 = len(self.tech_indicator_list)
        self.n_state_2 = len(self.tech_indicator_list_trend)
        self.n_clf = len(self.clf_list)
        self.n_state_1minute = len(self.minute1_indicator_list)
        self.n_state_3minute = len(self.minute3_indicator_list)
        self.n_state_5minute = len(self.minute5_indicator_list)

        self.trend_1 = subagent(
            self.n_state_1, self.n_state_2, self.n_state_1minute, self.n_state_3minute, self.n_state_5minute,
            self.look_back_window, self.n_action, 64).to(self.device)
        self.trend_2 = subagent(
            self.n_state_1, self.n_state_2, self.n_state_1minute, self.n_state_3minute, self.n_state_5minute,
            self.look_back_window, self.n_action, 64).to(self.device)

        self.vol_1 = subagent(
            self.n_state_1, self.n_state_2, self.n_state_1minute, self.n_state_3minute, self.n_state_5minute,
            self.look_back_window, self.n_action, 64).to(self.device)
        self.vol_2 = subagent(
            self.n_state_1, self.n_state_2, self.n_state_1minute, self.n_state_3minute, self.n_state_5minute,
            self.look_back_window, self.n_action, 64).to(self.device)

        self.liq_1 = subagent(
            self.n_state_1, self.n_state_2, self.n_state_1minute, self.n_state_3minute, self.n_state_5minute,
            self.look_back_window, self.n_action, 64).to(self.device)
        self.liq_2 = subagent(
            self.n_state_1, self.n_state_2, self.n_state_1minute, self.n_state_3minute, self.n_state_5minute,
            self.look_back_window, self.n_action, 64).to(self.device)

        model_list_trend = [
            f"./result/run_{args.run_count}/{args.dataset}/low_level/trend/trend_1_alpha_1.0_{args.trend_method}/best_model.pkl",
            f"./result/run_{args.run_count}/{args.dataset}/low_level/trend/trend_-1_alpha_1.0_{args.trend_method}/best_model.pkl",
        ]
        model_list_vol = [
            f"./result/run_{args.run_count}/{args.dataset}/low_level/vol/vol_1_alpha_1.0_{args.vol_method}/best_model.pkl",
            f"./result/run_{args.run_count}/{args.dataset}/low_level/vol/vol_-1_alpha_1.0_{args.vol_method}/best_model.pkl",
        ]
        model_list_liq = [
            f"./result/run_{args.run_count}/{args.dataset}/low_level/liq/liq_1_alpha_1.0_{args.liq_method}/best_model.pkl",
            f"./result/run_{args.run_count}/{args.dataset}/low_level/liq/liq_-1_alpha_1.0_{args.liq_method}/best_model.pkl",
        ]

        self.trend_1.load_state_dict(
            torch.load(model_list_trend[0], map_location=self.device))
        self.trend_2.load_state_dict(
            torch.load(model_list_trend[1], map_location=self.device))
        self.vol_1.load_state_dict(
            torch.load(model_list_vol[0], map_location=self.device))
        self.vol_2.load_state_dict(
            torch.load(model_list_vol[1], map_location=self.device))
        self.liq_1.load_state_dict(
            torch.load(model_list_liq[0], map_location=self.device)
        )
        self.liq_2.load_state_dict(
            torch.load(model_list_liq[1], map_location=self.device)
        )

        self.trend_1.eval()
        self.trend_2.eval()

        self.vol_1.eval()
        self.vol_2.eval()

        self.liq_1.eval()
        self.liq_2.eval()

        self.trend_agents = {
            1: self.trend_1,
            -1: self.trend_2
        }
        self.vol_agents = {
            1: self.vol_1,
            -1: self.vol_2,
        }
        self.liq_agents = {
            1: self.liq_1,
            -1: self.liq_2,
        }

        self.hyperagent = hyperagent(self.n_state_1, self.n_state_2, self.n_clf, self.n_state_1minute,
                                     self.n_state_3minute, self.n_state_5minute, self.look_back_window, self.n_action,
                                     32).to(self.device)
        self.hyperagent_target = hyperagent(self.n_state_1, self.n_state_2, self.n_clf, self.n_state_1minute,
                                            self.n_state_3minute, self.n_state_5minute, self.look_back_window,
                                            self.n_action, 32).to(self.device)
        self.hyperagent_target.load_state_dict(self.hyperagent.state_dict())
        self.update_times = args.update_times
        self.optimizer = torch.optim.Adam(self.hyperagent.parameters(),
                                          lr=args.lr)
        self.loss_func = nn.MSELoss()
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.n_step = args.n_step
        self.eval_update_freq = args.eval_update_freq
        self.buffer_size = args.buffer_size
        self.epsilon_start = args.epsilon_start
        self.epsilon_end = args.epsilon_end
        self.decay_length = args.decay_length
        self.epsilon_scheduler = LinearDecaySchedule(start_epsilon=self.epsilon_start, end_epsilon=self.epsilon_end,
                                                     decay_length=self.decay_length)
        self.epsilon = args.epsilon_start
        self.memory = episodicmemory(4320, 5, self.n_state_1, self.n_state_2, self.n_state_1minute,
                                     self.n_state_3minute, self.n_state_5minute, self.look_back_window, 64, self.device)

    def log_scalar(self, tag, value, epoch_counter=None, episode_counter=None, step_cpunter=None, update_counter=None):
        if tag in self.log_files:
            with open(self.log_files[tag], 'a') as f:
                if (
                        epoch_counter != None and episode_counter != None and step_cpunter != None and update_counter != None):
                    f.write(
                        f"Epoch {epoch_counter} | Episode {episode_counter} | Step {step_cpunter} | Update {update_counter} : {value}\n")
                elif (
                        epoch_counter != None and episode_counter != None and step_cpunter == None and update_counter == None):
                    f.write(f"Epoch {epoch_counter} | Episode {episode_counter}: {value}\n")
                elif (
                        epoch_counter != None and episode_counter == None and step_cpunter == None and update_counter == None):
                    f.write(f"Epoch {epoch_counter}: {value}\n")

    def calculate_q(self, w, qs):
        q_tensor = torch.stack(qs)
        q_tensor = q_tensor.permute(1, 0, 2)
        weights_reshaped = w.view(-1, 1, 6)
        combined_q = torch.bmm(weights_reshaped, q_tensor).squeeze(1)

        return combined_q

    def update(self, replay_buffer):
        # 保存当前模式状态
        was_training = {}
        for name, model in [('hyperagent', self.hyperagent), ('hyperagent_target', self.hyperagent_target),
                            ('trend_1', self.trend_1), ('trend_2', self.trend_2),
                            ('vol_1', self.vol_1), ('vol_2', self.vol_2),
                            ('liq_1', self.liq_1), ('liq_2', self.liq_2)]:
            was_training[name] = model.training

        self.hyperagent.train()

        try:
            batch, _, _ = replay_buffer.sample()
            batch = {k: v.to(self.device) for k, v in batch.items()}

            w_current = self.hyperagent(batch['state'], batch['state_trend'], batch['state_clf'],
                                        batch['state_minute1'], batch['state_minute3'], batch['state_minute5'],
                                        batch['previous_action'])
            w_next = self.hyperagent_target(batch['next_state'], batch['next_state_trend'], batch['next_state_clf'],
                                            batch['next_state_minute1'], batch['next_state_minute3'],
                                            batch['next_state_minute5'], batch['next_previous_action'])
            w_next_ = self.hyperagent(batch['next_state'], batch['next_state_trend'], batch['next_state_clf'],
                                      batch['next_state_minute1'], batch['next_state_minute3'],
                                      batch['next_state_minute5'], batch['next_previous_action'])

            with torch.no_grad():
                qs_current = [
                    self.trend_agents[1](batch['state'], batch['state_trend'], batch['state_minute1'],
                                         batch['state_minute3'], batch['state_minute5'], batch['previous_action']),
                    self.trend_agents[-1](batch['state'], batch['state_trend'], batch['state_minute1'],
                                          batch['state_minute3'], batch['state_minute5'], batch['previous_action']),
                    self.vol_agents[1](batch['state'], batch['state_trend'], batch['state_minute1'],
                                       batch['state_minute3'], batch['state_minute5'], batch['previous_action']),
                    self.vol_agents[-1](batch['state'], batch['state_trend'], batch['state_minute1'],
                                        batch['state_minute3'], batch['state_minute5'], batch['previous_action']),
                    self.liq_agents[1](batch['state'], batch['state_trend'], batch['state_minute1'],
                                       batch['state_minute3'], batch['state_minute5'], batch['previous_action']),
                    self.liq_agents[-1](batch['state'], batch['state_trend'], batch['state_minute1'],
                                        batch['state_minute3'], batch['state_minute5'], batch['previous_action'])
                ]
                qs_next = [
                    self.trend_agents[1](batch['next_state'], batch['next_state_trend'], batch['next_state_minute1'],
                                         batch['next_state_minute3'], batch['next_state_minute5'],
                                         batch['next_previous_action']),
                    self.trend_agents[-1](batch['next_state'], batch['next_state_trend'], batch['next_state_minute1'],
                                          batch['next_state_minute3'], batch['next_state_minute5'],
                                          batch['next_previous_action']),
                    self.vol_agents[1](batch['next_state'], batch['next_state_trend'], batch['next_state_minute1'],
                                       batch['next_state_minute3'], batch['next_state_minute5'],
                                       batch['next_previous_action']),
                    self.vol_agents[-1](batch['next_state'], batch['next_state_trend'], batch['next_state_minute1'],
                                        batch['next_state_minute3'], batch['next_state_minute5'],
                                        batch['next_previous_action']),
                    self.liq_agents[1](batch['next_state'], batch['next_state_trend'], batch['next_state_minute1'],
                                       batch['next_state_minute3'], batch['next_state_minute5'],
                                       batch['next_previous_action']),
                    self.liq_agents[-1](batch['next_state'], batch['next_state_trend'], batch['next_state_minute1'],
                                        batch['next_state_minute3'], batch['next_state_minute5'],
                                        batch['next_previous_action'])
                ]
            q_distribution = self.calculate_q(w_current, qs_current)
            q_current = q_distribution.gather(-1, batch['action']).squeeze(-1)
            a_argmax = self.calculate_q(w_next_, qs_next).argmax(dim=-1, keepdim=True)
            q_nexts = self.calculate_q(w_next, qs_next)
            q_target = batch['reward'] + self.gamma * (1 - batch['terminal']) * q_nexts.gather(-1, a_argmax).squeeze(-1)

            td_error = self.loss_func(q_current, q_target)
            memory_error = self.loss_func(q_current, batch['q_memory'])

            demonstration = batch['demo_action']
            KL_loss = F.kl_div(
                (q_distribution.softmax(dim=-1) + 1e-8).log(),
                (demonstration.softmax(dim=-1) + 1e-8),
                reduction="batchmean",
            )

            loss = td_error + args.alpha * memory_error + args.beta * KL_loss
            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.hyperagent.parameters(), 1)
            self.optimizer.step()
            for param, target_param in zip(self.hyperagent.parameters(), self.hyperagent_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            self.update_counter += 1
            return td_error.cpu(), memory_error.cpu(), KL_loss.cpu(), torch.mean(q_current.cpu()), torch.mean(
                q_target.cpu())

        finally:

            if not was_training['hyperagent']:
                self.hyperagent.eval()
            if not was_training['hyperagent_target']:
                self.hyperagent_target.eval()

            self.trend_1.eval()
            self.trend_2.eval()
            self.vol_1.eval()
            self.vol_2.eval()
            self.liq_1.eval()
            self.liq_2.eval()

    def act(self, state, state_trend, state_clf, state_minute1, state_minute3, state_minute5, info):
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        x3 = torch.FloatTensor(state_clf).unsqueeze(0).to(self.device)
        x4 = torch.FloatTensor(state_minute1).to(self.device)
        x5 = torch.FloatTensor(state_minute3).to(self.device)
        x6 = torch.FloatTensor(state_minute5).to(self.device)

        previous_action = torch.unsqueeze(
            torch.tensor(info["previous_action"]).long().to(self.device),
            0).to(self.device)
        if np.random.uniform() < (1 - self.epsilon):
            qs = [
                self.trend_agents[1](x1, x2, x4, x5, x6, previous_action),
                self.trend_agents[-1](x1, x2, x4, x5, x6, previous_action),
                self.vol_agents[1](x1, x2, x4, x5, x6, previous_action),
                self.vol_agents[-1](x1, x2, x4, x5, x6, previous_action),
                self.liq_agents[1](x1, x2, x4, x5, x6, previous_action),
                self.liq_agents[-1](x1, x2, x4, x5, x6, previous_action)
            ]
            # print(f"train qs:{qs}")
            w = self.hyperagent(x1, x2, x3, x4, x5, x6, previous_action)
            # print(f"train w:{w}")
            actions_value = self.calculate_q(w, qs)
            # print(f"train actions_value:{actions_value}")
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]
        else:
            action_choice = [0, 1]
            action = random.choice(action_choice)
        return action

    def act_test(self, state, state_trend, state_clf, state_minute1, state_minute3, state_minute5, info):
        with torch.no_grad():
            x1 = torch.FloatTensor(state).to(self.device)
            x2 = torch.FloatTensor(state_trend).to(self.device)
            x3 = torch.FloatTensor(state_clf).unsqueeze(0).to(self.device)
            x4 = torch.FloatTensor(state_minute1).to(self.device)
            x5 = torch.FloatTensor(state_minute3).to(self.device)
            x6 = torch.FloatTensor(state_minute5).to(self.device)

            previous_action = torch.unsqueeze(
                torch.tensor(info["previous_action"]).long().to(self.device),
                0).to(self.device)
            qs = [
                self.trend_agents[1](x1, x2, x4, x5, x6, previous_action),
                self.trend_agents[-1](x1, x2, x4, x5, x6, previous_action),
                self.vol_agents[1](x1, x2, x4, x5, x6, previous_action),
                self.vol_agents[-1](x1, x2, x4, x5, x6, previous_action),
                self.liq_agents[1](x1, x2, x4, x5, x6, previous_action),
                self.liq_agents[-1](x1, x2, x4, x5, x6, previous_action)
            ]
            # print(f"test qs:{qs}")
            w = self.hyperagent(x1, x2, x3, x4, x5, x6, previous_action)
            # print(f"test w:{w}")
            actions_value = self.calculate_q(w, qs)
            # print(f"test actions_value:{actions_value}")
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]
            return action

    def q_estimate(self, state, state_trend, state_clf, state_minute1, state_minute3, state_minute5, info):
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        x3 = torch.FloatTensor(state_clf).unsqueeze(0).to(self.device)
        x4 = torch.FloatTensor(state_minute1).to(self.device)
        x5 = torch.FloatTensor(state_minute3).to(self.device)
        x6 = torch.FloatTensor(state_minute5).to(self.device)

        previous_action = torch.unsqueeze(
            torch.tensor(info["previous_action"]).long().to(self.device),
            0).to(self.device)
        qs = [
            self.trend_agents[1](x1, x2, x4, x5, x6, previous_action),
            self.trend_agents[-1](x1, x2, x4, x5, x6, previous_action),
            self.vol_agents[1](x1, x2, x4, x5, x6, previous_action),
            self.vol_agents[-1](x1, x2, x4, x5, x6, previous_action),
            self.liq_agents[1](x1, x2, x4, x5, x6, previous_action),
            self.liq_agents[-1](x1, x2, x4, x5, x6, previous_action)
        ]
        w = self.hyperagent(x1, x2, x3, x4, x5, x6, previous_action)
        actions_value = self.calculate_q(w, qs)
        q = torch.max(actions_value, 1)[0].detach().cpu().numpy()

        return q

    def calculate_hidden(self, state, state_trend, state_minute1, state_minute3, state_minute5, info):
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        x3 = torch.FloatTensor(state_minute1).to(self.device)
        x4 = torch.FloatTensor(state_minute3).to(self.device)
        x5 = torch.FloatTensor(state_minute5).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor(info["previous_action"]).long().to(self.device),
            0).to(self.device)
        with torch.no_grad():
            hs = self.hyperagent.encode(x1, x2, x3, x4, x5, previous_action).cpu().numpy()
        return hs

    def train(self):
        best_model_path = os.path.join(self.model_path, 'best_model.pkl')
        epoch_counter = 0
        best_return_rate = -float('inf')
        best_model = None
        self.replay_buffer = ReplayBuffer_High(args, self.n_state_1, self.n_state_2, self.n_clf, self.n_state_1minute,
                                               self.n_state_3minute, self.n_state_5minute, self.look_back_window,
                                               self.n_action)
        for sample in range(self.epoch_number):
            print('epoch ', epoch_counter + 1)
            print(f'current time:{datetime.datetime.now(beijing_tz)}')
            step_counter = 0
            episode_counter = 0
            epoch_return_rate_train_list = []
            epoch_final_balance_train_list = []
            epoch_required_money_train_list = []
            epoch_reward_sum_train_list = []
            self.df = pd.read_feather(
                os.path.join(self.train_data_path, "train.feather"))

            train_env = Training_Env(
                df=self.df,
                tech_indicator_list=self.tech_indicator_list,
                tech_indicator_list_trend=self.tech_indicator_list_trend,
                clf_list=self.clf_list,
                minute1_indicator_list=self.minute1_indicator_list,
                minute3_indicator_list=self.minute3_indicator_list,
                minute5_indicator_list=self.minute5_indicator_list,
                transcation_cost=self.transcation_cost,
                back_time_length=self.back_time_length,
                look_back_window=self.look_back_window,
                max_holding_number=self.max_holding_number,
                initial_action=random.choices(range(self.n_action), k=1)[0],
                alpha=0)
            s, s2, s3, s_min1, s_min3, s_min5, done, info = train_env.reset()

            if done:
                continue

            episode_reward_sum = 0

            while True:
                a = self.act(s, s2, s3, s_min1, s_min3, s_min5, info)
                s_, s2_, s3_, s_min1_, s_min3_, s_min5_, r, done, info_ = train_env.step(a)
                hs = self.calculate_hidden(s, s2, s_min1, s_min3, s_min5, info)
                q = r + self.gamma * (1 - done) * self.q_estimate(s_, s2_, s3_, s_min1_, s_min3_, s_min5_, info_)
                q_memory = self.memory.query(hs, a)
                if np.isnan(q_memory):
                    q_memory = q
                self.replay_buffer.store_transition(s, s2, s3, s_min1, s_min3, s_min5, info['previous_action'],
                                                    info['q_value'], a, r, s_, s2_, s3_, s_min1_, s_min3_, s_min5_,
                                                    info_['previous_action'],
                                                    info_['q_value'], done, q_memory)
                self.memory.add(hs, a, q, s, s2, s_min1, s_min3, s_min5, info['previous_action'])
                episode_reward_sum += r

                s, s2, s3, s_min1, s_min3, s_min5, info = s_, s2_, s3_, s_min1_, s_min3_, s_min5_, info_
                step_counter += 1
                if step_counter % self.eval_update_freq == 0 and step_counter > (
                        self.batch_size + self.n_step):
                    for i in range(self.update_times):
                        td_error, memory_error, KL_loss, q_eval, q_target = self.update(self.replay_buffer)
                        if self.update_counter % self.q_value_memorize_freq == 1:
                            self.log_scalar("td_error", td_error, epoch_counter + 1, episode_counter + 1, step_counter,
                                            self.update_counter)
                            self.log_scalar("memory_error", td_error, epoch_counter + 1, episode_counter + 1,
                                            step_counter,
                                            self.update_counter)
                            self.log_scalar("KL_loss", KL_loss, epoch_counter + 1, episode_counter + 1, step_counter,
                                            self.update_counter)
                            self.log_scalar("total_loss", td_error + args.alpha * memory_error + args.beta * KL_loss,
                                            epoch_counter + 1,
                                            episode_counter + 1, step_counter, self.update_counter)
                            self.log_scalar("q_eval", q_eval, epoch_counter + 1, episode_counter + 1, step_counter,
                                            self.update_counter)
                            self.log_scalar("q_target", q_target, epoch_counter + 1, episode_counter + 1, step_counter,
                                            self.update_counter)

                            # self.writer.add_scalar(
                            #     tag="td_error",
                            #     scalar_value=td_error,
                            #     global_step=self.update_counter,
                            #     walltime=None)
                            # self.writer.add_scalar(
                            #     tag="memory_error",
                            #     scalar_value=memory_error,
                            #     global_step=self.update_counter,
                            #     walltime=None)
                            # self.writer.add_scalar(
                            #     tag="KL_loss",
                            #     scalar_value=KL_loss,
                            #     global_step=self.update_counter,
                            #     walltime=None)
                            # self.writer.add_scalar(
                            #     tag="q_eval",
                            #     scalar_value=q_eval,
                            #     global_step=self.update_counter,
                            #     walltime=None)
                            # self.writer.add_scalar(
                            #     tag="q_target",
                            #     scalar_value=q_target,
                            #     global_step=self.update_counter,
                            #     walltime=None)
                    if step_counter > 4320:
                        self.memory.re_encode(self.hyperagent)
                if done:
                    break
            episode_counter += 1

            final_balance, required_money = train_env.final_balance, train_env.required_money

            # self.log_scalar("return_rate_train", final_balance / (required_money), epoch_counter + 1, episode_counter)
            # self.log_scalar("final_balance_train", final_balance, epoch_counter + 1, episode_counter)
            # self.log_scalar("required_money_train", required_money, epoch_counter + 1, episode_counter)
            # self.log_scalar("reward_sum_train", episode_reward_sum, epoch_counter + 1, episode_counter)

            # self.writer.add_scalar(tag="return_rate_train",
            #                     scalar_value=final_balance / (required_money),
            #                     global_step=episode_counter,
            #                     walltime=None)
            # self.writer.add_scalar(tag="final_balance_train",
            #                     scalar_value=final_balance,
            #                     global_step=episode_counter,
            #                     walltime=None)
            # self.writer.add_scalar(tag="required_money_train",
            #                     scalar_value=required_money,
            #                     global_step=episode_counter,
            #                     walltime=None)
            # self.writer.add_scalar(tag="reward_sum_train",
            #                     scalar_value=episode_reward_sum,
            #                     global_step=episode_counter,
            #                     walltime=None)
            epoch_return_rate_train_list.append(final_balance / (required_money))
            epoch_final_balance_train_list.append(final_balance)
            epoch_required_money_train_list.append(required_money)
            epoch_reward_sum_train_list.append(episode_reward_sum)

            epoch_counter += 1
            self.epsilon = self.epsilon_scheduler.get_epsilon(epoch_counter)
            mean_return_rate_train = np.nanmean(epoch_return_rate_train_list)
            mean_final_balance_train = np.nanmean(epoch_final_balance_train_list)
            mean_required_money_train = np.nanmean(epoch_required_money_train_list)
            mean_reward_sum_train = np.nanmean(epoch_reward_sum_train_list)

            self.log_scalar("epoch_return_rate_train", mean_return_rate_train, epoch_counter)
            self.log_scalar("epoch_final_balance_train", mean_final_balance_train, epoch_counter)
            self.log_scalar("epoch_required_money_train", mean_required_money_train, epoch_counter)
            self.log_scalar("epoch_reward_sum_train", mean_reward_sum_train, epoch_counter)

            # self.writer.add_scalar(
            #         tag="epoch_return_rate_train",
            #         scalar_value=mean_return_rate_train,
            #         global_step=epoch_counter,
            #         walltime=None,
            #     )
            # self.writer.add_scalar(
            #     tag="epoch_final_balance_train",
            #     scalar_value=mean_final_balance_train,
            #     global_step=epoch_counter,
            #     walltime=None,
            #     )
            # self.writer.add_scalar(
            #     tag="epoch_required_money_train",
            #     scalar_value=mean_required_money_train,
            #     global_step=epoch_counter,
            #     walltime=None,
            #     )
            # self.writer.add_scalar(
            #     tag="epoch_reward_sum_train",
            #     scalar_value=mean_reward_sum_train,
            #     global_step=epoch_counter,
            #     walltime=None,
            #     )
            epoch_path = os.path.join(self.model_path, "val_epochs",
                                      "epoch_{}".format(epoch_counter))
            if not os.path.exists(epoch_path):
                os.makedirs(epoch_path)
            torch.save(self.hyperagent.state_dict(),
                       os.path.join(epoch_path, "trained_model.pkl"))
            val_path = os.path.join(epoch_path, "val")
            if not os.path.exists(val_path):
                os.makedirs(val_path)
            return_rate_eval = self.val_cluster(epoch_path, val_path)
            self.log_scalar("return_rate_eval", return_rate_eval, epoch_counter)
            if return_rate_eval > best_return_rate:
                best_return_rate = return_rate_eval
                best_model = self.hyperagent.state_dict()
                best_model_epoch = epoch_counter
                print("best model updated to epoch ", best_model_epoch)

                torch.save(best_model, best_model_path)

        self.test_cluster(best_model_path, self.result_path)

    def val_cluster(self, epoch_path, save_path):
        self.hyperagent.load_state_dict(
            torch.load(os.path.join(epoch_path, "trained_model.pkl")))
        self.hyperagent.eval()
        counter = False
        action_list = []
        reward_list = []
        final_balance_list = []
        required_money_list = []
        commission_fee_list = []
        self.df = pd.read_feather(
            os.path.join(self.val_data_path, "val.feather"))

        val_env = Testing_Env(
            df=self.df,
            tech_indicator_list=self.tech_indicator_list,
            tech_indicator_list_trend=self.tech_indicator_list_trend,
            clf_list=self.clf_list,
            minute1_indicator_list=self.minute1_indicator_list,
            minute3_indicator_list=self.minute3_indicator_list,
            minute5_indicator_list=self.minute5_indicator_list,
            transcation_cost=self.transcation_cost,
            back_time_length=self.back_time_length,
            look_back_window=self.look_back_window,
            max_holding_number=self.max_holding_number,
            initial_action=0)
        s, s2, s3, s_min1, s_min3, s_min5, done, info = val_env.reset()
        done = False
        action_list_episode = []
        reward_list_episode = []
        while not done:
            a = self.act_test(s, s2, s3, s_min1, s_min3, s_min5, info)
            s_, s2_, s3_, s_min1_, s_min3_, s_min5_, r, done, info_ = val_env.step(a)
            reward_list_episode.append(r)
            s, s2, s3, s_min1, s_min3, s_min5, info = s_, s2_, s3_, s_min1_, s_min3_, s_min5_, info_
            action_list_episode.append(a)
        portfit_magine, final_balance, required_money, commission_fee = val_env.get_final_return_rate(
            slient=True)
        final_balance = val_env.final_balance
        # action_list.append(action_list_episode)
        # reward_list.append(reward_list_episode)
        # final_balance_list.append(final_balance)
        # required_money_list.append(required_money)
        # commission_fee_list.append(commission_fee)

        action_list = np.array(action_list_episode)
        reward_list = np.array(reward_list_episode)
        final_balance_list = np.array(final_balance)
        required_money_list = np.array(required_money)
        commission_fee_list = np.array(commission_fee)
        np.save(os.path.join(save_path, "action_val.npy"), action_list)
        np.save(os.path.join(save_path, "reward_val.npy"), reward_list)
        np.save(os.path.join(save_path, "final_balance_val.npy"),
                final_balance_list)
        np.save(os.path.join(save_path, "require_money_val.npy"),
                required_money_list)
        np.save(os.path.join(save_path, "commission_fee_history_val.npy"),
                commission_fee_list)
        return_rate = final_balance / required_money
        return return_rate

    def test_cluster(self, best_model_path, save_path):
        test_result_path = os.path.join(save_path, "test_result")
        os.makedirs(test_result_path, exist_ok=True)

        self.hyperagent.load_state_dict(
            torch.load(best_model_path))
        self.hyperagent.eval()
        counter = False
        action_list = []
        reward_list = []
        final_balance_list = []
        required_money_list = []
        commission_fee_list = []
        self.df = pd.read_feather(
            os.path.join(self.test_data_path, "test.feather"))

        test_env = Testing_Env(
            df=self.df,
            tech_indicator_list=self.tech_indicator_list,
            tech_indicator_list_trend=self.tech_indicator_list_trend,
            clf_list=self.clf_list,
            minute1_indicator_list=self.minute1_indicator_list,
            minute3_indicator_list=self.minute3_indicator_list,
            minute5_indicator_list=self.minute5_indicator_list,
            transcation_cost=self.transcation_cost,
            back_time_length=self.back_time_length,
            look_back_window=self.look_back_window,
            max_holding_number=self.max_holding_number,
            initial_action=0)
        s, s2, s3, s_min1, s_min3, s_min5, done, info = test_env.reset()
        done = False
        action_list_episode = []
        reward_list_episode = []

        while not done:
            a = self.act_test(s, s2, s3, s_min1, s_min3, s_min5, info)
            s_, s2_, s3_, s_min1_, s_min3_, s_min5_, r, done, info_ = test_env.step(a)
            reward_list_episode.append(r)
            s, s2, s3, s_min1, s_min3, s_min5, info = s_, s2_, s3_, s_min1_, s_min3_, s_min5_, info_
            action_list_episode.append(a)

        portfit_magine, final_balance, required_money, commission_fee = test_env.get_final_return_rate(
            slient=True)
        final_balance = test_env.final_balance

        # action_list.append(action_list_episode)
        # reward_list.append(reward_list_episode)
        # final_balance_list.append(final_balance)
        # required_money_list.append(required_money)
        # commission_fee_list.append(commission_fee)

        action_list = np.array(action_list_episode)
        reward_list = np.array(reward_list_episode)
        final_balance_list = np.array(final_balance)
        required_money_list = np.array(required_money)
        commission_fee_list = np.array(commission_fee)

        np.save(os.path.join(test_result_path, "action.npy"), action_list)
        np.save(os.path.join(test_result_path, "reward.npy"), reward_list)
        np.save(os.path.join(test_result_path, "final_balance.npy"),
                final_balance_list)
        np.save(os.path.join(test_result_path, "require_money.npy"),
                required_money_list)
        np.save(os.path.join(test_result_path, "commission_fee_history.npy"),
                commission_fee_list)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    agent = DQN(args)
    agent.train()
