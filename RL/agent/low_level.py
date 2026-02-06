import pathlib
import sys
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import pickle
import os
from torch.utils.tensorboard import SummaryWriter
import warnings
from tqdm import tqdm
import datetime
import pytz
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = str(pathlib.Path(__file__).resolve().parents[3])
sys.path.append(ROOT)
sys.path.insert(0, ".")

from model.net import *
from env.low_level_env import Testing_Env, Training_Env
from RL.util.utili import get_ada, get_epsilon, LinearDecaySchedule
from RL.util.replay_buffer import ReplayBuffer

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--epoch_number", type=int, default=2)
parser.add_argument("--buffer_size", type=int, default=1000000, )
parser.add_argument("--dataset", type=str, default="BTCUSDT")
parser.add_argument("--q_value_memorize_freq", type=int, default=10, )
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--eval_update_freq", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--epsilon_start", type=float, default=0.5)
parser.add_argument("--epsilon_end", type=float, default=0.1)
parser.add_argument("--decay_length", type=int, default=5)
parser.add_argument("--update_times", type=int, default=10)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--transcation_cost", type=float, default=2.0 / 10000)
parser.add_argument("--back_time_length", type=int, default=1)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--n_step", type=int, default=1)
parser.add_argument("--label", type=str, default="label_1")
parser.add_argument("--clf", type=str, default="trend")
parser.add_argument("--alpha", type=float, default="0")
parser.add_argument("--device", type=str, default="cuda:0")
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


def calculate_alpha(diff, k):
    alpha = 16 * (1 - torch.exp(-k * diff))
    return torch.clip(alpha, 0, 16)


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
            os.makedirs(self.base_path, exist_ok=True)
        self.result_base_path = os.path.join(self.base_path, args.dataset, "low_level")
        if not os.path.exists(self.result_base_path):
            os.makedirs(self.result_base_path, exist_ok=True)

        if args.clf == 'trend':
            self.train_data_path = os.path.join(ROOT, "MedalHFT", "MyData", args.dataset, "trend_data",
                                                f"{args.trend_method}", "train")
            self.val_data_path = os.path.join(ROOT, "MedalHFT", "MyData", args.dataset, "trend_data",
                                              f"{args.trend_method}", "val")
            self.test_data_path = os.path.join(ROOT, "MedalHFT", "MyData", args.dataset, "trend_data",
                                               f"{args.trend_method}", "test")

            self.result_path = os.path.join(self.result_base_path, args.clf,
                                            f"{args.label}_alpha_{args.alpha}_{args.trend_method}")

            with open(os.path.join(self.train_data_path, 'trend_labels.pkl'), 'rb') as file:
                self.train_index = pickle.load(file)
            with open(os.path.join(self.val_data_path, 'trend_labels.pkl'), 'rb') as file:
                self.val_index = pickle.load(file)
            with open(os.path.join(self.test_data_path, 'trend_labels.pkl'), 'rb') as file:
                self.test_index = pickle.load(file)
        elif args.clf == 'vol':
            self.train_data_path = os.path.join(ROOT, "MedalHFT", "MyData", args.dataset, "vol_data",
                                                f"{args.vol_method}", "train")
            self.val_data_path = os.path.join(ROOT, "MedalHFT", "MyData", args.dataset, "vol_data",
                                              f"{args.vol_method}", "val")
            self.test_data_path = os.path.join(ROOT, "MedalHFT", "MyData", args.dataset, "vol_data",
                                               f"{args.vol_method}", "test")

            self.result_path = os.path.join(self.result_base_path, args.clf,
                                            f"{args.label}_alpha_{args.alpha}_{args.vol_method}")

            with open(os.path.join(self.train_data_path, 'vol_labels.pkl'), 'rb') as file:
                self.train_index = pickle.load(file)
            with open(os.path.join(self.val_data_path, 'vol_labels.pkl'), 'rb') as file:
                self.val_index = pickle.load(file)
            with open(os.path.join(self.test_data_path, 'vol_labels.pkl'), 'rb') as file:
                self.test_index = pickle.load(file)
        elif args.clf == 'liq':
            self.train_data_path = os.path.join(ROOT, "MedalHFT", "MyData", args.dataset, "liq_data",
                                                f"{args.liq_method}", "train")
            self.val_data_path = os.path.join(ROOT, "MedalHFT", "MyData", args.dataset, "liq_data",
                                              f"{args.liq_method}", "val")
            self.test_data_path = os.path.join(ROOT, "MedalHFT", "MyData", args.dataset, "liq_data",
                                               f"{args.liq_method}", "test")

            self.result_path = os.path.join(self.result_base_path, args.clf,
                                            f"{args.label}_alpha_{args.alpha}_{args.liq_method}")

            with open(os.path.join(self.train_data_path, 'liq_labels.pkl'), 'rb') as file:
                self.train_index = pickle.load(file)
            with open(os.path.join(self.val_data_path, 'liq_labels.pkl'), 'rb') as file:
                self.val_index = pickle.load(file)
            with open(os.path.join(self.test_data_path, 'liq_labels.pkl'), 'rb') as file:
                self.test_index = pickle.load(file)

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        self.label = int(args.label.split('_')[1])
        self.model_path = os.path.join(self.result_path)

        self.dataset = args.dataset
        self.clf = args.clf
        if "BTC" in self.dataset:
            self.max_holding_number = 0.01
        elif "ETH" in self.dataset:
            self.max_holding_number = 0.01
        elif "BNB" in self.dataset:
            self.max_holding_number = 0.1
        elif "SOL" in self.dataset:
            self.max_holding_number = 1
        elif "LTC" in self.dataset:
            self.max_holding_number = 1
        elif "LINK" in self.dataset:
            self.max_holding_number = 1
        else:
            raise Exception("we do not support other dataset yet")
        self.epoch_number = args.epoch_number

        # self.writer = SummaryWriter(self.log_path)

        self.log_path = os.path.join(self.model_path, "log")
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self.log_files = {}
        metrics = ["td_error", "KL_loss", "total_loss", "q_eval", "q_target", "return_rate_train", "return_rate_eval",
                   "final_balance_train", "required_money_train", "reward_sum_train",
                   "epoch_return_rate_train", "epoch_final_balance_train",
                   "epoch_required_money_train", "epoch_reward_sum_train"]

        for i, metric in enumerate(metrics):
            file_path = os.path.join(self.log_path, f"{i + 1}#{metric}.txt")
            self.log_files[metric] = file_path

            with open(file_path, 'w') as f:
                f.write(f"{metric} Log\n")
                f.write("=" * 50 + "\n")

        self.update_counter = 0
        self.q_value_memorize_freq = args.q_value_memorize_freq

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.tech_indicator_list = np.load('./MyData/feature_list/single_features.npy', allow_pickle=True).tolist()
        self.tech_indicator_list_trend = np.load('./MyData/feature_list/trend_features.npy', allow_pickle=True).tolist()
        self.minute1_indicator_list = np.load('./MyData/feature_list/minute1_features.npy', allow_pickle=True).tolist()
        self.minute3_indicator_list = np.load('./MyData/feature_list/minute3_features.npy', allow_pickle=True).tolist()
        self.minute5_indicator_list = np.load('./MyData/feature_list/minute5_features.npy', allow_pickle=True).tolist()

        self.transcation_cost = args.transcation_cost
        self.back_time_length = args.back_time_length
        self.look_back_window = args.look_back_window

        self.n_action = 2
        self.n_state_1 = len(self.tech_indicator_list)
        self.n_state_2 = len(self.tech_indicator_list_trend)
        self.n_state_1minute = len(self.minute1_indicator_list)
        self.n_state_3minute = len(self.minute3_indicator_list)
        self.n_state_5minute = len(self.minute5_indicator_list)

        self.eval_net = subagent(self.n_state_1, self.n_state_2, self.n_state_1minute, self.n_state_3minute,
                                 self.n_state_5minute, self.look_back_window, self.n_action, 64).to(self.device)
        self.target_net = subagent(self.n_state_1, self.n_state_2, self.n_state_1minute, self.n_state_3minute,
                                   self.n_state_5minute, self.look_back_window, self.n_action, 64).to(self.device)
        self.hardupdate()
        self.update_times = args.update_times
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(),
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

    def update(self, replay_buffer):
        self.eval_net.train()
        batch, _, _ = replay_buffer.sample()
        batch = {k: v.to(self.device) for k, v in batch.items()}
        a_argmax = self.eval_net(batch['next_state'], batch['next_state_trend'], batch['next_state_minute1'],
                                 batch['next_state_minute3'], batch['next_state_minute5'],
                                 batch['next_previous_action']).argmax(dim=-1, keepdim=True)
        q_target = batch['reward'] + self.gamma * (1 - batch['terminal']) * self.target_net(batch['next_state'],
                                                                                            batch['next_state_trend'],
                                                                                            batch['next_state_minute1'],
                                                                                            batch['next_state_minute3'],
                                                                                            batch['next_state_minute5'],
                                                                                            batch[
                                                                                                'next_previous_action']).gather(
            -1, a_argmax).squeeze(-1)

        q_distribution = self.eval_net(batch['state'], batch['state_trend'], batch['state_minute1'],
                                       batch['state_minute3'], batch['state_minute5'], batch['previous_action'])
        q_eval = q_distribution.gather(-1, batch['action']).squeeze(-1)

        td_error = self.loss_func(q_eval, q_target)

        demonstration = batch['demo_action']
        KL_loss = F.kl_div(
            (q_distribution.softmax(dim=-1) + 1e-8).log(),
            (demonstration.softmax(dim=-1) + 1e-8),
            reduction="batchmean",
        )

        alpha = args.alpha
        loss = td_error + alpha * KL_loss
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), 1)
        self.optimizer.step()
        for param, target_param in zip(self.eval_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.update_counter += 1
        self.eval_net.eval()
        return td_error.cpu(), KL_loss.cpu(), torch.mean(
            q_eval.cpu()), torch.mean(q_target.cpu())

    def hardupdate(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def act(self, state, state_trend, state_minute1, state_minute3, state_minute5, info):
        x1 = torch.FloatTensor(state).to(self.device)
        # print(f"x1 shape:{x1.shape}")
        x2 = torch.FloatTensor(state_trend).to(self.device)
        # print(f"x2 shape:{x2.shape}")
        x3 = torch.FloatTensor(state_minute1).to(self.device)
        # print(f"x3 shape:{x3.shape}")
        x4 = torch.FloatTensor(state_minute3).to(self.device)
        # print(f"x4 shape:{x4.shape}")
        x5 = torch.FloatTensor(state_minute5).to(self.device)
        # print(f"x5 shape:{x5.shape}")

        previous_action = torch.unsqueeze(
            torch.tensor(info["previous_action"]).long().to(self.device),
            0).to(self.device)
        if np.random.uniform() < (1 - self.epsilon):
            actions_value = self.eval_net(x1, x2, x3, x4, x5, previous_action)
            # print(f"action value:{actions_value}")
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]
        else:
            action_choice = [0, 1]
            action = random.choice(action_choice)
            # print(f"action:{action}")
        return action

    def act_test(self, state, state_trend, state_minute1, state_minute3, state_minute5, info):
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        x3 = torch.FloatTensor(state_minute1).to(self.device)
        x4 = torch.FloatTensor(state_minute3).to(self.device)
        x5 = torch.FloatTensor(state_minute5).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor(info["previous_action"]).long(), 0).to(self.device)
        actions_value = self.eval_net(x1, x2, x3, x4, x5, previous_action)
        # print(f"test action value:{actions_value}")
        action = torch.max(actions_value, 1)[1].data.cpu().numpy()
        action = action[0]
        # print(f"test action:{action}")
        return action

    def train(self):
        df_list = self.train_index[self.label]
        df_number = int(len(df_list))
        epoch_counter = 0
        self.replay_buffer = ReplayBuffer(args, self.n_state_1, self.n_state_2, self.n_state_1minute,
                                          self.n_state_3minute, self.n_state_5minute, self.look_back_window,
                                          self.n_action)
        best_return_rate = -float('inf')
        best_model = None
        best_model_epoch = 0
        for sample in tqdm(range(self.epoch_number)):

            epoch_return_rate_train_list = []
            epoch_final_balance_train_list = []
            epoch_required_money_train_list = []
            epoch_reward_sum_train_list = []
            step_counter = 0
            episode_counter = 0
            print('epoch ', epoch_counter + 1)
            print(f'current time:{datetime.datetime.now(beijing_tz)}')
            random_list = self.train_index[self.label]
            random.shuffle(random_list)
            random_position_list = random.choices(range(self.n_action), k=df_number)
            print(f"train_index:{random_list}")

            for i in tqdm(range(df_number)):
                df_index = random_list[i]
                print("training with df", df_index)
                self.df = pd.read_feather(
                    os.path.join(self.train_data_path, "df_{}.feather".format(df_index)))
                self.eval_net.eval()

                train_env = Training_Env(
                    df=self.df,
                    tech_indicator_list=self.tech_indicator_list,
                    tech_indicator_list_trend=self.tech_indicator_list_trend,
                    minute1_indicator_list=self.minute1_indicator_list,
                    minute3_indicator_list=self.minute3_indicator_list,
                    minute5_indicator_list=self.minute5_indicator_list,
                    transcation_cost=self.transcation_cost,
                    back_time_length=self.back_time_length,
                    look_back_window=self.look_back_window,
                    max_holding_number=self.max_holding_number,
                    initial_action=random_position_list[i],
                    alpha=0)
                s, s2, s_min1, s_min3, s_min5, done, info = train_env.reset()

                if done:
                    continue

                episode_reward_sum = 0

                while True:
                    a = self.act(s, s2, s_min1, s_min3, s_min5, info)
                    s_, s2_, s_min1_, s_min3_, s_min5_, r, done, info_ = train_env.step(a)
                    self.replay_buffer.store_transition(s, s2, s_min1, s_min3, s_min5, info['previous_action'],
                                                        info['q_value'], a, r, s_, s2_, s_min1_, s_min3_, s_min5_,
                                                        info_['previous_action'],
                                                        info_['q_value'], done)
                    episode_reward_sum += r

                    s, s2, s_min1, s_min3, s_min5, info = s_, s2_, s_min1_, s_min3_, s_min5_, info_
                    step_counter += 1
                    if step_counter % self.eval_update_freq == 0 and step_counter > (
                            self.batch_size + self.n_step):
                        for i in range(self.update_times):
                            td_error, KL_loss, q_eval, q_target = self.update(self.replay_buffer)
                            if self.update_counter % self.q_value_memorize_freq == 1:
                                self.log_scalar("td_error", td_error, epoch_counter + 1, episode_counter + 1,
                                                step_counter, self.update_counter)
                                self.log_scalar("KL_loss", KL_loss, epoch_counter + 1, episode_counter + 1,
                                                step_counter, self.update_counter)
                                self.log_scalar("total_loss", td_error + args.alpha * KL_loss, epoch_counter + 1,
                                                episode_counter + 1, step_counter, self.update_counter)
                                self.log_scalar("q_eval", q_eval, epoch_counter + 1, episode_counter + 1, step_counter,
                                                self.update_counter)
                                self.log_scalar("q_target", q_target, epoch_counter + 1, episode_counter + 1,
                                                step_counter, self.update_counter)

                                # self.writer.add_scalar(
                                #     tag="td_error",
                                #     scalar_value=td_error,
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
                    if done:
                        break
                episode_counter += 1
                final_balance, required_money = train_env.final_balance, train_env.required_money

                self.log_scalar("return_rate_train", final_balance / (required_money), epoch_counter + 1,
                                episode_counter)
                self.log_scalar("final_balance_train", final_balance, epoch_counter + 1, episode_counter)
                self.log_scalar("required_money_train", required_money, epoch_counter + 1, episode_counter)
                self.log_scalar("reward_sum_train", episode_reward_sum, epoch_counter + 1, episode_counter)

                # self.writer.add_scalar(tag="return_rate_train",
                #                    scalar_value=final_balance / (required_money),
                #                    global_step=episode_counter,
                #                    walltime=None)
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
            torch.save(self.eval_net.state_dict(),
                       os.path.join(epoch_path, "trained_model.pkl"))
            val_path = os.path.join(epoch_path, "val")
            if not os.path.exists(val_path):
                os.makedirs(val_path)
            return_rate_0 = self.val_cluster(epoch_path, val_path, 0)
            return_rate_1 = self.val_cluster(epoch_path, val_path, 1)
            return_rate_eval = (return_rate_0 + return_rate_1) / 2
            self.log_scalar("return_rate_eval", return_rate_eval, epoch_counter)
            print(f"epoch {epoch_counter} return_rate_eval: {return_rate_eval}")
            if return_rate_eval > best_return_rate:
                best_return_rate = return_rate_eval
                best_model = self.eval_net.state_dict()
                best_model_epoch = epoch_counter
                print("best model updated to epoch ", best_model_epoch)
                best_model_path = os.path.join(self.model_path, 'best_model.pkl')
                torch.save(best_model, best_model_path)
        print("best model epoch: ", best_model_epoch)

    def val_cluster(self, epoch_path, save_path, initial_action):
        self.eval_net.load_state_dict(
            torch.load(os.path.join(epoch_path, "trained_model.pkl")))
        self.eval_net.eval()
        df_list = self.val_index[self.label]
        df_number = int(len(df_list))
        action_list = []
        reward_list = []
        final_balance_list = []
        required_money_list = []
        commission_fee_list = []
        print(f"validation_index:{df_list}")
        for i in range(df_number):
            print("validating on df", df_list[i])
            self.df = pd.read_feather(
                os.path.join(self.val_data_path, "df_{}.feather".format(df_list[i])))

            val_env = Testing_Env(
                df=self.df,
                tech_indicator_list=self.tech_indicator_list,
                tech_indicator_list_trend=self.tech_indicator_list_trend,
                minute1_indicator_list=self.minute1_indicator_list,
                minute3_indicator_list=self.minute3_indicator_list,
                minute5_indicator_list=self.minute5_indicator_list,
                transcation_cost=self.transcation_cost,
                back_time_length=self.back_time_length,
                look_back_window=self.look_back_window,
                max_holding_number=self.max_holding_number,
                initial_action=initial_action)
            s, s2, s_min1, s_min3, s_min5, done, info = val_env.reset()
            if done:
                continue
            # done = False
            action_list_episode = []
            reward_list_episode = []
            while not done:
                a = self.act_test(s, s2, s_min1, s_min3, s_min5, info)
                s_, s2_, s_min1_, s_min3_, s_min5_, r, done, info_ = val_env.step(a)
                reward_list_episode.append(r)
                s, s2, s_min1, s_min3, s_min5, info = s_, s2_, s_min1_, s_min3_, s_min5_, info_
                action_list_episode.append(a)
            portfit_magine, final_balance, required_money, commission_fee = val_env.get_final_return_rate(
                slient=True)
            final_balance = val_env.final_balance
            required_money = val_env.required_money
            action_list.append(action_list_episode)
            reward_list.append(reward_list_episode)
            final_balance_list.append(final_balance)
            required_money_list.append(required_money)
            commission_fee_list.append(commission_fee)
        # action_list = np.array(action_list)
        # reward_list = np.array(reward_list)
        final_balance_list = np.array(final_balance_list)
        required_money_list = np.array(required_money_list)
        commission_fee_list = np.array(commission_fee_list)

        np.save(os.path.join(save_path, "action_val_{}.npy".format(initial_action)),
                np.array(action_list, dtype=object))
        np.save(os.path.join(save_path, "reward_val_{}.npy".format(initial_action)),
                np.array(reward_list, dtype=object))
        np.save(os.path.join(save_path, "final_balance_val_{}.npy".format(initial_action)),
                final_balance_list)
        np.save(os.path.join(save_path, "require_money_val_{}.npy".format(initial_action)),
                required_money_list)
        np.save(os.path.join(save_path, "commission_fee_history_val_{}.npy".format(initial_action)),
                commission_fee_list)
        return_rate_mean = np.nanmean(final_balance_list / required_money_list)
        np.save(os.path.join(save_path, "return_rate_mean_val_{}.npy".format(initial_action)),
                return_rate_mean)
        return return_rate_mean


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    agent = DQN(args)
    agent.train()
