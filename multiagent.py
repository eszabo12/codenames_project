

"""This is a minimal example of using Tianshou with MARL to train agents.

Author: Will (https://github.com/WillDudley)

Python version used: 3.8.10

Requirements:
pettingzoo == 1.22.0
git+https://github.com/thu-ml/tianshou
"""

import os
from typing import Optional, Tuple
import argparse
import gym
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net

from pettingzoo.classic import tictactoe_v3
from env import CodemasterEnv, GuesserEnv
words = [
        'vacuum', 'whip', 'moon', 'school', 'tube', 'lab', 'key', 'table', 'lead', 'crown',
        'bomb', 'bug', 'pipe', 'roulette', 'play', 'cloak', 'piano', 'beijing', 'bison',
        'boot', 'cap', 'car','change', 'circle', 'cliff', 'conductor', 'cricket', 'death', 'diamond',
        'figure', 'gas', 'india', 'jupiter', 'kid', 'king', 'lemon', 'litter', 'nut',
        'racket', 'row', 'scientist', 'shark', 'stream', 'swing', 'unicorn', 'witch', 'worm',
        'pistol', 'saturn', 'rock', 'superhero', 'mug', 'fighter', 'embassy', 'cell', 'state', 'beach',
        'capital', 'post', 'cast', 'soul', 'tower', 'green', 'plot', 'string', 'kangaroo', 'lawyer', 'fire',
        'robot', 'mammoth', 'hole', 'spider', 'bill', 'ivory', 'giant', 'bar', 'ray', 'drill', 'staff',
        'greece', 'press','pitch', 'nurse', 'contract', 'water', 'watch', 'amazon','spell', 'kiwi', 'ghost',
        'cold', 'doctor', 'port', 'bark','foot', 'luck', 'nail', 'ice', 'needle', 'disease', 'comic', 'pool',
        'field', 'star', 'cycle', 'shadow', 'fan', 'compound', 'heart', 'flute','millionaire', 'pyramid', 'africa',
        'robin', 'chest', 'casino','fish', 'oil', 'alps', 'brush', 'march', 'mint','dance', 'snowman', 'torch',
        'round', 'wake', 'satellite','calf', 'head', 'ground', 'club', 'ruler', 'tie','parachute', 'board',
        'paste', 'lock', 'knight', 'pit', 'fork', 'whale', 'scale', 'knife', 'plate','scorpion', 'bottle',
        'boom', 'bolt', 'fall', 'draft', 'hotel', 'game', 'mount', 'train', 'air', 'root', 'charge',
        'space', 'cat', 'olive', 'mouse', 'ham', 'washer', 'pound', 'fly', 'server','shop', 'engine',
        'box', 'shoe', 'tap', 'cross', 'rose', 'belt', 'thumb', 'gold', 'point', 'opera', 'pirate',
        'tag', 'olympus', 'cotton', 'glove', 'sink', 'carrot', 'jack', 'suit', 'glass', 'spot', 'straw', 'well',
        'pan', 'octopus', 'smuggler', 'grass', 'dwarf', 'hood', 'duck', 'jet', 'mercury',
    ]
def _get_agents(
    agent_learn: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = _get_env()
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gym.spaces.Dict)
        else env.observation_space
    )
    if agent_learn is None:
        # model
        net = Net(
            state_shape=observation_space["observation"].shape
            or observation_space["observation"].n,
            action_shape=env.action_space.shape or env.action_space.n,
            hidden_sizes=[128, 128, 128, 128],
            device="cuda" if torch.cuda.is_available() else "cpu",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=1e-4)
        agent_learn = DQNPolicy(
            model=net,
            optim=optim,
            discount_factor=0.9,
            estimation_step=3,
            target_update_freq=320,
        )

    if agent_opponent is None:
        agent_opponent = RandomPolicy()

    agents = [agent_opponent, agent_learn]
    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents


def _get_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    return PettingZooEnv(tictactoe_v3.env())


if __name__ == "__main__":
    default_single_word_label_scores = (1, 1.1, 1.1, 1.2)
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('embeddings', nargs='+',
                            help='an embedding method to use when playing codenames')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='print out verbose information'),
    parser.add_argument('--visualize', dest='visualize', action='store_true',
                        help='visualize the choice of clues with graphs')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='Write score breakdown to a file. You can specify what file is used with --debug-file, or one will be created for you')
    parser.add_argument('--no-heuristics', dest='no_heuristics', action='store_true',
                        help='Remove heuristics such as IDF and dict2vec')
    parser.add_argument('--debug-file', dest='debug_file', default=None,
                        help='Write score breakdown to debug file')
    parser.add_argument('--num-trials', type=int, dest='num_trials', default=1,
                        help='number of trials of the game to run')
    parser.add_argument('--split-multi-word', dest='split_multi_word', default=True)
    parser.add_argument('--disable-verb-split', dest='disable_verb_split', default=True)
    parser.add_argument('--kim-scoring-function', dest='use_kim_scoring_function', action='store_true',
                        help='use the kim 2019 et. al. scoring function'),
    parser.add_argument('--length-exp-scaling', type=int, dest='length_exp_scaling', default=None,
                        help='Rescale lengths using exponent')
    parser.add_argument('--single-word-label-scores', type=float, nargs=4, dest='single_word_label_scores',
                        default=default_single_word_label_scores,
                        help='main_single, main_multi, other_single, other_multi scores')
    parser.add_argument('--babelnet-api-key', type=str, dest='babelnet_api_key', default=None)
    parser.add_argument('--words-per-clue', type=int, default=2)
    parser.add_argument('--max_steps', type=int, default=25)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    # ======== Step 0: Arguments to environment setup =========
    args = parser.parse_args()
    tokenizer = {word: i for word, i in enumerate(words)}
    # ======== Step 1: Environment setup =========
    train_envs = DummyVectorEnv([PettingZooEnv(CodemasterEnv(words, tokenizer, args)) for _ in range(10)])
    test_envs = DummyVectorEnv([PettingZooEnv(GuesserEnv(words, tokenizer, args)) for _ in range(10)])

    # seed
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)

    # ======== Step 2: Agent setup =========
    policy, optim, agents = _get_agents()

    # ======== Step 3: Collector setup =========
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(20_000, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=64 * 10)  # batch size * training_num

    # ======== Step 4: Callback functions setup =========
    def save_best_fn(policy):
        model_save_path = os.path.join("log", "rps", "dqn", "policy.pth")
        os.makedirs(os.path.join("log", "rps", "dqn"), exist_ok=True)
        torch.save(policy.policies[agents[1]].state_dict(), model_save_path)

    def stop_fn(mean_rewards):
        return mean_rewards >= 1000

    def train_fn(epoch, env_step):
        policy.policies[agents[1]].set_eps(0.1)

    def test_fn(epoch, env_step):
        policy.policies[agents[1]].set_eps(0.05)

    def reward_metric(rews):
        return rews[:, 1]

    # ======== Step 5: Run the trainer =========
    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=50,
        step_per_epoch=1000,
        step_per_collect=50,
        episode_per_test=10,
        batch_size=64,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=0.1,
        test_in_train=False,
        reward_metric=reward_metric,
    )

    # return result, policy.policies[agents[1]]
    print(f"\n==========Result==========\n{result}")
    print("\n(the trained policy can be accessed via policy.policies[agents[1]])")