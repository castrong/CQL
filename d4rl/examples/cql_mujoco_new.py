import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector, CustomMDPPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.cql import CQLTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

import lqr_env


import argparse, os
from datetime import datetime
import numpy as np

import h5py
import d4rl, gym

def load_hdf5(dataset, replay_buffer):
    replay_buffer._observations = dataset['observations']
    replay_buffer._next_obs = dataset['next_observations']
    replay_buffer._actions = dataset['actions']
    replay_buffer._rewards = np.expand_dims(np.squeeze(dataset['rewards']), 1)
    replay_buffer._terminals = np.expand_dims(np.squeeze(dataset['terminals']), 1)  
    replay_buffer._size = dataset['terminals'].shape[0]
    print ('Number of terminals on: ', replay_buffer._terminals.sum())
    replay_buffer._top = replay_buffer._size

def experiment(variant):
    eval_env = gym.make(variant['env_name'])
    expl_env = eval_env
    
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    eval_env.seed(variant['seed']) # TODO: how should deal with seeding?


    # def seed_all(epoch):
    #     seed_shift = epoch * 9999
    #     mod_value = 999999
    #     env_seed = (seed + seed_shift) % mod_value
    #     eval_env_seed = (seed + 10000 + seed_shift) % mod_value
    #     torch.manual_seed(env_seed)
    #     np.random.seed(env_seed)
    #     #env.seed(env_seed) # Don't need to seed here bc it's not getting used in each epoch?
    #     #env.action_space.np_random.seed(env_seed)
    #     eval_env.seed(test_env_seed)
    #     eval_env.action_space.np_random.seed(test_env_seed)
       
    # seed_all(epoch=0)

    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M, M], 
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = CustomMDPPathCollector(
        eval_env,
    )
    buffer_filename = None
    if variant['buffer_filename'] is not None:
        buffer_filename = variant['buffer_filename']
    
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )


    # TODO: test this
    if variant['dataset_path'] is not None:
        with h5py.File(variant['dataset_path'], 'r') as f:
            dataset = {k: f[k][:] for k in f.keys()}

        # the possible keys should be 
        # observations
        # actions
        # rewards 
        # next observations 
        # terminals 
        # if there aren't rewards, next obsv, or terminals, generate those from the env
        fill_in = 'rewards' not in dataset or 'next_observations' not in dataset or 'terminals' not in dataset
        # fill_in = True
        if fill_in:
            print("FILLING IN, UNTESTED!!")
            n_data = dataset['observations'].shape[0]
            
            rewards = np.zeros(n_data, dtype=np.float32)
            next_obs = np.zeros_like(dataset['observations'])
            terminals = np.zeros(n_data, dtype=np.float32)

            for i in range(n_data):
                print("filling in data: ", i)
                eval_env.reset()
                eval_env.set_state(dataset['observations'][i])
                next_obs[i], rewards[i], terminals[i], _ = eval_env.step(dataset['actions'][i])

                print("obs: ", dataset['observations'][i])
                print("action: ", dataset['actions'][i])
                print("next obs saved: ", dataset['next_observations'][i])
                print("new next obs: ", next_obs[i])
                
                print("rewards saved: ", dataset['rewards'][i])
                print("new rewards: ", rewards[i])

                print("terminals saved: ", dataset['terminals'][i])
                print("new terminals: ", terminals[i])


            dataset['next_observations'] = next_obs 
            dataset['rewards'] = rewards
            dataset['terminals'] = terminals

        load_hdf5(dataset, replay_buffer)
    elif variant['load_buffer'] and buffer_filename is not None:
        replay_buffer.load_buffer(buffer_filename)
    elif 'random-expert' in variant['env_name']:
        load_hdf5(d4rl.basic_dataset(eval_env), replay_buffer)
    else:
        load_hdf5(d4rl.qlearning_dataset(eval_env), replay_buffer)
       
    trainer = CQLTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        eval_both=True,
        batch_rl=variant['load_buffer'],
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()

def enable_gpus(gpu_str):
    if (gpu_str is not ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="CQL",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(2E6),
        buffer_filename=None,
        load_buffer=None,
        dataset_path=None,
        env_name='Hopper-v2',
        sparse_reward=False,
        algorithm_kwargs=dict(
            num_epochs=500,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=1000,  
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            policy_lr=1E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,

            # Target nets/ policy vs Q-function update
            policy_eval_start=40000,
            num_qs=2,

            # target entropy
            target_entropy=None,

            # min Q
            temp=1.0,
            min_q_version=3,
            min_q_weight=1.0,

            # lagrange
            with_lagrange=True,   # Defaults to true
            lagrange_thresh=10.0,
            
            # extra params
            num_random=10,
            max_q_backup=False,
            deterministic_backup=False,
        ),
    )
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='hopper-medium-v0')
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--max_q_backup", type=str, default="False")          # if we want to try max_{a'} backups, set this to true
    parser.add_argument("--deterministic_backup", type=str, default="True")   # defaults to true, it does not backup entropy in the Q-function, as per Equation 3
    parser.add_argument("--policy_eval_start", default=40000, type=int)       # Defaulted to 20000 (40000 or 10000 work similarly)
    parser.add_argument('--min_q_weight', default=1.0, type=float)            # the value of alpha, set to 5.0 or 10.0 if not using lagrange
    parser.add_argument('--policy_lr', default=1e-4, type=float)              # Policy learning rate
    parser.add_argument('--min_q_version', default=3, type=int)               # min_q_version = 3 (CQL(H)), version = 2 (CQL(rho)) 
    parser.add_argument('--lagrange_thresh', default=5.0, type=float)         # the value of tau, corresponds to the CQL(lagrange) version
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--num_epochs', default=500, type=int)
    parser.add_argument('--exp_name', default='exp', type=str)
    parser.add_argument('--dataset_path', default=None, type=str)  # path to custom HDF5 dataset
    parser.add_argument('--use_automatic_entropy_tuning', default="True", type=str) 
    parser.add_argument('--target_entropy', default=None, type=float) 

    args = parser.parse_args()
    enable_gpus(args.gpu)
    variant['trainer_kwargs']['max_q_backup'] = (True if args.max_q_backup == 'True' else False)
    variant['trainer_kwargs']['deterministic_backup'] = (True if args.deterministic_backup == 'True' else False)
    variant['trainer_kwargs']['min_q_weight'] = args.min_q_weight
    variant['trainer_kwargs']['policy_lr'] = args.policy_lr
    variant['trainer_kwargs']['min_q_version'] = args.min_q_version
    variant['trainer_kwargs']['policy_eval_start'] = args.policy_eval_start
    variant['trainer_kwargs']['lagrange_thresh'] = args.lagrange_thresh
    if args.lagrange_thresh < 0.0:
        variant['trainer_kwargs']['with_lagrange'] = False

    variant['trainer_kwargs']['use_automatic_entropy_tuning'] = (args.use_automatic_entropy_tuning == "True")
    variant['trainer_kwargs']['target_entropy'] = args.target_entropy
    print("target entropy in main script: ", args.target_entropy)

    print("use automatic entropy tuning: ", args.use_automatic_entropy_tuning)

    variant['buffer_filename'] = None

    variant['algorithm_kwargs']['num_epochs'] = args.num_epochs
    variant['load_buffer'] = True
    variant['env_name'] = args.env
    variant['seed'] = args.seed
    variant['dataset_path'] = args.dataset_path

    timestamp = datetime.now().strftime('%m_%d_%Y:%H_%M_%S')
    exp_name = f"{timestamp}_{args.exp_name}"
    setup_logger(exp_name, variant=variant, base_log_dir=os.path.expanduser('./CQL_logs'), snapshot_mode="all")
    ptu.set_gpu_mode(True)
    experiment(variant)

# TODO: dataset generation for the LQR environment so that we can load it in. 
# make a dataset generation script that takes from our optimal controller stuff
# in the DDH, and that has as a parameter the amount to scale u by.

# from the dataset load the state, action pairs, and then use the environment to get the rest?
# oh that won't work with the done as well 
