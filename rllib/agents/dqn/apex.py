from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from ray.rllib.agents.dqn.dqn import DQNTrainer, DEFAULT_CONFIG as DQN_CONFIG
from ray.rllib.optimizers import AsyncReplayOptimizer
from ray.rllib.utils import merge_dicts
from ray.tune.result import DEFAULT_RESULTS_DIR

# yapf: disable
# __sphinx_doc_begin__
APEX_DEFAULT_CONFIG = merge_dicts(
    DQN_CONFIG,  # see also the options in dqn.py, which are also supported
    {
        "optimizer": merge_dicts(
            DQN_CONFIG["optimizer"], {
                "max_weight_sync_delay": 400,
                "num_replay_buffer_shards": 4,
                "debug": False
            }),
        "n_step": 3,
        "num_gpus": 1,
        "num_workers": 32,
        "buffer_size": 2000000,
        "learning_starts": 50000,
        "train_batch_size": 512,
        "sample_batch_size": 50,
        "target_network_update_freq": 500000,
        "timesteps_per_iteration": 25000,
        "per_worker_exploration": True,
        "worker_side_prioritization": True,
        "min_iter_time_s": 30,
    },
)
# __sphinx_doc_end__
# yapf: enable


def defer_make_workers(trainer, env_creator, policy, config):
    # Hack to workaround https://github.com/ray-project/ray/issues/2541
    # The workers will be creatd later, after the optimizer is created
    return trainer._make_workers(env_creator, policy, config, 0)


def make_async_optimizer(workers, config):
    assert len(workers.remote_workers()) == 0
    extra_config = config["optimizer"].copy()
    for key in [
            "prioritized_replay", "prioritized_replay_alpha",
            "prioritized_replay_beta", "prioritized_replay_eps"
    ]:
        if key in config:
            extra_config[key] = config[key]
    opt = AsyncReplayOptimizer(
        workers,
        learning_starts=config["learning_starts"],
        buffer_size=config["buffer_size"],
        train_batch_size=config["train_batch_size"],
        sample_batch_size=config["sample_batch_size"],
        **extra_config)
    workers.add_workers(config["num_workers"])
    opt._set_workers(workers.remote_workers())
    return opt


def update_target_based_on_num_steps_trained(trainer, fetches):
    # Ape-X updates based on num steps trained, not sampled
    if (trainer.optimizer.num_steps_trained -
            trainer.state["last_target_update_ts"] >
            trainer.config["target_network_update_freq"]):
        trainer.workers.local_worker().foreach_trainable_policy(
            lambda p, _: p.update_target())
        trainer.state["last_target_update_ts"] = (
            trainer.optimizer.num_steps_trained)
        trainer.state["num_target_updates"] += 1
        #import pdb
        #pdb.set_trace()
        #print("get trainer.workers.local_worker() stuff")
        #  save the weights in here!
    if (trainer.optimizer.num_steps_trained -
           trainer.state["last_network_save_ts"] >
            trainer.config["network_save_freq"]) and (trainer.optimizer.num_steps_trained > trainer.config["saving_starts"]):
        #net_save = trainer.workers.local_worker().policy_map['default_policy'].export_checkpoint('/home/william/ray_results/')

        #weights_to_save = trainer.workers.local_worker().policy_map['default_policy'].get_clean_weights()  # this will be used when we are building our own net to run the RLstrategy
        # np.save('/home/ubuntu/ray_results/net_weights' + str(trainer.state["last_network_save_ts"]) + '.npy', weights_to_save)

        flat_weights_to_save = trainer.workers.local_worker().policy_map['default_policy'].get_weights()
        np.save(DEFAULT_RESULTS_DIR + '/flat_net_weights.npy', flat_weights_to_save)  # need to see if I can find a way to get the right folder here.

        trainer.state["last_network_save_ts"] = (trainer.optimizer.num_steps_trained)
        # This will use properties in the dict that are set in the dqn class - so check there for changes regarding trainer.state etc.




APEX_TRAINER_PROPERTIES = {
    "make_workers": defer_make_workers,
    "make_policy_optimizer": make_async_optimizer,
    "after_optimizer_step": update_target_based_on_num_steps_trained,
}

ApexTrainer = DQNTrainer.with_updates(
    name="APEX", default_config=APEX_DEFAULT_CONFIG, **APEX_TRAINER_PROPERTIES)
