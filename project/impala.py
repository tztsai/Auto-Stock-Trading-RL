# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import argparse
import logging
import os
import pprint
import threading
import time
import timeit
import traceback
import typing

os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.

import torch
from torch import multiprocessing as mp
from torch import nn

from absl import flags

from utils import file_writer, prof, vtrace
from utils.environment import Environment


# yapf: disable
# parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")

flags.DEFINE_enum("mode", default="train",
                    enum_values=["train", "test", "test_render"],
                    help="Training or test mode.")
flags.DEFINE_integer("xpid", default=None,
                    help="Experiment id (default: None).")
flags.DEFINE_enum("device", default="cuda", enum_values=["cpu", "cuda"], help="Device to use.")

# Training settings.
flags.DEFINE_bool("disable_checkpoint", default=False,
                    help="Disable saving checkpoint.")
flags.DEFINE_string("savedir", default="impala_logs",
                    help="Root dir where experiment data will be saved.")
flags.DEFINE_integer("num_actors", default=4, 
                    help="Number of actors (default: 4).")
flags.DEFINE_integer("total_steps", default=int(1e6),
                    help="Total environment steps to train for.")
flags.DEFINE_integer("batch_size", default=8, 
                    help="Learner batch size.")
flags.DEFINE_integer("unroll_length", default=80, 
                    help="The unroll length (time dimension).")
flags.DEFINE_integer("num_buffers", default=None,
                    help="Number of shared-memory buffers.")
flags.DEFINE_integer("num_learner_threads", default=2,
                    help="Number learner threads.")
flags.DEFINE_bool("disable_cuda", default=False,
                    help="Disable CUDA.")

# Loss settings.
flags.DEFINE_float("entropy_cost", default=0.0006,
                    help="Entropy cost/multiplier.")
flags.DEFINE_float("baseline_cost", default=0.5,
                    help="Baseline cost/multiplier.")
flags.DEFINE_float("discounting", default=0.99,
                    help="Discounting factor.")
flags.DEFINE_enum("reward_clipping", default="none",
                    enum_values=["abs_one", "none"],
                    help="Reward clipping.")

# Optimizer settings.
flags.DEFINE_float("learning_rate", default=0.00048,
                    help="Learning rate.")
flags.DEFINE_float("alpha", default=0.99,
                    help="RMSProp smoothing constant.")
flags.DEFINE_float("momentum", default=0,
                    help="RMSProp momentum.")
flags.DEFINE_float("epsilon", default=0.01,
                    help="RMSProp epsilon.")
flags.DEFINE_float("grad_norm_clipping", default=40.0,
                    help="Global gradient norm clip.")
# yapf: enable


logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    # policy = F.softmax(logits, dim=-1)
    # log_policy = F.log_softmax(logits, dim=-1)
    # return torch.sum(policy * log_policy)
    dist = vtrace.logits_to_distribution(logits)
    return torch.sum(dist.entropy())


def compute_policy_gradient_loss(logits, actions, advantages):
    dist = vtrace.logits_to_distribution(logits)
    cross_entropy = -dist.log_prob(actions)
    # cross_entropy = F.nll_loss(
    #     F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
    #     target=torch.flatten(actions, 0, 1),
    #     reduction="none",
    # )
    # cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())


def act(
    flags,
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    model: torch.nn.Module,
    buffers: Buffers,
    initial_agent_state_buffers,
):
    try:
        logging.info("Actor %i started.", actor_index)
        timings = prof.Timings()  # Keep track of how fast things are.

        env = create_env(flags)
        seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
        env.gym_env.seed(seed)
        env_output = env.initial()
        agent_state = model.initial_state(batch_size=1)
        agent_output, unused_state = model(env_output, agent_state)
        
        while True:
            index = free_queue.get()
            if index is None: break

            # Write old rollout end.
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]
            for i, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][i][...] = tensor

            # Do new rollout.
            for t in range(flags.unroll_length):
                timings.reset()

                with torch.no_grad():
                    agent_output, agent_state = model(env_output, agent_state)

                timings.time("model")

                env_output = env.step(agent_output["action"])

                timings.time("step")

                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]

                timings.time("write")
            full_queue.put(index)

        if actor_index == 0:
            logging.info("Actor %i: %s", actor_index, timings.summary())

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e


def get_batch(
    flags,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    buffers: Buffers,
    initial_agent_state_buffers,
    timings,
    lock=threading.Lock(),
):
    with lock:
        timings.time("lock")
        indices = [full_queue.get() for _ in range(flags.batch_size)]
        timings.time("dequeue")
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers
    }
    initial_agent_state = (
        torch.cat(ts, dim=1)
        for ts in zip(*[initial_agent_state_buffers[m] for m in indices])
    )
    timings.time("batch")
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")
    batch = {k: t.to(device=flags.device, non_blocking=True) for k, t in batch.items()}
    initial_agent_state = tuple(
        t.to(device=flags.device, non_blocking=True) for t in initial_agent_state
    )
    timings.time("device")
    return batch, initial_agent_state


def learn(
    flags,
    actor_model,
    model,
    batch,
    initial_agent_state,
    optimizer,
    scheduler,
    lock=threading.Lock(),  # noqa: B008
):
    """Performs a learning (optimization) step."""
    with lock:
        learner_outputs, unused_state = model(batch, initial_agent_state)

        # Take final value function slice for bootstrapping.
        bootstrap_value = learner_outputs["baseline"][-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}

        rewards = batch["reward"]
        if flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(rewards, -1, 1)
        elif flags.reward_clipping == "none":
            clipped_rewards = rewards

        discounts = (~batch["done"]).float() * flags.discounting

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch["policy_logits"],
            target_policy_logits=learner_outputs["policy_logits"],
            actions=batch["action"],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs["baseline"],
            bootstrap_value=bootstrap_value,
        )

        pg_loss = compute_policy_gradient_loss(
            learner_outputs["policy_logits"],
            batch["action"],
            vtrace_returns.pg_advantages,
        )
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs["baseline"]
        )
        entropy_loss = flags.entropy_cost * compute_entropy_loss(
            learner_outputs["policy_logits"]
        )

        total_loss = pg_loss + baseline_loss + entropy_loss

        episode_returns = batch["episode_return"][batch["done"]]

        stats = {
            "episode_returns": tuple(episode_returns.cpu().numpy()),
            "mean_episode_return": torch.mean(episode_returns).item(),
            "total_loss": total_loss.item(),
            "pg_loss": pg_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)
        optimizer.step()
        scheduler.step()

        actor_model.load_state_dict(model.state_dict())
        return stats


def create_buffers(flags, obs_shape, num_actions) -> Buffers:
    T = flags.unroll_length
    specs = dict(
        frame=dict(size=(T + 1, *obs_shape), dtype=torch.uint8),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        policy_logits=dict(size=(T + 1, 2 * num_actions), dtype=torch.float32),
        baseline=dict(size=(T + 1,), dtype=torch.float32),
        last_action=dict(size=(T + 1, num_actions), dtype=torch.int64),
        action=dict(size=(T + 1, num_actions), dtype=torch.int64),
    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers


def train(flags):  # pylint: disable=too-many-branches, too-many-statements
    if flags.xpid is None:
        flags.xpid = "torchbeast-%s" % time.strftime("%Y%m%d-%H%M%S")
    plogger = file_writer.FileWriter(
        xpid=flags.xpid,
        xp_args=dict([k, getattr(flags, k)] for k in dir(flags)),
        rootdir=flags.savedir
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
    )

    if flags.num_buffers is None:  # Set sensible default for num_buffers.
        flags.num_buffers = max(2 * flags.num_actors, flags.batch_size)
    if flags.num_actors >= flags.num_buffers:
        raise ValueError("num_buffers should be larger than num_actors")
    if flags.num_buffers < flags.batch_size:
        raise ValueError("num_buffers should be larger than batch_size")

    T = flags.unroll_length
    B = flags.batch_size

    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        flags.device = torch.device("cuda")
    else:
        logging.info("Not using CUDA.")
        flags.device = torch.device("cpu")

    env = create_env(flags)
    model = Net(env)
    buffers = create_buffers(flags, env.observation_space.shape, model.num_actions)

    model.share_memory()

    # Add initial RNN state.
    initial_agent_state_buffers = []
    for _ in range(flags.num_buffers):
        state = model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)

    actor_processes = []
    ctx = mp.get_context("fork")
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(
                flags,
                i,
                free_queue,
                full_queue,
                model,
                buffers,
                initial_agent_state_buffers,
            ),
        )
        actor.start()
        actor_processes.append(actor)
        
    learner_model = Net(env).to(device=flags.device)

    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )

    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.total_steps) / flags.total_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger = logging.getLogger("logfile")
    stat_keys = [
        "total_loss",
        "mean_episode_return",
        "pg_loss",
        "baseline_loss",
        "entropy_loss",
    ]
    logger.info("# Step\t%s", "\t".join(stat_keys))

    timer = timeit.default_timer
    start_step = step = 0
    start_time = timer()

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal step, start_step, start_time
        timings = prof.Timings()
        while step < flags.total_steps:
            timings.reset()
            batch, agent_state = get_batch(
                flags,
                free_queue,
                full_queue,
                buffers,
                initial_agent_state_buffers,
                timings,
            )
            stats = learn(
                flags, model, learner_model, batch, agent_state, optimizer, scheduler
            )
            timings.time("learn")
            with lock:
                to_log = dict(step=step)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                step += T * B
                
            if stats.get("episode_returns", None):
                sps = (step - start_step) / (timer() - start_time)
                start_step = step
                start_time = timer()
                
                mean_return = (
                    "Return per episode: %.1f. " % stats["mean_episode_return"]
                )
                total_loss = stats.get("total_loss", float("inf"))
                logging.info(
                    "Steps %i @ %.1f SPS. Loss %f. %sStats:\n%s",
                    step,
                    sps,
                    total_loss,
                    mean_return,
                    pprint.pformat(stats),
                )

        if i == 0:
            logging.info("Batch and learn: %s", timings.summary())

    for m in range(flags.num_buffers):
        free_queue.put(m)

    threads = []
    for i in range(flags.num_learner_threads):
        thread = threading.Thread(
            target=batch_and_learn, name="batch-and-learn-%d" % i, args=(i,)
        )
        thread.start()
        threads.append(thread)

    def checkpoint():
        if flags.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", checkpointpath)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "flags": {k: getattr(flags, k) for k in dir(flags)}
            },
            checkpointpath,
        )

    try:
        last_checkpoint_time = timer()
        while step < flags.total_steps:
            time.sleep(5)
            if timer() - last_checkpoint_time > 10 * 60:  # Save every 10 min.
                checkpoint()
                last_checkpoint_time = timer()
    except KeyboardInterrupt:
        return  # Try joining actors then quit.
    else:
        for thread in threads:
            thread.join()
        logging.info("Learning finished after %d steps.", step)
    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)

    checkpoint()
    plogger.close()


def test(flags):
    if flags.xpid is None:
        checkpointpath = "./latest/model.tar"
    else:
        checkpointpath = os.path.expandvars(
            os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
        )

    env = create_env(flags)
    model = Net(env)
    model.eval()
    checkpoint = torch.load(checkpointpath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    observation = env.initial()
    episode_returns = list()  # the cumulative_return / initial_account
    episode_total_assets = list()
    episode_total_assets.append(env.initial_total_asset)
    episode_actions = list()
    
    while True:
        agent_outputs = model(observation)
        policy_outputs, _ = agent_outputs
        action = policy_outputs["action"]
        observation = env.step(action)
        episode_actions.append(env.action_np)

        episode_return = observation["episode_return"]
        episode_returns.append(episode_return)
        episode_total_assets.append(episode_return * env.initial_total_asset)
        
        if observation['done'].item():
            logging.info(
                "Episode ended after %d steps. Return: %.1f",
                observation["episode_step"].item(),
                observation["episode_return"].item(),
            )
            break
        
    print("Test Finished!")
    return episode_total_assets

class ActorNet(nn.Module):

    def __init__(self, env, net_dim=512):
        super().__init__()
        
        state_dim = env.observation_space.shape[-1]
        action_dim = env.action_space.shape[-1]

        self.net_state = nn.Sequential(nn.Linear(state_dim, net_dim),
                                       nn.ReLU(),
                                       nn.Linear(net_dim, net_dim),
                                       nn.ReLU())
        
        core_out_dim = net_dim + action_dim + 1
        
        self.core = nn.LSTM(core_out_dim, core_out_dim, 2)
        
        self.net_policy = nn.Sequential(
            nn.ReLU(),
            nn.Linear(core_out_dim, net_dim),
            nn.Hardswish(),
            nn.Linear(net_dim, action_dim)
        )
        self.net_action_logstd = nn.Sequential(
            nn.Hardswish(),
            nn.Linear(core_out_dim, action_dim)
        )
        self.baseline = nn.Sequential(
            nn.ReLU(),
            nn.Linear(core_out_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.observation_shape = env.observation_space.shape
        self.num_actions = action_dim
        
        print(self)

    def initial_state(self, batch_size):
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )
        
    def forward(self, inputs, core_state=None):
        x = inputs["frame"]
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1).float()
        x = self.net_state(x)

        core_input = torch.cat([
            x.view(T, B, -1),
            torch.clamp(inputs["reward"], -1, 1).view(T, B, 1).float(),
            inputs["last_action"].float(),
        ], dim=-1)
        
        core_output_list = []
        notdone = (~inputs["done"]).float()
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            # Reset core state to zero whenever an episode ended.
            # Make `done` broadcastable with (num_layers, B, hidden_size) states
            output, core_state = self.core(input.unsqueeze(0), core_state)
            core_state = tuple(nd.view(1, -1, 1) * s for s in core_state)
            core_output_list.append(output)
        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)

        action_mean = self.net_policy(core_output)
        baseline = self.baseline(core_output)

        action_std = self.net_action_logstd(core_output).clamp(-20, 2).exp()
        if self.training:
            action = torch.normal(action_mean, action_std).tanh()
        else:
            action = action_mean.tanh()
        
        policy_logits = torch.cat([
            action_mean.view(T, B, self.num_actions),
            action_std.view(T, B, self.num_actions)
        ], dim=-1)
        baseline = baseline.view(T, B)
        action = action.view(T, B, self.num_actions)

        return (dict(policy_logits=policy_logits,
                     baseline=baseline,
                     action=action),
                core_state)


Net = ActorNet


def create_env(flags):
    return Environment(Environment.default_env)


def set_env(env):
    Environment.default_env = env
