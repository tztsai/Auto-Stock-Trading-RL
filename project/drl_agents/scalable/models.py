import gym
import torch
import numpy as np
import tensorflow as tf
from absl import flags

# from .seed_rl.common.utils import EnvOutput
# from .seed_rl.common import actor
# from .seed_rl.agents.vtrace import learner as VTraceLearner
# from .seed_rl.dmlab.networks import ImpalaDeep as VTraceModel
# from .seed_rl.agents.sac import learner as SACLearner
# from .seed_rl.agents.sac.networks import ActorCriticLSTM as SACModel
# from .impala.environments import FlowEnvironment
from .torchbeast import monobeast #, test, act


Net, create_env, train = monobeast.Net, monobeast.create_env, monobeast.train

MODELS = dict(
    impala=Net
    # vtrace=VTraceModel,
    # r2d2=...,
    # sac=SACModel,
    # ppo=...
)

LEARNERS = dict(
    # vtrace=VTraceLearner,
    # r2d2=...,
    # sac=SACLearner,
    # ppo=...
)

FLAGS = flags.FLAGS
FLAGS.use_lstm = True

# Optimizer settings.
# flags.DEFINE_float('learning_rate', 0.00048, 'Learning rate.')
# flags.DEFINE_float('adam_epsilon', 3.125e-7, 'Adam epsilon.')


class DRLAgent:
    """Implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class
            user-defined class
        price_array: numpy array
            OHLC data
        tech_array: numpy array
            techical data
        turbulence_array: numpy array
            turbulence/risk data
    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env: gym.Env):
        self.env = env
        # self.price_array = price_array
        # self.tech_array = tech_array
        # self.turbulence_array = turbulence_array
        
    def get_model(self, *_, **__):
        return
    
    # def get_model(self, model_name, model_kwargs=None):
        # if model_name not in MODELS:
        #     raise NotImplementedError("NotImplementedError")

        # self.learner = LEARNERS[model_name]
        # self.model_cls = MODELS[model_name]
        # self.model_kwargs = model_kwargs or {}

        # return self.create_agent()

    # def create_agent(self, action_space, obs_space, parametric_action_distribution):
    #     if self.model_cls is VTraceModel:
    #         self.model_kwargs['num_actions'] = action_space.n
    #     return self.model_cls(**self.model_kwargs)

    # def create_env(self, *args):
    #     return self.env

    # def create_optimizer(self, final_iteration):
    #     # learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    #     #     FLAGS.learning_rate, final_iteration, 0)
    #     # optimizer = tf.keras.optimizers.Adam(learning_rate_fn, beta_1=0,
    #     #                                      epsilon=FLAGS.adam_epsilon)
    #     learning_rate_fn = lambda iteration: FLAGS.learning_rate
    #     optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)
    #     return optimizer, learning_rate_fn

    def train_model(self, *_, **__):
        create_env.instance = self.env
        train(FLAGS)
        # self.learner.learner_loop(
        #     self.create_env,
        #     self.create_agent,
        #     self.create_optimizer
        # )
    
    @staticmethod
    def DRL_prediction(model_path, environment):
        # load agent
        try:
            model = tf.saved_model.load(model_path)
        except BaseException:
            raise ValueError("Fail to load agent!")

        # test on the testing env
        action = None
        state = environment.reset()
        episode_returns = list()  # the cumulative_return / initial_account
        episode_total_assets = list()
        episode_total_assets.append(environment.initial_total_asset)
        done = False
        while not done:
            action = model(actions, env_outputs, state, is_training=False)  # FIXME
            actions.append(action)
            state, reward, done, _ = environment.step(action)
            env_output = EnvOutput(reward, done, state, None, None)
            env_outputs.append(env_output)

            total_asset = (
                environment.amount
                + (environment.price_ary[environment.day] * environment.stocks).sum()
            )
            episode_total_assets.append(total_asset)
            episode_return = total_asset / environment.initial_total_asset
            episode_returns.append(episode_return)

        print("episode_return", episode_return)
        print("Test Finished!")
        return episode_total_assets
