import torch
import numpy as np
import tensorflow as tf

from absl import app
from absl import flags

from .agents.vtrace import learner as VTraceLearner
from .agents.vtrace.networks import MLPandLSTM as VTraceModel


MODELS = dict(
    vtrace=VTraceModel,
    # r2d2=...,
    # sac=...,
    # ppo=...
)

LEARNERS = dict(
    vtrace=VTraceLearner,
    # r2d2=...,
    # sac=...,
    # ppo=...
)

MODEL_KWARGS = {}

FLAGS = flags.FLAGS

# Optimizer settings.
flags.DEFINE_float('learning_rate', 0.00048, 'Learning rate.')
flags.DEFINE_float('adam_epsilon', 3.125e-7, 'Adam epsilon.')


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

    def __init__(self, env, price_array, tech_array, turbulence_array):
        self.env = env
        self.price_array = price_array
        self.tech_array = tech_array
        self.turbulence_array = turbulence_array

    def get_model(self, model_name, model_kwargs=None):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        if model_kwargs is None:
           model_kwargs = MODEL_KWARGS[model_name]

        model_cls = MODELS[model_name]
        model = model_cls(**model_kwargs)
        
        self.learner = LEARNERS[model_name]

        return model
    
    def train_model(self, model, total_episodes=100):
        def create_env(*args):
            return self.env.copy()
        
        def create_agent(*args):
            return model
        
        def create_opt(final_iteration):
            learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
                FLAGS.learning_rate, final_iteration, 0)
            optimizer = tf.keras.optimizers.Adam(learning_rate_fn, beta_1=0,
                                                 epsilon=FLAGS.adam_epsilon)
            return optimizer, learning_rate_fn

        self.learner.learner_loop(
            create_env,
            create_agent,
            create_opt
        )

        return self.learner

    @staticmethod
    def DRL_prediction(model_name, cwd, net_dimension, environment):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        model = MODELS[model_name]()
        environment.env_num = 1

        # load agent
        try:
            state_dim = environment.state_dim
            action_dim = environment.action_dim

            agent = model
            net_dim = net_dimension

            agent.init(net_dim, state_dim, action_dim)
            agent.save_or_load_agent(cwd=cwd, if_save=False)
            act = agent.act
            device = agent.device

        except BaseException:
            raise ValueError("Fail to load agent!")

        # test on the testing env
        _torch = torch
        state = environment.reset()
        episode_returns = list()  # the cumulative_return / initial_account
        episode_total_assets = list()
        episode_total_assets.append(environment.initial_total_asset)
        with _torch.no_grad():
            for i in range(environment.max_step):
                s_tensor = _torch.as_tensor((state,), device=device)
                a_tensor = act(s_tensor)  # action_tanh = act.forward()
                action = (
                    a_tensor.detach().cpu().numpy()[0]
                )  # not need detach(), because with torch.no_grad() outside
                state, reward, done, _ = environment.step(action)

                total_asset = (
                    environment.amount
                    + (
                        environment.price_ary[environment.day] * environment.stocks
                    ).sum()
                )
                episode_total_assets.append(total_asset)
                episode_return = total_asset / environment.initial_total_asset
                episode_returns.append(episode_return)
                if done:
                    break
                
        print("Test Finished!")
        # return episode total_assets on testing data
        print("episode_return", episode_return)
        return episode_total_assets
