import torch
import gym


class Environment(gym.Wrapper):
    default_env = None
    
    def __init__(self, gym_env):
        super().__init__(gym_env)
        self.gym_env = gym_env
        # self.episode_return = None
        self.episode_step = None
        
    @staticmethod
    def _format_frame(frame):
        frame = torch.tensor(frame)
        return frame.view((1, 1) + frame.shape)  # (...) -> (T,B,...).
    
    @property
    def num_actions(self):
        return self.action_space.shape[-1]

    def initial(self):
        initial_reward = torch.zeros(1, 1)
        # This supports only single-tensor actions ATM.
        initial_last_action = torch.zeros(1, 1, self.num_actions, dtype=torch.int64)
        # self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        initial_done = torch.ones(1, 1, dtype=torch.uint8)
        initial_frame = self._format_frame(self.gym_env.reset())
        return dict(
            frame=initial_frame,
            reward=initial_reward,
            done=initial_done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            last_action=initial_last_action,
        )

    def step(self, action):
        self.action_np = action.detach().numpy().squeeze()
        frame, reward, done, _ = self.gym_env.step(self.action_np)
        self.episode_step += 1
        # self.episode_return += reward
        episode_step = self.episode_step
        episode_return = self.gym_env.total_asset / self.gym_env.initial_total_asset
        if done:
            frame = self.gym_env.reset()
            # self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)

        frame = self._format_frame(frame)
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)

        return dict(
            frame=frame,
            reward=reward,
            done=done,
            episode_return=episode_return,
            episode_step=episode_step,
            last_action=action,
        )

    def close(self):
        self.gym_env.close()

