# References:
# https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
# https://github.com/yfeng997/MadMario/
# https://stackoverflow.com/questions/52726475/display-openai-gym-in-jupyter-notebook-only
# https://github.com/uvipen/Super-mario-bros-PPO-pytorch/blob/master/src/env.py
# https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python


#%%bash
#pip install gym-super-mario-bros==7.4.0
#pip install tensordict==0.3.0
#pip install torchrl==0.3.0
#pip install torchvision
#pip install matplotlib
#pip install imageio

from mad_mario import *
from IPython.display import clear_output
import imageio


class MyMario(Mario):
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.exploration_rate_decay = 1 - ((1 - 0.99999975 ) * 10)
        self.exploration_rate_min = 0
        self.default_chkpoint = "chkpoint.chkpt"
        self.burnin = 200
        
        self.learn_from_death_count = 10

def load_mario(env, save_dir):
    mario = MyMario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

    # Load from default chkpt.
    if Path(mario.default_chkpoint).is_file():
        dic = torch.load(mario.default_chkpoint)
        mario.net.load_state_dict(dic["model"])
        mario.exploration_rate = dic["exploration_rate"]

    return mario

def train(is_eval = False, episodes = 1000):
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='rgb_array', apply_api_compatibility=True)
    # Limit the action-space to
    #   0. walk right
    #   1. jump right
    env = JoypadSpace(env, [["right"], ["right", "A"]])

    env.reset()
    next_state, reward, done, trunc, info = env.step(action=0)
    print(f"{next_state.shape},\n {reward},\n {done},\n {info}")


    # Apply Wrappers to environment
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    if gym.__version__ < '0.26':
        env = FrameStack(env, num_stack=4, new_step_api=True)
    else:
        env = FrameStack(env, num_stack=4)

    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    print()

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    mario = load_mario(env, save_dir)

    logger = MetricLogger(save_dir)

    images = []

    for e in range(episodes):

        state = env.reset()
        images = []

        # Play the game!
        while True:

            if is_eval:
                clear_output(wait=True)
                img = env.render()
                plt.imshow( img )
                plt.show()
                images.append(img.copy())

            # Run agent on the state
            action = mario.act(state)

            # Agent performs action
            next_state, reward, done, trunc, info = env.step(action)

            # Remember newer actions to learn more on new exploration
            if len(logger.moving_avg_ep_lengths) > 0:
                pos = (float)(logger.curr_ep_length) / (float)(logger.moving_avg_ep_lengths[-1])
                if pos > 0.8: pos = 1.0
                if pos < 0.2: pos = 0.2

                if np.random.rand() < pos:
                    mario.cache(state, next_state, action, reward, done)
            else:
                mario.cache(state, next_state, action, reward, done)

            if done == True:
                for idx in range(mario.learn_from_death_count):
                    mario.cache(state, next_state, action, reward, done)

            # Learn
            q, loss = mario.learn()
            
            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            state = next_state

            # Check if end of game
            if done or info["flag_get"]:
                break

        logger.log_episode()

        if (e % 20 == 0) or (e == episodes - 1):
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)    

        if (e % 200 == 0) or (e == episodes - 1):
            # Save to timestamped dir
            save_dir = Path("checkpoints_save") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            save_dir.mkdir(parents=True)
            mario.save_dir = save_dir
            mario.save()

            # Save to default dir
            torch.save(
                dict(model=mario.net.state_dict(), exploration_rate=mario.exploration_rate),
                mario.default_chkpoint,
            )
            print(f"MarioNet saved to {mario.default_chkpoint}")
    
    if is_eval:
        imageio.mimsave('movie.gif', images)


# For training
#train(is_eval = False, episodes = 1000)

# For evaluation
#train(is_eval = True, episodes = 1)