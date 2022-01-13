from harlow_test import test
from model import ActorCritic
import torch
import torch.optim as optim
import argparse
import deepmind_lab
import numpy as np
import six
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt


def _action(*entries):
    return np.array(entries, dtype=np.intc)


ACTIONS = {
    'look_left': _action(-5, 0, 0, 0, 0, 0, 0),
    'look_right': _action(5, 0, 0, 0, 0, 0, 0),
    'no-ops': _action(0, 0, 0, 0, 0, 0, 0)
}

ACTION_LIST = list(six.viewvalues(ACTIONS))


def train(n_episodes, length, width, height, fps, level, record, demo,
          demofiles, video):

    # Defaults parameters:
    #    gamma = 0.9
    #    lr = 0.0007
    #    betas = (0.05, 0.05)
    #    random_seed = 543

    loss_list = []
    BoxPlot_data = []
    save_results_to = "/home/damien/Documents/GitHub/lab/python/meta_rl/"

    gamma = 0.9
    learning_rate = 0.0007
    betas = (0.05, 0.05)
    random_seed = 543

    torch.manual_seed(random_seed)
    """Spins up an environment and runs the random agent."""
    config = {'fps': str(fps), 'width': str(width), 'height': str(height)}
    if record:
        config['record'] = record
    if demo:
        config['demo'] = demo
    if demofiles:
        config['demofiles'] = demofiles
    if video:
        config['video'] = video

    env = deepmind_lab.Lab(level, ['RGB_INTERLEAVED'], config=config)
    num_actions = len(ACTION_LIST)  # 3
    num_hidden_units = 256
    policy = ActorCritic(num_actions, num_hidden_units)
    optimizer = optim.RMSprop(policy.parameters(), lr=learning_rate)
    print(learning_rate, betas)

    running_reward = 0
    list_reward_500_episode = []
    list_reward_100_episode = []
    for i_episode in range(0, n_episodes):
        env.reset()
        # initial step
        episode_reward = 0
        state = np.float32(env.observations()['RGB_INTERLEAVED'])
        hidden_state = torch.zeros(1, 1, num_hidden_units)
        cell_state = torch.zeros(1, 1, num_hidden_units)
        for t in range(length):
            # Convert state into a batched tensor (batch size = 1)
            action, hidden_state, cell_state = policy(state, hidden_state,
                                                      cell_state)
            hidden_state = hidden_state.detach()
            cell_state = cell_state.detach()
            state = env.observations()['RGB_INTERLEAVED']
            #print(action)
            reward = env.step(ACTION_LIST[action])
            done = not env.is_running() or env.num_steps() == length - 1
            policy.rewards.append(reward)
            running_reward += reward
            episode_reward += reward
            if reward > 0:
                print("\033[34mAction Taken: " + ("Left" if action == 0 else (
                    "Right" if action == 1 else "No-Ops")) + "\033[0m")
                print("\033[1;32mTrial reward: " + str(reward) + "\033[0m")
            elif reward < 0:
                print("\033[34mAction Taken: " + ("Left" if action == 0 else (
                    "Right" if action == 1 else "No-Ops")) + "\033[0m")
                print("\033[1;31mTrial reward: " + str(reward) + "\033[0m")
            if done:
                break

        # Updating the policy :
        optimizer.zero_grad()
        loss = policy.calculateLoss(gamma)
        # we keep the loss the episode
        loss_list.append(loss.detach().numpy())
        list_reward_500_episode.append(episode_reward)

        loss.backward()
        optimizer.step()
        policy.clearMemory()
        policy.zero_grad()

        # saving the model
        if i_episode % 500 == 0:

            BoxPlot_data.append(list_reward_500_episode)
            list_reward_500_episode = []

            custom_params = {
                "axes.spines.right": False,
                "axes.spines.top": False
            }
            sns.set_theme(style="ticks", rc=custom_params)
            plt.boxplot(BoxPlot_data)
            plt.xlabel("500 Episodes")
            plt.ylabel("Reward")
            plt.savefig(save_results_to + "BoxPlot_rewards.pdf")
            plt.clf()

            torch.save(
                policy.state_dict(),
                '/home/damien/Documents/GitHub/lab/python/meta_rl/Harlow_episode_{}_parameters_{}_{}_{}.pth'
                .format(i_episode, learning_rate, betas[0], betas[1]))

        if i_episode % 100 == 0:

            list_reward_100_episode.append(running_reward / 100)

            custom_params = {
                "axes.spines.right": False,
                "axes.spines.top": False
            }
            sns.set_theme(style="ticks", rc=custom_params)
            plt.plot(range(len(list_reward_100_episode)),
                     list_reward_100_episode,
                     linewidth=0.5)
            plt.xlabel("100 Episodes")
            plt.ylabel("Reward")
            plt.savefig(save_results_to + "Graphe_rewards.pdf")
            plt.clf()

            print("\033[34mEpisode {}\tlength: {}\treward: {}\033[0m".format(
                i_episode, t + 2, running_reward / 100))
            running_reward = 0

        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(style="ticks", rc=custom_params)
        plt.plot(np.arange(len(loss_list)), loss_list, linewidth=0.5)
        plt.xlabel("Episodes")
        plt.ylabel("Loss")
        plt.savefig(save_results_to + "Graphe_loss.pdf")
        plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n_episodes',
                        type=int,
                        default=10000,
                        help='Number of episodes')
    parser.add_argument('--length',
                        type=int,
                        default=3600,
                        help='Number of steps to run the agent')
    parser.add_argument('--width',
                        type=int,
                        default=84,
                        help='Horizontal size of the observations')
    parser.add_argument('--height',
                        type=int,
                        default=84,
                        help='Vertical size of the observations')
    parser.add_argument('--fps',
                        type=int,
                        default=60,
                        help='Number of frames per second')
    parser.add_argument('--runfiles_path',
                        type=str,
                        default=None,
                        help='Set the runfiles path to find DeepMind Lab data')
    parser.add_argument('--level_script',
                        type=str,
                        default='contributed/psychlab/harlow',
                        help='The environment level script to load')
    parser.add_argument('--record',
                        type=str,
                        default=None,
                        help='Record the run to a demo file')
    parser.add_argument('--demo',
                        type=str,
                        default=None,
                        help='Play back a recorded demo file')
    parser.add_argument('--demofiles',
                        type=str,
                        default=None,
                        help='Directory for demo files')
    parser.add_argument('--video',
                        type=str,
                        default=None,
                        help='Record the demo run as a video')

    args = parser.parse_args()
    if args.runfiles_path:
        deepmind_lab.set_runfiles_path(args.runfiles_path)
    train(args.n_episodes, args.length, args.width, args.height, args.fps,
          args.level_script, args.record, args.demo, args.demofiles,
          args.video)
