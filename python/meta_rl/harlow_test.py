from model import ActorCritic
import torch
import argparse
import deepmind_lab
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import six


def _action(*entries):
    return np.array(entries, dtype=np.intc)


ACTIONS = {
    'look_left': _action(-5, 0, 0, 0, 0, 0, 0),
    'look_right': _action(5, 0, 0, 0, 0, 0, 0),
    'no-ops': _action(0, 0, 0, 0, 0, 0, 0)
}

ACTION_LIST = list(six.viewvalues(ACTIONS))


def test(n_episodes, name, length, width, height, fps, level, record, demo,
         demofiles, video, episode_range):
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

    save_results_to = "/home/damien/Documents/GitHub/lab/python/meta_rl/"

    policy.load_state_dict(
        torch.load(
            '/home/damien/Documents/GitHub/lab/python/meta_rl/{}'.format(
                name)))

    render = True
    save_gif = False

    rewards_episode_list = []

    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        running_reward = 0
        state = np.float32(env.observations()['RGB_INTERLEAVED'])
        hidden_state = torch.zeros(1, 1, num_hidden_units)
        cell_state = torch.zeros(1, 1, num_hidden_units)
        for t in range(length):
            action, hidden_state, cell_state = policy(state, hidden_state,
                                                      cell_state)
            hidden_state = hidden_state.detach()
            cell_state = cell_state.detach()
            state = env.observations()['RGB_INTERLEAVED']
            reward = env.step(ACTION_LIST[action])
            done = not env.is_running() or env.num_steps() == length - 1
            running_reward += reward
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
        rewards_episode_list.append(running_reward)

    plt.plot(np.arange(len(rewards_episode_list)), rewards_episode_list)
    plt.xlabel("Épisode")
    plt.ylabel("Reward")
    plt.title("Graphe des peformances après entrainement")
    plt.savefig(save_results_to + "Graphe_rewards_test_" + str(episode_range))
    plt.figure()
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n_episodes',
                        type=int,
                        default=20,
                        help='Number of episodes')
    parser.add_argument('--name',
                        type=str,
                        default='trainedfile.pth',
                        help='File to ')
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
    parser.add_argument('--episode_range',
                        type=int,
                        default=None,
                        help='Episode range of training')

    args = parser.parse_args()
    if args.runfiles_path:
        deepmind_lab.set_runfiles_path(args.runfiles_path)
    test(args.n_episodes, args.name, args.length, args.width, args.height,
         args.fps, args.level_script, args.record, args.demo, args.demofiles,
         args.video, args.episode_range)
