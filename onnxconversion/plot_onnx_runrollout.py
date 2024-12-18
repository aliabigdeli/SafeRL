import numpy as np
import onnxruntime as ort
from saferl.aerospace.tasks.rejoin.task import DubinsRejoin
from saferl.environment.utils import jsonify, is_jsonable
from saferl.environment.utils import YAMLParser, build_lookup
import jsonlines
import tqdm
import matplotlib.pyplot as plt




def plot_rollouts(env, num_rollouts=1, render=False):
    """
    A function to coordinate policy evaluation via RLLib API.

    Parameters
    ----------
    agent : ray.rllib.agents.trainer_template.PPO
        The trained agent which will be evaluated.
    env : BaseEnv
        The environment in which the agent will act.
    log_dir : str
        The path to the output directory in which evaluation logs will be run.
    num_rollouts : int
        The number of randomly initialized episodes conducted to evaluate the agent on.
    render : bool
        Flag to render the environment in a separate window during rollouts.
    """
    
    ort_session = ort.InferenceSession("f16dubins.onnx")

    # Get input and output names
    input_name = ort_session.get_inputs()[0].name
    # output_name = ort_session.get_outputs()[0].name
    output_names = [output.name for output in ort_session.get_outputs()]

    # initialize lists
    lead_x = []
    lead_y = []
    wingman_x = []
    wingman_y = []
    wingman_speed = []
    wingman_throttle = []
    wingman_untrimmed_throttle = []

    for i in tqdm.tqdm(range(num_rollouts)):
        # run until episode ends
        episode_reward = 0
        done = False
        obs = env.reset()
        step_num = 0

        while not done:
            # progress environment state
            # action = agent.compute_single_action(obs)
            obs = np.expand_dims(obs, axis=0).astype(np.float32)
            # action = ort_session.run([output_name], {input_name: obs})[0]
            action = ort_session.run(output_names, {input_name: obs})[0]
            action = (action[0][0], action[0][2])

            obs, reward, done, info = env.step(action)
            step_num += 1
            episode_reward += reward

            # store log contents in state
            state = {}
            if is_jsonable(info) is True:
                state["info"] = info
            else:
                state["info"] = jsonify(info)
            # state["actions"] = [float(i) for i in action[0]]
            state["actions"] = [float(i) for i in action]
            state["obs"] = obs.tolist()
            state["rollout_num"] = i
            state["step_number"] = step_num
            state["episode_reward"] = episode_reward

            # save data for plotting
            lead_x.append(info['lead']['x'])
            lead_y.append(info['lead']['y'])
            wingman_x.append(info['wingman']['x'])
            wingman_y.append(info['wingman']['y'])
            wingman_speed.append(info['wingman']['v'])
            wingman_throttle.append(info['wingman']['controller']['control'][1])
            wingman_untrimmed_throttle.append(info['wingman']['controller']['untrimmed_control'][1])

            if render:
                # attempt to render environment state
                env.render()
    
    # Setting up subplots
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[2, 1])

    ax1 = fig.add_subplot(gs[:, 0])  # Trajectory plot
    ax2 = fig.add_subplot(gs[0, 1])  # Speed plot
    ax3 = fig.add_subplot(gs[1, 1])  # Throttle plot
    
    # Plotting the trajectories
    ax1.plot(lead_x, lead_y, label='Lead', marker='o')
    ax1.plot(wingman_x, wingman_y, label='Follower', marker='o')
    ax1.scatter([lead_x[0], lead_x[-1]], [lead_y[0], lead_y[-1]], color='red')  # Start and End points for Lead
    ax1.scatter([wingman_x[0], wingman_x[-1]], [wingman_y[0], wingman_y[-1]], color='blue')  # Start and End points for Wingman
    ax1.text(lead_x[0], lead_y[0], 'Start')
    ax1.text(lead_x[-1], lead_y[-1], 'End')
    ax1.text(wingman_x[0], wingman_y[0], 'Start')
    ax1.text(wingman_x[-1], wingman_y[-1], 'End')
    ax1.set_title('Trajectory of Aircraft')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.legend()
    ax1.grid(True)

    # Plotting the speed of the follower
    ax2.plot(wingman_speed, 'g-', label='Follower Speed', marker='o')
    ax2.set_title('Speed of the Follower')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Speed')
    ax2.grid(True)

    # Plotting the throttle command
    ax3.plot(wingman_throttle, 'b-', label='Throttle')
    ax3.plot(wingman_untrimmed_throttle, 'b--', label='Untrimmed')
    ax3.set_title('Throttle Command to Follower')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Throttle Command')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    
    plt.savefig('onnx_runout_plot.png')
    plt.show()

def main():
    config_dir = 'rejoin_f16.yaml'
    parser = YAMLParser(yaml_file=config_dir, lookup=build_lookup())
    config = parser.parse_env()
    env = DubinsRejoin(config['env_config'])
    env.seed(0)
    plot_rollouts(env)

if __name__ == "__main__":
    main()