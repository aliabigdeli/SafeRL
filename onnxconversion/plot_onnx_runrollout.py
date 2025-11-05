import numpy as np
import onnxruntime as ort
from saferl.aerospace.tasks.rejoin.task import DubinsRejoin
from saferl.environment.utils import jsonify, is_jsonable
from saferl.environment.utils import YAMLParser, build_lookup
import jsonlines
import tqdm
import matplotlib.pyplot as plt
import csv
import argparse
import os



def plot_rollouts(env, seed, num_rollouts=1, render=False, model='f16dubins_50sec.onnx'):
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
    
    ort_session = ort.InferenceSession(model)

    # Get input and output names
    input_name = ort_session.get_inputs()[0].name
    # output_name = ort_session.get_outputs()[0].name
    output_names = [output.name for output in ort_session.get_outputs()]

    # initialize list to track rollout results
    rollout_results = []

    for i in tqdm.tqdm(range(num_rollouts)):
        # initialize lists for this rollout
        lead_x = []
        lead_y = []
        wingman_x = []
        wingman_y = []
        wingman_speed = []
        wingman_throttle = []
        wingman_untrimmed_throttle = []
        
        # run until episode ends
        episode_reward = 0
        done = False
        obs = env.reset()
        step_num = 0
        
        # track mission outcome for this rollout
        mission_successful = False
        failure_reason = None

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
            
            # check for mission success/failure status
            if 'success' in info and info['success']:
                mission_successful = True
            if 'failure' in info and info['failure']:
                failure_reason = info['failure']

            if render:
                # attempt to render environment state
                env.render()
        
        # Create individual plot for this rollout
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
        
        # Add success/failure status to trajectory plot title
        status_text = "SUCCESS" if mission_successful else f"FAILED ({failure_reason})" if failure_reason else "COMPLETED"
        ax1.set_title(f'Trajectory of Aircraft - Rollout {i+1} - {status_text}')
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
        
        # Check if rollout_plots directory exists, if not, create it
        if not os.path.exists('rollout_plots'):
            os.makedirs('rollout_plots')
        # Save individual rollout plot
        plt.savefig(f'rollout_plots/onnx_runout_plot_seed{seed}_rollout{i+1}.png')
        plt.show()
        plt.close()  # Close the figure to free memory
        
        # store rollout result
        rollout_results.append({
            'rollout_number': i + 1,
            'successful': mission_successful,
            'failure_reason': failure_reason if failure_reason else 'N/A',
            'final_reward': episode_reward,
            'total_steps': step_num
        })
    
    # Save rollout results to CSV
    csv_filename = f'rollout_results_seed{seed}_rollouts{num_rollouts}.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['rollout_number', 'successful', 'failure_reason', 'final_reward', 'total_steps']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in rollout_results:
            writer.writerow(result)
    
    print(f"Rollout results saved to {csv_filename}")
    print(f"Individual plots saved as onnx_runout_plot_seed{seed}_rollout[N].png")

def main():
    parser = argparse.ArgumentParser(description="Plot ONNX rollout results")
    parser.add_argument('-m', '--model', type=str, default='f16dubins_50sec.onnx', help='Path to model ONNX file')
    parser.add_argument('-c', '--config', type=str, default='rejoin_f16_50sec.yaml', help='Path to config YAML file')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed for environment')
    parser.add_argument('-n', '--num_rollouts', type=int, default=1, help='Number of rollouts to run')
    args = parser.parse_args()

    config_dir = args.config
    parser_yaml = YAMLParser(yaml_file=config_dir, lookup=build_lookup())
    config = parser_yaml.parse_env()
    env = DubinsRejoin(config['env_config'])
    seed = args.seed
    env.seed(seed)
    model = args.model
    plot_rollouts(env, seed, args.num_rollouts, model)

if __name__ == "__main__":
    main()