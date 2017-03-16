import pickle
import tensorflow as tf
import numpy as np
import gym
import argparse
import pandas as pd
import tf_util
import tqdm
import load_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    args = parser.parse_args()
    run_dagger(args.env, 20, 100, 10000)


def extract_imitation(env):
    dic_data = pickle.loads(open('imitation/original/{}.pkl'.format(env), 'rb').read())
    inputs = np.array(dic_data['observations'])
    outputs = np.array(dic_data['actions'])
    evaluations = pd.DataFrame({'steps': dic_data['steps'], 'expert returns': dic_data['returns']})
    return inputs, outputs, evaluations


def get_experts_actions(observations, policy):
    with tf.Session():
        tf_util.initialize()
        actions = []
        for obs in observations:
            action = policy(obs[None, :])
            actions.append(action)
    return np.array(actions)


def train_regressor(inputs, outputs, layers=[64, 64], activation_function=tf.nn.tanh, batch_size=100,
                    epochs=100, steps=10000):
    inputs_dim = inputs.shape[1]
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=inputs_dim)]
    outputs_dim = outputs.shape[2]
    estimator = tf.contrib.learn.DNNRegressor(
        feature_columns=feature_columns,
        hidden_units=layers,
        activation_fn=activation_function,
        label_dimension=outputs_dim
    )

    input_fn = tf.contrib.learn.io.numpy_input_fn({"": inputs}, outputs[:, 0, :],
                                                  batch_size=batch_size, num_epochs=epochs)
    estimator.fit(input_fn=input_fn, steps=steps)
    return estimator


def extract_stats(data):
    mean = data['returns'].mean()
    std = data['returns'].std()
    x = data['steps']
    pct_full_steps = (x / x.max()).mean()

    return pd.Series({
        'mean reward': mean,
        'std reward': std,
        'pct full rollout': pct_full_steps
    })


def run_dagger(env_name, num_runs, batch_size, steps):
    epochs = steps/batch_size
    inputs, outputs, evaluations = extract_imitation(env_name)

    stats = {}
    rewards = []
    step_count = []

    with tf.Session():
        env = gym.make(env_name)
        tf_util.initialize()
        policy = load_policy.load_policy('experts/{}.pkl'.format(env_name))
        model = train_regressor(inputs, outputs)
        for i in range(num_runs):
            np.random.shuffle(inputs)
            np.random.shuffle(outputs)
            input_fn = tf.contrib.learn.io.numpy_input_fn({"": inputs}, outputs[:, 0, :],
                                                          batch_size=batch_size, num_epochs=epochs)
            model.fit(input_fn=input_fn, steps=steps)
            data = run_regressor(model, env)
            new_inputs = data['observations']
            stats[i] = extract_stats(data)
            rewards.append(data['returns'])
            step_count.append(data['steps'])
            new_outputs = get_experts_actions(new_inputs, policy)
            inputs = np.append(inputs, new_inputs, axis=0)
            outputs = np.append(outputs, new_outputs, axis=0)
        reward_data = {'rewards': np.array(rewards), 'steps': np.array(step_count)}
        df = pd.DataFrame(stats).T
        df.index.name = 'iterations'
        df.to_csv('DAgger_results/{}_DAgger.csv'.format(env_name))
        pickle_name = 'DAgger_results/{}_DAgger_rewards.pkl'.format(env_name)
        pickle.dump(reward_data, open(pickle_name, 'wb+'))
    return


def run_regressor(model, env, num_rollouts=20, render=False):
    returns = []
    observations = []
    actions = []
    steps_numbers = []

    with tf.Session():
        max_steps = env.spec.timestep_limit
        tf_util.initialize()
        for i in tqdm.tqdm(range(num_rollouts)):
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = model.predict(obs[None, :], as_iterable=False)
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if render:
                    env.render()
                if steps >= max_steps:
                    break
            steps_numbers.append(steps)
            returns.append(totalr)

        model_data = {'observations': np.array(observations),
                        'actions': np.array(actions),
                        'returns': np.array(returns),
                        'steps': np.array(steps_numbers)}

    return model_data

if __name__ == '__main__':
    main()

