import gym
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import tf_util
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    args = parser.parse_args()
    inputs, outputs, evaluations = extract_imitation(args.env)
    model = train_regressor(inputs, outputs)


def extract_imitation(env):
    dic_data = pickle.loads(open('imitation/original/{}.pkl'.format(env), 'rb').read())
    inputs = np.array(dic_data['observations'])
    outputs = np.array(dic_data['actions'])
    evaluations = pd.DataFrame({'steps': dic_data['steps'], 'expert returns': dic_data['returns']})
    return inputs, outputs, evaluations


def train_regressor(inputs, outputs, layers=[64, 64], activation_function=tf.nn.tanh, batch_size=50,
                    epochs=200, steps=10000):
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


def run_regressor(expert_data, model, env, num_rollouts=20, render=False):
    env_gym = gym.make(env)
    max_steps = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
    returns = []
    observations = []
    actions = []
    steps_numbers = []
    with tf.Session():
        tf_util.initialize()
        for i in range(num_rollouts):
            obs = env_gym.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = model.predict(obs[None, :])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env_gym.step(action)
                totalr += r
                steps += 1
                if render:
                    env_gym.render()
                if steps >= max_steps:
                    break
            steps_numbers.append(steps)
            returns.append(totalr)

        model_data = {'observations': np.array(observations),
                        'actions': np.array(actions),
                        'returns': np.array(returns),
                        'steps': np.array(steps_numbers)}
        expert_data['model returns'] = pd.Series(model_data['returns'], index=expert_data.index)
        pickle.dump(model_data, open('imitation/original/{}.pkl'.format(env), 'wb+'))
    return report_results(expert_data, env)


def report_results(report_data, env):
    fig = plt.figure(figsize=(5, 5))
    plt.plot_date(report_data['steps'], report_data['expert returns'], 'r-')
    plt.plot_date(report_data['steps'], report_data['model returns'], 'g-')
    plt.show()
    return

if __name__ == '__main__':
    main()

