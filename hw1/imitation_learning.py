import gym
import load_policy
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
    train_regressor(inputs, outputs)


def extract_imitation(env):
    dic_data = pickle.loads(open('imitation/original/{}.pkl'.format(env), 'rb').read())
    inputs = np.array(dic_data['observations'])
    outputs = np.array(dic_data['actions'])
    evaluations = pd.DataFrame({'steps': dic_data['steps'], 'returns': dic_data['returns']})
    return inputs, outputs, evaluations


def train_regressor(inputs, outputs, layers=[64, 64],
                        activation_function=tf.nn.tanh, batch_size=50, epochs=200, steps=10000):
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

if __name__ == '__main__':
    main()

