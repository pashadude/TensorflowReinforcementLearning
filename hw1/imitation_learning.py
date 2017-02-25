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
    #print(evaluations)


def extract_imitation(env):
    dic_data = pickle.loads(open('imitation/original/{}.pkl'.format(env), 'rb').read())
    inputs = np.array(dic_data['observations'])
    outputs = np.array(dic_data['actions'])
    evaluations = pd.DataFrame({'steps': dic_data['steps'], 'returns': dic_data['returns']})
    return inputs, outputs, evaluations

def train_regressor(inputs,outputs,layer):
    inputs_dim = inputs.shape[1]
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=inputs_dim)]
    outputs_dim = outputs.shape[2]
    regressor = tf.contrib.learn.DNNRegressor(
        feature_columns=feature_columns,
        hidden_units=[inputs_dim, layer, outputs_dim],
        activation_fn=tf.nn.relu,
        label_dimension=outputs_dim
    )
    regressor.fit(x=tf.contrib.learn.infer_real_valued_columns_from_input(inputs),
               y=tf.contrib.learn.infer_real_valued_columns_from_input(outputs[:, 0, :]),
               steps=1000)
    loss = regressor.evaluate(x=tf.contrib.learn.infer_real_valued_columns_from_input(inputs),
               y=tf.contrib.learn.infer_real_valued_columns_from_input(outputs[:, 0, :]),
               steps=2)["loss"]
    return loss

if __name__ == '__main__':
    main()

