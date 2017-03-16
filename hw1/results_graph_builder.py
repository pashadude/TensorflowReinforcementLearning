import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    args = parser.parse_args()
    dic_data = pickle.loads(open('imitation/original/{}.pkl'.format(args.env), 'rb').read())
    expert_data = pd.DataFrame({'expert steps': dic_data['steps'], 'expert returns': dic_data['returns']})
    model_data = pickle.loads(open('imitation/tnn_imitation/{}.pkl'.format(args.env), 'rb').read())
    expert_data['imitation steps'] = pd.Series(model_data['steps'], index=expert_data.index)
    expert_data['imitation returns'] = pd.Series(model_data['returns'], index=expert_data.index)
    dagger_data = pickle.loads(open('DAgger_results/{}_DAgger_rewards.pkl'.format(args.env), 'rb').read())
    k = 0
    dagger_returns = []
    for i in dagger_data['rewards']:
        #print(dagger_data['steps'][k])
        dagger_series = i/dagger_data['steps'][k]
        dagger_returns.append(np.mean(dagger_series)*10)
    expert_data['dagger mean returns'] = pd.Series(dagger_returns, index=expert_data.index)
    expert_data['expert mean returns'] = expert_data['expert returns'] / expert_data['expert steps']
    expert_data['imitation mean returns'] = expert_data['imitation returns'] / expert_data['imitation steps']
    print(expert_data)
    plt.figure(figsize=(5, 5))
    plt.plot(np.arange(1.0, 21.0, 1.0), expert_data['expert mean returns'], 'r-')
    plt.plot(np.arange(1.0, 21.0, 1.0), expert_data['imitation mean returns'], 'g-')
    plt.plot(np.arange(1.0, 21.0, 1.0), expert_data['dagger mean returns'], 'b-')
    plt.ylabel('returns')
    plt.xlabel('rollouts')
    plt.title('{} returns: dagger (blue) vs imitation learning (green) vs expert (red)'.format(args.env))
    plt.show()


if __name__ == '__main__':
    main()
