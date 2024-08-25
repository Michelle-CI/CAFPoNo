from synthproblems import generate_samples
import numpy as np
import pandas as pd

def GP_sim(seed):
    data, model = generate_samples(seed=seed, num_variables=2, num_samples=1000)
    X, Y = data[:, 0], data[:, 1]
    adj = model.adj
    if adj[0, 1] == 1: label = 1
    if adj[1, 0] == 1: label = -1
    return X, Y, label

def simulate_dataset(N_dataset, random_state):
    data = []
    # rng = np.random.RandomState(random_state)
    for i in range(N_dataset):
        N = 1000
        X, Y, label  = GP_sim(seed=random_state + i)
        data.append(dict(A=X, B=Y, Target=label))
    df = pd.DataFrame(data)
    df['SampleID'] = np.arange(N_dataset)
    parsecell = lambda cell: ' '.join(map(str, cell))
    df[['A', 'B']] = df[['A', 'B']].applymap(parsecell)
    df.to_csv(f'data/Sim-PNL_GP.csv', index=False)
    return df

if __name__ == '__main__':
    simulate_dataset(N_dataset=100, random_state=987)