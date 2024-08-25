# Generate PNL: g(y) = h(x) + e
import numpy as np
import pandas as pd

def linear(h, rng):
    h = h / np.std(h)
    a = rng.uniform(0.2, 2)
    return a * h

def cube(X, rng=None):
    X = X / np.std(X)
    return X ** 3

def inverse(X, rng=None):
    X = X / np.std(X)
    return 1 / (X - np.min(X) + 1)

def nexp(X, rng=None):
    X = X / np.std(X)
    return np.exp(-X)

def log(X, rng=None):
    X = X / np.std(X)
    return np.log1p(X - np.min(X))
    
def sigmoid(X, rng=None):
    X = X / np.std(X)
    return 1 / (1 + np.exp(-X))

def square(X, rng=None):
    X = X / np.std(X)
    return X ** 2

def abs(X, rng=None):
    X = X / np.std(X)
    return np.abs(X)

def gaussian_noise(rng, noise_coeff, *size):
        return rng.randn(*size) * noise_coeff

def uniform_noise(rng, noise_coeff, *size):
    return rng.uniform(-1, 1, size=size) * noise_coeff

def laplace_noise(rng, noise_coeff, *size):
    return rng.laplace(size=size) * noise_coeff

def gumbel_noise(rng, noise_coeff, *size):
    return rng.gumbel(size=size) * noise_coeff

def exp_noise(rng, noise_coeff, *size):
    return rng.exponential(size=size) * noise_coeff

def simulate(N:int, random_state:int, noise_type:str, noise_coeff:float, **kwargs):
    if random_state is None:
        random_state = np.random.randint(2 ** 10)
    print(f'{random_state = }')
    rng = np.random.RandomState(random_state)
    # noise_type = [gaussian_noise, uniform_noise, laplace_noise]
    # noise = noise_type[rng.randint(len(noise_type))](rng, noise_coeff, N)
    noise_func = eval(noise_type)
    X = noise_func(rng, noise_coeff, N)

    # X = noise_type[rng.randint(len(noise_type))](rng, noise_coeff, N)
    g_types = [linear, cube, inverse, nexp, log, sigmoid]
    h_types = [square, abs, sigmoid]

    h_func = h_types[rng.randint(len(h_types))] 
    h_X = h_func(X, rng)
    g_func = g_types[rng.randint(len(g_types))] 
    noise = noise_func(rng, noise_coeff, N)
    Y = g_func(h_X + noise, rng)

    print(f'h(x) = {h_func.__name__}, g(.) = {g_func.__name__}')
    label = 1
    if rng.randint(2):
        X, Y, label = Y, X, -1
    return X, Y, label, noise

def simulate_dataset(N_dataset, noise_type, noise_coeff, random_state):
    data = []
    # rng = np.random.RandomState(random_state)
    for i in range(N_dataset):
        N = 1000
        X, Y, label, _ = simulate(N, random_state=random_state + i, noise_type=noise_type, noise_coeff=noise_coeff)
        data.append(dict(A=X, B=Y, Target=label))
    df = pd.DataFrame(data)
    df['SampleID'] = np.arange(N_dataset)
    parsecell = lambda cell: ' '.join(map(str, cell))
    df[['A', 'B']] = df[['A', 'B']].applymap(parsecell)
    df.to_csv(f'data/Sim-PNL_{noise_type}.csv', index=False)
    return df

if __name__ == '__main__':
    gaus_df = simulate_dataset(N_dataset=500, noise_type='gaussian_noise', noise_coeff=0.01, random_state=123)
    uni_df = simulate_dataset(N_dataset=500, noise_type='uniform_noise', noise_coeff=0.01, random_state=202)
    lap_df = simulate_dataset(N_dataset=500, noise_type='laplace_noise', noise_coeff=0.01, random_state=203)
    # gum_df = simulate_dataset(N_dataset=100, noise_type='gumbel_noise', noise_coeff=0.01, random_state=203)
    # exp_df = simulate_dataset(N_dataset=100, noise_type='exp_noise', noise_coeff=0.01, random_state=203)
    # all_df = pd.concat([gaus_df, uni_df, lap_df], axis=0)
    # all_df.to_csv(f'data/Sim-PNL_all.csv', index=False)