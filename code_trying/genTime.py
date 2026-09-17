import random
import pandas as pd
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def generateSeries(Z, T):
    preTime, postTime, counterpostTime = [], [], []
    for i in range(PARAM['N']):
        pre_rate = np.array([np.exp(PARAM['betaZ'] * Z[i] + PARAM['alphaP'] * np.sin(2 * np.pi * t / 7)) for t in
                             range(PARAM['pretime'])])
        post_rate = np.array(
            [np.exp(PARAM['betaZ'] * Z[i] + PARAM['betaT'] * T[i] + PARAM['alphaP'] * np.sin(2 * np.pi * t / 7)) for t
             in range(PARAM['posttime'])])
        counterpost_rate = np.array(
            [np.exp(PARAM['betaZ'] * Z[i] + PARAM['betaT'] * (1-T[i]) + PARAM['alphaP'] * np.sin(2 * np.pi * t / 7)) for t
             in range(PARAM['posttime'])])
        preTime.append(np.random.poisson(lam=pre_rate))
        postTime.append(np.random.poisson(lam=post_rate))
        counterpostTime.append(np.random.poisson(lam=counterpost_rate))
    return np.array(preTime), np.array(postTime), np.array(counterpostTime)

def main():
    Z = np.random.uniform(0,1,size=PARAM['N'])  # (1000,)
    T = np.array([np.random.binomial(1, p=Z[i]) for i in range(PARAM['N'])]) # (1000,)
    preTime, postTime, counterpostTime = generateSeries(Z,T) # (1000, 120), (1000, 20), (1000, 20)
    dt = np.concatenate([Z.reshape(-1,1), T.reshape(-1,1), preTime, postTime, counterpostTime], axis=1)
    columns = ['Z','T']+['pre_'+str(i) for i in range(1,PARAM['pretime']+1)]+['post_'+str(i) for i in range(1,PARAM['posttime']+1)]+['counterpost_'+str(i) for i in range(1,PARAM['posttime']+1)]
    data=pd.DataFrame(dt, columns=columns)
    data.to_csv('time_dt.csv', index=False)

SEED = 100
PARAM = {
    'N': 1000, # num of individuals
    'pretime': 120, # length of pre-treatment
    'posttime': 20, # length of post-treatment

    'betaZ': 1,
    'betaT': 1,
    'alphaP': 1,
}

if __name__ == "__main__":
    set_seed(SEED)
    main()