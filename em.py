import numpy as np
import math

# author : MouShuai
# time: 2016-09-16


def make_data(N, g_mu, b_mu, g_sigma, b_sigma, g_prop):
    X = np.zeros([N])
    for i in range(int(N*g_prop)):
        X[i] = np.random.normal(g_mu, g_sigma)
    for i in range(int(N*g_prop), N):
        X[i] = np.random.normal(b_mu, b_sigma)
    np.random.shuffle(X)
    return X


# only one dimension attribution: height
def compute_prob(x, mu, sigma):
    exponent = math.exp(-(math.pow(x-mu, 2))/(2*math.pow(sigma, 2)))
    prob = (1. / (math.sqrt(2*math.pi)*sigma)) * exponent
    return prob


# compute average value
def compute_avg_val(X, omega, n):
    sum_val = 0.0
    for i in range(len(X)):
        sum_val += omega[i] * X[i]
    return float(sum_val/n)


# compute the standard deviation
def compute_stdev_val(X, omega, mu, n):
    sum_val = 0.0
    for i in range(len(X)):
        sum_val += omega[i]*(X[i]-mu)**2
    return math.sqrt(float(sum_val/n))


# terminate condition
def is_terminate(old, new):
    old = np.matrix(old)
    new = np.matrix(new)
    tmp = old - new
    if tmp*tmp.T < 1e-4:
        return True
    return False


def compute_params(X):
    N = len(X)
    g_prop = 0.5
    b_prop = 0.5
    g_mu = min(X)
    b_mu = max(X)
    g_sigma = 1.0
    b_sigma = 1.0

    g_omega = range(N)
    b_omega = range(N)

    old = [g_prop, b_prop, g_mu, b_mu, g_sigma, b_sigma]
    new = []

    iters = 0
    while iters < 100:
        # E-step
        for i in range(N):
            g_omega[i] = g_prop * compute_prob(X[i], g_mu, g_sigma)
            b_omega[i] = b_prop * compute_prob(X[i], b_mu, b_sigma)
            sum_omega = g_omega[i] + b_omega[i]
            g_omega[i] /= sum_omega  # normalization
            b_omega[i] /= sum_omega  # normalization

        # M-setp
        g_num = sum(g_omega)
        g_prop = float(g_num) / float(N)
        b_num = sum(b_omega)
        b_prop = float(b_num) / float(N)

        # update average val and stdev val of two different clusters
        g_mu = compute_avg_val(X, g_omega, g_num)
        g_sigma = compute_stdev_val(X, g_omega, g_mu, g_num)
        b_mu = compute_avg_val(X, b_omega, b_num)
        b_sigma = compute_stdev_val(X, b_omega, b_mu, b_num)

        # whether terminate iterations or not
        new = [g_prop, b_prop, g_mu, b_mu, g_sigma, b_sigma]
        if is_terminate(old, new):
            break
        old = new
        iters += 1
        print (old)

    return new


if __name__ == '__main__':
    # Make a dataset in which each data is the height of a girl or a boy
    N = 200 # number of data
    g_mu = 160.0
    g_sigma = 2.0
    b_mu = 172.0
    b_sigma = 3.0
    g_prop = 0.3
    b_prop = 1.0 - g_prop
    params = []
    Heights = make_data(N, g_mu, b_mu, g_sigma, b_sigma, g_prop)
    params = compute_params(Heights)

    # print the initial params of two different distributions
    print ('The groundtruth params [g_mu, g_sigma, b_mu, b_sigma, g_prop, b_prop ]: '
           '[\t %.3f,\t%.3f,\t%.3f,\t%.3f,\t%.3f,\t%.3f]' % (g_prop, b_prop, g_mu, b_mu, g_sigma, b_sigma))

    # print the prediction params of two different distributions
    print ('The predicted params [g_mu, g_sigma, b_mu, b_sigma, g_prop, b_prop ]: '
           '[\t %.3f,\t%.3f,\t%.3f,\t%.3f,\t%.3f,\t%.3f]' % (params[0], params[1], params[2], params[3], params[4], params[5]))





