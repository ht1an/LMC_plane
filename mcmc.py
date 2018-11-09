import numpy as np
import scipy.stats as stats



def gauss2_model(y, p0, N, func, nwalkers):
    # MCMC sampling
    ndim = p0.shape[1]
    print(ndim)
    sampler = emcee.EnsembleSampler(nwalkers, \
                                    ndim, func, args=[y])
    N_burn_in = 2000

    pos, prob, state = sampler.run_mcmc(p0, N_burn_in)
    sampler.reset()

    sampler.run_mcmc(pos, N)

    samples = sampler.chain[:, N_burn_in:, :].reshape((-1, ndim))
    # corner.corner(samples)
    popt = np.median(samples, axis=0)
    pcov = np.zeros((ndim, ndim))
    for i in range(ndim):
        for j in range(ndim):
            pcov[i, j] = (np.sum((samples[:, i] - popt[i]) * \
                                 (samples[:, j] - popt[j]))) / len(samples)
    return popt, pcov, samples


def lnprob_gauss3(x, y):
    # n = np.float(len(y))
    f1 = x[0]
    f2 = x[1]
    mu1 = x[2]
    sig1 = x[3]
    mu2 = x[4]
    sig2 = x[5]
    mu3 = x[6]
    sig3 = x[7]

    if np.isinf(f1) or np.isinf(f2) or np.isinf(mu1) or np.isinf(sig1) or \
            np.isinf(mu2) or np.isinf(sig2) or np.isinf(mu3) or np.isinf(sig3) or \
            f1 < 0 or f1 > 1 or f2 < 0 or f2 > 1 or (f1 + f2) < 0 or (f1 + f2) > 1 or \
            sig1 < 0 or sig2 < 0 or sig3 < 0 or \
            sig1 > 200 or sig2 > 200 or sig3 > 200 or \
            mu1 > 100 or mu1 < -100 or mu3 > 0 or mu3 < -200 or mu2 > 300 or mu2 < 100:
        return -1e100
    g1 = f1 * stats.norm.pdf(y, mu1, sig1)  # np.exp(-(y-mu1)**2/(2*sig1**2))/(np.sqrt(2*np.pi)*sig1)
    g2 = f2 * stats.norm.pdf(y, mu2, sig2)  # np.exp(-(y-mu2)**2/(2*sig2**2))/(np.sqrt(2*np.pi)*sig2)
    g3 = (1 - f1 - f2) * stats.norm.pdf(y, mu3, sig3)  # np.exp(-(y-mu2)**2/(2*sig2**2))/(np.sqrt(2*np.pi)*sig2)
    g = g1 + g2 + g3
    # print g
    ind_g = (np.isinf(g) == False) & (np.isnan(g) == False) & (g > 0)
    logg = np.log(g[ind_g])
    ind_lg = logg > -1e100
    return np.sum(logg[ind_lg])