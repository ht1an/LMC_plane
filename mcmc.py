import numpy as np
import scipy.stats as stats
import emcee
import coord as C


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


def lnprob_gauss3(x, z):
    # n = np.float(len(y))
    Cra = x[0]   # Cra
    Cdec = x[1]   # Cdec
    Cpmra = x[2]  # Cpmra
    Cpmdec = x[3] # Cpmdec

    ra_tmp = z[:,0]
    dec_tmp = z[:,1]

    pmra_tmp = z[:,2]
    pmdec_tmp = z[:,3]

    xy = C.radec2xy(ra_tmp,dec_tmp,Cra,Cdec,degree=True)
    pmxy = C.PMradec2PMxy(pmra_tmp,pmdec_tmp,ra_tmp,dec_tmp,
                          Cra,Cdec,Cpmra,Cpmdec,degree=True)
    x = xy[:,0]
    y = xy[:,1]
    pmx = pmxy[:,0]
    pmy = pmxy[:,1]

    L = x*pmy-y*pmx
    L1 = L/np.abs(L)
    S = np.sum(L1)
    g = np.exp(S)
    return g