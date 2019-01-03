#!/usr/bin/python
# -*- coding: UTF-8 -*-
# this is used to use MCMC to constrain the inclination of the LMC plane
# this line is used for test github -- pycharm connection ----V2
import numpy as np
import matplotlib.pyplot as plt
import os
import emcee
import corner
import astropy.io.fits as fits
import coord as C
# ---------------------------------------------------------
def gauss2_model(y, p0, N, func, nwalkers):
    # MCMC sampling
    ndim = p0.shape[1]
    print(ndim)
    sampler = emcee.EnsembleSampler(nwalkers, \
                                    ndim, func, args=[y])
    N_burn_in = 1000
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

def lnprob_gauss3(xx, zz):
    # n = np.float(len(y))
    Cra = xx[0]   # Cra
    Cdec = xx[1]   # Cdec
    Cpmra = xx[2]  # Cpmra
    Cpmdec = xx[3] # Cpmdec

    if (Cra>85) or (Cra<75) or (Cdec>-67) or (Cdec<-72) or (Cpmra<1) or (Cpmra>3) or (Cpmdec<-1) or (Cpmdec>1):
        return -1e1000
    else:
        # print(Cra, Cdec, Cpmra, Cpmdec, '-------------')
        ra_tmp = zz[:,0]
        dec_tmp = zz[:,1]

        pmra_tmp = zz[:,2]
        pmdec_tmp = zz[:,3]

        x,y = C.radec2xy(ra_tmp,dec_tmp,Cra,Cdec,degree=True)
        pmx,pmy = C.PMradec2PMxy(pmra_tmp,pmdec_tmp,ra_tmp,dec_tmp,
                              Cra,Cdec,Cpmra,Cpmdec,degree=True)
        L = x*pmy-y*pmx
        L1 = np.ones_like(L)
        L1[L<0]=-1
        # S = np.abs(np.sum(L1))

        g = np.abs(np.sum(L))/np.sum(np.abs(L))
        return 10**(g)
# ----------------------------------------------------------------------------
# start the code
path = os.getcwd()
dpath = path+"/data/"
ppath = path+"/plots/"
dfn = "GDR2_Mgiant_candidate_color2_b20_gmag10_20_rest_4000_EBV_LMC.fits"

dt = fits.open(dpath+dfn)
data = dt[1].data
raGa,decGa = np.array(data["ra"]),np.array(data["dec"])
pmraGa,pmdecGa = np.array(data["pmra"]),np.array(data["pmdec"])
pmraGea,pmdecGea = np.array(data["pmra_error"]),np.array(data["pmdec_error"])
MagGa = np.array(data["phot_g_mean_mag"])
BPRPGa = np.array(data["bp_rp"])
lGa,bGa = np.array(data["l"]),np.array(data["b"])
S_IDa = data["source_id_1"]
AENa = data["astrometric_excess_noise"]
BEFa = data["phot_bp_rp_excess_factor"]
EBVa = data["EBV"]
rvGa,rveGa = data["radial_velocity"],data["radial_velocity_error"]

mag_range = np.array
min_mag = 13
max_mag = 16
ind = (AENa<0.25) & (BEFa<1.5) & (MagGa>min_mag) & (MagGa<max_mag)  #& (EBVa<0.8)
raG,decG = raGa[ind],decGa[ind]
pmraG,pmdecG = pmraGa[ind],pmdecGa[ind]
pmraGe,pmdecGe = pmraGea[ind], pmdecGea[ind]
MagG,BPRPG = MagGa[ind],BPRPGa[ind]
lG,bG = lGa[ind],bGa[ind]
S_ID = S_IDa[ind]
AEN = AENa[ind]
BEF = BEFa[ind]
EBV = EBVa[ind]

Cra_tmp,Cdec_tmp = 82,-70
# ind_LMC =

ds = np.sqrt((raG-Cra_tmp)**2*np.cos(np.deg2rad(Cdec_tmp))**2+(decG-Cdec_tmp)**2)
sdd = 6
ind_s = ds<sdd
plt.plot(raG,decG,'k.',alpha=0.1)
plt.plot(raG[ind_s],decG[ind_s],'r.',alpha=0.1)

plt.plot(Cra_tmp,Cdec_tmp,'bo')
plt.savefig(ppath+"ra_dec_selection.pdf")

nwalkers=50
ndim = 4
p0=np.zeros((nwalkers,ndim))
p0[:,0] = np.random.rand(nwalkers)*10+75
p0[:,1] = np.random.rand(nwalkers)*5-72
p0[:,2] = np.random.rand(nwalkers)*2+1
p0[:,3] = np.random.rand(nwalkers)*2-1
Ns = len(raG[ind_s])
zz = np.zeros((Ns,ndim))
zz[:,0] = raG[ind_s]
zz[:,1] = decG[ind_s]
zz[:,2] = pmraG[ind_s]
zz[:,3] = pmdecG[ind_s]
# print(np.max(zz[:,0]),np.max(zz[:,1]),np.max(zz[:,2]),np.max(zz[:,3]))
Mpopt3,Mpcov3, samples3 = gauss2_model(zz,p0,2000,lnprob_gauss3,nwalkers)

ss = np.sqrt(np.diag(Mpcov3))
print("min_mag:",)
print('Cra=%(f1).3f\pm%(f1e).3f' % {'f1':Mpopt3[0],'f1e':ss[0]})
print('Cdec=%(v1).3f \pm %(v1e).3f km/s' % {'v1':Mpopt3[1],'v1e':ss[1]})
print('Cpmra=%(v2).3f \pm %(v2e).3f km/s' % {'v2':Mpopt3[2],'v2e':ss[2]})
print('Cpmdec=%(v3).3f \pm %(v3e).3f km/s' % {'v3':Mpopt3[3],'v3e':ss[3]})
fig = corner.corner(samples3,labels=["$CRA$","$Cdec$","$CPMRA$","$CPMDEC$"],\
                   truths=[78.77,-69.01,1.85,0.234],fontsize=15)
plt.savefig(ppath+"coner_results_{min_mag}_{max_mag}.pdf".format(**locals()))

fig = plt.figure(figsize=(6,4))
plt.plot(zz[:,0],zz[:,1],'k.',alpha=0.1)
plt.quiver(zz[:,0],zz[:,1],zz[:,2]-Mpopt3[2],zz[:,3]-Mpopt3[3])
plt.plot(Mpopt3[0],Mpopt3[1],'ro',markersize=25)
plt.xlabel("RA")
plt.ylabel("Dec")
fig.savefig(ppath+"RA_DEC_{min_mag}_{max_mag}.pdf".format(**locals()))
