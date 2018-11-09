#!/usr/bin/python
# -*- coding: UTF-8 -*-

# this is used to use MCMC to constrain the inclination of the LMC plane

import numpy as np
import matplotlib.pyplot as plt
import math as M
import sys
import os
import emcee
import corner
import astropy.io.fits as fits
import mcmc
import coord as C


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
ind = (AENa<0.25) & (BEFa<1.5) & (EBVa<0.8)
raG,decG = raGa[ind],decGa[ind]
pmraG,pmdecG = pmraGa[ind],pmdecGa[ind]
pmraGe,pmdecGe = pmraGea[ind], pmdecGea[ind]
MagG,BPRPG = MagGa[ind],BPRPGa[ind]
lG,bG = lGa[ind],bGa[ind]
S_ID = S_IDa[ind]
AEN = AENa[ind]
BEF = BEFa[ind]
EBV = EBVa[ind]

Cra_tmp,Cdec_tmp = 78.77,-69.01
ind_LMC =




