import numpy as np

# this is used for coordinate conversion

def radec2xy(ra,dec,Cra,Cdec,degree=True):
    # this is based on Equ2 in GDR2Helmi
    # always return degree
    if degree:
        ra = np.deg2rad(ra)
        dec = np.deg2rad(dec)
        Cra = np.deg2rad(Cra)
        Cdec = np.deg2rad(Cdec)

    x = np.cos(dec)*np.sin(ra-Cra)
    y = np.sin(dec)*np.cos(Cdec) - np.cos(dec)*np.sin(Cdec)*np.cos(ra-Cra)
    return np.rad2deg(x),np.rad2deg(y)

def PMradec2PMxy(PMra,PMdec,ra,dec,Cra,Cdec,CPMra=0,CPMdec=0,degree=True):
    # this is based on Equ2 in GDR2Helmi
    if degree:
        ra = np.deg2rad(ra)
        dec = np.deg2rad(dec)
        Cra = np.deg2rad(Cra)
        Cdec = np.deg2rad(Cdec)

    PMrac = PMra-CPMra
    PMdecc = PMdec-CPMdec

    PMx = PMrac*np.cos(ra-Cra) - PMdecc*np.sin(dec)*np.sin(ra-Cra)
    PMy = PMrac*np.sin(Cdec)*np.sin(ra-Cra) + \
          PMdecc*(np.cos(dec)*np.cos(Cdec) +
                  np.sin(dec)*np.sin(Cdec)*np.cos(ra-Cra))
    return PMx,PMy


