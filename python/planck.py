import os, healpy, numpy
from astropy.io import fits

maps = {
    'cmb-nilc':'COM_CompMap_CMB-nilc_2048_R1.11.fits',
    'cmb-sevem':'COM_CompMap_CMB-sevem_2048_R1.11.fits',
    'cmb-smica':'COM_CompMap_CMB-smica_2048_R1.11.fits',
    'dust-commrul_0256':'COM_CompMap_dust-commrul_0256_R1.00.fits',
    'dust-commrul_2048':'COM_CompMap_dust-commrul_2048_R1.00.fits',
    'lensing':'COM_CompMap_Lensing_2048_R1.10.fits',
    'Lfreqfor-commrul_256':'COM_CompMap_Lfreqfor-commrul_0256_R1.00.fits',
    'Lfreqfor-commrul_2048':'COM_CompMap_Lfreqfor-commrul_2048_R1.00.fits',
    'rulerminimal':'COM_CompMap_Mask-rulerminimal_2048_R1.00.fits',
    'co-type1':'HFI_CompMap_CO-Type1_2048_R1.10.fits',
    'co-type2':'HFI_CompMap_CO-Type2_2048_R1.10.fits',
    'co-type3':'HFI_CompMap_CO-Type3_2048_R1.10.fits',
    'dust':'HFI_CompMap_DustOpacity_2048_R1.10.fits',
    'dust1.2':'HFI_CompMap_ThermalDustModel_2048_R1.20.fits',
    '100':'HFI_SkyMap_100_2048_R1.10_nominal.fits',
    '143':'HFI_SkyMap_143_2048_R1.10_nominal.fits',
    '217':'HFI_SkyMap_217_2048_R1.10_nominal.fits',
    '353':'HFI_SkyMap_353_2048_R1.10_nominal.fits',
    '545':'HFI_SkyMap_545_2048_R1.10_nominal.fits',
    '857':'HFI_SkyMap_857_2048_R1.10_nominal.fits',
    '030':'LFI_SkyMap_030_1024_R1.10_nominal.fits',
    '044':'LFI_SkyMap_044_1024_R1.10_nominal.fits',
    '070':'LFI_SkyMap_070_1024_R1.10_nominal.fits',
    'nilcspectralindex': 'COM_CompMap_Dust-GNILC-Model-Spectral-Index_2048_R2.00.fits',
    'commander2.0': 'COM_CompMap_ThermalDust-commander_2048_R2.00.fits',
}


def getval(l, b, map='dust', field='ebv', nest=True):
    planck_dir = os.environ['PLANCK_DIR']
    if map in maps:
        map = maps[map]
    map = fits.getdata(os.path.join(planck_dir, map))
    map = map[field]
    t, p = ((90.-b)*numpy.pi/180., l*numpy.pi/180.)
    return healpy.get_interp_val(map, t, p, nest=nest)

def getval_dpf(l, b):
    planck_dir = os.environ['PLANCK_DIR']
    map = fits.getdata(os.path.join(planck_dir, '../dust/Planckdust_1024_v2.fits'), 4)
    t, p = ((90.-b)*numpy.pi/180., l*numpy.pi/180.)
    return healpy.get_interp_val(map, t, p, nest=True)*7500
