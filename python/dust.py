import os
import numpy
from scipy.ndimage import map_coordinates
import pdb
from astropy import wcs
from astropy.io import fits


def getval(l, b, map='sfd', size=None, order=0):
    """Return SFD at the Galactic coordinates l, b.

    Example usage:
    h, w = 1000, 4000
    b, l = numpy.mgrid[0:h,0:w]
    l = 180.-(l+0.5) / float(w) * 360.
    b = 90. - (b+0.5) / float(h) * 180.
    ebv = dust.getval(l, b)
    imshow(ebv, aspect='auto', norm=matplotlib.colors.LogNorm())
    """
    l = numpy.atleast_1d(l)
    b = numpy.atleast_1d(b)
    if map == 'sfd':
        map = 'dust'
    if map in ['dust', 'd100', 'i100', 'i60', 'mask', 'temp', 'xmap']:
        fname = 'SFD_'+map
    else:
        fname = map
    maxsize = {'d100': 1024, 'dust': 4096, 'i100': 4096, 'i60': 4096,
               'mask': 4096}
    if size is None and map in maxsize:
        size = maxsize[map]
    if size is not None:
        fname = fname + '_%d' % size
    fname = os.path.join(os.environ['DUST_DIR'], 'maps', fname)
    if not os.access(fname+'_ngp.fits', os.F_OK):
        raise Exception('Map file %s not found' % (fname+'_ngp.fits'))
    if l.shape != b.shape:
        raise ValueError('l.shape must equal b.shape')
    out = numpy.zeros_like(l, dtype='f4')
    for pole in ['ngp', 'sgp']:
        m = (b >= 0) if pole == 'ngp' else b < 0
        if numpy.any(m):
            hdulist = fits.open(fname+'_%s.fits' % pole)
            imwcs = wcs.WCS(hdulist[0].header)
            x, y = imwcs.wcs_world2pix(l[m], b[m], 0)
            out[m] = map_coordinates(hdulist[0].data, [y, x], order=order,
                                     mode='nearest')
    return out


def wgetval(l, b, **kw):
    import os
    import sys
    try:
        import wssa_utils
    except:
        raise ImportError('wgetval requires wssa_utils')
    from lsd.builtins.misc import galequ
    if os.environ.get('WISE_TILE', None) is None:
        os.environ['WISE_TILE'] = '/n/fink1/ameisner/tile-combine'
        sys.path.append('/n/home09/ameisner/wssa_utils/python')
    return wssa_utils.wssa_getval(*galequ(l, b), **kw)


def fm_unred(wave, flux, ebv, *args, **kwargs):
    '''
    NAME:
     FM_UNRED
    PURPOSE:
     Deredden a flux vector using the Fitzpatrick (1999) parameterization
    EXPLANATION:
     The R-dependent Galactic extinction curve is that of Fitzpatrick & Massa
     (Fitzpatrick, 1999, PASP, 111, 63; astro-ph/9809387 ).
     Parameterization is valid from the IR to the far-UV (3.5 microns to 0.1
     microns).  UV extinction curve is extrapolated down to 912 Angstroms.

    CALLING SEQUENCE:
     fm_unred( wave, flux, ebv [, 'LMC2', 'AVGLMC', 'ExtCurve', R_V = ,
                                   gamma =, x0=, c1=, c2=, c3=, c4= ])
    INPUT:
      wave - wavelength vector (Angstroms)
      flux - calibrated flux vector, same number of elements as "wave"
      ebv  - color excess E(B-V), scalar.  If a negative "ebv" is supplied,
              then fluxes will be reddened rather than dereddened.

    OUTPUT:
      Unreddened flux vector, same units and number of elements as "flux"

    OPTIONAL INPUT KEYWORDS
      R_V - scalar specifying the ratio of total to selective extinction
               R(V) = A(V) / E(B - V).  If not specified, then R = 3.1
               Extreme values of R(V) range from 2.3 to 5.3

      'AVGLMC' - if set, then the default fit parameters c1,c2,c3,c4,gamma,x0
             are set to the average values determined for reddening in the
             general Large Magellanic Cloud (LMC) field by Misselt et al.
            (1999, ApJ, 515, 128)
      'LMC2' - if set, then the fit parameters are set to the values determined
             for the LMC2 field (including 30 Dor) by Misselt et al.
             Note that neither /AVGLMC or /LMC2 will alter the default value
             of R_V which is poorly known for the LMC.

      The following five input keyword parameters allow the user to customize
      the adopted extinction curve.  For example, see Clayton et al. (2003,
      ApJ, 588, 871) for examples of these parameters in different interstellar
      environments.

      x0 - Centroid of 2200 A bump in microns (default = 4.596)
      gamma - Width of 2200 A bump in microns (default = 0.99)
      c3 - Strength of the 2200 A bump (default = 3.23)
      c4 - FUV curvature (default = 0.41)
      c2 - Slope of the linear UV extinction component
           (default = -0.824 + 4.717 / R)
      c1 - Intercept of the linear UV extinction component
           (default = 2.030 - 3.007 * c2)

    OPTIONAL OUTPUT KEYWORD:
      'ExtCurve' - If this keyword is set, fm_unred will return two arrays.
                  First array is the unreddend flux vector.  Second array is
                  the E(wave-V)/E(B-V) extinction curve, interpolated onto the
                  input wavelength vector.

    EXAMPLE:
       Determine how a flat spectrum (in wavelength) between 1200 A and 3200 A
       is altered by a reddening of E(B-V) = 0.1.  Assume an "average"
       reddening for the diffuse interstellar medium (R(V) = 3.1)

       >>> w = 1200 + arange(40)*50       #Create a wavelength vector
       >>> f = w*0 + 1                    #Create a "flat" flux vector
       >>> fnew = fm_unred(w, f, -0.1)    #Redden (negative E(B-V)) flux vector
       >>> plot(w, fnew)

    NOTES:
       (1) The following comparisons between the FM curve and that of Cardelli,
           Clayton, & Mathis (1989), (see ccm_unred.pro):
           (a) - In the UV, the FM and CCM curves are similar for R < 4.0, but
                 diverge for larger R
           (b) - In the optical region, the FM more closely matches the
                 monochromatic extinction, especially near the R band.
       (2)  Many sightlines with peculiar ultraviolet interstellar extinction
               can be represented with the FM curve, if the proper value of
               R(V) is supplied.
    REQUIRED MODULES:
       scipy, numpy
    REVISION HISTORY:
       Written   W. Landsman        Raytheon  STX   October, 1998
       Based on FMRCurve by E. Fitzpatrick (Villanova)
       Added /LMC2 and /AVGLMC keywords,  W. Landsman   August 2000
       Added ExtCurve keyword, J. Wm. Parker   August 2000
       Assume since V5.4 use COMPLEMENT to WHERE  W. Landsman April 2006
       Ported to Python, C. Theissen August 2012
    '''

    from scipy.interpolate import CubicSpline

    lmc2_set, avglmc_set, extcurve_set = None, None, None
    R_V, gamma, x0, c1, c2, c3, c4 = None, None, None, None, None, None, None

    x = 10000. / numpy.array([wave])  # Convert to inverse microns
    curve = x * 0.

    for arg in args:
        if arg.lower() == 'lmc2':
            lmc2_set = 1
        if arg.lower() == 'avglmc':
            avglmc_set = 1
        if arg.lower() == 'extcurve':
            extcurve_set = 1

    for key in kwargs:
        if key.lower() == 'r_v':
            R_V = kwargs[key]
        if key.lower() == 'x0':
            x0 = kwargs[key]
        if key.lower() == 'gamma':
            gamma = kwargs[key]
        if key.lower() == 'c4':
            c4 = kwargs[key]
        if key.lower() == 'c3':
            c3 = kwargs[key]
        if key.lower() == 'c2':
            c2 = kwargs[key]
        if key.lower() == 'c1':
            c1 = kwargs[key]

    if R_V == None:
        R_V = 3.1

    if lmc2_set == 1:
        if x0 == None: x0 = 4.626
        if gamma == None: gamma =  1.05	
        if c4 == None: c4 = 0.42   
        if c3 == None: c3 = 1.92	
        if c2 == None: c2 = 1.31
        if c1 == None: c1 = -2.16
    elif avglmc_set == 1:
        if x0 == None: x0 = 4.596  
        if gamma == None: gamma = 0.91
        if c4 == None: c4 = 0.64  
        if c3 == None: c3 =  2.73	
        if c2 == None: c2 = 1.11
        if c1 == None: c1 = -1.28
    else:
        if x0 == None: x0 = 4.596  
        if gamma == None: gamma = 0.99
        if c4 == None: c4 = 0.41
        if c3 == None: c3 =  3.23	
        if c2 == None: c2 = -0.824 + 4.717 / R_V
        if c1 == None: c1 = 2.030 - 3.007 * c2
    
    # Compute UV portion of A(lambda)/E(B-V) curve using FM fitting function and 
    # R-dependent coefficients
 
    xcutuv = 10000.0 / 2700.0
    xspluv = 10000.0 / numpy.array([2700.0, 2600.0])
   
    iuv = x >= xcutuv
    iuv_comp = ~iuv

    if len(x[iuv]) > 0: xuv = numpy.concatenate( (xspluv, x[iuv]) )
    else: xuv = xspluv.copy()

    yuv = c1  + c2 * xuv
    yuv = yuv + c3 * xuv**2 / ( ( xuv**2 - x0**2 )**2 + ( xuv * gamma )**2 )

    filter1 = xuv.copy()
    filter1[xuv <= 5.9] = 5.9
    
    yuv = yuv + c4 * ( 0.5392 * ( filter1 - 5.9 )**2 + 0.05644 * ( filter1 - 5.9 )**3 )
    yuv = yuv + R_V
    yspluv = yuv[0:2].copy()                  # save spline points
    
    if len(x[iuv]) > 0: curve[iuv] = yuv[2:len(yuv)]      # remove spline points

    # Compute optical portion of A(lambda)/E(B-V) curve
    # using cubic spline anchored in UV, optical, and IR

    xsplopir = numpy.concatenate(([0], 10000.0 / numpy.array([26500.0, 12200.0, 6000.0, 5470.0, 4670.0, 4110.0])))
    ysplir   = numpy.array([0.0, 0.26469, 0.82925]) * R_V / 3.1
    ysplop   = [numpy.polyval(numpy.array([2.13572e-04, 1.00270, -4.22809e-01]), R_V ), 
                numpy.polyval(numpy.array([-7.35778e-05, 1.00216, -5.13540e-02]), R_V ),
                numpy.polyval(numpy.array([-3.32598e-05, 1.00184, 7.00127e-01]), R_V ),
                numpy.polyval(numpy.array([-4.45636e-05, 7.97809e-04, -5.46959e-03, 1.01707, 1.19456] ), R_V ) ]
    
    ysplopir = numpy.concatenate( (ysplir, ysplop) )
    
    if len(iuv_comp) > 0:
        cubic = CubicSpline(numpy.concatenate( (xsplopir,xspluv) ),
                            numpy.concatenate( (ysplopir,yspluv) ), bc_type='natural')
        curve[iuv_comp] = cubic( x[iuv_comp] )

    # Now apply extinction correction to input flux vector
    curve = ebv * curve[0]
    flux = flux * 10.**(0.4 * curve)
    if extcurve_set is None:
        return flux
    else:
        ExtCurve = curve/ebv - R_V
        return flux, ExtCurve


def af99(lam, rv=3.1):
    """Return A(lam) from F99

    Returns the extinction A(lam) at wavelength lam (angstroms) from the
    Fitzpatrick (1999) prescription.  A(lam) is normalized to 1 at 5420 A
    (~V).  Note this normalization is arbitrary!

    Args:
        lam: wavelengths at which A(lam) should be evaluated, angstroms
        rv: R_V parameter of the reddening curve

    Returns:
        A(lam, rv)/A(5420 A, rv)
    """
    lam = numpy.atleast_1d(lam)

    flux = numpy.ones_like(lam)
    rflux = fm_unred(lam, flux, -1./rv, R_V=rv)
    ext = -2.5*numpy.log10(rflux/flux)
    ext5420 = -2.5*numpy.log10(fm_unred(5420, 1, -1./rv, R_V=rv))
    return ext/ext5420

# Optical/NIR coefficients from Cardelli (1989)
c1_ccm = [1., 0.17699, -0.50447, -0.02427, 0.72085, 0.01979, -0.77530, 0.32999]
c2_ccm = [0., 1.41338, 2.28305, 1.07233, -5.38434, -0.62251, 5.30260, -2.09002]

# Optical/NIR coefficents from O'Donnell (19accm94)
c1_odonnell = [1., 0.104, -0.609, 0.701, 1.137, -1.718, -0.827, 1.647, -0.505]
c2_odonnell = [0., 1.952, 2.908, -3.989, -7.985, 11.102, 5.491, -10.805, 3.347]


def extinction_ccm(wavelength, a_v=None, ebv=None, r_v=3.1,
                   optical_coeffs='ccm'):
    r"""The Cardelli, Clayton, and Mathis (1989) extinction function.

    This function returns the total extinction A(\lambda) at the given
    wavelengths, given either the total V band extinction `a_v` or the
    selective extinction E(B-V) `ebv`, where `a_v = r_v * ebv`.

    Adopted from Kyle Barbary's sncosmo.

    Parameters
    ----------
    wavelength : float or list_like
        Wavelength(s) in Angstroms. Values must be between 909.1 and 33,333.3,
        the range of validity of the extinction curve paramterization.
    a_v or ebv: float
        Total V band extinction or selective extinction E(B-V) (must specify
        exactly one).
    r_v : float, optional
        Ratio of total to selective extinction: R_V = A_V / E(B-V).
        Default is 3.1.
    optical_coeffs : {'odonnell', 'ccm'}, optional
        If 'odonnell' (default), use the updated parameters for the optical
        given by O'Donnell (1994) [2]_. If 'ccm', use the original paramters
        given by Cardelli, Clayton and Mathis (1989) [1]_.

    Returns
    -------
    extinction_ratio : float or `~numpy.ndarray`
        Ratio of total to selective extinction: A(wavelengths) / E(B - V)
        at given wavelength(s).

    Notes
    -----
    In [1]_ the mean :math:`R_V`-dependent extinction law, is parameterized
    as

    .. math::

       <A(\lambda)/A_V> = a(x) + b(x) / R_V

    where the coefficients a(x) and b(x) are functions of
    wavelength. At a wavelength of approximately 5494.5 Angstroms (a
    characteristic wavelength for the V band), a(x) = 1 and b(x) = 0,
    so that A(5494.5 Angstroms) = A_V. This function returns

    .. math::

       A(\lambda) = A_V * (a(x) + b(x) / R_V)

    where `A_V` can either be specified directly or via `E(B-V)`
    (by defintion, `A_V = R_V * E(B-V)`). The flux transmission fraction
    as a function of wavelength can then be obtained by

    .. math::

       T(\lambda) = (10^{-0.4 A(\lambda)})

    The extinction scales linearly with `a_v` or `ebv`, so one can compute
    t ahead of time for a given set of wavelengths with `a_v=1` and then
    scale by `a_v` later:
    `t_base = 10 ** (-0.4 * extinction_ccm(wavelengths, a_v=1.))`, then later:
    `t = numpy.power(t_base, a_v)`. Similarly for `ebv`.

    For an alternative to the CCM curve, see the extinction curve
    given in Fitzpatrick (1999) [6]_.

    **Notes from the IDL routine CCM_UNRED:**

    1. The CCM curve shows good agreement with the Savage & Mathis (1979)
       [5]_ ultraviolet curve shortward of 1400 A, but is probably
       preferable between 1200 and 1400 A.
    2. Many sightlines with peculiar ultraviolet interstellar extinction
       can be represented with a CCM curve, if the proper value of
       R(V) is supplied.
    3. Curve is extrapolated between 912 and 1000 A as suggested by
       Longo et al. (1989) [3]_.
    4. Valencic et al. (2004) [4]_ revise the ultraviolet CCM
       curve (3.3 -- 8.0 um-1).  But since their revised curve does
       not connect smoothly with longer and shorter wavelengths, it is
       not included here.


    References
    ----------
    .. [1] Cardelli, Clayton, and Mathis 1989, ApJ, 345, 245
    .. [2] O'Donnell 1994, ApJ, 422, 158
    .. [3] Longo et al. 1989, ApJ, 339,474
    .. [4] Valencic et al. 2004, ApJ, 616, 912
    .. [5] Savage & Mathis 1979, ARA&A, 17, 73
    .. [6] Fitzpatrick 1999, PASP, 111, 63
    """

    if (a_v is None) and (ebv is None):
        raise ValueError('Must specify either a_v or ebv')
    if (a_v is not None) and (ebv is not None):
        raise ValueError('Cannot specify both a_v and ebv')
    if a_v is None:
        a_v = r_v * ebv

    wavelength = numpy.asarray(wavelength)
    in_ndim = wavelength.ndim

    x = 1.e4 / wavelength.ravel()  # Inverse microns.
    if ((x < 0.3) | (x > 11.)).any():
        raise ValueError("extinction only defined in wavelength range"
                         " [909.091, 33333.3].")

    a = numpy.empty(x.shape, dtype=numpy.float)
    b = numpy.empty(x.shape, dtype=numpy.float)

    # Infrared
    idx = x < 1.1
    if idx.any():
        a[idx] = 0.574 * x[idx] ** 1.61
        b[idx] = -0.527 * x[idx] ** 1.61

    # Optical/NIR
    idx = (x >= 1.1) & (x < 3.3)
    if idx.any():
        xp = x[idx] - 1.82

        if optical_coeffs == 'odonnell':
            c1, c2 = c1_odonnell, c2_odonnell
        elif optical_coeffs == 'ccm':
            c1, c2 = c1_ccm, c2_ccm
        else:
            raise ValueError('Unrecognized optical_coeffs: {0!r}'
                             .format(optical_coeffs))

        # we need to flip the coefficients, because in polyval
        # c[0] corresponds to x^(N-1), but above, c[0] corresponds to x^0
        a[idx] = numpy.polyval(numpy.flipud(c1), xp)
        b[idx] = numpy.polyval(numpy.flipud(c2), xp)

    # Mid-UV
    idx = (x >= 3.3) & (x < 8.)
    if idx.any():
        xp = x[idx]
        a[idx] = 1.752 - 0.316 * xp - 0.104 / ((xp - 4.67)**2 + 0.341)
        b[idx] = -3.090 + 1.825 * xp + 1.206 / ((xp - 4.67)**2 + 0.263)

    idx = (x > 5.9) & (x < 8.)
    if idx.any():
        xp = x[idx] - 5.9
        a[idx] += -0.04473 * xp**2 - 0.009779 * xp**3
        b[idx] += 0.2130 * xp**2 + 0.1207 * xp**3

    # Far-UV
    idx = x >= 8.
    if idx.any():
        xp = x[idx] - 8.
        c1 = [-1.073, -0.628,  0.137, -0.070]
        c2 = [13.670,  4.257, -0.420,  0.374]
        a[idx] = numpy.polyval(numpy.flipud(c1), xp)
        b[idx] = numpy.polyval(numpy.flipud(c2), xp)

    extinction = (a + b / r_v) * a_v
    if in_ndim == 0:
        return extinction[0]
    return extinction


def accm(lam, rv=3.1, **kw):
    out = numpy.zeros(len(lam), dtype='f8')
    lbd = 908
    ubd = 33333
    moutofrange = (lam < lbd) | (lam > ubd)
    normalization = extinction_ccm(5420., a_v=True, r_v=rv, **kw)
    out[~moutofrange] = (extinction_ccm(lam[~moutofrange], a_v=True,
                                        r_v=rv, **kw) /
                         normalization).astype('f4')
    if numpy.any(lam < lbd):
        m = lam < lbd
        out[m] = (af99(lam[m], rv=rv)/af99(lbd, rv=rv) *
                  (extinction_ccm(lbd, a_v=True, r_v=rv, **kw)/normalization)).astype('f4')
    if numpy.any(lam > ubd):
        m = lam > ubd
        out[m] = (ext_f09(lam[m], rv=rv)/ext_f09(ubd, rv=rv) *
                  (extinction_ccm(ubd, a_v=True, r_v=rv, **kw)/normalization)).astype('f4')
    return out


def aplaw(lam, alpha=2., rv=None):
    """Power law extinction curve: lam^-alpha"""
    return lam**(-alpha)/5420.**(-alpha)


def af04(lam, rv=3.1):
    self = af04
    if not getattr(self, 'csfits', None):
        self.csfits = {}
    cs = self.csfits.get(rv, None)
    if not cs:
        fname = os.environ['REDSTDDUST_DIR']+'/f04/F04_CURVE_%3.1f.dat' % rv
        curve = numpy.genfromtxt(fname, skip_header=14, comments='%',
                                 dtype=[('oneoverlam', 'f4'),
                                        ('klamminusv', 'f4')])
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(curve['oneoverlam'],
                         curve['klamminusv']+rv)
        self.csfits[rv] = cs
    return cs(10000./lam)/cs(10000./5420.)


def ala5495(lam, rv=3.1):
    """Return A(lam) from Maiz Apellaniz et al. (2014) extinction laws.

    Returns the extinction A(lam) at wavelength lam (angstroms) from the
    Maiz Apellaniz et al. (2014) extinction laws.  A(lam) is normalized
    to 1 at 5495 A (~V).  Note that this normalization is arbitrary!

    Args:
        lam: wavelengths at which A(lam) should be evaluated, angstroms
        rv: R_V (R_5495) parameter of the reddening curve

    Returns:
        A(lam, rv)/A(5495, rv)
    """
    from scipy.interpolate import CubicSpline
    if len(numpy.atleast_1d(lam)) == 0:
        return lam
    x = 10000./lam
    if numpy.min(x) < 0.3 or numpy.max(x) > 10:
        raise ValueError('Wavelength not implemented.')
    # Infrared
    ai = 0.574*x**1.61
    bi = -0.527*x**1.61
    # Optical
    x1 = numpy.array([1.0])
    xi1 = x1[0]
    x2 = numpy.array([1.15,1.81984,2.1,2.27015,2.7])
    x3 = numpy.array([3.5 ,3.9 ,4.0,4.1 ,4.2])
    xi3 = x3[len(x3)-1]
    a1v = 0.574 *x1**1.61
    a1d = 0.574*1.61*xi1**0.61
    b1v = -0.527 *x1**1.61
    b1d = -0.527*1.61*xi1**1.61
    a2v = (1 + 0.17699*(x2-1.82) - 0.50447*(x2-1.82)**2 - 0.02427*(x2-1.82)**3 + 0.72085*(x2-1.82)**4
           + 0.01979*(x2-1.82)**5 - 0.77530*(x2-1.82)**6 + 0.32999*(x2-1.82)**7) + numpy.array([0.0,0.0,-0.011,0.0,0.0])
    b2v = (1.41338*(x2-1.82) + 2.28305*(x2-1.82)**2 + 1.07233*(x2-1.82)**3 - 5.38434*(x2-1.82)**4
           - 0.62251*(x2-1.82)**5 + 5.30260*(x2-1.82)**6 - 2.09002*(x2-1.82)**7 + numpy.array([0.0,0.0,+0.091,0.0,0.0]))
    a3v = 1.752 - 0.316*x3 - 0.104/ (( x3-4.67)*( x3-4.67) + 0.341) + numpy.array([0.442,0.341,0.130,0.020,0.000])
    a3d = -0.316 + 0.104*2*(xi3-4.67)/((xi3-4.67)*(xi3-4.67) + 0.341)**2
    b3v = -3.090 + 1.825*x3 + 1.206/ (( x3-4.62)*( x3-4.62) + 0.263) - numpy.array([1.256,1.021,0.416,0.064,0.000])
    b3d =  1.825 - 1.206*2*(xi3-4.62)/((xi3-4.62)*(xi3-4.62) + 0.263)**2
    acs = CubicSpline(numpy.concatenate([x1, x2, x3]), numpy.concatenate([a1v, a2v, a3v]), bc_type=((1, a1d), (1, a3d)))
    bcs = CubicSpline(numpy.concatenate([x1, x2, x3]), numpy.concatenate([b1v, b2v, b3v]), bc_type=((1, b1d), (1, b3d)))
    av = acs(x)
    bv = bcs(x)
    # Ultraviolet
    y = x - 5.9
    fa = (-0.04473*y**2 - 0.009779*y**3)*((x <= 8.0) & (x >= 5.9))
    fb = (  0.2130*y**2 + 0.1207  *y**3)*((x <= 8.0) & (x >= 5.9))
    au =  1.752 - 0.316*x - 0.104/((x-4.67)*(x-4.67) + 0.341) + fa
    bu = -3.090 + 1.825*x + 1.206/((x-4.62)*(x-4.62) + 0.263) + fb
    # Far ultraviolet
    y = x - 8.0
    af = -1.073 - 0.628*y + 0.137*y**2 - 0.070*y**3
    bf = 13.670 + 4.257*y - 0.420*y**2 + 0.374*y**3
    # Final result
    a = ai*(x < xi1) + av*((x >= xi1) & (x < xi3)) + au*((x >= xi3) & (x < 8.0)) + af*(x >= 8.0)
    b = bi*(x < xi1) + bv*((x >= xi1) & (x < xi3)) + bu*((x >= xi3) & (x < 8.0)) + bf*(x >= 8.0)
    return a + b/rv


def ext_f09(lam, rv=3.1, alpha=2.4, join=6000):
    """Return A(lam) from Fitzpatrick & Massa (2009) extinction law.

    Returns the extinction A(lam) at wavelength lam (angstroms) from the
    Fitzpatrick & Massa (2009) extinction law.  A(lam) is normalized to 1
    at 5420 A (~V).  Note that this normalization is arbitrary!

    Args:
        lam: wavelengths at which A(lam) should be evaluated, angstroms
        rv: R_V parameter of the reddening curve.
        alpha: alpha parameter of the reddening curve (see FM2009); controls
            the slope in the infrared.

    Returns:
        A(lam, rv, alpha)/A(5420, rv, alpha)
    """

    lam = numpy.atleast_1d(lam)
    alam = af99(lam, rv=rv)

    def f09(lam):
        # Fitzpatrick 2009 eq. 5 translated into A_lam/A_V
        return (0.349+2.087*rv)/(1+(lam/10000./0.507)**(alpha))/rv
    alam2 = f09(lam)
    alamj = af99(join, rv=rv)
    alam2j = f09(join)
    alam /= alamj
    alam2 /= alam2j
    if 5420 < join:
        norm = af99(5420, rv=rv)/alamj
    else:
        norm = f09(5420)/alam2j
    m = lam > join
    alam[m] = alam2[m]
    alam /= norm
    return alam


def ext_f09_efstweaks(lam, **kw):
    alam = ext_f09(lam, **kw)
    breakwave = 9000
    m = (lam > breakwave)
    alambreak = ext_f09(breakwave, **kw)
    weight = 1.04
    alam[m] = (weight*alam[m]+(1-weight)*alambreak).astype('f4')
#     m = (lam > 11000) & (lam < 14000)
#     alam[m] = alam[m] * 1.06
#     m = (lam > 14000) & (lam < 19000)
#     alam[m] = alam[m] * 1.17
#     m = (lam > 19000) & (lam < 50000)
#     alam[m] = alam[m] * 1.4
    m = (lam > 30000)
    alam[m] = alam[m] - 0.005
    m = (lam > 9500) & (lam < 11000)
    alam[m] -= 0.012
    # discontinuous, garbage, but let's me play around a bit...
    return alam


def ext_f09_efstweaks2(lam, **kw):
    alam = ext_f09(lam, **kw)
    breakwave = 10000
    m = (lam > breakwave)
    alambreak = ext_f09(breakwave, **kw)
    weight = 0.9
    alam[m] = (weight*alam[m]+(1-weight)*alambreak).astype('f4')
#     m = (lam > 11000) & (lam < 14000)
#     alam[m] = alam[m] * 1.06
#     m = (lam > 14000) & (lam < 19000)
#     alam[m] = alam[m] * 1.17
#     m = (lam > 19000) & (lam < 50000)
#     alam[m] = alam[m] * 1.4
    m = (lam > 30000)
    alam[m] = alam[m] * 0.9
    # discontinuous, garbage, but let's me play around a bit...
    return alam


def ext_ala14(lam, rv=3.1, join=10000./0.3-0.0001):
    """Return A(lam) from Maiz Apellaniz et al. (2014) extinction laws.

    Returns the extinction A(lam) at wavelength lam (angstroms) from the
    Maiz Apellaniz et al. (2014) extinction laws.  A(lam) is normalized
    to 1 at 5420 A (~V).  Note that this normalization is arbitrary!
    Joins to FM09 at 3.3 microns.

    Args:
        lam: wavelengths at which A(lam) should be evaluated, angstroms
        rv: R_V (R_5495) parameter of the reddening curve

    Returns:
        A(lam, rv)/A(5420, rv)
    """
    m = lam < join
    alam = (lam * 0.).astype('f8')
    ala5420 = ala5495(5420, rv=rv)
    alam[m] = ala5495(lam[m], rv=rv)/ala5420
    alam[~m] = ext_f09(lam[~m], rv=rv)
    # now some joining and normalization logic
    alamj1 = ala5495(join, rv=rv)/ala5420
    alamj2 = ext_f09(join, rv=rv)
    alam[~m] = alam[~m] * alamj1/alamj2
    return alam


def ext_ala142(lam, rv=3.1):
    return ext_ala14(lam, rv=rv, join=6900.)


def compute_rf(ws, ss, wf, sf, av, divide_by_av=True, urmag=None,
               reddeninglaw=af99, deriv=False, **kw):
    """Return A_f/E(B-V) for a given spectrum and filter from F99

    Returns the extinction at A_V in a given filter
    (wf, sf) for a given spectrum (ws, ss).  An F99 reddening law with
    given R_V is assumed.

    Args:
        ws: wavelength (angstroms) at which spectrum is tabulated
        ss: flux (energy/unit wavelength/unit time/unit area)
        wf: wavelength (angstroms) at which filter is tabulated
        sf: filter throughput (probability a photon is registered)
        av: the dust column of interest, units A_5420 (mag)
        divide_by_av: flag indicating whether or not A_f should be
                       normalized to A_V
        **kw: extra parameters to pass to reddening law

    Returns:
        A_filter/A_V
    """
    import filters
    if urmag is None and not deriv:
        urmag = filters.intfilter(ws, ss, wf, sf)
    extmag = reddeninglaw(ws, **kw)*av
    med_extmag = numpy.median(extmag)
    extmag -= med_extmag  # get rid of gray component; add back in below
    # makes the algorithm somewhat more robust against underflow
    ssr = ss*10.**(-extmag/2.5)
    rmag = filters.intfilter(ws, ssr, wf, sf)+med_extmag
    if deriv:
        eps = 0.001
        extmag += med_extmag
        extmag *= (1 + eps/av)
        med_extmag = numpy.median(extmag)
        extmag -= med_extmag
        ssr = ss*10.**(-extmag/2.5)
        urmag = filters.intfilter(ws, ssr, wf, sf)+med_extmag
    res = rmag-urmag
    if divide_by_av and not deriv:
        res /= av
    if deriv:
        res /= (-eps)
    return res


def compute_rf_filters(ws, ss, filters, av, urmag=None,
                       specregions=None, **kw):
    if urmag is None:
        urmag = [None]*len(filters)
    if specregions is None:
        specregions = [[0, len(ws)] for f in filters]
    f = filters
    return numpy.array([compute_rf(ws[specregions[i][0]:specregions[i][1]],
                                   ss[specregions[i][0]:specregions[i][1]],
                                   f[i][0], f[i][1], av,
                                   urmag=urmag[i], **kw)
                        for i in xrange(len(filters))])


def construct_reddening_grid(reddeninglaw, lam=None, rv=None):
    if lam is None:
        lam = 3000+numpy.arange((60000-3000)/100+1, dtype='f4')*100
    if rv is None:
        rv = 0.01+numpy.arange(20/0.01, dtype='f4')*0.01
    out = []
    for rv0 in rv:
        out.append(reddeninglaw(lam, rv=rv0))
    return lam, rv, numpy.vstack(out)


def construct_reddening_grid_alpha(reddeninglaw, lam=None, alpha=None):
    if lam is None:
        lam = 3000+numpy.arange((60000-3000)/100+1, dtype='f4')*100
    if alpha is None:
        alpha = 0.01+numpy.arange(5/0.01, dtype='f4')*0.01
    out = []
    for alpha0 in alpha:
        out.append(reddeninglaw(lam, alpha=alpha0))
    return lam, alpha, numpy.vstack(out)


def ext_grid(lam, rv=3.1, grid=None):
    if grid is None:
        raise ValueError('Must set grid.')
    rvpix = (numpy.interp(rv, grid[1], numpy.arange(len(grid[1]))) *
             numpy.ones(len(lam)))
    lampix = numpy.interp(lam, grid[0], numpy.arange(len(grid[0])))
    return map_coordinates(grid[2], [rvpix, lampix], order=1, mode='nearest')


def read_ext_fm04_files():
    files = os.listdir(os.path.join(os.environ['DUST_DIR'], '../fm04'))
    import re
    from scipy.interpolate import CubicSpline
    splines = {}
    for f in files:
        match = re.match('F04_CURVE_([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)\.dat', f)
        if match is None:
            continue
        rv = float(match.group(1))
        curve = numpy.genfromtxt(os.path.join(os.environ['DUST_DIR'], '../fm04', f),
                                 skip_header=14, dtype=[('ilam', 'f4'), ('elammvebv', 'f4')],
                                 comments='%')
        splines[rv] = CubicSpline(curve['ilam'], curve['elammvebv']+rv)
    return splines


def make_ext_fm04_grid(lam=None):
    splines = read_ext_fm04_files()
    if lam is None:
        lam = 3000+numpy.arange((60000-3000)/100+1, dtype='f4')*100
    rv = numpy.sort(splines.keys())
    out = []
    for rv0 in rv:
        spl = splines[rv0]
        out.append(spl(10000./lam)/spl(10000./5420))
    return lam, rv, numpy.vstack(out)
