import os, pywcs, pyfits, numpy
from scipy.ndimage import map_coordinates
import pdb

def get_red_fac(version=-1, ps=False, lsst=False): # dumb interface
    self = get_red_fac
    if (version == -1) and getattr(self, 'rf', None) is not None:
        return self.rf
    if version == 0:
        rf = {'u':5.155, 'g':3.793, 'r':2.751, 'i':2.086, 'z':1.479, 'y':1.}
    else:
        #this is version from IDL
        # that was trained up on blue tip stuff, forcing z band to agree with F99
        # but getting all other information from blue tip
        # we are replacing that here with the S10 F99-based table
        #rf = {'u':4.292, 'g':3.286, 'r':2.282, 'i':1.714, 'z':1.266, 'y':1.}
        rf  = {'u':4.239, 'g':3.303, 'r':2.285, 'i':1.698, 'z':1.263 }
    if ps:
        rf = {'g':3.172, 'r':2.271, 'i':1.682, 'z':1.322, 'y':1.087 }

    return rf

def set_red_fac(red_fac=None, mode=None, rv=3.1):
    if red_fac is not None and mode is not None:
        raise ValueError('Must set only one of red_fac and mode')
    if red_fac is None and mode is None:
        raise ValueError('Must set one of red_fac and mode')
    if red_fac is not None:
        get_red_fac.rf = red_fac
        return
    if mode == 'lsst':
        get_red_fac.rf = {'u':4.145, 'g':3.237, 'r':2.273, 'i':1.684,
                          'z':1.323, 'y':1.088}
    elif mode == 'ps':
        if rv == 3.1:
            get_red_fac.rf = {'g':3.172, 'r':2.271, 'i':1.682, 'z':1.322,
                              'y':1.087}
        elif rv == 2.1:
            get_red_fac.rf = {'g':3.634, 'r':2.241, 'i':1.568, 'z':1.258,
                              'y':1.074}
        elif rv == 4.1:
            get_red_fac.rf = {'g':2.958, 'r':2.284, 'i':1.734, 'z':1.352,
                              'y':1.094}
        elif rv == 5.1:
            get_red_fac.rf = {'g':2.835, 'r':2.292, 'i':1.765, 'z':1.369,
                              'y':1.097}
        else:
            raise Exception('Do not have that R_V ready.')
            
    elif mode == 'sdss':
        get_red_fac.rf = get_red_fac(version=1)
    else:
        raise Exception('bug!')
    if rv != 3.1 and mode != 'ps':
        raise Exception('Only have the different R_Vs for PS1')

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
    maxsize = { 'd100':1024, 'dust':4096, 'i100':4096, 'i60':4096,
                'mask':4096 }
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
            hdulist = pyfits.open(fname+'_%s.fits' % pole)
            wcs = pywcs.WCS(hdulist[0].header)
            x, y = wcs.wcs_sky2pix(l[m], b[m], 0)
            out[m] = map_coordinates(hdulist[0].data, [y, x], order=order, mode='nearest')
    return out

def wgetval(l, b, **kw):
    import os, sys
    from lsd.builtins.misc import galequ, equgal
    if os.environ.get('WISE_TILE', None) is None:
         os.environ['WISE_TILE'] = '/n/fink1/ameisner/tile-combine'
         sys.path.append('/n/home09/ameisner/wssa_utils/python')
    import wssa_utils
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
    
    # Import needed modules
    import cubicspline

    # Set defaults
    lmc2_set, avglmc_set, extcurve_set = None, None, None
    R_V, gamma, x0, c1, c2, c3, c4 = None, None, None, None, None, None, None
    
    x = 10000. / numpy.array([wave])                # Convert to inverse microns
    curve = x * 0.

    # Read in keywords
    for arg in args:
        if arg.lower() == 'lmc2': lmc2_set = 1
        if arg.lower() == 'avglmc': avglmc_set = 1
        if arg.lower() == 'extcurve': extcurve_set = 1
        
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

    if R_V == None: R_V = 3.1

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
        cubic = cubicspline.CubicSpline(numpy.concatenate( (xsplopir,xspluv) ),
                       numpy.concatenate( (ysplopir,yspluv) ))
        curve[iuv_comp] = cubic( x[iuv_comp] )

    # Now apply extinction correction to input flux vector
    curve = ebv * curve[0]
    flux = flux * 10.**(0.4 * curve)
    if extcurve_set == None:
        return flux
    else:
        ExtCurve = curve - R_V
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

def af04(lam, rv=3.1):
    self = af04
    if not getattr(self, 'csfits', None):
        self.csfits = { }
    cs = self.csfits.get(rv, None)
    if not cs:
        fname = os.environ['REDSTDDUST_DIR']+'/f04/F04_CURVE_%3.1f.dat' % rv
        curve = numpy.genfromtxt(fname, skip_header=14, comments='%',
                                 dtype=[('oneoverlam', 'f4'),
                                        ('klamminusv', 'f4')])
        import cubicspline
        cs = cubicspline.CubicSpline(curve['oneoverlam'],
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
    import cubicspline
    if len(numpy.atleast_1d(lam)) == 0:
        return lam
    x = 10000./lam
    if numpy.min(x) < 0.3  or numpy.max(x) > 10:
        raise ValueError('Wavelength not implemented.')
    # Infrared
    ai =  0.574*x**1.61
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
    acs = cubicspline.CubicSpline(numpy.concatenate([x1, x2, x3]), numpy.concatenate([a1v, a2v, a3v]), yp=[a1d, a3d])
    bcs = cubicspline.CubicSpline(numpy.concatenate([x1, x2, x3]), numpy.concatenate([b1v, b2v, b3v]), yp=[b1d, b3d])
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
    alam  /= alamj
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
    #eps = 0.0001
    #join = 10000./0.3-eps
    m = lam < join
    alam = (lam * 0.).astype('f8')
    ala5420 = ala5495(5420, rv=rv)
    alam[m] = ala5495(lam[m], rv=rv)/ala5420
    alam[~m] = ext_f99(lam[~m], rv=rv)
    # now some joining and normalization logic
    alamj1 = ala5495(join, rv=rv)/ala5420
    alamj2 = ext_f99(join, rv=rv)
    alam[~m] = alam[~m] * alamj1/alamj2
    return alam

def ext_ala142(lam, rv=3.1):
    return ext_ala14(lam, rv=rv, join=6900.)

def compute_rf(ws, ss, wf, sf, av, divide_by_av=True, urmag=None,
               reddeninglaw=af99, **kw):
    """Return A_f/E(B-V) for a given spectrum and filter from F99

    Returns the extinction per E(B-V) (actually in units of dust column,
    scaled to E(B-V) in the low extinction limit) in a given filter
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
    import filters, util_efs
    if urmag is None:
        urmag = filters.intfilter(ws, ss, wf, sf)
    extmag = reddeninglaw(ws, **kw)*av
    med_extmag = numpy.median(extmag)
    extmag -= med_extmag # get rid of gray component; add back in below
    # makes the algorithm somewhat more robust against underflow
    ssr = ss*10.**(-extmag/2.5)
    rmag = filters.intfilter(ws, ssr, wf, sf)+med_extmag
    res = rmag-urmag
    #print res, urmag, rmag
    if divide_by_av:
        res /= av
    return res

def compute_rf_filters(ws, ss, filters, av, urmag=None,
                       specregions=None, **kw):
    if urmag is None:
        urmag = [None]*len(filters)
    if specregions is None:
        specregions = [ [0, len(ws)] for f in filters ]
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
    rvpix  = numpy.interp(rv, grid[1], numpy.arange(len(grid[1])))*numpy.ones(len(lam))
    lampix = numpy.interp(lam, grid[0], numpy.arange(len(grid[0])))
    return map_coordinates(grid[2], [rvpix, lampix], order=1, mode='nearest')
