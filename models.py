"""Model classes for 3D density distribution."""

__author__  = "Robert Nikutta, Claudia Agliozzo"
__version__ = "2015-05-02"

import numpy as N

#######################################
# Models for 3D density distributions #
#######################################

class TruncatedNormalShell:

    def __init__(self,r,width,clip=(0.,1.),offsets=(0.,0.),weight=1.):
        
        """Truncated Normal Shell

        A spherical shell with radius 'r', and Gaussian density
        fall-off from r. The width of the Normal is 'width'. The PDF
        of the Normal is truncated at 'clip' values.
        
        Parameters:
        -----------
        r : float
           Radius at which the shell is centered, in fractions of
           unity, i.e. between 0 and 1.

        width : float
           Thickness of the shell, in same units as r.

        clip : 2-tuple of floats
           Where to clip the Gaussian left and right. Default is (0,1).

        offsets : 2-tuple of floats
           x and y offsets of the shall center from (0.,0). Positive
           values are to the right and up, negative to the left and
           down. In units if unity (remember that the image is within
           [-1,1]. Default offsets: (0.,0.)

        weight : float
           Normalize the total (relative) mass contained in the shell
           to this value. The total mass is the sum of rho over all
           pixels (in 3D). This is useful if you e.g. want to have
           more than one shell, and wish to distribute different
           amounts of mass inside each one.

        """

        self.r = r
        self.width = width
        self.clipa, self.clipb = clip
        self.deltax, self.deltay = offsets
        self.weight = weight

        self.sanity()

#        self.parameters = {'r',)

    def __call__(self,x,y,z):

        """Return density rho at (x,y,z)"""

#        self.rho = self.get_rho_in_voxel(x,y,z)
        self.rho = self.get_rho(x,y,z)
        self.normalize()


    def sanity(self):

        assert (self.clipa >= 0.)
        assert (self.clipb <= 1.)
        assert (self.clipb > self.clipa)
        assert (self.width >= 0.)


    def normalize(self):

        self.mass = self.rho.sum() / self.weight
        self.rho /= self.mass

    
    def get_rho(self,x,y,z):

        r = get_r((x-self.deltax,y-self.deltay,z),mode=2)  # mode=1,2 are fast, 3,4are slow

        return self.get_pdf(r)


    def get_pdf(self,x):

        from scipy.stats import truncnorm

        # Because of the non-standard way that Scipy defines
        # distributions, we compute the shape parameters for a
        # truncated Normal, with mean mu, std-deviation sigma, and
        # clipped left and right at clipa and clipb.
        mu, sig, clipa, clipb = self.r, self.width, self.clipa, self.clipb
        a, b = (clipa - mu) / sig, (clipb - mu) / sig
        rv = truncnorm(a, b, loc=mu, scale=sig)
        pdf = rv.pdf(x)

        return pdf


class HardEdgeShell:

    def __init__(self,rin,width,offsets=(0.,0.),weight=1.):
        
        """Truncated Normal Shell

        A spherical shell with radius 'r', and Gaussian density
        fall-off from r. The width of the Normal is 'width'. The PDF
        of the Normal is truncated at 'clip' values.
        
        Parameters:
        -----------
        rin : float
           Radius at which the shell is centered, in fractions of
           unity, i.e. between 0 and 1.

        width : float
           Thickness of the shell, in same units as r.

        offsets : 2-tuple of floats
           x and y offsets of the shall center from (0.,0). Positive
           values are to the right and up, negative to the left and
           down. In units if unity (remember that the image is within
           [-1,1]. Default offsets: (0.,0.)

        weight : float
           Normalize the total (relative) mass contained in the shell
           to this value. The total mass is the sum of rho over all
           pixels (in 3D). This is useful if you e.g. want to have
           more than one shell, and wish to distribute different
           amounts of mass inside each one.

        """

        self.rin = rin
        self.width = width
        self.rout = self.rin + self.width
        self.deltax, self.deltay = offsets
        self.weight = weight

        self.sanity()

#        self.parameters = {'r',)

    def __call__(self,x,y,z):

        """Return density rho at (x,y,z)"""

#        self.rho = self.get_rho_in_voxel(x,y,z)
        self.rho = self.get_rho(x,y,z)
        self.normalize()


    def sanity(self):

        assert (self.width > 0.)


    def normalize(self):

        self.mass = self.rho.sum() / self.weight
        self.rho /= self.mass

    
    def get_rho(self,x,y,z):

        r = get_r((x-self.deltax,y-self.deltay,z),mode=2)  # mode=1,2 are fast, 3,4are slow
        co = (r >= self.rin) & (r <= self.rout)
        res = N.zeros(r.shape)
        res[co] = 1.

        return res



class HardEdgeShell2:

# TODO: write this class

    def __init__(self,r,dr):
        
        """Hard Edge Shell
        
        Parameters:
        -----------
        r : float
           Outer radius of the shell, in fractions of unity, i.e. between 0 and 1.

        dr : float
           Radial thickness of the shell, in same units as r.

        """
        
        self.rout, self.dr = r, dr
        self.sanity()
        self.rin = self.rout - self.dr

    def sanity(self):

        assert (self.rout > 0. and self.rout <= 1.), "Shell radius must be within (0,1.]"
        assert (self.dr > 0. and self.dr <= 1.), "Shell thickness must be positive and at most = r"


    def get_rho(self,x,y,z):

        r, theta, phi = cartesian2spherical(x,y,z)


####################
# Helper functions #
####################

def mirror(a,ax=-1):
    
    """Expand a 3D cube by mirroring it along one axis.

    The dimension to mirror must have odd number of cells, and the
    last column of cells will not be mirrored, i.e. the mirroring
    happens 'about the last colum'.

    Example: a 3-by-3 array is mirrored 3 consecutive times, with the
    axis specified each tim by the 'ax' argument:

                                      XXXXX           XXXXXXXXX
         XXX   ax=-1   XXXXX   ax=0   XXXXX   ax=1    XXXXXXXXX
         XXX --------> XXXXX -------> XXXXX ------->  XXXXXXXXX
         XXX           XXXXX          XXXXX           XXXXXXXXX
                                      XXXXX           XXXXXXXXX

    Note that for a 2D array ax=-1 and ax=1 are equivalent.

    """

    pass  # write it


def get_r(coords,mode=2): # mode=1):

    """Get Euclidean distance r from Cartesian coordinates x,y,[[z],..]

    Parameters:
    -----------
    
    coords : seq
        'coords' is a sequence of Cartesian coordinates, of arbitrary
        dimensions (e.g. (x,y), or (x,y,z), etc.), from which the
        Euclidean distance r will be computed.

        x,y,z,... can be single values, 1D arrays, 3D arrays, etc.

    mode : int
        Several implementations for computing r are provided. Those
        which use numpy.sum() seem to suffer from speed loss. mode=1
        or 2 is recommended. mode=2 is the default.

    """

    if mode == 1:

        """Fast outer loop, in-place summation."""
   
        shape_ = coords[0].shape

        def square(a):
            for x in N.nditer(a, flags=['external_loop'], order='C'):
                aux = x**2

            return aux

        res = 0
        for c in coords:
            res += square(c)

        return N.sqrt(res.reshape(shape_))

    elif mode == 2:

        """Explicit summation"""

        res = 0
        for c in coords:
            res += c**2

        return N.sqrt(res)

    elif mode == 3:

        """Numpy's sum() is slow, if not paying attention to RAM layout."""

        return N.sqrt(N.sum([c**2 for c in coords],axis=0))

    elif mode == 4:

        """Numpy's sum() is slow, if not paying attention to RAM layout."""

        return N.sqrt(N.sum(N.power(coords,2),axis=0))


def cartesian2spherical(x,y,z):

    r = N.sqrt(x**2+y**2+z**2)
    theta = N.arctan2(N.sqrt(x**2+y**2)/z)
    phi = N.arctan2(y/x)

    return r, theta, phi
    

