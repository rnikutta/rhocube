"""Model classes for 3D density distribution."""

__author__  = "Robert Nikutta, Claudia Agliozzo"
__version__ = "2016-03-10"

import numpy as N

from numpy import cross,eye,dot
from scipy import ndimage
from scipy.linalg import expm3,norm
from rhocube import Cube

#######################################
# Models for 3D density distributions #
#######################################

class TruncatedNormalShell(Cube):

    def __init__(self,npix,transform=None,buildkdtree=False,computeR=True):
        
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

        xoff, yoff : floats
           x and y offsets of the shell center from (0,0). Positive
           values are to the right and up, negative to the left and
           down. In units if unity (remember that the image is within
           [-1,1]. Defaults: xoff = yoff = 0.

        weight : float
           Normalize the total (relative) mass contained in the shell
           to this value. The total mass is the sum of rho over all
           pixels (in 3D). This is useful if you e.g. want to have
           more than one component, and wish to distribute different
           amounts of mass inside each one.

        """

        Cube.__init__(self,npix,transform=transform,buildkdtree=buildkdtree,computeR=computeR)

        
    def __call__(self,r,width,clipa=0.,clipb=1.,xoff=0.,yoff=0.,weight=1.,smooth=1.):

        """Return density rho at (x,y,z)"""

        self.r = r
        self.width = width
        self.clipa = clipa
        self.clipb = clipb
        self.xoff = xoff
        self.yoff = yoff
        self.weight = weight

        self.sanity()

        self.get_rho()
        self.shift()

#        self.normalize()

        return self.rho


    def sanity(self):

        # CAREFUL ASSERTIONS
        # lower cut clipa must be smaller than r
        # lower cut clipa can be as small as zero
        # upper cut clipb can be as low as r
        # upper cub clipb can be in principle larger than unity (but we'll default to 1.0)
        # width must be a positive number
        assert (0. < self.clipa < self.r < self.clipb), "self.clipa = %.3f, self.r = %.3f, self.clipb = %.3f" % (self.clipa,self.r,self.clipb)  # radial distance relations that must hold: 0. <= clipa < r < clipb [<= 1.]
        assert (self.width > 0.), "self.width = %.3f" % self.width


    def get_rho(self):

        """Compute rho(x,y,z) in every voxel.
        """

        self.rho = self.get_pdf(self.R)


    def get_pdf(self,x):

        """Distribution of density according to a Gaussian with (mu,sig) =
        (r,width).
        """

        from scipy.stats import truncnorm

        # Because of the non-standard way that Scipy defines
        # distributions, we compute the shape parameters for a
        # truncated Normal, with mean mu, std-deviation sigma, and
        # clipped left and right at clipa and clipb.
        mu, sig = self.r, self.width
        a, b = (self.clipa - mu) / sig, (self.clipb - mu) / sig
        rv = truncnorm(a, b, loc=mu, scale=sig)
        pdf = rv.pdf(x)

        return pdf
