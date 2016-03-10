"""Compute a model 3D density distribution, and 2D dz-integrated map."""

__author__  = "Robert Nikutta, Claudia Agliozzo"
__version__ = "2015-05-15"

import numpy as N
import pyfits
#import models
import pymc
import numpy.ma as ma
from numpy import pi, exp, sqrt, array, ndarray, zeros, sign, linspace
import scipy.stats as stats
from scipy.special import erf, erfc
from scipy import spatial, ndimage
from collections import OrderedDict

#R1def get_tail(arg,mode='sigma'):
#R1
#R1    if mode == 'sigma':
#R1        fraction = erf(arg/sqrt(2))
#R1    elif mode == 'fraction':
#R1        fraction = arg
#R1
#R1    tail = (1.0-fraction)/2.
#R1
#R1    return tail
#R1
#R1
#R1def get_quantiles(sample,sigmas=1.):
#R1
#R1    tail = get_tail(sigmas,mode='sigma')
#R1    left, median, right = stats.mstats.mquantiles(sample,prob=[tail,0.5,1.-tail])
#R1
#R1    return left, median, right


        

def get_r(coords,mode=2):

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

    from math import sqrt as msqrt
    
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

        res = 0.
        for c in coords:
            res += c*c

        return N.sqrt(res)

    elif mode == 3:

        """Numpy's sum() is slow, if not paying attention to RAM layout."""

        return N.sqrt(N.sum([c**2 for c in coords],axis=0))

    elif mode == 4:

        """Numpy's sum() is slow, if not paying attention to RAM layout."""

        return N.sqrt(N.sum(N.power(coords,2),axis=0))


class NullTransform:

    def __init__(self):
        pass

    def __call__(self,x):
        return x

    def _inverse(self,x):
        return x
        
    
class Quad:

    def __init__(self):
        pass

    def __call__(self,x):
        return x**2

    def _inverse(self,x):
        return N.sqrt(x)

    
class Power:

    def __init__(self,pow):
        self.pow = float(pow)
        self.invpow = 1./pow

    def __call__(self,x):
        return x**self.pow

    def inverse(self,x):
        return x**self.invpow

    
class Cube:

    """Generic Cube of nx*ny*nz pixels.

    Models should inherit from this class, as it provides common
    members and methods, e.g. normalize(), set_rho(), shift(),
    rotate3d(), and can provide a 3D R cube, as well as a kdtree for
    density fields computed around parametric curves.

    """

    def __init__(self,npix,transform=None,normalize='none',buildkdtree=False,computeR=False):

        """Initialize a 3D cube of voxels using X,Y,Z corrdinate arrays (each
        a 3D array).
        """

        global models
        import models
        
        self.npix = npix
        if self.npix % 2 == 0:   # make sure we have a central pixel, i.e. an odd number of pixels along the axes
            self.npix += 1
            
        self.pixelscale = 2./float(self.npix)

        # center -/+ 1 radius to each side
        self.x = N.linspace(-1.,1,self.npix)

        self.X, self.Y, self.Z = N.meshgrid(self.x,self.x,self.x,indexing='xy')  # three 3d coordinate arrays to

        if computeR is True:
            self.R = get_r((self.X,self.Y,self.Z),mode=2)  # mode=1,2 are fast, 3,4are slow
        
        if buildkdtree is True:
            self.build_kdtree()

        self.ndshape = self.X.shape

        self.extent = (self.x.min(),self.x.max(),self.x.min(),self.x.max())

        self.transform = transform # func ref

        self.rho = self.set_rho(val=0.)
        
        
    def build_kdtree(self):

        print "Building kdtree"
        mat = N.vstack((self.X.flatten(order='F'),self.Y.flatten(order='F'),self.Z.flatten(order='F'))).T
        self.kdtree = spatial.cKDTree(mat)   # kd-tree is in unit hypercell space


    def kdtree_query(self,x,r=0):
        
        """Generalized kd-tree query.

        Either finds nearest point to a vector x in the kd-tree (when
        radius r=0), or all points within a hyper-sphere with radius r
        around point x (when r > 0).

        Parameters
        ----------
        x : array
            n-dimensional point in the hyperspace mapped by self.kdtree

        r : float
            Either 0. or greater than 0. If r=0, finds the single
            closest point to x. If r>0, finds all points with sphere
            of radius r around point x.

        Returns
        -------
        Nothing.

        Stores
        ------
        self.idx1d : array
            1-dimensional array of idices corresponding to points
            found by the query.

        self.dist : float
            The actual distance from x to the closest found
            neighbor. Only returned when r = 0, i.e. single-point
            query.

        """

        if r > 0:
            dist = None
            idx1d = self.kdtree.query_ball_point(x,r)
        else:
            dist, idx1d = self.kdtree.query(x)
        
        idx1d = N.atleast_1d(idx1d)

        return dist, idx1d

    
    def set_rho(self,val=0.):

        """(Re)set all voxels in self.rho (3D cube) to 'val' (default: 0.)
        """

        self.rho = val*N.ones(self.X.shape)


    def normalize(self):

        """Normalize the sum of all voxel values to self.weight.
        """

        self.mass = self.rho.sum() / self.weight
        self.rho /= self.mass


    def shift(self):

        """Shift density distribution by xoff and yoff. If rotation is needed,
        shift first, then rotate.
        """

        if hasattr(self,'xoff'):
            self.rho = ndimage.shift(self.rho, (0,self.xoff/self.pixelscale,0), order=1)
            
        if hasattr(self,'yoff'):
            self.rho = ndimage.shift(self.rho, (self.yoff/self.pixelscale,0,0), order=1)

            
    def rotate3d(self):

        if hasattr(self,'tiltx') and self.tiltx != 0:
            self.rho = ndimage.rotate(self.rho, self.tiltx, axes=(0,2), reshape=False, order=1)

        if hasattr(self,'tilty') and self.tilty != 0:
            self.rho = ndimage.rotate(self.rho, self.tilty, axes=(1,2), reshape=False, order=1)

        if hasattr(self,'tiltz') and self.tiltz != 0:
            self.rho = ndimage.rotate(self.rho, self.tiltz, axes=(0,1), reshape=False, order=1)
            
            


def savefile(image,outname,header=None):


    def savefits(image,outname):
        # TODO: put some values in the FITS header
        import pyfits
        hdu = pyfits.PrimaryHDU(image)

        print "hdu.header: ", hdu.header

        if header is not None:
            for k,v in header.items():
                hdu.header[k] = v

        try:
            hdu.writeto(outname)
        except IOError:
            print "Can not write output file %s" % outname
            raise

        

    print "Saving %s file..." % outname
    savefits(image,outname)
