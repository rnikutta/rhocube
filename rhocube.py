"""Compute a model 3D density distribution, and 2D dz-integrated map."""

__author__  = "Robert Nikutta, Claudia Agliozzo"
__version__ = "2016-10-12"

# IMPORTS
import numpy as N
from scipy import spatial, ndimage


# CLASSES
class NullTransform:

    """Does nothing."""

    def __init__(self):
        pass

    def __call__(self,x):
        return x
    

class PowerTransform:

    """Power-Law Tranform of rho(x,y,z).

    Parameters:
    -----------
    pow : float
        The power-law exponent in rho**pow. The default is pow=1.,
        which does not transform rho(x,y,z) at all. Give pow=2. to
        square rho, pow=0.5 to take the square root of rho, etc.

    The inverse function is computed automatically.

    """
    
    def __init__(self,pow=1.):
        self.pow = float(pow)
        self.invpow = 1./pow

    def __call__(self,x):
        return x**self.pow

    def _inverse(self,x):
        return x**self.invpow

    
class LogTransform:

    """Logarithmic transform of rho(x,y,z)

    Parameters:
    -----------
    base : float | 'ln'
        If float, it is the base of the logarithm to be
        applied. Base=10. is the default. Natural logarithm is used if
        base='ln'.

    The inverse function is computed automatically.

    Be careful that log(1.) is zero, and this class takes no
    precautions against this case!

    """
    
    def __init__(self,base=10.):
        self.base = base

    def __call__(self,x):

        if self.base != 'ln':
            return N.log10(x) / N.log10(self.base)
        else:
            return N.log(x)
            
    def inverse(self,x):
        return self.base**x
    

class GenericTransform:

    """Generic transform of rho(x,y,z) (any argument-free function from Numpy).

    Parameters:
    -----------
    func : str
        The name in the Numpy module of the functions to be applied to
        rho(x,y,z). In principle any parameter-free function in numpy
        can be called. The default is 'sin'.

    inversefunc : str | None
        The Numpy-name of the inverse function of 'func', or None. The
        default is 'arcsin', the inverse function of the default
        func='sin'.

    """
    
    def __init__(self,func='sin',inversefunc='arcsin'):
        self.func = getattr(N,func)
        self.inversefunc = getattr(N,inversefunc)

    def __call__(self,x):
        return self.func(x)

    def _inverse(self,x):

        if self.inversefunc is not None:
            return self.inversefunc(x)
        else:
            return None

        
class Cube:

    """Generic cube of n**3 pixels.

    3D density geometry models should inherit from this class, as it
    provides common members and methods, e.g. normalize(), set_rho(),
    shift(), rotate3d(), etc. `Cube` can provide a unit 3D R-cube
    (Euclidean distances of all voxel centers from (0,0,0), as well as
    a kdtree for density fields computed around parametric curves.
    """

    def __init__(self,npix,transform=None,smoothing=1.,buildkdtree=False,computeR=False):

        """Initialize a unit 3D cube of voxels.

        Parameters
        ----------
        npix : int
            Number of cells along every dimension x,y,z. Must be odd
            to ensure that there is a single central voxel. If npix is
            not odd, it will be made odd by adding +1.

        transform : class instance | None
            If not None, transform is the instance of a class that
            will be applied to the resulting 3D density distribution,
            and before any further computations. Several built-in
            transform classes are provided, e.g. PowerTransform() (see
            doc string there). If the class also provides a _inverse()
            function, then the entire 3D cube rho(x,y,z) with correct
            scaling can also be computed. If None (the default), no
            transform of rho will be performed.

        smoothing : float | None
            If a float, the value is the width (in standard
            deviations) of a 3D Gaussian kernel that rho(x,y,z) will
            be convolved with, resulting in a smoothed 3D density
            distribution. smoothing=1. is default and does not alter
            the resulting structure significantly. If None, no
            smoothing will be applied. Note that smoothing does
            preserve the total \sum_i rho(x,y,z) where i runs over all
            voxels.

        buildkdtree : bool
            If True, a k-d tree of the voxel center positions in the
            cube will be computed. This is very helpful in speeding up
            computations of voxel proximity to some parametric curve
            (some models make use of it, e.g. Helix3D). The cost of
            building the tree is O(n log^2 n), i.e. it is a rather
            expensive function of npix (npix=101 is safe on modern
            PCs, npix=301 might be too expensive for most users to
            wait for). Once build, subsequent lookups in the tree are
            much faster. The default is False.

        """

        global models
        import models
        
        self.npix = npix
        if self.npix % 2 == 0:   # make sure we have a central pixel, i.e. an odd number of pixels along the axes
            self.npix += 1
            
        self.pixelscale = 2./float(self.npix)

        # center -/+ 1 radius to each side
        self.x = N.linspace(-1.,1,self.npix)

        self.X, self.Y, self.Z = N.meshgrid(self.x,self.x,self.x,indexing='xy')  # three 3d coordinate arrays

        if computeR is True:
            self.R = get_r((self.X,self.Y,self.Z),mode=2)  # mode=1,2 are fast, 3,4are slow
        
        if buildkdtree is True:
            self.build_kdtree()

        self.ndshape = self.X.shape

        self.extent = (self.x.min(),self.x.max(),self.x.min(),self.x.max())

        self.transform = transform   # a function reference
        if self.transform == None:
            self.transform = NullTransform()
        
        self.rho = self.set_rho(val=0.)

        self.smoothing = smoothing

        
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
        idx1d : array
            1-dimensional array of idices corresponding to points
            found by the query.

        dist : float
            The actual distance from x to the closest found
            neighbor. Only meaningful when r = 0, i.e. single-point
            query, otherwise dist=None is returned.

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


    def smooth(self):

        """Smooth 3D density array self.rho by convolving with a Gaussian
        kernel of width self.smoothing (measured in std deviations)."""
        
        if hasattr(self,'smoothing') and self.smoothing is not None:
            self.rho = ndimage.gaussian_filter(self.rho,sigma=self.smoothing)
            

    def apply_rho_ops(self):

        """Convenience function to apply all post rho-comuptatations common to
        all models.

        Specifically, the operations, in this order, are:
            transform(rho)   # if any 'transform' class was passed
            shift            # if xoff and/or yoff are not zero
            rotate3D         # if any of the tilt angles tiltx, tilty, tiltz are not zero
            smooth           # if 'smoothing' parameter is not None
            normalize        # if 'weight' parameter is not None

        After all these operations, the function also computes the
        newest z-integrated image, rho_surface(x,y) = \int dx rho(x,y,z)
        """

        if self.transform is not None:
            self.rho = self.transform(self.rho)  # apply transform() if not None
            mask = N.isnan(self.rho) | N.isinf(self.rho)   # find Nan and Inf
            self.rho[mask] = 0.                            # and replace them with zeros

        self.shift()
        self.rotate3d()
        self.smooth()
        
        if hasattr(self,'weight') and self.weight is not None:
            self.normalize()

        # compute image by summing rho(x,y,z) over z
        self.image = N.sum(self.rho,axis=-1)
        

# FUNCTIONS
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
