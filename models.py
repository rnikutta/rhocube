"""Model classes for 3D density distribution."""

__author__  = "Robert Nikutta, Claudia Agliozzo"
__version__ = "2015-05-11"

import numpy as N

from numpy import cross,eye,dot
from scipy.linalg import expm3,norm

#######################################
# Models for 3D density distributions #
#######################################


class Cube3D:

    """Generic Cube of nx*ny*nz pixels. Models should inherit from this
    class, as it provides common members and methods,
    e.g. normalize(), set_rho(), shift(), and rotate().
    """

    def __init__(self,X,Y,Z):

        """Initialize a 3D cube of voxels using X,Y,Z corrdinate arrays (each
        a 3D array).
        """

#        self.X, self.Y, self.Z = X, Y, Z
        self.X, self.Y, self.Z = X.copy(), Y.copy(), Z.copy()
        self.X2 = self.X * self.X
#        self.Y2 = self.Y * self.Y
        self.Z2 = self.Z * self.Z
        self.rho = self.set_rho(val=0.)
        

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

        if abs(self.xoff) > 0:
            self.X -= self.xoff

        if abs(self.yoff) > 0:
            self.Y -= self.yoff


    def rotate2d(self,A,B,theta):

        """Intrinsic rotation in 2D.

        For rotation in 3D about one of the principal axes, only the
        coordinates of the two other axes are needed (the coordinates
        along the rotation axis won't be affected.

        Parameters:
        -----------
        A, B : {floats, float arrays}
            A, B are coordinate arrays (which can be 3D arrays) of the
            two axes affected by the rotation to be performed. E.g. if
            the rotation is about the x-axis, only the y and z
            coordinates will be affected, and thus A and B should be
            arrays (of any shape) containing the y and z coordinates,
            etc.

        theta : float
            Rotation angle in radians.

        Example:
        --------

        Given x, y, z coordinates, rotate the coordinate system by 30
        degrees about the x-axis:

            x, y, z = N.linspace(-1,1,11), N.linspace(0,10,21), N.linspace(3.5,4.5,15)
            X, Y, Z = N.meshgrid(x,y,z)            # 3D array cubes of x, y, values at every voxel
            Xp = X                                 # Xp (X-prime) is invariant
            Yp, Zp = rotate2d(Y,Z,N.radians(30))   # rotated Y and Z coordinates

        """

        cos_ = N.cos(theta)
        sin_ = N.sin(theta)
        Ap = A*cos_ - B*sin_
        Bp = A*sin_ + B*cos_

        return Ap, Bp


    def rotate3d(self):
        
        """Rotate the unshifted rho (i.e. one that's centered on (0,0) by
        angles tiltx and tiltz. tilty is only needed for density
        distributions that are not axisymmetric around the y axis,
        maybe we'll implement such distros later.
        """

        if self.tiltx > 0.:

            # when rotating about the x-axis (pointing to the right in the image), the X coordinates are invariant.
            self.Y, self.Z = self.rotate2d(self.Y,self.Z,N.radians(self.tiltx))

        if self.tiltz > 0.:

            # when rotating about the z-axis (pointing towards observer), the Z coordinates are invariant.
            self.X, self.Y = self.rotate2d(self.X,self.Y,N.radians(self.tiltz))


class ConstantDensityShell(Cube3D):

    # supply X,Y,Z to init instead of the param values. SUpply the param values instead to call (which may use update)

    def __init__(self,X,Y,Z):
        
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
           amounts of mass inside each one. Default: weight=1.

        """

        Cube3D.__init__(self,X,Y,Z)


    def __call__(self,rin,rout,xoff=0.,yoff=0.,weight=1):

        """Return density rho at (x,y,z)"""

        self.rin = rin
        self.rout = rout
        self.xoff = xoff
        self.yoff = yoff
        self.weight = weight

        self.sanity()

        self.shift()

        self.get_rho()  # get_rho should set self.rho (3D)
        self.normalize()

        return self.rho


    def sanity(self):

        """Sanity checks for constant density-edge shell.
        """

        assert (0. < self.rin < self.rout)  # this automatically asserts that the shell thickness is finite and positive


    def get_rho(self):

        """Compute rho(x,y,z) in every voxel.
        """

        r = get_r((self.X,self.Y,self.Z),mode=2)  # mode=1,2 are fast, 3,4are slow
        co = (r >= self.rin) & (r <= self.rout)
        self.set_rho(val=0.)  # default is 0.
        self.rho[co] = 1.


class ConstantDensityTorus(Cube3D):

    """Torus as a ring with circular cross-section.

    Parameters:
    -----------
    r : float
       Torus radius

    rcross : float
       Torus ring cross-section radius

    xoff, yoff : floats
        x and y offsets of the torus center from (0,0). Positive
        values are to the right and up, negative to the left and
        down. In units if unity (remember that the image is within
        [-1,1]. Defaults: xoff = yoff = 0.

    tiltx, tiltz : floats
        The tilt angles along the x and z axes, respectively. Looking
        at the plane of the sky, x is to the right, y is up, and z is
        toward the observer. Thus tiltx tilts the torus axis towards
        and from the observer, and tiltz tilts the torus axis in the
        plane of the sky. tiltx = tiltz = 0 results in the torus seen
        edge-on and its axis pointing up in the image. tiltx = 90 &
        tiltz = 0 points the torus axis towards the observer (pole-on
        view).

    weight : float
        Normalize the total (relative) mass contained in the torus to
        this value. The total mass is the sum of rho over all pixels
        (in 3D). This is useful if you e.g. want to have more than one
        component, and wish to distribute different amounts of mass
        inside each one. Default: weight=1.

    """

    def __init__(self,X,Y,Z):

        Cube3D.__init__(self,X,Y,Z)


    def __call__(self,r,rcross,xoff=0.,yoff=0.,tiltx=0.,tiltz=0,weight=1.):

        """Return density rho at (x,y,z)"""

        self.r = r
        self.rcross = rcross
        self.xoff = xoff
        self.yoff = yoff
        self.tiltx = tiltx
        self.tiltz = tiltz
        self.weight = weight

        self.sanity()

        self.shift()
        self.rotate3d()

        self.get_rho()
        self.normalize()

        return self.rho


    def sanity(self):

        assert (0. < self.rcross <= self.r)
        
#        assert (0. <= self.tiltx <= 90.)
        assert (0. <= self.tiltx <= 180.)
        assert (0. <= self.tiltz <= 180.)


    def get_rho(self):

        # A point (x,y,z) is inside the torus when:
        #
        #    (x^2 + y^2 + z^2 + r^2 - rcross^2)^2 - 4 * r^2 * (x^2 + z^2) < 0


        # To speed up computation a bit (the following expression are used twice each in the formula below)
        r2 = self.r**2
        X2 = self.X**2
        Z2 = self.Z**2
        co = (X2 + self.Y**2 + Z2 + r2 - self.rcross**2)**2 - 4 * r2 * (X2 + Z2) < 0
#        co = (self.X**2 + self.Y**2 + self.Z**2 + r2 - self.rcross**2)**2 - 4 * r2 * (self.X**2 + self.Z**2) < 0
        
        self.set_rho(val=0.)  # default is 0.
        self.rho[co] = 1.


class TruncatedNormalShell(Cube3D):

    # all models should inherit from Cube3D; the models only provide what's unique to them (e.g. get_rho(), sanity())

    # self.get_rho() should compute and set self.rho (3D)
    # __call__(*args) should return self.rho (3D)


    def __init__(self,X,Y,Z):
        
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

        Cube3D.__init__(self,X,Y,Z)


    def __call__(self,r,width,clipa=0.,clipb=1.,xoff=0.,yoff=0.,weight=1.):

        """Return density rho at (x,y,z)"""

        self.r = r
        self.width = width
        self.clipa = clipa
        self.clipb = clipb
        self.xoff = xoff
        self.yoff = yoff
        self.weight = weight

        self.sanity()

        self.shift()

        self.get_rho()
        self.normalize()

        return self.rho


    def sanity(self):

        # CAREFUL ASSERTIONS
        # lower cut clipa must be smaller than r
        # lower cut clipa can be as small as zero
        # upper cut clipb can be as low as r
        # upper cub clipb can be in principle larger than unity (but we'll default to 1.0)
        # width must be a positive number
        assert (0. <= self.clipa < self.r <= self.clipb)  # radial distance relations that must hold: 0. <= clipa < r < clipb [<= 1.]
        assert (self.width > 0.)


    def get_rho(self):

        """Compute rho(x,y,z) in every voxel.
        """

        r = get_r((self.X,self.Y,self.Z),mode=2)  # mode=1,2 are fast, 3,4are slow

        self.rho = self.get_pdf(r)


    def get_pdf(self,x):

        """Distribution of density according to a Gaussian with (mu,sig) =
        (r,width).
        """

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
    

