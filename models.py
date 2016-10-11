"""Model classes for 3D density distribution."""

__author__  = "Robert Nikutta, Claudia Agliozzo"
__version__ = "2016-10-10"

# imports
from rhocube import Cube
import numpy as N


def spiral3D(h,Rbase,nturns,rtube,envelope='dualcone'):

        """Function to compute a helical parametric curve along the outline of
        a dual cone or a cylinder.
        """
        
        a = nturns * 2*N.pi/h
        delta = rtube/3.
        th = a*h # / (2*N.pi)
        nsteps = int(th/N.float(delta))
        t = N.linspace(-th,th,2*nsteps+1)
        z = t/a

        if envelope == 'dualcone':
            zprogression = z*(Rbase/h)
        elif envelope == 'cylinder':
            zprogression = Rbase/h
        else:
            raise Exception, "Invalid value for 'envelope'. Must be either of: ['dualcone','cylinder']."
        
        x = zprogression * N.cos(N.abs(t))
        y = zprogression * N.sin(N.abs(t))

        return x, y, z


class Helix3D(Cube):

    def __init__(self,npix,transform=None,smoothing=1.,snakefunc=spiral3D,envelope='dualcone'):
        
        """Helical tube winding along a dual cone, with constant density inside the tube.


        Scratch:

        ROTATE
        ndimage.rotate() uses angles in degrees.
        Rotation of 3D array cube.rho by angle A in the plane defined by axes:

        axes=(0,2) --> our x axis
        axes=(0,1) --> our z axis
        axes=(1,2) --> our y axis

        So indexing seems to be (y,x,z)

        SHIFT
        ndimage.shift() works in pixel units.

        
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

        Cube.__init__(self,npix,transform=transform,smoothing=smoothing,buildkdtree=True)
        
        self.z = N.unique(self.Z.flatten())
        self.snakefunc = snakefunc
        self.envelope = envelope
        

    def __call__(self,h,nturns,rtube,tiltx=0.,tilty=0.,tiltz=0.,xoff=0.,yoff=0.,weight=None):

        """Return density rho at (x,y,z)

        TODO: automatically determine args (their names), and produce
        self.ARG members, and use those in

        """

        self.h = h
        self.Rbase = self.h
        self.nturns = nturns
        self.rtube = rtube
        
        self.tiltx = tiltx
        self.tilty = tilty
        self.tiltz = tiltz
        self.xoff = xoff
        self.yoff = yoff
        
        self.weight = weight
        self.sanity()
        self.get_rho()  # get_rho should set self.rho (3D)
        self.apply_rho_ops()  # shift, rotate3d, smooth, in that order

        return self.rho
    

    def sanity(self):

        """Sanity checks.
        """

        pass  # not yet implemented
    

    def get_rho(self):

        """Compute rho(x,y,z) in every voxel.
        """
        
        self.x, self.y, self.z = self.snakefunc(self.h,self.Rbase,self.nturns,self.rtube,self.envelope)

        co = N.zeros(N.prod(self.ndshape),dtype=N.float)  # get a cube full of zeros

        # for evey voxel quickly determine if it's close enough to the helix center line
        for j,pt in enumerate(zip(self.x,self.y,self.z)):
            idxes = self.kdtree_query(pt,self.rtube)[1]
            co[idxes] = 1.

        self.rho = co.reshape(self.ndshape)


        
class PowerLawShell(Cube):

    def __init__(self,npix,transform=None,smoothing=1.,exponent=-1.):
        
        """Power-law shell.

        A spherical shell with inner and outer radii, and radial power-law density fall-off.

        See __call__ doc string for shell parameters.

        """
        
        Cube.__init__(self,npix,transform=transform,smoothing=smoothing,buildkdtree=False,computeR=True)

        self.exponent = exponent

        
    def __call__(self,rin,rout,xoff=0.,yoff=0.,weight=None):

        """Return density rho at all voxels (x,y,z).

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

        weight : float or None
           Normalize the total (relative) mass contained in the shell
           to this value. The total mass is the sum of rho over all
           pixels (in 3D). This is useful if you e.g. want to have
           more than one component, and wish to distribute different
           amounts of mass inside each one. Default: weight=1.

        """

        self.rin = rin
        self.rout = rout

        # helper arrays for finding the edges of the shell in get_rho()
        self.Rin = self.rin * N.ones(self.X.shape)
        self.Rout = self.rout * N.ones(self.X.shape)
        self.xoff = xoff
        self.yoff = yoff
        self.weight = weight

        self.sanity()

        self.get_rho()  # get_rho sets self.rho (3D)
        self.apply_rho_ops()  # shift, rotate3d, smooth, in that order
        
        return self.rho


    def sanity(self):

        """Sanity checks."""

        assert (0. < self.rin < self.rout)  # this automatically asserts that the shell thickness is finite and positive


    def get_rho(self):

        """Compute rho(x,y,z) in every voxel."""

        self.rho = self.R**self.exponent
        co = ((self.R >= self.rin) & (self.R <= self.rout)) | N.isclose(self.R,self.Rout) | N.isclose(self.R,self.Rin)  # isclose also captures pixels at the very edge of the shell
        self.rho[~co] = 0.



class TruncatedNormalShell(Cube):

    def __init__(self,npix,transform=None,smoothing=1.):
        
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

        Cube.__init__(self,npix,transform=transform,smoothing=smoothing,buildkdtree=False,computeR=True)

        
    def __call__(self,r,width,clipa=0.,clipb=1.,xoff=0.,yoff=0.,weight=None):

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
        self.apply_rho_ops()  # shift, rotate3d, smooth, in that order

        return self.rho


    def sanity(self):

        # CAREFUL ASSERTIONS
        # lower cut clipa must be smaller than r
        # lower cut clipa can be as small as zero
        # upper cut clipb can be as low as r
        # upper cub clipb can be in principle larger than unity (but we'll default to 1.0)
        # width must be a positive number
        assert (0. < self.clipa < self.r < self.clipb)  # radial distance relations that must hold: 0. <= clipa < r < clipb [<= 1.]
        assert (self.width > 0.)


    def get_rho(self):

        """Compute rho(x,y,z) in every voxel.
        """

        self.rho = self.get_pdf(self.R)


    def get_pdf(self,x):

        """Distribution of density according to a Gaussian with (mu,sig) = (r,width).
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



class ConstantDensityTorus(Cube):


    def __init__(self,npix,transform=None,smoothing=1.):

        """Torus as a ring with circular cross-section.
        """        

        Cube.__init__(self,npix,transform=transform,smoothing=smoothing,buildkdtree=False,computeR=True)


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

        self.get_rho()
        self.apply_rho_ops()  # shift, rotate3d, smooth, in that order

        return self.rho


    def sanity(self):

        assert (0. < self.rcross <= self.r)
       
        assert (0. <= self.tiltx <= 180.)
        assert (0. <= self.tiltz <= 180.)


    def get_rho(self):

        """ 
        A point (x,y,z) is inside the torus when:
        
            (x^2 + y^2 + z^2 + r^2 - rcross^2)^2 - 4 * r^2 * (x^2 + z^2) < 0

        """
        
        # To speed up computation a bit (the following expression are used twice each in the formula below)
        r2 = self.r**2
        X2 = self.X**2
        Z2 = self.Z**2
        co = (X2 + self.Y**2 + Z2 + r2 - self.rcross**2)**2 - 4 * r2 * (X2 + Z2) < 0
#        co = (self.X**2 + self.Y**2 + self.Z**2 + r2 - self.rcross**2)**2 - 4 * r2 * (self.X**2 + self.Z**2) < 0
        
        self.set_rho(val=0.)  # default is 0.
        self.rho[co] = 1.



class ConstantDensityDualCone(Cube):

    def __init__(self,npix,transform=None,smoothing=1.):
        
        """ConstantDensityDualCone
        """

        Cube.__init__(self,npix,transform=transform,smoothing=1.,buildkdtree=False,computeR=True)


    def __call__(self,r,theta,tiltx=0.,tiltz=0,xoff=0.,yoff=0.,weight=None):

        """Return density rho at (x,y,z)"""

        self.r = r
        self.theta_deg = theta
        self.theta_rad = N.radians(self.theta_deg)
        self.tiltx = tiltx
        self.tiltz = tiltz

        self.xoff = xoff
        self.yoff = yoff
        self.weight = weight

        self.get_rho()  # get_rho should set self.rho (3D)
        self.apply_rho_ops()  # shift, rotate3d, smooth, in that order

        return self.rho


    def sanity(self):

        """Sanity checks for constant density-edge shell.
        """

        pass  # not yet implemented


    def get_rho(self):

        """Compute rho(x,y,z) in every voxel.
        """

        # cone formula
        aux = ((self.X**2 + self.Z**2) * N.cos(self.theta_rad)**2 - (self.Y*N.sin(self.theta_rad))**2)
        co1 = (aux <= 0) | N.isclose(aux,0)

        # radial caps
        co2 = (N.sqrt(self.X**2 + self.Y**2 + self.Z**2) <= self.r)

        # overall
        coall = co1 & co2 #| coaux
       
        # set all occupied voxels to one
        self.set_rho(val=0.)  # default is 0.
        self.rho[coall] = 1.

