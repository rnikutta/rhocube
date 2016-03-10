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

def get_tail(arg,mode='sigma'):

    if mode == 'sigma':
        fraction = erf(arg/sqrt(2))
    elif mode == 'fraction':
        fraction = arg

    tail = (1.0-fraction)/2.

    return tail


def get_quantiles(sample,sigmas=1.):

    tail = get_tail(sigmas,mode='sigma')
    left, median, right = stats.mstats.mquantiles(sample,prob=[tail,0.5,1.-tail])

    return left, median, right


def output_fits(image,fitsfile='foo.fits'):

    data = image.data
    mask = image.mask
    
    hdu = pyfits.PrimaryHDU(data)

#    hdu.writeto(fitsfile)
    hdulist = pyfits.HDUList([hdu])

    scidata = hdulist[0].data
    scidata[mask] = N.nan

    hdulist.writeto(fitsfile)

    # Header keywords
#    hdr = pyfits.getheader(fitsfile, 1)  # get first extension's header
##    filter = hdr['filter']         # get the value of the keyword "filter'
##    val = hdr[10]                  # get the 11th keyword's value
##    hdr['filter'] = 'FW555'        # change the keyword value


    print "FITS file written to %s" % fitsfile






def make_testdata(case='CDShell'):

    from collections import OrderedDict

    if case == 'TNShell':
        themodels = (('TruncatedNormalShell',0.3,0.05,0,1,0.5,0.2,1.),)
    elif case == 'CDTorus':
        themodels = (('ConstantDensityTorus',0.5,0.2,-0.2,0.1,50.,55.,1.),)

    cube = Cube(200,themodels)

    # (1/sig^2) * randn + loc
#    image = (1./N.sqrt(cube.allmap)) * N.random.randn(*cube.allmap.shape) + cube.allmap

#    mask = (cube.allmap == 0)
#    scale = N.sqrt(cube.allmap)
#    scale[mask] = 1.
#    image = (1./scale) * N.random.randn(*cube.allmap.shape) + cube.allmap
#    image[mask] = 0.

#    mask = (cube.allmap == 0)
#    min_ = cube.allmap[~mask].min()
#    scale = N.sqrt(cube.allmap)
#    scale[mask] = min_
#    image = (1./scale) * N.random.randn(*cube.allmap.shape) + cube.allmap
#    image[image < 0] = 0.

#    sig = 0.1*cube.allmap.max()
    mask = (cube.allmap > 0.)
    sig = 0.1*cube.allmap[mask].max()    # noise level
#    print sig, cube.allmap.max()
    image = sig * N.random.randn(*cube.allmap.shape) + cube.allmap
#    image[image < 0] = 0.
    

    model = cube.models[0]

    if case == 'TNShell':
        keywords = ('model','radius','width','clipa','clipb','xoff','yoff','weight')
        values = ( (model.__class__.__name__,'model type'),\
                   (model.r,'shell radius'),\
                   (model.width,'shell thickness'),\
                   (model.clipa,'lower clip radius'),\
                   (model.clipb,'upper clip radius'),\
                   (model.xoff,'x offset of shell center'),\
                   (model.yoff,'y offset of shell center'),\
                   (model.weight,'relative normalization of total mass in shell') )

        header = OrderedDict(zip(keywords,values))
        
    elif case == 'CDTorus':
        keywords = ('model','r','rcross','xoff','yoff','tiltx','tiltz','weight')
        values = ( (model.__class__.__name__,'model type'),\
                   (model.r,'torus major radius'),\
                   (model.rcross,'torus tube cross-section'),\
                   (model.xoff,'x offset of torus center'),\
                   (model.yoff,'y offset of torus center'),\
                   (model.tiltx,'tilt around x-axis (to the right) in degrees'),\
                   (model.tiltz,'tilt around z-axis (to the observer) in degrees'),\
                   (model.weight,'relative normalization of total mass in torus') )

        header = OrderedDict(zip(keywords,values))
        

    savefile(image,'%s.fits' % case,header=header)




class MCMC:

    def __init__(self,themodels,priors=None,nsamples=100):

        self.themodels = themodels
        self.priors = priors
        self.nsamples = nsamples

        self._sanity()

        self._setup()


    def _sanity(self):

        if self.priors is not None:
            assert (len(self.priors) == len(self.themodels))


    def _setup(self):
    
        """Set up PyMC model, parameters, etc."""
 
        pass
#        for comp in self.themodels:
            





#ORIGclass Cube:
#ORIG
#ORIG    
#ORIG    """3D Cube to hold the total rho(x,y,z).
#ORIG
#ORIG    Example:
#ORIG    --------
#ORIG    Two different shells:
#ORIG
#ORIG    cube = rhocube.Cube( 200, ( ('TruncatedNormalShell',0.4,0.03,0,1,0.5,0.2,1),\
#ORIG                                ('ConstantDensityShell',0.3,0.5,,-0.2,-0.4,2) ) )
#ORIG
#ORIG    pylab.imshow(cube.allmap,origin='lower',extent=extent,cmap=matplotlib.cm.gray)
#ORIG
#ORIG    """
#ORIG
#ORIG    def __init__(self,npix,themodels,normalize='none'):  # TODO: get npix from the FITS file to be fitted
#ORIG
#ORIG        """
#ORIG
#ORIG        Example:
#ORIG        --------
#ORIG
#ORIG        themodels = (\
#ORIG                     ('TruncatedNormalShell',0.8,0.02,0,1,0.,0.,1.),\
#ORIG                     ('ConstantDensityShell',0.4,0.6,,(-0.3,0.2),0.5)
#ORIG                    )
#ORIG
#ORIG        """
#ORIG
#ORIG        self.npix = npix
#ORIG        if self.npix % 2 == 0:   # make sure we have a central pixel, i.e. an odd number of pixels along the axes
#ORIG            self.npix += 1
#ORIG
#ORIG        self.cube = N.zeros((self.npix,self.npix,self.npix))
#ORIG
#ORIG        # center -/+ 1 radius to each side
#ORIG        self.x = N.linspace(-1.,1,self.npix)
#ORIG        self.X, self.Y, self.Z = N.meshgrid(self.x,self.x,self.x,indexing='xy')  # three 3d coordinate arrays to 
#ORIG
#ORIG        self.extent = (self.x.min(),self.x.max(),self.x.min(),self.x.max())
#ORIG
#ORIG        # put this into __call__?
#ORIG        # also store in init func ref to a 'transform' function?
#ORIG        # TODO: separate this in init and call
#ORIG        self.models = []
#ORIG        for m in themodels:
#ORIG            print m
#ORIG
#ORIG            typ, args = m[0], m[1:]
#ORIG#P            print "typ: ", typ
#ORIG#P            print "args: ", args
#ORIG            model = getattr(models,typ)(self.X, self.Y, self.Z)
#ORIG            model(*args)  # model.rho (3D) is normalized to model_.weight
#ORIG
#ORIG            model.rhomap = N.sum(model.rho**2,axis=-1)
#ORIG
#ORIG#            if normalize == 'peak':
#ORIG#                model_.rhomap /= model_.rhomap.max()
#ORIG#                model_.rhomap *= model_.weight
#ORIG
#ORIG            self.models.append(model)
#ORIG
#ORIG
#ORIG#        self.allmap = N.sum([m.rhomap for m in self.models],axis=0)
#ORIG
#ORIG#        self.image = N.sum([m.rhomap for m in self.models],axis=0)
#ORIG        self.image = N.sum([m.rhomap for m in self.models],axis=0)
#ORIG        self.mask = (self.image > 0.)
#ORIG
#ORIG
#ORIG    def __call__(self):
#ORIG
#ORIG        """When called with parameters"""
#ORIG
#ORIG
#ORIG    def add_noise(self,type='gauss',mode='std',magnitude=0.1):
#ORIG
#ORIG        """Add random noise to final image.
#ORIG
#ORIG        Leaves self.image alone, and creates self.errimage (just the
#ORIG        errors image) and self.noisyimage (self.image +
#ORIG        self.errimage).
#ORIG
#ORIG        Parameters:
#ORIG        -----------
#ORIG        type : str
#ORIG            Noise statistic. Currently only 'gauss' is. implemented
#ORIG
#ORIG        mode : str
#ORIG            Use one of 'std', 'max', 'mean', 'median' of all non-zero
#ORIG            valued image pixels to compute the error amplitude.
#ORIG
#ORIG        magnitude : float
#ORIG            Scaling factor applied to the error amplitude (defined by
#ORIG            'mode'). The error image will be
#ORIG
#ORIG               error = magnitude * mode(image) * RND
#ORIG        
#ORIG            where RND is a random variate per pixel drawn from the
#ORIG            noise statistic defined by 'type'. The noisy image will be
#ORIG
#ORIG               noisyimage = error + image
#ORIG        """
#ORIG
#ORIG        self.rnd = N.random.randn
#ORIG        if type != 'gauss':
#ORIG            warnings.warn("Only 'gauss' implemented as noise type at the moment.")
#ORIG
#ORIG        assert (mode in ('std','max','median','mean'))
#ORIG
#ORIG        self.mag = magnitude
#ORIG        self.errmode = getattr(N,mode)
#ORIG        self.sig = self.mag * self.errmode(self.image[self.mask])
#ORIG        self.errimage = self.sig * self.rnd(*self.image.shape)
#ORIG        self.noisyimage = self.errimage + self.image


#class Tube:
#
#    def __init__(self):
#        pass
    

class Spiral:

    def __init__(self,h,Rbase,nturns,envelope='cone',dual=True):

        self.h = h
        self.Rbase = Rbase
        self.a = self.nturns * 2*N.pi/self.h

        th = self.a*self.h # / (2*N.pi)
        nsteps = int(th/N.float(delta))
        t = N.linspace(-th,th,2*nsteps+1)
        self.z = t/self.a

        if envelope == 'cone':
            multiplier = z
        elif envelope == 'cylinder':
            multiplier = 1.
        else:
            raise Exception, "Invalid value for 'envelope'. Must be either of: ['cone','cylinder']."
            
        self.x = multiplier*self.Rbase*N.cos(N.abs(t))
        self.y = multiplier*self.Rbase*N.sin(N.abs(t))

        

class Model:

#    def __init__(self,*themodels):
    def __init__(self,themodels):

        self.themodels = themodels
        print "self.themodels = ", self.themodels
        self.npix = self.themodels[0].npix


    def __call__(self,*args):

        """When called with parameters"""

        # put this into __call__?
        # also store in init func ref to a 'transform' function?
        # TODO: separate this in init and call
        self.totalrho = N.zeros((self.npix,self.npix,self.npix))

        self.models = []
        for j,model in enumerate(self.themodels):

            model(*args[j])  # model.rho (3D) is normalized to model_.weight

            self.totalrho += model.rho
#            model.rhomap = N.sum(model.rho**2,axis=-1)

#            if normalize == 'peak':
#                model_.rhomap /= model_.rhomap.max()
#                model_.rhomap *= model_.weight

#            self.models.append(model)  # CAUTION: not yet tested

        self.rhoraw = self.totalrho.copy()

#        if self.transform is not None:
#            self.rho = self.transform(self.rho.copy())

        self.image_raw = N.sum(self.rhoraw,axis=-1)
        self.image = N.sum(self.totalrho,axis=-1)  #self.image_raw.copy()
        
#G        self.image = self.image / self.image.max()
#        self.image = self.image / self.image.sum()

##        self.image = N.sum([m.rhomap for m in self.models],axis=0)
#        self.image = N.sum([m.rhomap for m in self.models],axis=0)
#?        self.mask = (self.image > 0.)
    


class CubeOLD:

    """Generic Cube of nx*ny*nz pixels. Models should inherit from this
    class, as it provides common members and methods,
    e.g. normalize(), set_rho(), shift(), and rotate().
    """

#    def __init__(self,X,Y,Z):
    def __init__(self,npix,transform=None,normalize='none'):

        """Initialize a 3D cube of voxels using X,Y,Z corrdinate arrays (each
        a 3D array).
        """

        global models
        import models
        
        self.npix = npix
        if self.npix % 2 == 0:   # make sure we have a central pixel, i.e. an odd number of pixels along the axes
            self.npix += 1

            

        # center -/+ 1 radius to each side
        self.x = N.linspace(-1.,1,self.npix)

#        self.reset_grid()
        self.X, self.Y, self.Z = N.meshgrid(self.x,self.x,self.x,indexing='ij')  # three 3d coordinate arrays to 
        self.build_kdtree()

        
        
#        self.X, self.Y, self.Z = N.meshgrid(self.x,self.x,self.x,indexing='xy')  # three 3d coordinate arrays to 
##        self.X, self.Y, self.Z = N.meshgrid(self.x,self.x,self.x,indexing='xy',sparse=True)  # three 3d coordinate arrays to
#
#        self.distance = N.sqrt(self.X**2+self.Y**2+self.Z**2)

        self.ndshape = self.X.shape

        self.extent = (self.x.min(),self.x.max(),self.x.min(),self.x.max())

        self.transform = transform # func ref


##        self.X, self.Y, self.Z = X, Y, Z
#        self.X, self.Y, self.Z = X.copy(), Y.copy(), Z.copy()
#        self.R = N.sqrt(self.X**2+self.Y**2+self.Z**2)
##        self.X2 = self.X * self.X
##        self.Y2 = self.Y * self.Y
##        self.Z2 = self.Z * self.Z
##        self.XYZ2sum = self.X2 + self.Y2 + self.Z2

        self.rho = self.set_rho(val=0.)
        

#CALL    def __call__(self,themodels):
#CALL
#CALL        """When called with parameters"""
#CALL
#CALL        # put this into __call__?
#CALL        # also store in init func ref to a 'transform' function?
#CALL        # TODO: separate this in init and call
#CALL        self.rho = N.zeros((self.npix,self.npix,self.npix))
#CALL
#CALL        self.models = []
#CALL        for m in themodels:
#CALL#            print m
#CALL
#CALL            typ, args = m[0], m[1:]
#CALL            print "typ: ", typ
#CALL            print "args: ", args
#CALL#            model = getattr(models,typ)(self.X, self.Y, self.Z)
#CALL            model = getattr(models,typ)()
#CALL            model(*args)  # model.rho (3D) is normalized to model_.weight
#CALL
#CALL            self.rho += model.rho
#CALL#            model.rhomap = N.sum(model.rho**2,axis=-1)
#CALL
#CALL#            if normalize == 'peak':
#CALL#                model_.rhomap /= model_.rhomap.max()
#CALL#                model_.rhomap *= model_.weight
#CALL
#CALL            self.models.append(model)  # CAUTION: not yet tested
#CALL
#CALL        self.rhoraw = self.rho.copy()
#CALL
#CALL        if self.transform is not None:
#CALL            self.rho = self.transform(self.rho.copy())
#CALL
#CALL        self.image_raw = N.sum(self.rhoraw,axis=-1)
#CALL        self.image = N.sum(self.rho,axis=-1)  #self.image_raw.copy()
#CALL#G        self.image = self.image / self.image.max()
#CALL#        self.image = self.image / self.image.sum()
#CALL
#CALL##        self.image = N.sum([m.rhomap for m in self.models],axis=0)
#CALL#        self.image = N.sum([m.rhomap for m in self.models],axis=0)
#CALL#?        self.mask = (self.image > 0.)



#    def reset_grid(self):
#
#        self.X, self.Y, self.Z = N.meshgrid(self.x,self.x,self.x,indexing='xy')  # three 3d coordinate arrays to 
##        self.X, self.Y, self.Z = N.meshgrid(self.x,self.x,self.x,indexing='xy',sparse=True)  # three 3d coordinate arrays to
#
#
#        self.build_kdtree()
#
#
##        self.distance = N.sqrt(self.X**2+self.Y**2+self.Z**2)
#       
##        self.set_rho(val=0.)
        


    def build_kdtree(self):

#        self.mg = N.meshgrid(*self.midpoints.unit,indexing='ij')
#        midgrid = N.vstack(self.mg).reshape(self.ndim,-1).T
#        self.kdtree = S.spatial.cKDTree(midgrid)   # kd-tree is in unit hypercell space

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

        print "In shift(). hasattr(self,'X'), hasattr(self,'xoff') = ", hasattr(self,'X'), hasattr(self,'xoff')

        if hasattr(self,'xoff'):
            self.X -= self.xoff

        if hasattr(self,'yoff'):
            self.Y -= self.yoff


            

#    def rotate2d(self,A,B,theta):
#
#        """Intrinsic rotation in 2D.
#
#        For rotation in 3D about one of the principal axes, only the
#        coordinates of the two other axes are needed (the coordinates
#        along the rotation axis won't be affected.
#
#        Parameters:
#        -----------
#        A, B : {floats, float arrays}
#            A, B are coordinate arrays (which can be 3D arrays) of the
#            two axes affected by the rotation to be performed. E.g. if
#            the rotation is about the x-axis, only the y and z
#            coordinates will be affected, and thus A and B should be
#            arrays (of any shape) containing the y and z coordinates,
#            etc.
#
#        theta : float
#            Rotation angle in radians.
#
#        Example:
#        --------
#
#        Given x, y, z coordinates, rotate the coordinate system by 30
#        degrees about the x-axis:
#
#            x, y, z = N.linspace(-1,1,11), N.linspace(0,10,21), N.linspace(3.5,4.5,15)
#            X, Y, Z = N.meshgrid(x,y,z)            # 3D array cubes of x, y, values at every voxel
#            Xp = X                                 # Xp (X-prime) is invariant
#            Yp, Zp = rotate2d(Y,Z,N.radians(30))   # rotated Y and Z coordinates
#
#        """
#
#        cos_ = N.cos(theta)
#        sin_ = N.sin(theta)
#        Ap = A*cos_ - B*sin_
#        Bp = A*sin_ + B*cos_
#
#        return Ap, Bp


    def get_trigs(self,deg):

        rad = N.radians(deg)
        
        return N.sin(rad), N.cos(rad)
    

    def get_Rx(self,deg):

        sin, cos = self.get_trigs(deg)
        
        Rx = N.array([  [1,   0,    0],\
                        [0, cos, -sin],\
                        [0, sin,  cos] ])

        return Rx
    
        
    def get_Ry(self,deg):

        sin, cos = self.get_trigs(deg)
        
        Ry = N.array([  [cos,  0, sin],\
                        [0,    1,   0],\
                        [-sin, 0, cos] ])
        return Ry
    

    def get_Rz(self,deg):

        sin, cos = self.get_trigs(deg)
        
        Rz = N.array([  [cos, -sin, 0],\
                        [sin,  cos, 0],\
                        [0,      0, 1] ])

        return Rz
    
    

    
    def get_3d_extrinsic_rotation_matrix(self,alpha,beta,gamma):

        Rx = self.get_Rx(alpha)
        Ry = self.get_Ry(beta)
        Rz = self.get_Rz(gamma)

        R = N.dot(N.dot(Rz,Ry),Rx)  # 3D rotation matrix

        return R

    
#    def rotate3d_extrinsic(self,alpha,beta,gamma):
    def rotate3d_extrinsic(self):


#        self.reset_grid()
        
        alpha, beta, gamma = self.tiltx, 0., self.tiltz
        self.R = self.get_3d_extrinsic_rotation_matrix(alpha,beta,gamma)
#        self.Ri = N.linalg.inv(self.R)
        self.Ri = self.R.T
#        print "R, Ri = ", self.R, self.Ri
        
        Xflat = self.X.flatten()
        Yflat = self.Y.flatten()
        Zflat = self.Z.flatten()
        vectors = N.array((Xflat,Yflat,Zflat))

#        newvectors = N.zeros((len(vectors),3))
#
#        for j,v in enumerate(vectors):
##            nv = N.dot(R,v)
##            print "j, v, nv: ", j, v, nv
##            newvectors[j,:] = N.dot(self.R,v) #nv
#            newvectors[j,:] = N.dot(self.Ri,v) #nv
#
##        self.Xnew = newvectors[:,0].reshape(self.X.shape)
##        self.Ynew = newvectors[:,1].reshape(self.X.shape)
##        self.Znew = newvectors[:,2].reshape(self.X.shape)

        newvectors = N.dot(self.Ri,vectors) #nv

        self.X = newvectors[0,:].reshape(self.X.shape)
        self.Y = newvectors[1,:].reshape(self.X.shape)
        self.Z = newvectors[2,:].reshape(self.X.shape)

#        self.reset_grid()

#        super(HERE,self).build_kdtree()  # The kdtree must be updated

#        self.X2 = self.X * self.X
#        self.Y2 = self.Y * self.Y
#        self.Z2 = self.Z * self.Z
#
#        self.XYZ2sum = self.X2 + self.Y2 + self.Z2


#    def rotate3d(self):
#        
#        """Rotate the unshifted rho (i.e. one that's centered on (0,0) by
#        angles tiltx and tiltz. tilty is only needed for density
#        distributions that are not axisymmetric around the y axis,
#        maybe we'll implement such distros later.
#        """
#
#        if self.tiltx != 0.:
#
#            # when rotating about the x-axis (pointing to the right in the image), the X coordinates are invariant.
#            self.Y, self.Z = self.rotate2d(self.Y,self.Z,N.radians(self.tiltx))
##            self.Y, self.Z = self.rotate2d(self.Z,self.Y,N.radians(self.tiltx))
#
#        if self.tiltz != 0.:
#
#            # when rotating about the z-axis (pointing towards observer), the Z coordinates are invariant.
#            self.X, self.Y = self.rotate2d(self.X,self.Y,N.radians(self.tiltz))



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


class Container:

    def __init__(self,paramnames):

        self.params = OrderedDict(zip(paramnames,[None]*len(paramnames)))
        self.set_all()

    def __call__(self,args):

        print "len(args), len(self.params.keys()) = ", len(args), len(self.params.keys())
        print "args = ", args
        print "self.params.keys() = ", self.params.keys()
        
        if len(args) == len(self.params.keys()):
            try:
                self.update(args)
            except:
                raise
        
    def update(self,args):
        self.params.update(zip(self.params.keys(),args))
        self.set_all()

    def set_all(self):
        for k,v in self.params.items():
            self.set(k,v)
            
    def set(self,obj,val):
        setattr(self,obj,val)




    
class Cube:

    """Generic Cube of nx*ny*nz pixels. Models should inherit from this
    class, as it provides common members and methods,
    e.g. normalize(), set_rho(), shift(), and rotate().
    """

    def __init__(self,npix,transform=None,normalize='none',buildkdtree=False,computeR=False):
#    def __init__(self,npix,paramnames,transform=None,normalize='none',buildkdtree=False,computeR=False):

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
        
#        self.params = Container(paramnames)
        

        
#CALL    def __call__(self,themodels):
#CALL
#CALL        """When called with parameters"""
#CALL
#CALL        # put this into __call__?
#CALL        # also store in init func ref to a 'transform' function?
#CALL        # TODO: separate this in init and call
#CALL        self.rho = N.zeros((self.npix,self.npix,self.npix))
#CALL
#CALL        self.models = []
#CALL        for m in themodels:
#CALL#            print m
#CALL
#CALL            typ, args = m[0], m[1:]
#CALL            print "typ: ", typ
#CALL            print "args: ", args
#CALL#            model = getattr(models,typ)(self.X, self.Y, self.Z)
#CALL            model = getattr(models,typ)()
#CALL            model(*args)  # model.rho (3D) is normalized to model_.weight
#CALL
#CALL            self.rho += model.rho
#CALL#            model.rhomap = N.sum(model.rho**2,axis=-1)
#CALL
#CALL#            if normalize == 'peak':
#CALL#                model_.rhomap /= model_.rhomap.max()
#CALL#                model_.rhomap *= model_.weight
#CALL
#CALL            self.models.append(model)  # CAUTION: not yet tested
#CALL
#CALL        self.rhoraw = self.rho.copy()
#CALL
#CALL        if self.transform is not None:
#CALL            self.rho = self.transform(self.rho.copy())
#CALL
#CALL        self.image_raw = N.sum(self.rhoraw,axis=-1)
#CALL        self.image = N.sum(self.rho,axis=-1)  #self.image_raw.copy()
#CALL#G        self.image = self.image / self.image.max()
#CALL#        self.image = self.image / self.image.sum()
#CALL
#CALL##        self.image = N.sum([m.rhomap for m in self.models],axis=0)
#CALL#        self.image = N.sum([m.rhomap for m in self.models],axis=0)
#CALL#?        self.mask = (self.image > 0.)



#    def reset_grid(self):
#
#        self.X, self.Y, self.Z = N.meshgrid(self.x,self.x,self.x,indexing='xy')  # three 3d coordinate arrays to 
##        self.X, self.Y, self.Z = N.meshgrid(self.x,self.x,self.x,indexing='xy',sparse=True)  # three 3d coordinate arrays to
#
#
#        self.build_kdtree()
#
#
##        self.distance = N.sqrt(self.X**2+self.Y**2+self.Z**2)
#       
##        self.set_rho(val=0.)
        


    def build_kdtree(self):

#        self.mg = N.meshgrid(*self.midpoints.unit,indexing='ij')
#        midgrid = N.vstack(self.mg).reshape(self.ndim,-1).T
#        self.kdtree = S.spatial.cKDTree(midgrid)   # kd-tree is in unit hypercell space

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

#        print "In shift(). hasattr(self,'X'), hasattr(self,'xoff') = ", hasattr(self,'X'), hasattr(self,'xoff')

        if hasattr(self,'xoff'):
#            "has xoff"
            self.rho = ndimage.shift(self.rho, (0,self.xoff/self.pixelscale,0), order=1)
            
        if hasattr(self,'yoff'):
#            "has yoff"
            self.rho = ndimage.shift(self.rho, (self.yoff/self.pixelscale,0,0), order=1)

            
    def rotate3d(self):

        if hasattr(self,'tiltx') and self.tiltx != 0:
            self.rho = ndimage.rotate(self.rho, self.tiltx, axes=(0,2), reshape=False, order=1)

        if hasattr(self,'tilty') and self.tilty != 0:
            self.rho = ndimage.rotate(self.rho, self.tilty, axes=(1,2), reshape=False, order=1)

        if hasattr(self,'tiltz') and self.tiltz != 0:
            self.rho = ndimage.rotate(self.rho, self.tiltz, axes=(0,1), reshape=False, order=1)
            
            



class CubePREVIOUS:

    
    """3D Cube to hold the total rho(x,y,z).

    Example:
    --------
    Two different shells:

    cube = rhocube.Cube( 200, ( ('TruncatedNormalShell',0.4,0.03,0,1,0.5,0.2,1),\
                                ('ConstantDensityShell',0.3,0.5,,-0.2,-0.4,2) ) )

    pylab.imshow(cube.allmap,origin='lower',extent=extent,cmap=matplotlib.cm.gray)

    """

    def __init__(self,npix,transform=None,normalize='none'):  # TODO: get npix from the FITS file to be fitted

        """

        Example:
        --------

        themodels = (\
                     ('TruncatedNormalShell',0.8,0.02,0,1,0.,0.,1.),\
                     ('ConstantDensityShell',0.4,0.6,,(-0.3,0.2),0.5)
                    )

        """

        self.npix = npix
#        self.npix = npix + 1  # CAUTION
        if self.npix % 2 == 0:   # make sure we have a central pixel, i.e. an odd number of pixels along the axes
            self.npix += 1


        # center -/+ 1 radius to each side
        self.x = N.linspace(-1.,1,self.npix)
        self.X, self.Y, self.Z = N.meshgrid(self.x,self.x,self.x,indexing='xy')  # three 3d coordinate arrays to 
#        self.X, self.Y, self.Z = N.meshgrid(self.x,self.x,self.x,indexing='xy',sparse=True)  # three 3d coordinate arrays to 

        self.ndshape = self.X.shape

        self.extent = (self.x.min(),self.x.max(),self.x.min(),self.x.max())

        self.transform = transform # func ref



        
        
    def __call__(self,themodels):

        """When called with parameters"""

        # put this into __call__?
        # also store in init func ref to a 'transform' function?
        # TODO: separate this in init and call
        self.rho = N.zeros((self.npix,self.npix,self.npix))

        self.models = []
        for m in themodels:
#            print m

            typ, args = m[0], m[1:]
#P            print "typ: ", typ
#P            print "args: ", args
            model = getattr(models,typ)(self.X, self.Y, self.Z)
            model(*args)  # model.rho (3D) is normalized to model_.weight

            self.rho += model.rho
#            model.rhomap = N.sum(model.rho**2,axis=-1)

#            if normalize == 'peak':
#                model_.rhomap /= model_.rhomap.max()
#                model_.rhomap *= model_.weight

            self.models.append(model)  # CAUTION: not yet tested

        self.rhoraw = self.rho.copy()

        if self.transform is not None:
            self.rho = self.transform(self.rho.copy())

        self.image_raw = N.sum(self.rhoraw,axis=-1)
        self.image = N.sum(self.rho,axis=-1)  #self.image_raw.copy()
#G        self.image = self.image / self.image.max()
#        self.image = self.image / self.image.sum()

##        self.image = N.sum([m.rhomap for m in self.models],axis=0)
#        self.image = N.sum([m.rhomap for m in self.models],axis=0)
#?        self.mask = (self.image > 0.)


    def add_noise(self,type='gauss',mode='std',magnitude=0.1):

        """Add random noise to final image.

        Leaves self.image alone, and creates self.errimage (just the
        errors image) and self.noisyimage (self.image +
        self.errimage).

        Parameters:
        -----------
        type : str
            Noise statistic. Currently only 'gauss' is. implemented

        mode : str
            Use one of 'std', 'max', 'mean', 'median' of all non-zero
            valued image pixels to compute the error amplitude.

        magnitude : float
            Scaling factor applied to the error amplitude (defined by
            'mode'). The error image will be

               error = magnitude * mode(image) * RND
        
            where RND is a random variate per pixel drawn from the
            noise statistic defined by 'type'. The noisy image will be

               noisyimage = error + image
        """

        self.rnd = N.random.randn
        if type != 'gauss':
            warnings.warn("Only 'gauss' implemented as noise type at the moment.")

        assert (mode in ('std','max','median','mean'))

        self.mag = magnitude
        self.errmode = getattr(N,mode)
        self.sig = self.mag * self.errmode(self.image[self.mask])
        self.errimage = self.sig * self.rnd(*self.image.shape)
        self.noisyimage = self.errimage + self.image



#def savefile(image,outnames=('out.fits','out.png')):
#def savefile(image,outname='out.fits'):
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


#    def savepng(image,outname,**kwargs):
#        # TODO: get some kwargs from user and apply them, e.g. cmap etc.
#        import pylab as p
#
#        p.imshow(image,origin='lower',extent=extent,cmap=matplotlib.cm.grey_r)
#
#        p.savefig(outname,dpi=100)
#        
#
#    for outname in outnames:
#
#        print "Saving %s file..." % outname
#
#        if outname.endswith('fits'):
#            savefits(image,outname)
#
#        elif outname.endswith('png'):
#            savepng(image,outname)
