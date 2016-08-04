"""Model classes for 3D density distribution."""

__author__  = "Robert Nikutta, Claudia Agliozzo"
__version__ = "2015-05-11"

import numpy as N

from numpy import cross,eye,dot
from scipy import ndimage
from scipy.linalg import expm3,norm

#from rhocube import CubeNDI #*
from rhocube import Cube
#import rhocube

#######################################
# Models for 3D density distributions #
#######################################


#class Cube3D_Smart:
#
#    """Generic Cube of nx*ny*nz pixels. Models should inherit from this
#    class, as it provides common members and methods,
#    e.g. normalize(), set_rho(), shift(), and rotate().
#    """
#
#    def __init__(self,npixX,Y,Z):
#
#        """Initialize a 3D cube of voxels using X,Y,Z corrdinate arrays (each
#        a 3D array).
#        """
#
##        self.X, self.Y, self.Z = X, Y, Z
#        self.X, self.Y, self.Z = X.copy(), Y.copy(), Z.copy()
#        self.R = N.sqrt(self.X**2+self.Y**2+self.Z**2)
#        self.X2 = self.X * self.X
##        self.Y2 = self.Y * self.Y
#        self.Z2 = self.Z * self.Z
#        self.rho = self.set_rho(val=0.)
#        
#
#    def set_rho(self,val=0.):
#
#        """(Re)set all voxels in self.rho (3D cube) to 'val' (default: 0.)
#        """
#
#        self.rho = val*N.ones(self.X.shape)
#
#
#    def normalize(self):
#
#        """Normalize the sum of all voxel values to self.weight.
#        """
#
#        self.mass = self.rho.sum() / self.weight
#        self.rho /= self.mass
#
#
#    def shift(self):
#
#        """Shift density distribution by xoff and yoff. If rotation is needed,
#        shift first, then rotate.
#        """
#
##GOOD        if abs(self.xoff) > 0:
##GOOD            self.X -= self.xoff
##GOOD
##GOOD        if abs(self.yoff) > 0:
##GOOD            self.Y -= self.yoff
#
#            
#        if hasattr(self,'xoff'):
#            self.X -= self.xoff
#
#        if hasattr(self,'yoff'):
#            self.Y -= self.yoff
#
#            
#
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
#
#
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
#
#        if self.tiltz != 0.:
#
#            # when rotating about the z-axis (pointing towards observer), the Z coordinates are invariant.
#            self.X, self.Y = self.rotate2d(self.X,self.Y,N.radians(self.tiltz))


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
        self.R = N.sqrt(self.X**2+self.Y**2+self.Z**2)
#        self.X2 = self.X * self.X
#        self.Y2 = self.Y * self.Y
#        self.Z2 = self.Z * self.Z
#        self.XYZ2sum = self.X2 + self.Y2 + self.Z2

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

#GOOD        if abs(self.xoff) > 0:
#GOOD            self.X -= self.xoff
#GOOD
#GOOD        if abs(self.yoff) > 0:
#GOOD            self.Y -= self.yoff

            
        if hasattr(self,'xoff'):
            self.X -= self.xoff

        if hasattr(self,'yoff'):
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


#TMP        super(HERE,self).build_kdtree()  # The kdtree must be updated

        
#        self.X2 = self.X * self.X
#        self.Y2 = self.Y * self.Y
#        self.Z2 = self.Z * self.Z
#
#        self.XYZ2sum = self.X2 + self.Y2 + self.Z2

    def rotate3d(self):
        
        """Rotate the unshifted rho (i.e. one that's centered on (0,0) by
        angles tiltx and tiltz. tilty is only needed for density
        distributions that are not axisymmetric around the y axis,
        maybe we'll implement such distros later.
        """

        if self.tiltx != 0.:

            # when rotating about the x-axis (pointing to the right in the image), the X coordinates are invariant.
            self.Y, self.Z = self.rotate2d(self.Y,self.Z,N.radians(self.tiltx))
#            self.Y, self.Z = self.rotate2d(self.Z,self.Y,N.radians(self.tiltx))

        if self.tiltz != 0.:

            # when rotating about the z-axis (pointing towards observer), the Z coordinates are invariant.
            self.X, self.Y = self.rotate2d(self.X,self.Y,N.radians(self.tiltz))




#PREVIOUSclass Cube:
#PREVIOUS
#PREVIOUS    """Generic Cube of nx*ny*nz pixels. Models should inherit from this
#PREVIOUS    class, as it provides common members and methods,
#PREVIOUS    e.g. normalize(), set_rho(), shift(), and rotate().
#PREVIOUS    """
#PREVIOUS
#PREVIOUS    def __init__(self,X,Y,Z):
#PREVIOUS
#PREVIOUS        """Initialize a 3D cube of voxels using X,Y,Z corrdinate arrays (each
#PREVIOUS        a 3D array).
#PREVIOUS        """
#PREVIOUS
#PREVIOUS        self.npix = npix
#PREVIOUS        if self.npix % 2 == 0:   # make sure we have a central pixel, i.e. an odd number of pixels along the axes
#PREVIOUS            self.npix += 1
#PREVIOUS
#PREVIOUS
#PREVIOUS        # center -/+ 1 radius to each side
#PREVIOUS        self.x = N.linspace(-1.,1,self.npix)
#PREVIOUS        self.X, self.Y, self.Z = N.meshgrid(self.x,self.x,self.x,indexing='xy')  # three 3d coordinate arrays to 
#PREVIOUS#        self.X, self.Y, self.Z = N.meshgrid(self.x,self.x,self.x,indexing='xy',sparse=True)  # three 3d coordinate arrays to
#PREVIOUS
#PREVIOUS        self.R = N.sqrt(self.X**2+self.Y**2+self.Z**2)
#PREVIOUS
#PREVIOUS        self.ndshape = self.X.shape
#PREVIOUS
#PREVIOUS        self.extent = (self.x.min(),self.x.max(),self.x.min(),self.x.max())
#PREVIOUS
#PREVIOUS        self.transform = transform # func ref
#PREVIOUS
#PREVIOUS
#PREVIOUS##        self.X, self.Y, self.Z = X, Y, Z
#PREVIOUS#        self.X, self.Y, self.Z = X.copy(), Y.copy(), Z.copy()
#PREVIOUS#        self.R = N.sqrt(self.X**2+self.Y**2+self.Z**2)
#PREVIOUS##        self.X2 = self.X * self.X
#PREVIOUS##        self.Y2 = self.Y * self.Y
#PREVIOUS##        self.Z2 = self.Z * self.Z
#PREVIOUS##        self.XYZ2sum = self.X2 + self.Y2 + self.Z2
#PREVIOUS
#PREVIOUS        self.rho = self.set_rho(val=0.)
#PREVIOUS        
#PREVIOUS
#PREVIOUS    def __call__(self,themodels):
#PREVIOUS
#PREVIOUS        """When called with parameters"""
#PREVIOUS
#PREVIOUS        # put this into __call__?
#PREVIOUS        # also store in init func ref to a 'transform' function?
#PREVIOUS        # TODO: separate this in init and call
#PREVIOUS        self.rho = N.zeros((self.npix,self.npix,self.npix))
#PREVIOUS
#PREVIOUS        self.models = []
#PREVIOUS        for m in themodels:
#PREVIOUS#            print m
#PREVIOUS
#PREVIOUS            typ, args = m[0], m[1:]
#PREVIOUS#P            print "typ: ", typ
#PREVIOUS#P            print "args: ", args
#PREVIOUS            model = getattr(models,typ)(self.X, self.Y, self.Z)
#PREVIOUS            model(*args)  # model.rho (3D) is normalized to model_.weight
#PREVIOUS
#PREVIOUS            self.rho += model.rho
#PREVIOUS#            model.rhomap = N.sum(model.rho**2,axis=-1)
#PREVIOUS
#PREVIOUS#            if normalize == 'peak':
#PREVIOUS#                model_.rhomap /= model_.rhomap.max()
#PREVIOUS#                model_.rhomap *= model_.weight
#PREVIOUS
#PREVIOUS            self.models.append(model)  # CAUTION: not yet tested
#PREVIOUS
#PREVIOUS        self.rhoraw = self.rho.copy()
#PREVIOUS
#PREVIOUS        if self.transform is not None:
#PREVIOUS            self.rho = self.transform(self.rho.copy())
#PREVIOUS
#PREVIOUS        self.image_raw = N.sum(self.rhoraw,axis=-1)
#PREVIOUS        self.image = N.sum(self.rho,axis=-1)  #self.image_raw.copy()
#PREVIOUS#G        self.image = self.image / self.image.max()
#PREVIOUS#        self.image = self.image / self.image.sum()
#PREVIOUS
#PREVIOUS##        self.image = N.sum([m.rhomap for m in self.models],axis=0)
#PREVIOUS#        self.image = N.sum([m.rhomap for m in self.models],axis=0)
#PREVIOUS#?        self.mask = (self.image > 0.)
#PREVIOUS
#PREVIOUS
#PREVIOUS
#PREVIOUS    def set_rho(self,val=0.):
#PREVIOUS
#PREVIOUS        """(Re)set all voxels in self.rho (3D cube) to 'val' (default: 0.)
#PREVIOUS        """
#PREVIOUS
#PREVIOUS        self.rho = val*N.ones(self.X.shape)
#PREVIOUS
#PREVIOUS
#PREVIOUS    def normalize(self):
#PREVIOUS
#PREVIOUS        """Normalize the sum of all voxel values to self.weight.
#PREVIOUS        """
#PREVIOUS
#PREVIOUS        self.mass = self.rho.sum() / self.weight
#PREVIOUS        self.rho /= self.mass
#PREVIOUS
#PREVIOUS
#PREVIOUS    def shift(self):
#PREVIOUS
#PREVIOUS        """Shift density distribution by xoff and yoff. If rotation is needed,
#PREVIOUS        shift first, then rotate.
#PREVIOUS        """
#PREVIOUS
#PREVIOUS#GOOD        if abs(self.xoff) > 0:
#PREVIOUS#GOOD            self.X -= self.xoff
#PREVIOUS#GOOD
#PREVIOUS#GOOD        if abs(self.yoff) > 0:
#PREVIOUS#GOOD            self.Y -= self.yoff
#PREVIOUS
#PREVIOUS            
#PREVIOUS        if hasattr(self,'xoff'):
#PREVIOUS            self.X -= self.xoff
#PREVIOUS
#PREVIOUS        if hasattr(self,'yoff'):
#PREVIOUS            self.Y -= self.yoff
#PREVIOUS
#PREVIOUS            
#PREVIOUS
#PREVIOUS    def rotate2d(self,A,B,theta):
#PREVIOUS
#PREVIOUS        """Intrinsic rotation in 2D.
#PREVIOUS
#PREVIOUS        For rotation in 3D about one of the principal axes, only the
#PREVIOUS        coordinates of the two other axes are needed (the coordinates
#PREVIOUS        along the rotation axis won't be affected.
#PREVIOUS
#PREVIOUS        Parameters:
#PREVIOUS        -----------
#PREVIOUS        A, B : {floats, float arrays}
#PREVIOUS            A, B are coordinate arrays (which can be 3D arrays) of the
#PREVIOUS            two axes affected by the rotation to be performed. E.g. if
#PREVIOUS            the rotation is about the x-axis, only the y and z
#PREVIOUS            coordinates will be affected, and thus A and B should be
#PREVIOUS            arrays (of any shape) containing the y and z coordinates,
#PREVIOUS            etc.
#PREVIOUS
#PREVIOUS        theta : float
#PREVIOUS            Rotation angle in radians.
#PREVIOUS
#PREVIOUS        Example:
#PREVIOUS        --------
#PREVIOUS
#PREVIOUS        Given x, y, z coordinates, rotate the coordinate system by 30
#PREVIOUS        degrees about the x-axis:
#PREVIOUS
#PREVIOUS            x, y, z = N.linspace(-1,1,11), N.linspace(0,10,21), N.linspace(3.5,4.5,15)
#PREVIOUS            X, Y, Z = N.meshgrid(x,y,z)            # 3D array cubes of x, y, values at every voxel
#PREVIOUS            Xp = X                                 # Xp (X-prime) is invariant
#PREVIOUS            Yp, Zp = rotate2d(Y,Z,N.radians(30))   # rotated Y and Z coordinates
#PREVIOUS
#PREVIOUS        """
#PREVIOUS
#PREVIOUS        cos_ = N.cos(theta)
#PREVIOUS        sin_ = N.sin(theta)
#PREVIOUS        Ap = A*cos_ - B*sin_
#PREVIOUS        Bp = A*sin_ + B*cos_
#PREVIOUS
#PREVIOUS        return Ap, Bp
#PREVIOUS
#PREVIOUS
#PREVIOUS    def get_trigs(self,deg):
#PREVIOUS
#PREVIOUS        rad = N.radians(deg)
#PREVIOUS        
#PREVIOUS        return N.sin(rad), N.cos(rad)
#PREVIOUS    
#PREVIOUS
#PREVIOUS    def get_Rx(self,deg):
#PREVIOUS
#PREVIOUS        sin, cos = self.get_trigs(deg)
#PREVIOUS        
#PREVIOUS        Rx = N.array([  [1,   0,    0],\
#PREVIOUS                        [0, cos, -sin],\
#PREVIOUS                        [0, sin,  cos] ])
#PREVIOUS
#PREVIOUS        return Rx
#PREVIOUS    
#PREVIOUS        
#PREVIOUS    def get_Ry(self,deg):
#PREVIOUS
#PREVIOUS        sin, cos = self.get_trigs(deg)
#PREVIOUS        
#PREVIOUS        Ry = N.array([  [cos,  0, sin],\
#PREVIOUS                        [0,    1,   0],\
#PREVIOUS                        [-sin, 0, cos] ])
#PREVIOUS        return Ry
#PREVIOUS    
#PREVIOUS
#PREVIOUS    def get_Rz(self,deg):
#PREVIOUS
#PREVIOUS        sin, cos = self.get_trigs(deg)
#PREVIOUS        
#PREVIOUS        Rz = N.array([  [cos, -sin, 0],\
#PREVIOUS                        [sin,  cos, 0],\
#PREVIOUS                        [0,      0, 1] ])
#PREVIOUS
#PREVIOUS        return Rz
#PREVIOUS    
#PREVIOUS    
#PREVIOUS
#PREVIOUS    
#PREVIOUS    def get_3d_extrinsic_rotation_matrix(self,alpha,beta,gamma):
#PREVIOUS
#PREVIOUS        Rx = self.get_Rx(alpha)
#PREVIOUS        Ry = self.get_Ry(beta)
#PREVIOUS        Rz = self.get_Rz(gamma)
#PREVIOUS
#PREVIOUS        R = N.dot(N.dot(Rz,Ry),Rx)  # 3D rotation matrix
#PREVIOUS
#PREVIOUS        return R
#PREVIOUS
#PREVIOUS    
#PREVIOUS#    def rotate3d_extrinsic(self,alpha,beta,gamma):
#PREVIOUS    def rotate3d_extrinsic(self):
#PREVIOUS
#PREVIOUS
#PREVIOUS        alpha, beta, gamma = self.tiltx, 0., self.tiltz
#PREVIOUS        self.R = self.get_3d_extrinsic_rotation_matrix(alpha,beta,gamma)
#PREVIOUS#        self.Ri = N.linalg.inv(self.R)
#PREVIOUS        self.Ri = self.R.T
#PREVIOUS#        print "R, Ri = ", self.R, self.Ri
#PREVIOUS        
#PREVIOUS        Xflat = self.X.flatten()
#PREVIOUS        Yflat = self.Y.flatten()
#PREVIOUS        Zflat = self.Z.flatten()
#PREVIOUS        vectors = N.array((Xflat,Yflat,Zflat))
#PREVIOUS
#PREVIOUS#        newvectors = N.zeros((len(vectors),3))
#PREVIOUS#
#PREVIOUS#        for j,v in enumerate(vectors):
#PREVIOUS##            nv = N.dot(R,v)
#PREVIOUS##            print "j, v, nv: ", j, v, nv
#PREVIOUS##            newvectors[j,:] = N.dot(self.R,v) #nv
#PREVIOUS#            newvectors[j,:] = N.dot(self.Ri,v) #nv
#PREVIOUS#
#PREVIOUS##        self.Xnew = newvectors[:,0].reshape(self.X.shape)
#PREVIOUS##        self.Ynew = newvectors[:,1].reshape(self.X.shape)
#PREVIOUS##        self.Znew = newvectors[:,2].reshape(self.X.shape)
#PREVIOUS
#PREVIOUS        newvectors = N.dot(self.Ri,vectors) #nv
#PREVIOUS
#PREVIOUS        self.X = newvectors[0,:].reshape(self.X.shape)
#PREVIOUS        self.Y = newvectors[1,:].reshape(self.X.shape)
#PREVIOUS        self.Z = newvectors[2,:].reshape(self.X.shape)
#PREVIOUS
#PREVIOUS
#PREVIOUS        super(HERE,self).build_kdtree()  # The kdtree must be updated
#PREVIOUS
#PREVIOUS        
#PREVIOUS#        self.X2 = self.X * self.X
#PREVIOUS#        self.Y2 = self.Y * self.Y
#PREVIOUS#        self.Z2 = self.Z * self.Z
#PREVIOUS#
#PREVIOUS#        self.XYZ2sum = self.X2 + self.Y2 + self.Z2
#PREVIOUS
#PREVIOUS    def rotate3d(self):
#PREVIOUS        
#PREVIOUS        """Rotate the unshifted rho (i.e. one that's centered on (0,0) by
#PREVIOUS        angles tiltx and tiltz. tilty is only needed for density
#PREVIOUS        distributions that are not axisymmetric around the y axis,
#PREVIOUS        maybe we'll implement such distros later.
#PREVIOUS        """
#PREVIOUS
#PREVIOUS        if self.tiltx != 0.:
#PREVIOUS
#PREVIOUS            # when rotating about the x-axis (pointing to the right in the image), the X coordinates are invariant.
#PREVIOUS            self.Y, self.Z = self.rotate2d(self.Y,self.Z,N.radians(self.tiltx))
#PREVIOUS#            self.Y, self.Z = self.rotate2d(self.Z,self.Y,N.radians(self.tiltx))
#PREVIOUS
#PREVIOUS        if self.tiltz != 0.:
#PREVIOUS
#PREVIOUS            # when rotating about the z-axis (pointing towards observer), the Z coordinates are invariant.
#PREVIOUS            self.X, self.Y = self.rotate2d(self.X,self.Y,N.radians(self.tiltz))



#GOODclass ConstantDensityShell(Cube3D):
#GOOD
#GOOD    # supply X,Y,Z to init instead of the param values. SUpply the param values instead to call (which may use update)
#GOOD
#GOOD    def __init__(self,X,Y,Z):
#GOOD        
#GOOD        """Truncated Normal Shell
#GOOD
#GOOD        A spherical shell with radius 'r', and Gaussian density
#GOOD        fall-off from r. The width of the Normal is 'width'. The PDF
#GOOD        of the Normal is truncated at 'clip' values.
#GOOD        
#GOOD        Parameters:
#GOOD        -----------
#GOOD        rin : float
#GOOD           Radius at which the shell is centered, in fractions of
#GOOD           unity, i.e. between 0 and 1.
#GOOD
#GOOD        width : float
#GOOD           Thickness of the shell, in same units as r.
#GOOD
#GOOD        xoff, yoff : floats
#GOOD           x and y offsets of the shell center from (0,0). Positive
#GOOD           values are to the right and up, negative to the left and
#GOOD           down. In units if unity (remember that the image is within
#GOOD           [-1,1]. Defaults: xoff = yoff = 0.
#GOOD
#GOOD        weight : float
#GOOD           Normalize the total (relative) mass contained in the shell
#GOOD           to this value. The total mass is the sum of rho over all
#GOOD           pixels (in 3D). This is useful if you e.g. want to have
#GOOD           more than one component, and wish to distribute different
#GOOD           amounts of mass inside each one. Default: weight=1.
#GOOD
#GOOD        """
#GOOD
#GOOD        Cube3D.__init__(self,X,Y,Z)
#GOOD
#GOOD
#GOOD    def __call__(self,rin,rout,xoff=0.,yoff=0.,weight=1):
#GOOD
#GOOD        """Return density rho at (x,y,z)"""
#GOOD
#GOOD        self.rin = rin
#GOOD        self.rout = rout
#GOOD
#GOOD        # helper arrays for finding the edges of the shell in get_rho()
#GOOD        self.Rin = self.rin * N.ones(self.X.shape)
#GOOD        self.Rout = self.rout * N.ones(self.X.shape)
#GOOD
#GOOD        self.xoff = xoff
#GOOD        self.yoff = yoff
#GOOD        self.weight = weight
#GOOD
#GOOD        self.sanity()
#GOOD
#GOOD        self.shift()
#GOOD
#GOOD        self.get_rho()  # get_rho should set self.rho (3D)
#GOOD#        self.normalize()
#GOOD
#GOOD        return self.rho
#GOOD
#GOOD
#GOOD    def sanity(self):
#GOOD
#GOOD        """Sanity checks for constant density-edge shell.
#GOOD        """
#GOOD
#GOOD        assert (0. < self.rin < self.rout)  # this automatically asserts that the shell thickness is finite and positive
#GOOD
#GOOD
#GOOD    def get_rho(self):
#GOOD
#GOOD        """Compute rho(x,y,z) in every voxel.
#GOOD        """
#GOOD
#GOOD        r = get_r((self.X,self.Y,self.Z),mode=2)  # mode=1,2 are fast, 3,4are slow
#GOOD#        co = (r >= self.rin) & (r <= self.rout)
#GOOD#        Rin = self.rin * N.ones(r.shape)
#GOOD#        Rout = self.rout * N.ones(r.shape)
#GOOD        co = ((r >= self.rin) & (r <= self.rout)) | N.isclose(r,self.Rout) | N.isclose(r,self.Rin)  # isclose also captures pixels at the very edge of the shell
#GOOD        self.set_rho(val=0.)  # default is 0.
#GOOD        self.rho[co] = 1.


    

class ConstantDensityShell(Cube):

    # supply X,Y,Z to init instead of the param values. SUpply the param values instead to call (which may use update)

#    def __init__(self,npix,paramnames,transform=None,buildkdtree=False,computeR=True):
    def __init__(self,npix,transform=None,buildkdtree=False,computeR=True):
        
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

#        Cube.__init__(self,npix,paramnames,transform=transform,buildkdtree=buildkdtree,computeR=computeR)
        Cube.__init__(self,npix,transform=transform,buildkdtree=buildkdtree,computeR=computeR)
        
#    def __call__(self,rin,rout,xoff=0.,yoff=0.,weight=1,smooth=1.):
    def __call__(self,rin,rout,xoff=0.,yoff=0.,weight=1,smooth=1.):

        """Return density rho at (x,y,z)"""

        self.rin = rin
        self.rout = rout

        # helper arrays for finding the edges of the shell in get_rho()
        self.Rin = self.rin * N.ones(self.X.shape)
        self.Rout = self.rout * N.ones(self.X.shape)

        self.xoff = xoff
        self.yoff = yoff
        self.weight = weight

        self.sanity()

        self.get_rho()  # get_rho should set self.rho (3D)
        self.shift()

#        self.normalize()

        return self.rho


    def sanity(self):

        """Sanity checks for constant density-edge shell.
        """

        assert (0. < self.rin < self.rout)  # this automatically asserts that the shell thickness is finite and positive


    def get_rho(self):

        """Compute rho(x,y,z) in every voxel.
        """

#        r = get_r((self.X,self.Y,self.Z),mode=2)  # mode=1,2 are fast, 3,4are slow
        co = ((self.R >= self.rin) & (self.R <= self.rout)) | N.isclose(self.R,self.Rout) | N.isclose(self.R,self.Rin)  # isclose also captures pixels at the very edge of the shell
        self.set_rho(val=0.)  # default is 0.
        self.rho[co] = 1.


class PowerLawShell(Cube):

    def __init__(self,npix,transform=None,buildkdtree=False,computeR=True):
        
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

#        Cube.__init__(self,npix,paramnames,transform=transform,buildkdtree=buildkdtree,computeR=computeR)
        Cube.__init__(self,npix,transform=transform,buildkdtree=buildkdtree,computeR=computeR)
        
#    def __call__(self,rin,rout,xoff=0.,yoff=0.,weight=1,smooth=1.):
    def __call__(self,rin,rout,exponent=-1.,xoff=0.,yoff=0.,weight=1,smooth=1.):

        """Return density rho at (x,y,z)"""

        self.rin = rin
        self.rout = rout
        self.exponent = exponent

        # helper arrays for finding the edges of the shell in get_rho()
        self.Rin = self.rin * N.ones(self.X.shape)
        self.Rout = self.rout * N.ones(self.X.shape)

        self.xoff = xoff
        self.yoff = yoff
        self.weight = weight

        self.sanity()

        self.get_rho()  # get_rho should set self.rho (3D)
        self.shift()

#        self.normalize()

        return self.rho


    def sanity(self):

        """Sanity checks for constant density-edge shell.
        """

        assert (0. < self.rin < self.rout)  # this automatically asserts that the shell thickness is finite and positive


    def get_rho(self):

        """Compute rho(x,y,z) in every voxel.
        """

#        print "self.exponent: ", self.exponent

        self.rho = self.R**self.exponent
#        r = get_r((self.X,self.Y,self.Z),mode=2)  # mode=1,2 are fast, 3,4are slow
        co = ((self.R >= self.rin) & (self.R <= self.rout)) | N.isclose(self.R,self.Rout) | N.isclose(self.R,self.Rin)  # isclose also captures pixels at the very edge of the shell
#        self.set_rho(val=0.)  # default is 0.
#        self.rho[co] = 1.
        self.rho[~co] = 0.




        
#GOODclass Helix3D(Cube3D):
#GOOD
#GOOD    # supply X,Y,Z to init instead of the param values. SUpply the param values instead to call (which may use update)
#GOOD
#GOOD    def __init__(self,X,Y,Z):
#GOOD        
#GOOD        """Truncated Normal Shell
#GOOD
#GOOD        A spherical shell with radius 'r', and Gaussian density
#GOOD        fall-off from r. The width of the Normal is 'width'. The PDF
#GOOD        of the Normal is truncated at 'clip' values.
#GOOD        
#GOOD        Parameters:
#GOOD        -----------
#GOOD        rin : float
#GOOD           Radius at which the shell is centered, in fractions of
#GOOD           unity, i.e. between 0 and 1.
#GOOD
#GOOD        width : float
#GOOD           Thickness of the shell, in same units as r.
#GOOD
#GOOD        xoff, yoff : floats
#GOOD           x and y offsets of the shell center from (0,0). Positive
#GOOD           values are to the right and up, negative to the left and
#GOOD           down. In units if unity (remember that the image is within
#GOOD           [-1,1]. Defaults: xoff = yoff = 0.
#GOOD
#GOOD        weight : float
#GOOD           Normalize the total (relative) mass contained in the shell
#GOOD           to this value. The total mass is the sum of rho over all
#GOOD           pixels (in 3D). This is useful if you e.g. want to have
#GOOD           more than one component, and wish to distribute different
#GOOD           amounts of mass inside each one. Default: weight=1.
#GOOD
#GOOD        """
#GOOD
#GOOD        Cube3D.__init__(self,X,Y,Z)
#GOOD
#GOOD#        self.masterrho = None
#GOOD
#GOOD    def __call__(self,rwind,rtube,pitch,tiltx=0.,tiltz=0.,weight=1,smooth=1.):
#GOOD
#GOOD        """Return density rho at (x,y,z)"""
#GOOD
#GOOD        self.rwind = rwind   # winding radius (in x,y)
#GOOD        self.rtube = rtube   # tube radius
#GOOD        self.pitch = pitch
#GOOD        self.tiltx = tiltx  # degrees
#GOOD        self.tiltz = tiltz  # degrees
#GOOD        self.weight = weight
#GOOD
#GOOD        self.smooth = smooth
#GOOD        
#GOOD        
#GOOD#        self.sanity()
#GOOD
#GOOD#        self.shift()
#GOOD
#GOOD
#GOOD        self.get_rho()  # get_rho should set self.rho (3D)
#GOOD#        self.normalize()
#GOOD
#GOOD###        self.rotate3d_extrinsic()
#GOOD
#GOOD        return self.rho
#GOOD
#GOOD
#GOOD    def sanity(self):
#GOOD
#GOOD        """Sanity checks for constant density-edge shell.
#GOOD        """
#GOOD
#GOOD        pass
#GOOD#        assert (0. < self.rin < self.rout)  # this automatically asserts that the shell thickness is finite and positive
#GOOD
#GOOD
#GOOD    def get_rho(self):
#GOOD
#GOOD        """Compute rho(x,y,z) in every voxel.
#GOOD        """
#GOOD        
#GOOD#        a = self.pitch/(2*N.pi)
#GOOD##        za = self.Z/N.float(a)
#GOOD#        t = N.unique(self.X.flatten())/N.float(a)
#GOOD#        x = self.rwind*N.cos(t)
#GOOD#        y = self.rwind*N.sin(t)
#GOOD#        z = a*t
#GOOD
#GOOD
#GOOD
#GOOD        a = self.pitch/(2*N.pi)
#GOOD        
#GOOD##        za = self.Z/N.float(a)
#GOOD#        t = self.Z/N.float(a)
#GOOD#        x = self.rwind*N.cos(t)
#GOOD#        y = self.rwind*N.sin(t)
#GOOD#        z = a*t
#GOOD
#GOOD        z = N.unique(self.Z.flatten())
#GOOD        t = z/N.float(a)
#GOOD        x = self.rwind*N.cos(t)
#GOOD        y = self.rwind*N.sin(t)
#GOOD#        z = a*t
#GOOD
#GOOD        print "t.shape,x.shape,y.shape,z.shape = ", t.shape,x.shape,y.shape,z.shape
#GOOD#        print "tS.shape,xS.shape,yS.shape,zS.shape = ", tS.shape,xS.shape,yS.shape,zS.shape
#GOOD        
#GOOD        
#GOOD        
#GOOD        self.set_rho(val=0.)  # default is 0.
#GOOD
#GOOD#        co = (((self.X - x)**2 + (self.Y - y)**2 + (self.Z - z)**2) <= self.rtube**2)
#GOOD
#GOOD        co = N.zeros(self.X.shape,dtype=N.bool)
#GOOD
#GOOD#        for pt in zip(xS,yS,zS):
#GOOD        for j in xrange(x.size):
#GOOD#            print "j, x[j],y[j],z[j] = ", j, x[j],y[j],z[j]
#GOOD
#GOOD            coaux = (((self.X - xS[j])**2 + (self.Y - yS[j])**2 + (self.Z - zS[j])**2) <= self.rtube**2)
#GOOD            co = co | coaux
#GOOD        
#GOOD        
#GOOD        
#GOOD#        co = ((x**2 + y**2 + z**2) <= self.rtube**2)
#GOOD        self.rho[co] = 1.
#GOOD
#GOOD##        co = (((self.X - self.rwind*N.cos(za))**2 + (self.Y - self.rwind*N.sin(za))**2) <= self.rtube**2)
#GOOD#        co = (((self.X - self.rwind*N.cos(t))**2 + (self.Y - self.rwind*N.sin(t))**2 + (self.Z-a*t)**2) <= self.rtube**2)
#GOOD##        co = (((self.X - self.rwind*N.cos(za))**2 + (self.Y - self.rwind*N.sin(za))**2 + (self.Z)**2) <= self.rtube**2)
#GOOD##        co = ( ( (self.X - self.rwind*N.cos(za))**2 + (self.Y - self.rwind*N.sin(za))**2) <= self.rtube**2 ) & N.abs(self.Z-self.tube)
#GOOD#        self.rho[co] = 1.
#GOOD##        self.masterrho = self.rho.copy()
#GOOD##        print "self.rho.sum(), mean, std =                                                      ", self.rho.sum(), self.rho.mean(), self.rho.std()
#GOOD##        co = (self.rho>0.)
#GOOD##        print "rho[rho>0].size/rho.size = ", self.rho[co].size/float(self.rho.size)
#GOOD
#GOOD        if self.smooth is not None:
#GOOD            self.rho = ndimage.gaussian_filter(self.rho,sigma=self.smooth,mode='nearest')
#GOOD#            self.rho = ndimage.gaussian_filter(self.masterrho,sigma=self.smooth)
#GOOD            
#GOOD#            print "GAUSSIAN sig, self.rho.sum(), mean, std =                                     ", self.smooth, self.rho.sum(), self.rho.mean(), self.rho.std()
#GOOD#            co = (self.rho>0.)
#GOOD#            print "rho[rho>0].size/rho.size = ", self.rho[co].size/float(self.rho.size)
#GOOD
#GOOD
#GOOD    def get_rhoOLD(self):
#GOOD
#GOOD        """Compute rho(x,y,z) in every voxel.
#GOOD        """
#GOOD        
#GOOD        a = self.pitch/(2*N.pi)
#GOOD#        za = self.Z/N.float(a)
#GOOD        t = self.Z/N.float(a)
#GOOD
#GOOD        self.set_rho(val=0.)  # default is 0.
#GOOD        
#GOOD#        co = (((self.X - self.rwind*N.cos(za))**2 + (self.Y - self.rwind*N.sin(za))**2) <= self.rtube**2)
#GOOD        co = (((self.X - self.rwind*N.cos(t))**2 + (self.Y - self.rwind*N.sin(t))**2 + (self.Z-a*t)**2) <= self.rtube**2)
#GOOD#        co = (((self.X - self.rwind*N.cos(za))**2 + (self.Y - self.rwind*N.sin(za))**2 + (self.Z)**2) <= self.rtube**2)
#GOOD#        co = ( ( (self.X - self.rwind*N.cos(za))**2 + (self.Y - self.rwind*N.sin(za))**2) <= self.rtube**2 ) & N.abs(self.Z-self.tube)
#GOOD        self.rho[co] = 1.
#GOOD#        self.masterrho = self.rho.copy()
#GOOD#        print "self.rho.sum(), mean, std =                                                      ", self.rho.sum(), self.rho.mean(), self.rho.std()
#GOOD#        co = (self.rho>0.)
#GOOD#        print "rho[rho>0].size/rho.size = ", self.rho[co].size/float(self.rho.size)
#GOOD
#GOOD        if self.smooth is not None:
#GOOD            self.rho = ndimage.gaussian_filter(self.rho,sigma=self.smooth)
#GOOD#            self.rho = ndimage.gaussian_filter(self.masterrho,sigma=self.smooth)
#GOOD            
#GOOD#            print "GAUSSIAN sig, self.rho.sum(), mean, std =                                     ", self.smooth, self.rho.sum(), self.rho.mean(), self.rho.std()
#GOOD#            co = (self.rho>0.)
#GOOD#            print "rho[rho>0].size/rho.size = ", self.rho[co].size/float(self.rho.size)
#GOOD
#GOOD
#GOOD
#GOOD#class ConicalHelix3DOLD(Cube3D):
#GOOD
#GOOD    # supply X,Y,Z to init instead of the param values. SUpply the param values instead to call (which may use update)
#GOOD
#GOOD    def __init__(self,X,Y,Z):
#GOOD        
#GOOD        """Truncated Normal Shell
#GOOD
#GOOD        A spherical shell with radius 'r', and Gaussian density
#GOOD        fall-off from r. The width of the Normal is 'width'. The PDF
#GOOD        of the Normal is truncated at 'clip' values.
#GOOD        
#GOOD        Parameters:
#GOOD        -----------
#GOOD        rin : float
#GOOD           Radius at which the shell is centered, in fractions of
#GOOD           unity, i.e. between 0 and 1.
#GOOD
#GOOD        width : float
#GOOD           Thickness of the shell, in same units as r.
#GOOD
#GOOD        xoff, yoff : floats
#GOOD           x and y offsets of the shell center from (0,0). Positive
#GOOD           values are to the right and up, negative to the left and
#GOOD           down. In units if unity (remember that the image is within
#GOOD           [-1,1]. Defaults: xoff = yoff = 0.
#GOOD
#GOOD        weight : float
#GOOD           Normalize the total (relative) mass contained in the shell
#GOOD           to this value. The total mass is the sum of rho over all
#GOOD           pixels (in 3D). This is useful if you e.g. want to have
#GOOD           more than one component, and wish to distribute different
#GOOD           amounts of mass inside each one. Default: weight=1.
#GOOD
#GOOD        """
#GOOD
#GOOD        Cube3D.__init__(self,X,Y,Z)
#GOOD
#GOOD#        self.masterrho = None
#GOOD
#GOOD        self.z = N.unique(self.Z.flatten())
#GOOD
#GOOD
#GOOD    def __call__(self,rtube,pitch,DR=1.,tiltx=0.,tiltz=0.,weight=1,smooth=1.):
#GOOD
#GOOD        """Return density rho at (x,y,z)"""
#GOOD
#GOOD#        self.rwind = rwind   # winding radius (in x,y)
#GOOD        self.rtube = rtube   # tube radius
#GOOD        self.pitch = pitch
#GOOD        self.DR = DR
#GOOD        self.tiltx = tiltx  # degrees
#GOOD        self.tiltz = tiltz  # degrees
#GOOD        self.weight = weight
#GOOD
#GOOD        self.smooth = smooth
#GOOD        
#GOOD        
#GOOD#        self.sanity()
#GOOD
#GOOD#        self.shift()
#GOOD
#GOOD        self.rotate3d_extrinsic()
#GOOD
#GOOD        self.get_rho()  # get_rho should set self.rho (3D)
#GOOD#        self.normalize()
#GOOD
#GOOD        return self.rho
#GOOD
#GOOD
#GOOD    def sanity(self):
#GOOD
#GOOD        """Sanity checks for constant density-edge shell.
#GOOD        """
#GOOD
#GOOD        pass
#GOOD#        assert (0. < self.rin < self.rout)  # this automatically asserts that the shell thickness is finite and positive
#GOOD
#GOOD
#GOOD    def get_rho(self):
#GOOD
#GOOD        """Compute rho(x,y,z) in every voxel.
#GOOD        """
#GOOD
#GOOD        a = self.pitch/(2*N.pi)
#GOOD
#GOOD        t = self.z/N.float(a)
#GOOD        x = self.DR*t*N.cos(t)
#GOOD        y = self.DR*t*N.sin(t)
#GOOD
#GOOD
#GOOD        self.set_rho(val=0.)  # default is 0.
#GOOD        
#GOOD        co = N.zeros(self.X.shape,dtype=N.bool)
#GOOD
#GOOD        for j in xrange(x.size):
#GOOD            coaux = (((self.X - x[j])**2 + (self.Y - y[j])**2 + (self.Z - self.z[j])**2) <= self.rtube**2)
#GOOD            co = co | coaux
#GOOD
#GOOD        print "(self.rho>0).size = "
#GOOD        self.rho[co] = 1.
#GOOD
#GOOD        if self.smooth is not None:
#GOOD            self.rho = ndimage.gaussian_filter(self.rho,sigma=self.smooth)
#GOOD
#GOOD
#GOOD    def get_rhoOLD(self):
#GOOD
#GOOD        """Compute rho(x,y,z) in every voxel.
#GOOD        """
#GOOD
#GOOD        a = self.pitch/(2*N.pi)
#GOOD        za = self.Z/N.float(a)
#GOOD
#GOOD
#GOOD        self.set_rho(val=0.)  # default is 0.
#GOOD        
#GOOD#        co = (((self.X - self.rwind*N.cos(za))**2 + (self.Y - self.rwind*N.sin(za))**2) <= self.rtube**2)
#GOOD        co = (((self.X - self.DR*self.Z*N.cos(za))**2 + (self.Y - self.DR*self.Z*N.sin(za))**2) <= self.rtube**2)
#GOOD
#GOOD        
#GOOD
#GOOD#        co = (   ((self.X - self.DR*self.Z*N.cos(za))**2 + (self.Y - self.DR*self.Z*N.sin(za))**2 + self.Z**2) <= self.rtube**2)
#GOOD        
#GOOD        self.rho[co] = 1.
#GOOD#        self.masterrho = self.rho.copy()
#GOOD#        print "self.rho.sum(), mean, std =                                                      ", self.rho.sum(), self.rho.mean(), self.rho.std()
#GOOD#        co = (self.rho>0.)
#GOOD#        print "rho[rho>0].size/rho.size = ", self.rho[co].size/float(self.rho.size)
#GOOD
#GOOD        if self.smooth is not None:
#GOOD            self.rho = ndimage.gaussian_filter(self.rho,sigma=self.smooth)
#GOOD#            self.rho = ndimage.gaussian_filter(self.masterrho,sigma=self.smooth)
#GOOD            
#GOOD#            print "GAUSSIAN sig, self.rho.sum(), mean, std =                                     ", self.smooth, self.rho.sum(), self.rho.mean(), self.rho.std()
#GOOD#            co = (self.rho>0.)
#GOOD#            print "rho[rho>0].size/rho.size = ", self.rho[co].size/float(self.rho.size)


#class ConicalHelix3D(Cube):
#
#    # supply X,Y,Z to init instead of the param values. SUpply the param values instead to call (which may use update)
#
##    def __init__(self,X,Y,Z):
#    def __init__(self,npix,transform=None):
#        
#        """Helical tube winding along a dual cone, with constant density inside the tube.
#
#
#        Scratch:
#
#        ROTATE
#        ndimage.rotate() uses angles in degrees.
#        Rotation of 3D array cube.rho by angle A in the plane defined by axes:
#
#        axes=(0,2) --> our x axis
#        axes=(0,1) --> our z axis
#        axes=(1,2) --> our y axis
#
#        So indexing seems to be (y,x,z)
#
#        SHIFT
#        ndimage.shift() works in pixel units.
#
#        
#        Parameters:
#        -----------
#        rin : float
#           Radius at which the shell is centered, in fractions of
#           unity, i.e. between 0 and 1.
#
#        width : float
#           Thickness of the shell, in same units as r.
#
#        xoff, yoff : floats
#           x and y offsets of the shell center from (0,0). Positive
#           values are to the right and up, negative to the left and
#           down. In units if unity (remember that the image is within
#           [-1,1]. Defaults: xoff = yoff = 0.
#
#        weight : float
#           Normalize the total (relative) mass contained in the shell
#           to this value. The total mass is the sum of rho over all
#           pixels (in 3D). This is useful if you e.g. want to have
#           more than one component, and wish to distribute different
#           amounts of mass inside each one. Default: weight=1.
#
#        """
#
##        Cube3D.__init__(self,X,Y,Z)
#        CubeNDI.__init__(self,npix,transform=transform)
#
#        self.z = N.unique(self.Z.flatten())
#
#
##    def __call__(self,rtube,pitch,DR=1.,tiltx=0.,tiltz=0.,weight=1,smooth=1.):
##    def __call__(self,h,Rbase,a,rtube,tiltx=0.,tiltz=0.,weight=1,smooth=1.):
#    def __call__(self,h,Rbase,nturns,rtube,tiltx=0.,tiltz=0.,xoff=0.,yoff=0.,weight=1,smooth=1.):
#
#        """Return density rho at (x,y,z)"""
#
#        self.reset_grid()
#        
#        self.h = h
#        self.Rbase = Rbase / self.h
#
#        self.nturns = nturns
##        self.a = nturns * (2*N.pi)  #/ self.h
#
#        self.rtube = rtube
#        self.tiltx = tiltx
#        self.tiltz = tiltz
#        self.xoff = xoff
#        self.yoff = yoff
#        
#        self.weight = weight
#        self.smooth = smooth
#        
##        self.sanity()
#        self.get_rho()  # get_rho should set self.rho (3D)
#        self.shift()
#        self.rotate3d_extrinsic()
#        print "self.X = ", self.X
##        self.get_rho()  # get_rho should set self.rho (3D)
##        self.normalize()
#
#        return self.rho
#
#    def sanity(self):
#
#        """Sanity checks for constant density-edge shell.
#        """
#
##        assert (0. < self.rin < self.rout)  # this automatically asserts that the shell thickness is finite and positive
#        pass
#
#    
##GOOD    def get_rho(self):
##GOOD
##GOOD        """Compute rho(x,y,z) in every voxel.
##GOOD        """
##GOOD
##GOOD        self.a = self.nturns * 2*N.pi/self.h
##GOOD
##GOOD        
##GOOD        delta = self.rtube/3.
##GOOD        th = self.a*self.h # / (2*N.pi)
##GOOD#        nsteps = int(2*th/delta)
##GOOD        nsteps = int(th/N.float(delta))
##GOOD#        print "nsteps = ", nsteps
##GOOD        
##GOOD        t = N.linspace(-th,th,2*nsteps+1)
##GOOD#        t = N.linspace(0,th,nsteps)
##GOOD        z = t/self.a
##GOOD        x = z*self.Rbase*N.cos(N.abs(t))
##GOOD        y = z*self.Rbase*N.sin(N.abs(t))
##GOOD
##GOOD#        x2 = x**2
##GOOD#        y2 = y**2
##GOOD#        z2 = z**2
##GOOD
##GOOD#        xyz2sum = x2 + y2 + z2
##GOOD
##GOOD        rtube2 = self.rtube**2
##GOOD
##GOOD        self.set_rho(val=0.)  # default is 0.
##GOOD
##GOOD#        co = (((self.X - x)**2 + (self.Y - y)**2 + (self.Z - z)**2) <= self.rtube**2)
##GOOD        co = N.zeros(self.X.shape,dtype=N.bool)
##GOOD
##GOOD#        for j,z_ in enumerate(z):
##GOOD#        for j in xrange(z.size):
##GOOD        for j,(x_,y_,z_) in enumerate(zip(x,y,z)):
##GOOD#            if N.abs(z_) <= self.h:
##GOOD#                coaux = (((self.X - x[j])**2 + (self.Y - y[j])**2 + (self.Z - z_)**2) <= self.rtube**2)
##GOOD                coaux = (((self.X - x_)**2 + (self.Y - y_)**2 + (self.Z - z_)**2) <= rtube2)
##GOOD#                coaux = ( ((self.X2 - 2*self.X*x[j] + x2[j]) + (self.Y2 - 2*self.Y*y[j] + y2[j]) + (self.Z2 - 2*self.Z*z[j] + z2[j])) <= rtube2 )
##GOOD#                coaux = ((self.XYZ2sum + xyz2sum[j] - 2*(self.X*x[j] + self.Y*y[j] + self.Z*z[j]))  <= rtube2 )
##GOOD#                coaux = ((self.XYZ2sum + xyz2sum[j] - 2*(self.X*x_ + self.Y*y_ + self.Z*z_))  <= rtube2 )
##GOOD                co = co | coaux
##GOOD
##GOOD#        print "(self.rho>0).size = "
##GOOD        self.rho[co] = 1.
##GOOD
##GOOD        if self.smooth is not None:
##GOOD            self.rho = ndimage.gaussian_filter(self.rho,sigma=self.smooth)
#
#
#
#    def get_rho(self):
#
#        """Compute rho(x,y,z) in every voxel.
#        """
#
#        self.a = self.nturns * 2*N.pi/self.h
#
#        
#        delta = self.rtube/3.
#        th = self.a*self.h # / (2*N.pi)
##        nsteps = int(2*th/delta)
#        nsteps = int(th/N.float(delta))
##        print "nsteps = ", nsteps
#        
#        t = N.linspace(-th,th,2*nsteps+1)
##        t = N.linspace(0,th,nsteps)
#        z = t/self.a
#        x = z*self.Rbase*N.cos(N.abs(t))
#        y = z*self.Rbase*N.sin(N.abs(t))
#
##        x2 = x**2
##        y2 = y**2
##        z2 = z**2
#
##        xyz2sum = x2 + y2 + z2
#
#        rtube2 = self.rtube**2
#
###        self.set_rho(val=0.)  # default is 0.
#
##        co = (((self.X - x)**2 + (self.Y - y)**2 + (self.Z - z)**2) <= self.rtube**2)
##        co = N.zeros(self.X.shape,dtype=N.bool)
#
#        co = N.zeros(N.prod(self.ndshape),dtype=N.float)
#
##        for j,(x_,y_,z_) in enumerate(zip(x,y,z)):
#        for j,pt in enumerate(zip(x,y,z)):
#
#            idxes = self.kdtree_query(pt,self.rtube)[1]
#            co[idxes] = 1.
#
#
#        self.rho = co.reshape(self.ndshape)
#
##        self.rho[co] = 1.
#
#        if self.smooth is not None:
#            self.rho = ndimage.gaussian_filter(self.rho,sigma=self.smooth)

            
def spiral3D(h,Rbase,nturns,rtube,envelope='dualcone'):


        a = nturns * 2*N.pi/h
        delta = rtube/3.
        th = a*h # / (2*N.pi)
        nsteps = int(th/N.float(delta))
        t = N.linspace(-th,th,2*nsteps+1)
        z = t/a

        if envelope == 'dualcone':
#            print "envelope = 'dualcone'"
            zprogression = z*(Rbase/h)
        elif envelope == 'cylinder':
#            print "envelope = 'cylinder'"
            zprogression = Rbase/h
        else:
            raise Exception, "Invalid value for 'envelope'. Must be either of: ['dualcone','cylinder']."
        
        x = zprogression * N.cos(N.abs(t))
        y = zprogression * N.sin(N.abs(t))

        return x, y, z


class Spiral3D:

#    def __init__(self,h,Rbase,nturns,envelope='dualcone',dual=True):
#    def __init__(self,h,Rbase,nturns,envelope='dualcone'):
    def __init__(self,envelope='dualcone'):
        """Set up invariant elements of the spiral."""

        self.envelope = envelope

##############        
#        self.h = h
#        self.Rbase = Rbase # / self.h
#        self.nturns = nturns
#        self.a = self.nturns * 2*N.pi/self.h
#
#        th = self.a*self.h # / (2*N.pi)
#        nsteps = int(th/N.float(delta))
#        t = N.linspace(-th,th,2*nsteps+1)
#        self.z = t/self.a
#
#        if envelope == 'dualcone':
#            multiplier = z
#        elif envelope == 'cylinder':
#            multiplier = 1.
#        else:
#            raise Exception, "Invalid value for 'envelope'. Must be either of: ['dualcone','cylinder']."
#        
##        self.x = multiplier*self.Rbase*N.cos(N.abs(t))
##        self.y = multiplier*self.Rbase*N.sin(N.abs(t))
#        zprogression = multiplier*(self.Rbase/self.h)
#        self.x = zprogression * N.cos(N.abs(t))
#        self.y = zprogression * N.sin(N.abs(t))


#    def __call__(self,h,Rbase,nturns,rtube):
    def __call__(self):
        """Actually compute the spiral."""

#        self.h = h
#        self.Rbase = Rbase # / self.h
#        self.nturns = nturns
        self.a = self.nturns * 2*N.pi/self.h
#        self.rtube = rtube
        
        delta = self.rtube/3.
        th = self.a*self.h # / (2*N.pi)
        nsteps = int(th/N.float(delta))
        t = N.linspace(-th,th,2*nsteps+1)
        self.z = t/self.a

        if self.envelope == 'dualcone':
            print "self.envelope = 'dualcone'"
            zprogression = self.z*(self.Rbase/self.h)
#            multiplier = self.z
        elif self.envelope == 'cylinder':
            print "self.envelope = 'cylinder'"
#            multiplier = 1.
            zprogression = self.Rbase
        else:
            raise Exception, "Invalid value for 'envelope'. Must be either of: ['dualcone','cylinder']."
        
#        self.x = multiplier*self.Rbase*N.cos(N.abs(t))
#        self.y = multiplier*self.Rbase*N.sin(N.abs(t))
#TMP2        zprogression = multiplier*(self.Rbase/self.h)
        self.x = zprogression * N.cos(N.abs(t))
        self.y = zprogression * N.sin(N.abs(t))
        


#TMP        self.h = h
#TMP        self.Rbase = Rbase # / self.h
#TMP        self.nturns = nturns
#TMP        self.a = self.nturns * 2*N.pi/self.h
#TMP        self.rtube = rtube
#TMP        
#TMP        delta = self.rtube/3.
#TMP        th = self.a*self.h # / (2*N.pi)
#TMP        nsteps = int(th/N.float(delta))
#TMP        t = N.linspace(-th,th,2*nsteps+1)
#TMP        self.z = t/self.a
#TMP
#TMP        if self.envelope == 'dualcone':
#TMP            multiplier = self.z
#TMP        elif self.envelope == 'cylinder':
#TMP            multiplier = 1.
#TMP        else:
#TMP            raise Exception, "Invalid value for 'envelope'. Must be either of: ['dualcone','cylinder']."
#TMP        
#TMP#        self.x = multiplier*self.Rbase*N.cos(N.abs(t))
#TMP#        self.y = multiplier*self.Rbase*N.sin(N.abs(t))
#TMP        zprogression = multiplier*(self.Rbase/self.h)
#TMP        self.x = zprogression * N.cos(N.abs(t))
#TMP        self.y = zprogression * N.sin(N.abs(t))
        

        
#TMP#class ConicalHelix3DNDI(CubeNDI,Spiral):
#TMPclass SomeObject(CubeNDI,Spiral):
#TMP
#TMP    # supply X,Y,Z to init instead of the param values. SUpply the param values instead to call (which may use update)
#TMP
#TMP    def __init__(self,npix,transform=None):
#TMP        
#TMP        """Helical tube winding along a dual cone, with constant density inside the tube.
#TMP
#TMP
#TMP        Scratch:
#TMP
#TMP        ROTATE
#TMP        ndimage.rotate() uses angles in degrees.
#TMP        Rotation of 3D array cube.rho by angle A in the plane defined by axes:
#TMP
#TMP        axes=(0,2) --> our x axis
#TMP        axes=(0,1) --> our z axis
#TMP        axes=(1,2) --> our y axis
#TMP
#TMP        So indexing seems to be (y,x,z)
#TMP
#TMP        SHIFT
#TMP        ndimage.shift() works in pixel units.
#TMP
#TMP        
#TMP        Parameters:
#TMP        -----------
#TMP        rin : float
#TMP           Radius at which the shell is centered, in fractions of
#TMP           unity, i.e. between 0 and 1.
#TMP
#TMP        width : float
#TMP           Thickness of the shell, in same units as r.
#TMP
#TMP        xoff, yoff : floats
#TMP           x and y offsets of the shell center from (0,0). Positive
#TMP           values are to the right and up, negative to the left and
#TMP           down. In units if unity (remember that the image is within
#TMP           [-1,1]. Defaults: xoff = yoff = 0.
#TMP
#TMP        weight : float
#TMP           Normalize the total (relative) mass contained in the shell
#TMP           to this value. The total mass is the sum of rho over all
#TMP           pixels (in 3D). This is useful if you e.g. want to have
#TMP           more than one component, and wish to distribute different
#TMP           amounts of mass inside each one. Default: weight=1.
#TMP
#TMP        """
#TMP
#TMP#        Cube3D.__init__(self,X,Y,Z)
#TMP        CubeNDI.__init__(self,npix,transform=transform)
#TMP
#TMP#!!        self.z = N.unique(self.Z.flatten())
#TMP
#TMP
#TMP#    def __call__(self,rtube,pitch,DR=1.,tiltx=0.,tiltz=0.,weight=1,smooth=1.):
#TMP#    def __call__(self,h,Rbase,a,rtube,tiltx=0.,tiltz=0.,weight=1,smooth=1.):
#TMP    def __call__(self,h,Rbase,nturns,rtube,tiltx=0.,tiltz=0.,xoff=0.,yoff=0.,weight=1,smooth=1.):
#TMP
#TMP        """Return density rho at (x,y,z)
#TMP
#TMP
#TMP        TODO: automatically determine args (their names), and produce
#TMP        self.ARG members, and use those in
#TMP
#TMP        """
#TMP
#TMP#        self.reset_grid()
#TMP        
#TMP        self.h = h
#TMP        self.Rbase = Rbase / self.h
#TMP
#TMP        self.nturns = nturns
#TMP#        self.a = nturns * (2*N.pi)  #/ self.h
#TMP
#TMP        self.rtube = rtube
#TMP        self.tiltx = tiltx
#TMP        self.tiltz = tiltz
#TMP        self.xoff = xoff
#TMP        self.yoff = yoff
#TMP        
#TMP        self.weight = weight
#TMP        self.smooth = smooth
#TMP        
#TMP#        self.sanity()
#TMP        self.get_rho()  # get_rho should set self.rho (3D)
#TMP        self.shift()
#TMP        self.rotate3d()
#TMP        print "self.X = ", self.X
#TMP#        self.get_rho()  # get_rho should set self.rho (3D)
#TMP#        self.normalize()
#TMP
#TMP        return self.rho
#TMP
#TMP    def sanity(self):
#TMP
#TMP        """Sanity checks for constant density-edge shell.
#TMP        """
#TMP
#TMP#        assert (0. < self.rin < self.rout)  # this automatically asserts that the shell thickness is finite and positive
#TMP        pass
#TMP
#TMP    
#TMP
#TMP    def get_rho(self):
#TMP
#TMP        """Compute rho(x,y,z) in every voxel.
#TMP        """
#TMP
#TMP        self.a = self.nturns * 2*N.pi/self.h
#TMP
#TMP        
#TMP        delta = self.rtube/3.
#TMP        th = self.a*self.h # / (2*N.pi)
#TMP#        nsteps = int(2*th/delta)
#TMP        nsteps = int(th/N.float(delta))
#TMP#        print "nsteps = ", nsteps
#TMP        
#TMP        t = N.linspace(-th,th,2*nsteps+1)
#TMP#        t = N.linspace(0,th,nsteps)
#TMP        z = t/self.a
#TMP        x = z*self.Rbase*N.cos(N.abs(t))
#TMP        y = z*self.Rbase*N.sin(N.abs(t))
#TMP
#TMP        rtube2 = self.rtube**2
#TMP
#TMP        co = N.zeros(N.prod(self.ndshape),dtype=N.float)
#TMP
#TMP        for j,pt in enumerate(zip(x,y,z)):
#TMP
#TMP            idxes = self.kdtree_query(pt,self.rtube)[1]
#TMP            co[idxes] = 1.
#TMP
#TMP
#TMP        self.rho = co.reshape(self.ndshape)
#TMP
#TMP#        self.rho[co] = 1.
#TMP
#TMP        if self.smooth is not None:
#TMP            self.rho = ndimage.gaussian_filter(self.rho,sigma=self.smooth)


#class ConicalHelix3DNDI(CubeNDI,Spiral3D):
class ConicalHelix3DNDI(Cube):

    # supply X,Y,Z to init instead of the param values. SUpply the param values instead to call (which may use update)

    def __init__(self,npix,transform=None,buildkdtree=True,snakefunc=spiral3D,envelope='dualcone'):
        
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

#        Cube3D.__init__(self,X,Y,Z)
        Cube.__init__(self,npix,transform=transform,buildkdtree=buildkdtree)
#        Spiral3D.__init__(self,envelope=envelope)
        
        self.z = N.unique(self.Z.flatten())

        self.snakefunc = snakefunc
        self.envelope = envelope
        

#    def __call__(self,rtube,pitch,DR=1.,tiltx=0.,tiltz=0.,weight=1,smooth=1.):
#    def __call__(self,h,Rbase,a,rtube,tiltx=0.,tiltz=0.,weight=1,smooth=1.):
#G    def __call__(self,h,Rbase,nturns,rtube,tiltx=0.,tilty=0.,tiltz=0.,xoff=0.,yoff=0.,weight=1,smooth=1.):
    def __call__(self,h,nturns,rtube,tiltx=0.,tilty=0.,tiltz=0.,xoff=0.,yoff=0.,weight=1,smooth=1.):

        """Return density rho at (x,y,z)


        TODO: automatically determine args (their names), and produce
        self.ARG members, and use those in

        """

        self.h = h
#        self.Rbase = Rbase
        self.Rbase = self.h
        self.nturns = nturns
        self.rtube = rtube
        
#        self.get_rho()
        

#TMP
#TMP
#TMP
#TMP
#TMP        
#TMP#        self.reset_grid()
#TMP        
#TMP        self.h = h
#TMP        self.Rbase = Rbase / self.h
#TMP
#TMP        self.nturns = nturns
#TMP#        self.a = nturns * (2*N.pi)  #/ self.h

#TMP2        self.rtube = rtube
        self.tiltx = tiltx
        self.tilty = tilty
        self.tiltz = tiltz
        self.xoff = xoff
        self.yoff = yoff
        
        self.weight = weight
        self.smooth = smooth
        
#        self.sanity()
        self.get_rho()  # get_rho should set self.rho (3D)
        self.rotate3d()
        self.shift()
#        print "self.X = ", self.X

#        self.get_rho()  # get_rho should set self.rho (3D)
#        self.normalize()

        return self.rho
    

    def sanity(self):

        """Sanity checks for constant density-edge shell.
        """

#        assert (0. < self.rin < self.rout)  # this automatically asserts that the shell thickness is finite and positive
        pass

    

    def get_rho(self):

        """Compute rho(x,y,z) in every voxel.
        """
        
        self.x, self.y, self.z = self.snakefunc(self.h,self.Rbase,self.nturns,self.rtube,self.envelope)


        co = N.zeros(N.prod(self.ndshape),dtype=N.float)

#        for j,pt in enumerate(zip(x,y,z)):
        for j,pt in enumerate(zip(self.x,self.y,self.z)):

#            print "pt = ", pt
            
            idxes = self.kdtree_query(pt,self.rtube)[1]
#            print "idxes = ", idxes
            co[idxes] = 1.


        self.rho = co.reshape(self.ndshape)

#        self.rho[co] = 1.

        if self.smooth is not None:
            self.rho = ndimage.gaussian_filter(self.rho,sigma=self.smooth)



#class ConstantDensityDualCone(CubeNDI):
class ConstantDensityDualCone(Cube):

    # supply X,Y,Z to init instead of the param values. SUpply the param values instead to call (which may use update)

#    def __init__(self,X,Y,Z):
    def __init__(self,npix,transform=None,buildkdtree=False):
        
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

#        Cube3D.__init__(self,X,Y,Z)
#        CubeNDI.__init__(self,npix,transform=transform,buildkdtree=buildkdtree)
        Cube.__init__(self,npix,transform=transform,buildkdtree=buildkdtree)


    def __call__(self,r,theta,tiltx=0.,tiltz=0,xoff=0.,yoff=0.,weight=1,smooth=1.):

        """Return density rho at (x,y,z)"""

        self.r = r
        self.theta_deg = theta
        self.theta_rad = N.radians(self.theta_deg)
        self.tiltx = tiltx
        self.tiltz = tiltz

        # helper arrays for finding the edges of the shell in get_rho()
#        self.Rin = self.rin * N.ones(self.X.shape)
#        self.Rout = self.rout * N.ones(self.X.shape)

        self.xoff = xoff
        self.yoff = yoff
        self.weight = weight

        self.smooth = smooth

        self.get_rho()  # get_rho should set self.rho (3D)
        self.shift()
        self.rotate3d()

##        self.sanity()
#
#        self.shift()
##        self.rotate3d()
#        self.rotate3d_extrinsic()
#
#        self.get_rho()  # get_rho should set self.rho (3D)
##        self.normalize()

        return self.rho


    def sanity(self):

        """Sanity checks for constant density-edge shell.
        """

        pass
#        assert (0. < self.rin < self.rout)  # this automatically asserts that the shell thickness is finite and positive


    def get_rho(self):

        """Compute rho(x,y,z) in every voxel.
        """

        # cone formula
        aux = ((self.X**2 + self.Z**2) * N.cos(self.theta_rad)**2 - (self.Y*N.sin(self.theta_rad))**2)
        co1 = (aux <= 0) | N.isclose(aux,0)
#        co1 = (((self.X**2 + self.Z**2) * N.cos(self.theta_rad)**2 - (self.Y*N.sin(self.theta_rad))**2) <= 0.)

        # radial cap
        co2 = (N.sqrt(self.X**2 + self.Y**2 + self.Z**2) <= self.r)

#        coaux = N.isclose(self.r,self.R)
        
        # overall
        coall = co1 & co2 #| coaux
       
        
#        r = get_r((self.X,self.Y,self.Z),mode=2)  # mode=1,2 are fast, 3,4are slow
#        co = ((r >= self.rin) & (r <= self.rout)) | N.isclose(r,self.Rout) | N.isclose(r,self.Rin)  # isclose also captures pixels at the very edge of the shell
        self.set_rho(val=0.)  # default is 0.
        self.rho[coall] = 1.

        if self.smooth is not None:
            self.rho = ndimage.gaussian_filter(self.rho,sigma=self.smooth)


#class TruncatedNormalShell(CubeNDI):
class TruncatedNormalShell(Cube):

    # all models should inherit from Cube3D; the models only provide what's unique to them (e.g. get_rho(), sanity())

    # self.get_rho() should compute and set self.rho (3D)
    # __call__(*args) should return self.rho (3D)


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

#        CubeNDI.__init__(self,npix,transform=transform,buildkdtree=buildkdtree,computeR=computeR)
        Cube.__init__(self,npix,transform=transform,buildkdtree=buildkdtree,computeR=computeR)


#    def __call__(self,r,width,clipa=0.,clipb=1.,xoff=0.,yoff=0.,weight=1.):
#    def __call__(self,r,width,xoff=0.,yoff=0.,weight=1.):
    def __call__(self,r,width,clipa=0.,clipb=1.,xoff=0.,yoff=0.,weight=1.,smooth=1.):

        """Return density rho at (x,y,z)"""

        self.r = r
        self.width = width
#        self.clip = clip
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
        assert (0. < self.clipa < self.r < self.clipb)  # radial distance relations that must hold: 0. <= clipa < r < clipb [<= 1.]
#        assert (0. < self.r < 1.)  # radial distance relations that must hold: 0. <= clipa < r < clipb [<= 1.]
        
#        assert (0. <= self.r <= 1.)  # radial distance relations that must hold: 0. <= clipa < r < clipb [<= 1.]
        assert (self.width > 0.)


    def get_rho(self):

        """Compute rho(x,y,z) in every voxel.
        """

#        r = get_r((self.X,self.Y,self.Z),mode=2)  # mode=1,2 are fast, 3,4are slow

#        self.rho = self.get_pdf(r)
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
#GOOD        mu, sig, clipa, clipb = self.r, self.width, self.clipa, self.clipb
#GOOD        a, b = (clipa - mu) / sig, (clipb - mu) / sig
#GOOD        rv = truncnorm(a, b, loc=mu, scale=sig)
#GOOD        pdf = rv.pdf(x)
#GOOD
#GOOD        return pdf

        mu, sig = self.r, self.width
#        clipa = max(mu-3*sig,0.)
#        clipb = min(mu+3*sig,1.)
        a, b = (self.clipa - mu) / sig, (self.clipb - mu) / sig
        rv = truncnorm(a, b, loc=mu, scale=sig)
        pdf = rv.pdf(x)

        
#        mu, sig = self.r, self.width
#        clipa = max(mu-self.clip*sig,0.)
##        clipb = min(mu+self.clip*sig,1.)
#        clipb = mu+self.clip*sig
#        a, b = (clipa - mu) / sig, (clipb - mu) / sig
#        rv = truncnorm(a, b, loc=mu, scale=sig)
#        pdf = rv.pdf(x)

        return pdf

    



#SAVEclass ConstantDensityDualCone(Cube3D):
#SAVE
#SAVE    # supply X,Y,Z to init instead of the param values. SUpply the param values instead to call (which may use update)
#SAVE
#SAVE    def __init__(self,X,Y,Z):
#SAVE        
#SAVE        """Truncated Normal Shell
#SAVE
#SAVE        A spherical shell with radius 'r', and Gaussian density
#SAVE        fall-off from r. The width of the Normal is 'width'. The PDF
#SAVE        of the Normal is truncated at 'clip' values.
#SAVE        
#SAVE        Parameters:
#SAVE        -----------
#SAVE        rin : float
#SAVE           Radius at which the shell is centered, in fractions of
#SAVE           unity, i.e. between 0 and 1.
#SAVE
#SAVE        width : float
#SAVE           Thickness of the shell, in same units as r.
#SAVE
#SAVE        xoff, yoff : floats
#SAVE           x and y offsets of the shell center from (0,0). Positive
#SAVE           values are to the right and up, negative to the left and
#SAVE           down. In units if unity (remember that the image is within
#SAVE           [-1,1]. Defaults: xoff = yoff = 0.
#SAVE
#SAVE        weight : float
#SAVE           Normalize the total (relative) mass contained in the shell
#SAVE           to this value. The total mass is the sum of rho over all
#SAVE           pixels (in 3D). This is useful if you e.g. want to have
#SAVE           more than one component, and wish to distribute different
#SAVE           amounts of mass inside each one. Default: weight=1.
#SAVE
#SAVE        """
#SAVE
#SAVE        Cube3D.__init__(self,X,Y,Z)
#SAVE
#SAVE
#SAVE    def __call__(self,r,theta,tiltx=0.,tiltz=0,xoff=0.,yoff=0.,weight=1):
#SAVE
#SAVE        """Return density rho at (x,y,z)"""
#SAVE
#SAVE        self.r = r
#SAVE        self.theta_deg = theta
#SAVE        self.theta_rad = N.radians(self.theta_deg)
#SAVE        self.tiltx = tiltx
#SAVE        self.tiltz = tiltz
#SAVE
#SAVE        # helper arrays for finding the edges of the shell in get_rho()
#SAVE#        self.Rin = self.rin * N.ones(self.X.shape)
#SAVE#        self.Rout = self.rout * N.ones(self.X.shape)
#SAVE
#SAVE        self.xoff = xoff
#SAVE        self.yoff = yoff
#SAVE        self.weight = weight
#SAVE
#SAVE#        self.sanity()
#SAVE
#SAVE        self.shift()
#SAVE#        self.rotate3d()
#SAVE        self.rotate3d_extrinsic()
#SAVE
#SAVE        self.get_rho()  # get_rho should set self.rho (3D)
#SAVE#        self.normalize()
#SAVE
#SAVE        return self.rho
#SAVE
#SAVE
#SAVE    def sanity(self):
#SAVE
#SAVE        """Sanity checks for constant density-edge shell.
#SAVE        """
#SAVE
#SAVE        pass
#SAVE#        assert (0. < self.rin < self.rout)  # this automatically asserts that the shell thickness is finite and positive
#SAVE
#SAVE
#SAVE    def get_rho(self):
#SAVE
#SAVE        """Compute rho(x,y,z) in every voxel.
#SAVE        """
#SAVE
#SAVE        # cone formula
#SAVE        aux = ((self.X**2 + self.Z**2) * N.cos(self.theta_rad)**2 - (self.Y*N.sin(self.theta_rad))**2)
#SAVE        co1 = (aux <= 0) | N.isclose(aux,0)
#SAVE#        co1 = (((self.X**2 + self.Z**2) * N.cos(self.theta_rad)**2 - (self.Y*N.sin(self.theta_rad))**2) <= 0.)
#SAVE
#SAVE        # radial cap
#SAVE        co2 = (N.sqrt(self.X**2 + self.Y**2 + self.Z**2) <= self.r)
#SAVE
#SAVE#        coaux = N.isclose(self.r,self.R)
#SAVE        
#SAVE        # overall
#SAVE        coall = co1 & co2 #| coaux
#SAVE       
#SAVE        
#SAVE#        r = get_r((self.X,self.Y,self.Z),mode=2)  # mode=1,2 are fast, 3,4are slow
#SAVE#        co = ((r >= self.rin) & (r <= self.rout)) | N.isclose(r,self.Rout) | N.isclose(r,self.Rin)  # isclose also captures pixels at the very edge of the shell
#SAVE        self.set_rho(val=0.)  # default is 0.
#SAVE        self.rho[coall] = 1.
#SAVE
#SAVE

#GOODclass ConstantDensityTorus(Cube3D):
#GOOD
#GOOD    """Torus as a ring with circular cross-section.
#GOOD
#GOOD    Parameters:
#GOOD    -----------
#GOOD    r : float
#GOOD       Torus radius
#GOOD
#GOOD    rcross : float
#GOOD       Torus ring cross-section radius
#GOOD
#GOOD    xoff, yoff : floats
#GOOD        x and y offsets of the torus center from (0,0). Positive
#GOOD        values are to the right and up, negative to the left and
#GOOD        down. In units if unity (remember that the image is within
#GOOD        [-1,1]. Defaults: xoff = yoff = 0.
#GOOD
#GOOD    tiltx, tiltz : floats
#GOOD        The tilt angles along the x and z axes, respectively. Looking
#GOOD        at the plane of the sky, x is to the right, y is up, and z is
#GOOD        toward the observer. Thus tiltx tilts the torus axis towards
#GOOD        and from the observer, and tiltz tilts the torus axis in the
#GOOD        plane of the sky. tiltx = tiltz = 0 results in the torus seen
#GOOD        edge-on and its axis pointing up in the image. tiltx = 90 &
#GOOD        tiltz = 0 points the torus axis towards the observer (pole-on
#GOOD        view).
#GOOD
#GOOD    weight : float
#GOOD        Normalize the total (relative) mass contained in the torus to
#GOOD        this value. The total mass is the sum of rho over all pixels
#GOOD        (in 3D). This is useful if you e.g. want to have more than one
#GOOD        component, and wish to distribute different amounts of mass
#GOOD        inside each one. Default: weight=1.
#GOOD
#GOOD    """
#GOOD
#GOOD    def __init__(self,X,Y,Z):
#GOOD
#GOOD        Cube3D.__init__(self,X,Y,Z)
#GOOD
#GOOD
#GOOD    def __call__(self,r,rcross,xoff=0.,yoff=0.,tiltx=0.,tiltz=0,weight=1.):
#GOOD
#GOOD        """Return density rho at (x,y,z)"""
#GOOD
#GOOD        self.r = r
#GOOD        self.rcross = rcross
#GOOD        self.xoff = xoff
#GOOD        self.yoff = yoff
#GOOD        self.tiltx = tiltx
#GOOD        self.tiltz = tiltz
#GOOD        self.weight = weight
#GOOD
#GOOD        self.sanity()
#GOOD
#GOOD        self.shift()
#GOOD        self.rotate3d()
#GOOD
#GOOD        self.get_rho()
#GOOD        self.normalize()
#GOOD
#GOOD        return self.rho
#GOOD
#GOOD
#GOOD    def sanity(self):
#GOOD
#GOOD        assert (0. < self.rcross <= self.r)
#GOOD        
#GOOD#        assert (0. <= self.tiltx <= 90.)
#GOOD        assert (0. <= self.tiltx <= 180.)
#GOOD        assert (0. <= self.tiltz <= 180.)
#GOOD
#GOOD
#GOOD    def get_rho(self):
#GOOD
#GOOD        # A point (x,y,z) is inside the torus when:
#GOOD        #
#GOOD        #    (x^2 + y^2 + z^2 + r^2 - rcross^2)^2 - 4 * r^2 * (x^2 + z^2) < 0
#GOOD
#GOOD
#GOOD        # To speed up computation a bit (the following expression are used twice each in the formula below)
#GOOD        r2 = self.r**2
#GOOD        X2 = self.X**2
#GOOD        Z2 = self.Z**2
#GOOD        co = (X2 + self.Y**2 + Z2 + r2 - self.rcross**2)**2 - 4 * r2 * (X2 + Z2) < 0
#GOOD#        co = (self.X**2 + self.Y**2 + self.Z**2 + r2 - self.rcross**2)**2 - 4 * r2 * (self.X**2 + self.Z**2) < 0
#GOOD        
#GOOD        self.set_rho(val=0.)  # default is 0.
#GOOD        self.rho[co] = 1.
#GOOD
#GOOD
#GOODclass TruncatedNormalShell(Cube3D):
#GOOD
#GOOD    # all models should inherit from Cube3D; the models only provide what's unique to them (e.g. get_rho(), sanity())
#GOOD
#GOOD    # self.get_rho() should compute and set self.rho (3D)
#GOOD    # __call__(*args) should return self.rho (3D)
#GOOD
#GOOD
#GOOD    def __init__(self,X,Y,Z):
#GOOD        
#GOOD        """Truncated Normal Shell
#GOOD
#GOOD        A spherical shell with radius 'r', and Gaussian density
#GOOD        fall-off from r. The width of the Normal is 'width'. The PDF
#GOOD        of the Normal is truncated at 'clip' values.
#GOOD        
#GOOD        Parameters:
#GOOD        -----------
#GOOD        r : float
#GOOD           Radius at which the shell is centered, in fractions of
#GOOD           unity, i.e. between 0 and 1.
#GOOD
#GOOD        width : float
#GOOD           Thickness of the shell, in same units as r.
#GOOD
#GOOD        clip : 2-tuple of floats
#GOOD           Where to clip the Gaussian left and right. Default is (0,1).
#GOOD
#GOOD        xoff, yoff : floats
#GOOD           x and y offsets of the shell center from (0,0). Positive
#GOOD           values are to the right and up, negative to the left and
#GOOD           down. In units if unity (remember that the image is within
#GOOD           [-1,1]. Defaults: xoff = yoff = 0.
#GOOD
#GOOD        weight : float
#GOOD           Normalize the total (relative) mass contained in the shell
#GOOD           to this value. The total mass is the sum of rho over all
#GOOD           pixels (in 3D). This is useful if you e.g. want to have
#GOOD           more than one component, and wish to distribute different
#GOOD           amounts of mass inside each one.
#GOOD
#GOOD        """
#GOOD
#GOOD        Cube3D.__init__(self,X,Y,Z)
#GOOD
#GOOD
#GOOD#    def __call__(self,r,width,clipa=0.,clipb=1.,xoff=0.,yoff=0.,weight=1.):
#GOOD#    def __call__(self,r,width,xoff=0.,yoff=0.,weight=1.):
#GOOD    def __call__(self,r,width,clipa=0.,clipb=1.,xoff=0.,yoff=0.,weight=1.):
#GOOD#    def __call__(self,r,width,clip=1.,xoff=0.,yoff=0.,weight=1.):
#GOOD
#GOOD        """Return density rho at (x,y,z)"""
#GOOD
#GOOD        self.r = r
#GOOD        self.width = width
#GOOD#        self.clip = clip
#GOOD        self.clipa = clipa
#GOOD        self.clipb = clipb
#GOOD        self.xoff = xoff
#GOOD        self.yoff = yoff
#GOOD        self.weight = weight
#GOOD
#GOOD        self.sanity()
#GOOD
#GOOD        self.shift()
#GOOD
#GOOD        self.get_rho()
#GOOD        self.normalize()
#GOOD
#GOOD        return self.rho
#GOOD
#GOOD
#GOOD    def sanity(self):
#GOOD
#GOOD        # CAREFUL ASSERTIONS
#GOOD        # lower cut clipa must be smaller than r
#GOOD        # lower cut clipa can be as small as zero
#GOOD        # upper cut clipb can be as low as r
#GOOD        # upper cub clipb can be in principle larger than unity (but we'll default to 1.0)
#GOOD        # width must be a positive number
#GOOD        assert (0. < self.clipa < self.r < self.clipb)  # radial distance relations that must hold: 0. <= clipa < r < clipb [<= 1.]
#GOOD#        assert (0. < self.r < 1.)  # radial distance relations that must hold: 0. <= clipa < r < clipb [<= 1.]
#GOOD        
#GOOD#        assert (0. <= self.r <= 1.)  # radial distance relations that must hold: 0. <= clipa < r < clipb [<= 1.]
#GOOD        assert (self.width > 0.)
#GOOD
#GOOD
#GOOD    def get_rho(self):
#GOOD
#GOOD        """Compute rho(x,y,z) in every voxel.
#GOOD        """
#GOOD
#GOOD        r = get_r((self.X,self.Y,self.Z),mode=2)  # mode=1,2 are fast, 3,4are slow
#GOOD
#GOOD        self.rho = self.get_pdf(r)
#GOOD
#GOOD
#GOOD    def get_pdf(self,x):
#GOOD
#GOOD        """Distribution of density according to a Gaussian with (mu,sig) =
#GOOD        (r,width).
#GOOD        """
#GOOD
#GOOD        from scipy.stats import truncnorm
#GOOD
#GOOD        # Because of the non-standard way that Scipy defines
#GOOD        # distributions, we compute the shape parameters for a
#GOOD        # truncated Normal, with mean mu, std-deviation sigma, and
#GOOD        # clipped left and right at clipa and clipb.
#GOOD#GOOD        mu, sig, clipa, clipb = self.r, self.width, self.clipa, self.clipb
#GOOD#GOOD        a, b = (clipa - mu) / sig, (clipb - mu) / sig
#GOOD#GOOD        rv = truncnorm(a, b, loc=mu, scale=sig)
#GOOD#GOOD        pdf = rv.pdf(x)
#GOOD#GOOD
#GOOD#GOOD        return pdf
#GOOD
#GOOD        mu, sig = self.r, self.width
#GOOD#        clipa = max(mu-3*sig,0.)
#GOOD#        clipb = min(mu+3*sig,1.)
#GOOD        a, b = (self.clipa - mu) / sig, (self.clipb - mu) / sig
#GOOD        rv = truncnorm(a, b, loc=mu, scale=sig)
#GOOD        pdf = rv.pdf(x)
#GOOD
#GOOD        
#GOOD#        mu, sig = self.r, self.width
#GOOD#        clipa = max(mu-self.clip*sig,0.)
#GOOD##        clipb = min(mu+self.clip*sig,1.)
#GOOD#        clipb = mu+self.clip*sig
#GOOD#        a, b = (clipa - mu) / sig, (clipb - mu) / sig
#GOOD#        rv = truncnorm(a, b, loc=mu, scale=sig)
#GOOD#        pdf = rv.pdf(x)
#GOOD
#GOOD        return pdf
#GOOD
#GOOD    
#GOOD
#GOODclass NormalShell(Cube3D):
#GOOD
#GOOD    # all models should inherit from Cube3D; the models only provide what's unique to them (e.g. get_rho(), sanity())
#GOOD
#GOOD    # self.get_rho() should compute and set self.rho (3D)
#GOOD    # __call__(*args) should return self.rho (3D)
#GOOD
#GOOD
#GOOD    def __init__(self,X,Y,Z):
#GOOD        
#GOOD        """Truncated Normal Shell
#GOOD
#GOOD        A spherical shell with radius 'r', and Gaussian density
#GOOD        fall-off from r. The width of the Normal is 'width'. The PDF
#GOOD        of the Normal is truncated at 'clip' values.
#GOOD        
#GOOD        Parameters:
#GOOD        -----------
#GOOD        r : float
#GOOD           Radius at which the shell is centered, in fractions of
#GOOD           unity, i.e. between 0 and 1.
#GOOD
#GOOD        width : float
#GOOD           Thickness of the shell, in same units as r.
#GOOD
#GOOD        clip : 2-tuple of floats
#GOOD           Where to clip the Gaussian left and right. Default is (0,1).
#GOOD
#GOOD        xoff, yoff : floats
#GOOD           x and y offsets of the shell center from (0,0). Positive
#GOOD           values are to the right and up, negative to the left and
#GOOD           down. In units if unity (remember that the image is within
#GOOD           [-1,1]. Defaults: xoff = yoff = 0.
#GOOD
#GOOD        weight : float
#GOOD           Normalize the total (relative) mass contained in the shell
#GOOD           to this value. The total mass is the sum of rho over all
#GOOD           pixels (in 3D). This is useful if you e.g. want to have
#GOOD           more than one component, and wish to distribute different
#GOOD           amounts of mass inside each one.
#GOOD
#GOOD        """
#GOOD
#GOOD        Cube3D.__init__(self,X,Y,Z)
#GOOD
#GOOD
#GOOD#    def __call__(self,r,width,clipa=0.,clipb=1.,xoff=0.,yoff=0.,weight=1.):
#GOOD    def __call__(self,r,width,xoff=0.,yoff=0.,weight=1.):
#GOOD
#GOOD        """Return density rho at (x,y,z)"""
#GOOD
#GOOD        self.r = r
#GOOD        self.width = width
#GOOD        self.xoff = xoff
#GOOD        self.yoff = yoff
#GOOD        self.weight = weight
#GOOD
#GOOD        self.sanity()
#GOOD
#GOOD        self.shift()
#GOOD
#GOOD        self.get_rho()
#GOOD        self.normalize()
#GOOD
#GOOD        return self.rho
#GOOD
#GOOD
#GOOD    def sanity(self):
#GOOD
#GOOD        # CAREFUL ASSERTIONS
#GOOD        # lower cut clipa must be smaller than r
#GOOD        # lower cut clipa can be as small as zero
#GOOD        # upper cut clipb can be as low as r
#GOOD        # upper cub clipb can be in principle larger than unity (but we'll default to 1.0)
#GOOD        # width must be a positive number
#GOOD#        assert (0. <= self.clipa < self.r <= self.clipb)  # radial distance relations that must hold: 0. <= clipa < r < clipb [<= 1.]
#GOOD        assert (0. < self.r < 1.)  # radial distance relations that must hold: 0. <= clipa < r < clipb [<= 1.]
#GOOD        assert (self.width > 0.)
#GOOD
#GOOD
#GOOD    def get_rho(self):
#GOOD
#GOOD        """Compute rho(x,y,z) in every voxel.
#GOOD        """
#GOOD
#GOOD        r = get_r((self.X,self.Y,self.Z),mode=2)  # mode=1,2 are fast, 3,4are slow
#GOOD
#GOOD        self.rho = self.get_pdf(r)
#GOOD
#GOOD
#GOOD    def get_pdf(self,x):
#GOOD
#GOOD        """Distribution of density according to a Gaussian with (mu,sig) =
#GOOD        (r,width).
#GOOD        """
#GOOD
#GOOD        from scipy.stats import truncnorm
#GOOD
#GOOD        # Because of the non-standard way that Scipy defines
#GOOD        # distributions, we compute the shape parameters for a
#GOOD        # truncated Normal, with mean mu, std-deviation sigma, and
#GOOD        # clipped left and right at clipa and clipb.
#GOOD        mu, sig = self.r, self.width
#GOOD        a, b = (0. - mu) / sig, (1. - mu) / sig
#GOOD        rv = truncnorm(0., 1., loc=mu, scale=sig)
#GOOD        pdf = rv.pdf(x)
#GOOD
#GOOD        return pdf


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


def cartesian2spherical(x,y,z):

    r = N.sqrt(x**2+y**2+z**2)
    theta = N.arctan2(N.sqrt(x**2+y**2)/z)
    phi = N.arctan2(y/x)

    return r, theta, phi
    

