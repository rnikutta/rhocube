"""Compute a model 3D density distribution, and 2D dz-integrated map."""

__author__  = "Robert Nikutta, Claudia Agliozzo"
__version__ = "2015-04-27"

import numpy as N
import pyfits
import models


#def density_cube():
#
#    cube = Cube(10)
#    model = models.HardEdgeShell(0.7,0.1)  # a shell from with inner/outer radii from 0.6-0.7


class Cube:

    def __init__(self,npix,themodels,normalize='none'):  # TODO: get npix from the FITS file to be fitted

        """

        Example:
        --------

        themodels = (\
                     ('TruncatedNormalShell',0.8,0.02,(0,1),1.),\
                     ('TruncatedNormalShell',0.4,0.04,(0,1),0.5)
                    )

        """

        self.npix = npix
        if self.npix % 2 == 0:
            self.npix += 1

        self.cube = N.zeros((self.npix,self.npix,self.npix))

        # center -/+ 1 radius to each side
        self.x = N.linspace(-1.,1,self.npix)
        self.X, self.Y, self.Z = N.meshgrid(self.x,self.x,self.x,indexing='xy')  # three 3d coordinate arrays to 

        self.models = []
        for m in themodels:
            print m

            typ, args = m[0], m[1:]
            model = getattr(models,typ)(*args)
            model(self.X, self.Y, self.Z)  # model.rho (3D) is normalized to model_.weight

            model.rhomap = N.sum(model.rho,axis=-1)

#            if normalize == 'peak':
#                model_.rhomap /= model_.rhomap.max()
#                model_.rhomap *= model_.weight

            self.models.append(model)


        self.allmap = N.sum([m.rhomap for m in self.models],axis=0)


def savefits(image,outname='out.out'):

    import pyfits

    hdu = pyfits.PrimaryHDU(image)

    try:
        hdu.writeto(outname)
    except IOError:
        print "Can not write output file %s" % outname
        raise
        
