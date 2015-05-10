"""Compute a model 3D density distribution, and 2D dz-integrated map."""

__author__  = "Robert Nikutta, Claudia Agliozzo"
__version__ = "2015-05-10"

import numpy as N
import pyfits
import models
import pymc


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
            


class Data:

    def __init__(self,fitsfile='testdata.fits'):

        hdulist = pyfits.open(fitsfile)
        header = hdulist[0].header

        self.npix = header['NAXIS1']

        self.data2d = hdulist[0].data
        self.data1d = self.data2d.flatten(order='C')

        self.scale = self.data1d.sum()



class Cube:

    """3D Cube to hold the total rho(x,y,z).

    Example:
    --------
    Two different shells:

    cube = rhocube.Cube( 200, ( ('TruncatedNormalShell',0.4,0.03,0,1,0.5,0.2,1),\
                                ('ConstantDensityShell',0.3,0.5,,-0.2,-0.4,2) ) )

    pylab.imshow(cube.allmap,origin='lower',extent=extent,cmap=matplotlib.cm.gray)

    """

    def __init__(self,npix,themodels,normalize='none'):  # TODO: get npix from the FITS file to be fitted

        """

        Example:
        --------

        themodels = (\
                     ('TruncatedNormalShell',0.8,0.02,0,1,0.,0.,1.),\
                     ('ConstantDensityShell',0.4,0.6,,(-0.3,0.2),0.5)
                    )

        """

        self.npix = npix
        if self.npix % 2 == 0:   # make sure we have a central pixel, i.e. an odd number of pixels along the axes
            self.npix += 1

        self.cube = N.zeros((self.npix,self.npix,self.npix))

        # center -/+ 1 radius to each side
        self.x = N.linspace(-1.,1,self.npix)
        self.X, self.Y, self.Z = N.meshgrid(self.x,self.x,self.x,indexing='xy')  # three 3d coordinate arrays to 

        self.extent = (self.x.min(),self.x.max(),self.x.min(),self.x.max())

        # TODO: separate this in init and call
        self.models = []
        for m in themodels:
            print m

            typ, args = m[0], m[1:]
            print "typ: ", typ
            print "args: ", args
            model = getattr(models,typ)(self.X, self.Y, self.Z)
            model(*args)  # model.rho (3D) is normalized to model_.weight

            model.rhomap = N.sum(model.rho,axis=-1)

#            if normalize == 'peak':
#                model_.rhomap /= model_.rhomap.max()
#                model_.rhomap *= model_.weight

            self.models.append(model)


        self.allmap = N.sum([m.rhomap for m in self.models],axis=0)


    def __call__(self):

        """When called with parameters"""






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
