"""Continuous model parameter variation.

This tests shows why fractional (as opposed to integer) variation of
model parameters in necessary. A Gaussian shell is computed centered
on (0,0), and another version of it but, shifted by just 1/4 pixel to
the right. Their difference is clearly visible in the right panel of
the produced plot.

Usage:

python ./subpixel.py

Then inspect subpixel.pdf

"""

__author__ = "Robert Nikutta <robert.nikutta@gmail.com>"
__version__ = "20150818"

import sys
sys.path.append('..')
sys.path.append('../helpers')
import rhocube
from plothelpers import *
import pylab as p

def test():
       
    # two objects to hold cubes
    npix = 201
    print "Setting up two cubes with (%d)^3 pixels each" % npix
    cube1 = rhocube.Cube(npix)
    cube2 = rhocube.Cube(npix)

    # quarter pixel shift in +x direction
    xoff = 0.25*(2/float(npix))

    print "Computing centered shell"
    cube1((('TruncatedNormalShell',0.5,0.1,0.1,1,0,0,1.),))     # a centered Gaussian shell
    print "Computing shell shifted by 1/4 pixel"
    cube2((('TruncatedNormalShell',0.5,0.1,0.1,1,xoff,0,1.),))  # same shell but shifted 1/4 pixel to the right

    # plot
    print "Plotting...", 

    fig, ax1, ax2, ax3 = setup_figparams(figsize=(9,3),layout=(1,3))  # layout = (nrows,ncols)

    kwargs = {'origin':'lower','extent':cube1.extent,'cmap':p.cm.Blues,'interpolation':'none'}
    im1 = ax1.imshow(cube1.image,**kwargs)               # centered shell
    im2 = ax2.imshow(cube2.image,**kwargs)               # shifted shell
    im3 = ax3.imshow(cube1.image-cube2.image,**kwargs)   # their difference

    for ax in (ax1,ax2,ax3):
        axcross(ax)
        ax.set_xlabel('x')

    ax1.set_ylabel('y')
    ax1.set_title('Centered shell')
    ax2.set_title('Shell shifted by 1/4 pixel to the right')
    ax3.set_title('Their difference')
    
    p.subplots_adjust(left=0.07,right=0.98,top=0.95,bottom=0.09,wspace=0.25)
    
    pdffile = 'subpixel.pdf'
    print "Saving plot to %s" % pdffile
    p.savefig(pdffile)
    print "Done."


if __name__ == '__main__':
    test()
