"""Creates a 4x5 gallery of a TruncatedNormalShell with progressively
increasing image resolution.
"""

__author__ = 'Robert Nikutta <roert.nikutta@gmail.com>'
__version__ = '2015-11-26' # yyy-mm-dd

import sys
sys.path.append('..')
sys.path.append('../helpers')
import rhocube
import plothelpers
from numpy import arange
import pylab as p
from matplotlib.ticker import NullLocator


def make_gallery_resolution(cmap='Blues_r',savefig=None):

    model = (('TruncatedNormalShell',0.6,0.07,0,0,1.),)
    npixels = arange(5,45,2)  # all odd integers 5...43
    
    fig = p.figure(figsize=(11,9))

    for j,npix in enumerate(npixels):
        ax = fig.add_subplot(4,5,j+1)
        cube = rhocube.Cube(npix)
        cube(model)
        im = ax.imshow(cube.image,origin='lower',extent=(-1,1,-1,1),cmap=p.cm.get_cmap(cmap),interpolation='none')
        ax.set_xlim(-1.1,1.1)
        ax.set_ylim(-1.1,1.1)
        ax.xaxis.set_major_locator(NullLocator())
        ax.yaxis.set_major_locator(NullLocator())
        ax.text(0.1,0.1,'%d' % npix,color='w',fontsize=10,transform=ax.transAxes)

    p.subplots_adjust(hspace=0.1,wspace=0.08,left=0.02,right=0.98,top=0.98,bottom=0.03)
    p.show()
        
    if savefig is not None:
        p.savefig(savefig)
        

def make_gallery_degeneracy(cmap='Blues_r',savefig='TruncatedNormalShell_gallery_degeneracy.pdf'):
    
    n = 5

    fig = plothelpers.setup_figure(figsize=(5,5),fontsize=9,returnaxes=False)
    

#    fontsize = 9
#    p.rcParams['axes.labelsize'] = fontsize
#    p.rcParams['axes.titlesize'] = fontsize
#    p.rcParams['font.size'] =  fontsize
#    p.rcParams['legend.fontsize'] = fontsize
#    p.rcParams['xtick.labelsize'] = fontsize
#    p.rcParams['ytick.labelsize'] = fontsize
#
#    # don't use Type 3 fonts (requirement by MNRAS)
#    p.rcParams['ps.useafm'] = True
#    p.rcParams['pdf.use14corefonts'] = True
#    p.rcParams['text.usetex'] = True
#    p.rcParams['font.family'] = 'sans-serif'
#    p.rcParams['text.latex.preamble'] = [
#        r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
#        r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
#    ]
#
#    
#    fig = p.figure(figsize=(5,5))
    grid = ImageGrid(fig, 111, # similar to subplot(111)
                 nrows_ncols = (n, n), # creates 2x3 grid of axes
                 direction = 'row',
                 axes_pad=0., # pad between axes in inch.
                 cbar_mode = 'none',
             )

    r = N.linspace(0.1,0.9,n)
    width = N.linspace(0.1,0.9,n)
    clipa = 1e-3
    clipb = 1.
    cube = rhocube.Cube(101)

    c = 0
    for iw,w_ in enumerate(width):
        for ir,r_ in enumerate(r):
            clipa = 1e-3
            clipb = 1.
            ax = grid[c]
            c += 1
            model = (('TruncatedNormalShell',r_,w_,clipa,clipb,0,0,1.),)
            cube(model)
            im = ax.imshow(cube.image,origin='lower',extent=(-1,1,-1,1),cmap=p.cm.get_cmap(cmap),interpolation='none')
            ax.set_xlim(-1.0,1.0)
            ax.set_ylim(-1.0,1.0)
            ax.xaxis.set_major_locator(NullLocator())
            ax.yaxis.set_major_locator(NullLocator())
            if iw == 0:
                ax.set_title('radius = %.1f' % r_)
            if ir == 0:
                ax.set_ylabel('width = %.1f' % w_)

    p.subplots_adjust(hspace=0.,wspace=0.0,left=0.05,right=0.98,top=0.95,bottom=0.03)
#    p.show()
        
    if savefig is not None:
        p.savefig(savefig)


if __name__ == '__main__':

    make_gallery()



