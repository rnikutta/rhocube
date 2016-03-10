"""Common plotting helper functions."""

__author__ = "Robert Nikutta <robert.nikutta@gmail.com>"
__version__ = "20150818"

import pylab as p


def axcross(ax):

    """Plot an (x,y) coordinate cross on supplied ax instance."""

    ax.axvline(0,ls='-',color='k',lw=0.5,alpha=0.2)
    ax.axhline(0,ls='-',color='k',lw=0.5,alpha=0.2)


def setup_figure(figsize=(5,5),layout=(1,1),fontsize=10,returnaxes=True):

    """Set up basic figure size and layout and some common style
       parameters. Return a figure objects and (if requested) a list
       of axes.
    """
    
    fontsize = fontsize
    p.rcParams['font.family'] = 'sans-serif'
    p.rcParams['axes.linewidth'] = 1.
    p.rcParams['axes.labelsize'] = fontsize
    p.rcParams['axes.titlesize'] =  fontsize
    p.rcParams['font.size'] =  fontsize
    p.rcParams['xtick.labelsize'] = fontsize
    p.rcParams['ytick.labelsize'] = fontsize

    # don't use Type 3 fonts (requirement by MNRAS); you'll need dvipng installed
    p.rcParams['ps.useafm'] = True
    p.rcParams['pdf.use14corefonts'] = True
    p.rcParams['text.usetex'] = True
    p.rcParams['font.family'] = 'sans-serif'
    p.rcParams['text.latex.preamble'] = [
        r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
        r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
    ] 

    fig = p.figure(figsize=figsize)


    if returnaxes is False:
        return fig
    else:
        nrows, ncols = layout
        for iax in xrange(nrows*ncols):
            ax = fig.add_subplot(nrows, ncols, iax+1)
            returnvals.append(ax)
            
        return [fig] + returnvals


#def setup_figparams(figsize=(5,5),layout=(1,1)):
#
#    """Set up basic figure size and layout and some common style parameters."""
#    
#    fontsize = 10
#    p.rcParams['font.family'] = 'sans-serif'
#    p.rcParams['axes.linewidth'] = 1.
#    p.rcParams['axes.labelsize'] = fontsize
#    p.rcParams['axes.titlesize'] =  fontsize
#    p.rcParams['font.size'] =  fontsize
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
#    fig = p.figure(figsize=figsize)
#
#    returnvals = [fig]
#
#    nrows, ncols = layout
#    for iax in xrange(nrows*ncols):
#        ax = fig.add_subplot(nrows, ncols, iax+1)
#        returnvals.append(ax)
#
#    return returnvals
    

