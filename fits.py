"""Functions that use rhocube core functionality to fit data for our LBV paper (Agliozzo+2015)."""

import pyfits
import numpy as N
import rhocube
import models
import pymc
import numpy.ma as ma
from numpy import pi, exp, sqrt, array, ndarray, zeros, sign, linspace
import scipy.stats as stats
from scipy.special import erf, erfc
from collections import OrderedDict
import matplotlib as mpl
import pylab as p


#GOODdef get_ne_cube(data,sig,tracesdict,chi2trace,outfits=None):
#GOOD
#GOOD
#GOOD    idx = N.argmin(chi2trace)
#GOOD    pixelscale = (data.px2pc*data.npix/2.)
#GOOD    bestvals = [t[idx] for t in tracesdict.values()]
#GOOD    bestvals[0] = bestvals[0]/pixelscale
#GOOD    bestvals = bestvals[:-1]
#GOOD    
#GOOD    cube = rhocube.Cube(data.npix,transform=Quad())
#GOOD    image2dA, image2d_nonmaskedA, cubeA, scaleA = modelfunc_cddc(cube,data.data2d,sig.data2d,*bestvals,returnall=True)
#GOOD    cubeA.transform._inverse(scaleA)
#GOOD    
#GOOD    Sne = cube.transform._inverse(scaleA)
#GOOD    AUX = Sne*cube.rhoraw
#GOOD
#GOOD    if outfits is not None:
#GOOD        comments = ['radius (pc)','cone full aperture angle (deg)','deg','deg','pc','pc','Msun']
#GOOD        keyvalcom = [(i[0],(i[1][idx],comments[j])) for j,i in enumerate(tracesdict.items())]
#GOOD        header = dict(keyvalcom)
#GOOD        header['CDELT1'] = (data.px2pc,'parsec per pixel')
#GOOD        try:
#GOOD            rhocube.savefile(N.transpose(AUX,axes=(2,0,1)),outfits,header=header)
#GOOD        except:
#GOOD            raise
#GOOD            
#GOOD    return AUX


def get_ne_cube(data,sig,tracesdict,chi2trace,outfits=None):

    pixelscale = (data.px2pc*data.npix/2.)
    
    # MAP model
    idx = N.argmin(chi2trace)
    print "chi2trace[idx] = ", chi2trace[idx]
    bestvals = [t[idx] for t in tracesdict.values()]
    bestvals = [bv/pixelscale for bv in bestvals]
    print "bestvals = ", bestvals
    bestvals = bestvals[:-1]  # leave out Mion

#    # Median model
#    bestvals = [N.median(tr)/pixelscale for par,tr in tracesdict.items() if par in ('r','width','clipa','clipb','xoff','yoff')]
#    bestvals = bestvals + [1,1.]

    # set up model
    mod = models.TruncatedNormalShell(data.npix,transform=Quad())

    # call it with best values
    mod(*bestvals)
    image = N.sum(mod.transform(mod.rho),axis=-1)
    image2d, dummy, scale = get_scale(image,data.data2d,sig.data2d, returnall=True)

#    cube = rhocube.Cube(data.npix,transform=Quad())
#    image2dA, image2d_nonmaskedA, cubeA, scaleA = modelfunc_cddc(cube,data.data2d,sig.data2d,*bestvals,returnall=True)
#    cubeA.transform._inverse(scaleA)
    
#    Sne = cube.transform._inverse(scaleA)
    Sne = mod.transform._inverse(scale)
#    AUX = Sne*mod.rhoraw
    AUX = Sne*mod.rho

    if outfits is not None:
        comments = ['shell radius (pc)','Gaussian shell width (pc)','lower clip radius (pc)','upper clip radius (pc)','pc','pc','Msun']
        keyvalcom = [(i[0],(i[1][idx],comments[j])) for j,i in enumerate(tracesdict.items())]
        header = dict(keyvalcom)
        header['CDELT1'] = (data.px2pc,'parsec per pixel')
        try:
            rhocube.savefile(N.transpose(AUX,axes=(2,0,1)),outfits,header=header)
        except:
            raise
            
    return AUX


def get_stats(tracesdict,chi2redtrace,printtable=False):

    import scipy.stats as stats
    
    idx = chi2redtrace.argmin()
    print "idx, chi2redtrace[idx] = ", idx, chi2redtrace[idx]

    if printtable is True:
        print "parameter     MAP       median  +    -"
    bestvals = []
    quantiles = OrderedDict()
    tail = get_tail(1.)
    for k,v in tracesdict.items():
        bestval = v[idx]
        bestvals.append((k,bestval))
        quantiles_ = stats.mstats.mquantiles(v,prob=[tail,0.5,1.-tail])
        quantiles[k] = quantiles_

        if printtable is True:
            print "%7s      % .2f     % .2f   % .2f   %.2f" % (k,bestval,quantiles_[1],quantiles_[2]-quantiles_[1],quantiles_[1]-quantiles_[0])


    print "XXX bestvals = ", bestvals
    return bestvals, quantiles


def plot_MAP_posteriors_kde(data,sig,tracesdict,chi2trace,vars,xlabels,modeltype='ConstantDensityShell'):

    """Plot data, MAP model, marginalized posterior, mean radial profiles, and compute best model cube.

    Parameters:
    -----------

    data,sig,tracesdict,chi2trace : As provided by fit_* function

    """

    import pandas as pd
    import seaborn as sb
    import pylab as p
    from matplotlib.ticker import NullFormatter
    from matplotlib.ticker import MaxNLocator, LinearLocator, FormatStrFormatter


    data2d = data.data2d
   
    df = pd.DataFrame(tracesdict)

    idx = N.argmin(chi2trace)
    theta_map = [v[idx] for v in tracesdict.values()]
    print "theta_map = ", theta_map
    
    bestvals, quantiles = get_stats(tracesdict,chi2trace,printtable=True)
    print "bestvals =", bestvals
    print "quantiles.items() = ", quantiles.items()

    medians = [v[1] for v in quantiles.values()]
    print "medians = ", medians

    
    modinstance = getattr(models,modeltype)
    mod = modinstance(data.npix,transform=Quad())  # CDS
    image2d_best_map, cube_best_map, scale_best_map, theta_best_map = S61_get_best_model(data,sig,tracesdict,chi2trace,mod,theta=theta_map[:-1]+[1,1])

    
##TNS    vars = ['r','width','clipa','clipb','Mion']
#    vars = ['rin','rout','Mion']  # CDS
##TNS    xlabels = [r'radius (pc)',r'width (pc)',r'lower clip radius (pc)',r'upper clip radius (pc)',r'M$_{\rm ion}$ (M$_{\odot}$)']
#    xlabels = [r'inner radius (pc)',r'outer radius (pc)',r'M$_{\rm ion}$ (M$_{\odot}$)']  #CDS

    fontsize = 12
    p.rcParams['legend.fontsize'] = fontsize
    p.rcParams['axes.labelsize'] = fontsize
    p.rcParams['axes.titlesize'] = fontsize
    p.rcParams['text.fontsize'] =  fontsize-1
    p.rcParams['legend.fontsize'] = fontsize-2
    p.rcParams['xtick.labelsize'] = fontsize
    p.rcParams['ytick.labelsize'] = fontsize
#    p.rcParams['font.family'] = 'sans-serif'
    p.rcParams['font.family'] = 'serif'
#latex    # don't use Type 3 fonts (requirement by MNRAS); you'll need dvipng installed
#latex    p.rcParams['ps.useafm'] = True
#latex    p.rcParams['pdf.use14corefonts'] = True
#latex    p.rcParams['text.usetex'] = True
#latex    p.rcParams['font.family'] = 'sans-serif'
#latex    p.rcParams['text.latex.preamble'] = [
#latex        r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
#latex        r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
#latex    ] 

#TNS    f, axes = p.subplots(2, 4, figsize=(12, 6), sharex=False, sharey=False)
    ncols = int(N.ceil((len(vars)+3)/2.))
#    f, axes = p.subplots(2, 3, figsize=(10, 6), sharex=False, sharey=False)
    f, axes = p.subplots(2, ncols, figsize=(3*ncols, 2*3), sharex=False, sharey=False)
    lw = 2.
    al = 0.4

    cmap = p.cm.rainbow
    
    # IMAGE DATA
    ticks = N.array((-1,-0.5,0,0.5,1.)) * (data.px2pc*data.npix/2.)
    ticklabels = ["%.1f" % t for t in ticks]
    extent = [ticks[0],ticks[-1],ticks[0],ticks[-1]]

    ax00 = axes[0,0]
    ax00.imshow(data2d,origin='lower',extent=extent,cmap=cmap,interpolation='none')
    ax00.text(0.04,0.9,'DATA',ha='left',transform=ax00.transAxes,color='k')
    ax00.text(0.97,0.9,'(0)',ha='right',transform=ax00.transAxes,color='k')
    ax00.set_xlabel('x (pc)')
    ax00.set_ylabel('y (pc)')
    ax00.set_xticks(ticks)
    ax00.set_xticklabels(ticklabels)
    ax00.set_yticks(ticks)
    ax00.set_yticklabels(ticklabels)
    
    
    # IMAGE MAP MODEL
    ax10 = axes[1,0]
    ax10.imshow(image2d_best_map,origin='lower',extent=extent,cmap=cmap,interpolation='none')

    # TODO: automate this
    if modeltype == 'ConstantDensityShell':
        # use this for CDS
        center = (theta_best_map[2],theta_best_map[3])
        circ1 = p.Circle(center,theta_best_map[0],ec='k',fc='none',ls='solid',lw=0.7,alpha=0.5)
        circ2 = p.Circle(center,theta_best_map[1],ec='k',fc='none',ls='dashed',lw=0.7,alpha=0.5)
        ax10.add_patch(circ1)
        ax10.add_patch(circ2)
    elif modeltype == 'TruncatedNormalShell':
        # use this for TNS
        center = (theta_best_map[4],theta_best_map[5])
        circ1 = p.Circle(center,theta_best_map[0],ec='k',fc='none',ls='solid',lw=0.7,alpha=0.5)
        circ2a = p.Circle(center,theta_best_map[0]+theta_best_map[1],ec='k',fc='none',ls='dashed',lw=0.7,alpha=0.5)
        if theta_best_map[0]-theta_best_map[1] > 0:
            circ2b = p.Circle(center,theta_best_map[0]-theta_best_map[1],ec='k',fc='none',ls='dashed',lw=0.7,alpha=0.5)
        else:
            circ2b = None
            
        circ3 = p.Circle(center,theta_best_map[3],ec='k',fc='none',ls='dotted',lw=0.7,alpha=0.5)
        ax10.add_patch(circ1)
        ax10.add_patch(circ2a)
        if circ2b is not None:
            ax10.add_patch(circ2b)

        ax10.add_patch(circ3)

    
    ax10.set_xticks(ticks)
    ax10.set_xticklabels(ticklabels)
    ax10.set_yticks(ticks)
    ax10.set_yticklabels(ticklabels)

    ax10.text(0.04,0.9,'MAP MODEL',ha='left',transform=ax10.transAxes,color='k')
    ax10.text(0.97,0.9,'(1)',ha='right',transform=ax10.transAxes,color='k')
    ax10.set_xlabel('x (pc)')
    ax10.set_ylabel('y (pc)')
    ax10.set_xticks(ticks)
    ax10.set_xticklabels(ticklabels)
    ax10.set_yticks(ticks)
    ax10.set_yticklabels(ticklabels)


    # TODO: try to automate this
    if modeltype == 'TruncatedNormalShell':
        axorder = [(0,1),(1,1),(0,2),(1,2),(0,3)]
    elif modeltype == 'ConstantDensityShell':
        axorder = [(0,1),(1,1),(0,2)]
    
    for j,jax in enumerate(axorder):
        print j, jax

        ax = axes[jax]
        ax.text(0.97,0.9,'(%d)' % (j+2),ha='right',transform=ax.transAxes,color='k')
        var = vars[j]
        sample = df[var]
        bestval = sample[idx]
        med = N.median(sample)
            
        if j < 5:
            sb.distplot(sample,norm_hist=1,bins=12,ax=ax,\
                        hist_kws={'histtype':'step','lw':lw,'color':'r','alpha':al},
                        kde_kws={'lw':0.5*lw,'color':'k','alpha':0.8})

        line = ax.axvline(med,ls=':',lw=lw,color='g',label='distribution median',alpha=0.8)
        dashes = [2,2,2,2] # 10 points on, 5 off, 100 on, 5 off
        line.set_dashes(dashes)
        
        line = ax.axvline(bestval,ls='--',lw=lw,color='b',label='MAP model',alpha=0.8)
        dashes = [6,3,6,3] # 10 points on, 5 off, 100 on, 5 off
        line.set_dashes(dashes)

        ax.set_ylabel('marginalized posterior')
        if j == ncols:
#            leg = ax.legend(loc='center',bbox_to_anchor=(1.65,0.5),frameon=True)
            leg = ax.legend(loc='lower right',frameon=True)
            leg.legendPatch.set_alpha(0.8)
            leg.get_frame().set_facecolor('1.')
            leg.get_frame().set_edgecolor('0.3')

        
        ax.set_xlabel(xlabels[j])
        ax.set_xlim(xmin=-0.01)
        fmt = '%g'
        ax.set_xlim(xmin=0.)
        ax.set_aspect('auto')

        ax.xaxis.set_major_locator(LinearLocator(5))
        ax.xaxis.set_major_formatter(FormatStrFormatter(fmt))
        ax.yaxis.set_major_locator(LinearLocator(5))
#        ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))
        ax.yaxis.set_major_formatter(NullFormatter())


    # mean profiles
    auxdata = data2d.data
    co = N.isnan(auxdata)
    auxdata[co] = 0.
    yscale = 1000.
    meanprofile_data = get_mean_profile(auxdata) / yscale
    meanprofile_map = get_mean_profile(image2d_best_map) / yscale
    ax = axes[-1,-1]
    ax.plot(meanprofile_data,'b-',lw=2,alpha=0.6,label='data')
    ax.plot(meanprofile_map,'r-',lw=2,alpha=0.6,label='MAP model')
    ax.text(0.97,0.9,'(%d)' % (2*ncols-1),ha='right',transform=ax.transAxes,color='k')
    ax.set_ylim(0,1.01*max(meanprofile_data.max(),meanprofile_map.max()))
    ax.set_xlabel('radial coordinate (pc)')
    ax.set_xticks(N.linspace(0,40,5))
    ax.set_xticklabels(ticklabels)
    ax.set_ylabel(r'$\rm EM\ (%d\ \times\ pc\ cm^{-6}\ px^{-1})$' % yscale)
    ax.yaxis.set_major_locator(LinearLocator(5))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    leg = ax.legend(loc='lower left',frameon=1,title='Mean profiles',handlelength=1.,borderaxespad=0.1)
    leg.legendPatch.set_alpha(0.8)
    leg.get_frame().set_facecolor('1.')
    leg.get_frame().set_edgecolor('0.3')
    #    axes[-1, -1].axis('off')
        
    f.subplots_adjust(left=0.065,right=0.975,top=0.96,bottom=0.1,hspace=0.3,wspace=0.4)

    f.savefig('S61_posteriors_MAPmodels_CDS.png',dpi=300)

    return cube_best_map,scale_best_map,   meanprofile_data


def get_profile(image,pa=0.):

    """Parameters
    ----------

    image : 2d array
    
    pa : float
        Position angle in degrees. 0. mean North, and positive values
        mean counter-clockwise (i.e. in the astronomical sense) from
        the North positions.

    """

    from scipy import ndimage

    rotimage = ndimage.rotate(image,pa,reshape=False)
    midcol = image.shape[0]/2
    profile = rotimage[midcol,:]

    return profile


def get_mean_profile(image,delta=10.):

    nsteps = int(180. / float(delta))

    profiles = []
    for j in xrange(nsteps-1):
        angle = (j+1)*delta
        profile = get_profile(image,pa=angle)
        profiles.append(profile)

    meanprofile = N.mean(profiles,axis=0)

    return meanprofile
    


def get_Mion_trace(data,sig2d,tracesdict,mod):

    k = tracesdict.keys()[0]
    n = tracesdict[k].size
    traces = tracesdict.values()

    Mion_trace = N.zeros(n)
    
    for j in xrange(n):
        if j % 10 == 0:
            print "%d of %d" % (j,n)
            
        args = [t[j] for t in traces]
        mod(*args)
        image = N.sum(mod.transform(mod.rho),axis=-1)
        image_masked, image_nonmasked, scale = get_scale(image,data.data2d,sig2d, returnall=True)
        
        Mion_trace[j] = get_Mion(data,mod,scale)

    return Mion_trace



def get_Mion(data,cube,scale):

    cminpc = 3.0857e18  # 1 parsec in cm
    mproton = 1.672621777e-27  # proton mass in kg
    Msun = 1.98855e30          # solar mass in kg

    Sne = cube.transform._inverse(scale)
#    AUX = Sne*cube.rhoraw 
    AUX = Sne*cube.rho 

#    print "avg. electron density per cm^3 = ", N.mean(AUX) # * (data.px2pc*cminpc)**3)
    
    Mion = AUX.sum() * (data.px2pc*cminpc)**3 * mproton/Msun  # number of particles Ne = \int dV ne(x,y,z) * mp/Msun

    return Mion


def get_rho_cube(data,cube,scale):

    Sne = cube.transform._inverse(scale)
    AUX = Sne*cube.rhoraw 

    cminpc = 3.0857e18  # 1 parsec in cm
    mproton = 1.672621777e-27  # proton mass in kg
    Msun = 1.98855e30          # solar mass in kg

    
    Mion = AUX.sum() * (data.px2pc*cminpc)**3 * mproton/Msun  # number of particles Ne = \int dV ne(x,y,z) * mp/Msun

    return Mion


#MASKEDdef get_Mion_masked(data,cube,scale):
#MASKED
#MASKED    cminpc = 3.0857e18  # 1 parsec in cm
#MASKED    mproton = 1.672621777e-27  # proton mass in kg
#MASKED    Msun = 1.98855e30          # solar mass in kg
#MASKED
#MASKED    Sne = cube.transform._inverse(scale)
#MASKED    AUX = Sne*cube.rhoraw
#MASKED
#MASKED    mask2d = data.data2d.mask
#MASKED    rhomasked = ma.array(AUX, mask=N.tile(mask2d, (AUX.shape[0],1)))
#MASKED    
#MASKED    Mion = rhomasked.sum() * (data.px2pc*cminpc)**3 * mproton/Msun  # number of particles Ne = \int dV ne(x,y,z) * mp/Msun
#MASKED
#MASKED    return Mion


def dopickle(file,objects,mode='wb',protocol=2):  # file: string, objects: tuple

    import cPickle as pickle
    
    if mode == 'wb':  #  write file
        try:
            with open(file,mode) as f:
                pickle.dump(objects,f,protocol=protocol)
        except:
            raise


def dounpickle(file,mode='wb'):  # file: string, objects: tuple

    import cPickle as pickle
    
    if mode == 'rb':  #  write file
        try:
            with open(file,mode) as f:
                objects = pickle.load(f)
        except:
            raise

        return objects


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


class Data:

    def __init__(self,fitsfile='tests/TNShell.fits'):

        hdulist = pyfits.open(fitsfile)
        header = hdulist[0].header

        self.npix = header['NAXIS1']
        self.px2pc = abs(float(header['CDELT1']))

        self.data2d = ma.masked_invalid(N.squeeze(hdulist[0].data))  # MASKED ARRAY (NANs, INFs); the N.squeeze removes length-one dimensions, if any are present
#        self.SD = self.data2d.sum()
        self.shape_ = self.data2d.shape
        self.data1d = self.data2d.flatten(order='C')

        self.scale = self.data1d.sum()

        hdulist.close()



# TRANSFORM FUNCTIONS TO APPLY TO RHO(X,Y,Z) BEFORE INTEGRATING

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

    

# HELPER FUNCTIONS

def optimalscale(d,e,m):
    """Calculate optimal scaling of model fluxes m to data fluxes f (given data errors e),
       such that chi2 is minimized.
       
       d   data fluxes
       e   data flux errors
    m   model fluxes
    """

#    print "d,m,e = ", d,m,e
#    aux1 = N.sum(d*m/e**2.)
#    aux2 = N.sum(m**2./e**2.)
#    scale = aux1 / aux2

    mask = d.mask


    aux1 = N.sum(d[~mask]*m[~mask]/e[~mask]**2.)
    aux2 = N.sum(m[~mask]**2./e[~mask]**2.)
    scale = aux1 / aux2

    
    return scale


def chi2(d,e,m):
    """Calculate chi2, given data fluxes d, their errors e, and model fluxes m.
    
    d   data fluxes
    e   data flux errors
    m   model fluxes
    """

    return N.sum((m-d)**2./e**2.)



# FITTING FUNCTIONS

def fit_S61_CDS(datafile='/home/robert/science/ownpapers/lbv-paper/data-em-maps/em_map_s61-8GHz.fits',\
                sigmafile='/home/robert/science/ownpapers/lbv-paper/data-em-maps/err_em_map_s61-8GHz.fits',\
                nsample=500,nburn=100,physical_units=True,return_Mion=True,pickle=None,\
                modeltype='ConstantDensityShell'):

    # get image, flatten, normalize
    print "Reading data file"
    data = Data(fitsfile=datafile)
    data2d = data.data2d.copy()
    print "data.npix = ", data.npix

    # get uncertainties images, flatten, normalize same as image
    print "Reading uncertainties file"
    sig = Data(sigmafile)
    sig2d = sig.data2d.copy()  # CAUTION!

    co = (data.data2d.mask == False)
    nmeas = (data.data2d.mask[co]).size  # number of non-masked pixels = number of data points
    
    # model
    print "Defining model"
    npix = data.npix
    npix2 = npix/2
    px = 2/float(npix)

    # factor to translate from parsecs to pixels on cube with [-1,1] ranges
    factor = (data.px2pc*data.npix/2.)
             
    # ///// SET UP PRIORS
    eps = 0.001  # to avoid numerical armageddons
    rin = pymc.Uniform('rin',0.05/factor,0.15/factor)  # r=mu of the Gaussian; values are in parsec, factor converts them to unit hypercube
    rout = pymc.Uniform('rout',rin+eps,(0.45/factor))#,value=0.05)  # width=sigma of the Gaussian
    # offsets drawn from narrow Gaussian, truncated at [-2,+2] pixels from (x,y)=(0,0)
    xoff = pymc.TruncatedNormal('xoff',0,1./0.02**2,-0.02,0.02)
    yoff = pymc.TruncatedNormal('yoff',0,1./0.02**2,-0.02,0.02)
    # ///// END OF SETTING UP PRIORS

    # model: truncated Gaussian shell
#    mod = models.ConstantDensityShell(npix,transform=Quad())  # Quad() will square ne before integration

    modinstance = getattr(models,modeltype)
    mod = modinstance(data.npix,transform=Quad())  # CDS
    
    # MCMC sampling happens inside here
    @pymc.deterministic()
    def modeled_data(rin=rin,rout=rout,xoff=xoff,yoff=yoff):

        mod(rin,rout,xoff,yoff,1,1.)
        image = N.sum(mod.transform(mod.rho),axis=-1)
        image2d, dummy, scale = get_scale(image,data2d,sig2d, returnall=True)

        return image2d
   
    
    # MCMC model
    modely = pymc.Normal('modely',mu=modeled_data,tau=1./sig2d**2,value=data2d,observed=True)
    model = pymc.Model([rin,rout,xoff,yoff,modely])
    M = pymc.MCMC(model)

    
    # adaptive stepping methods, can lead to better convergence of MCMC chains
    for par in (rin,rout,xoff,yoff):
        M.use_step_method(pymc.AdaptiveMetropolis,[par],delay=100,shrink_if_necessary=True,greedy=False)

    # run the MCMC sampling
    M.sample(iter=nsample,burn=nburn,thin=1)

    # convenience
    from collections import OrderedDict
    params = ('rin','rout','xoff','yoff')
    tracesdict = OrderedDict([(par,M.trace(par).gettrace()) for par in params])

    chi2red_trace, idx, chi2red_min = get_logp_trace(data2d,sig2d,tracesdict,mod)

    if return_Mion is True:
        print "Computing Mion"
        Mion_trace = get_Mion_trace(data,sig2d,tracesdict,mod)
        tracesdict['Mion'] = Mion_trace

#    if physical_units is True:
    if physical_units is not None:
        print "Converting traces to physical units"
        for k,v in tracesdict.items():
            if k in params:
                print k
                v = v * factor  # now in pc
                tracesdict[k] = v

    # pickling means to "save a state" in Python; the pickle file will
    # contain all we need if we want to run the analysis, plotting,
    # etc. later/again.
    if pickle is not None:
        print "Pickling results to file %s" % pickle
        dopickle(pickle,(data,sig,tracesdict,chi2red_trace))


    # return these for the plotting routine
    vars = ['rin','rout','Mion']  # CDS
    xlabels = [r'inner radius (pc)',r'outer radius (pc)',r'M$_{\rm ion}$ (M$_{\odot}$)']  #CDS
    
    return data, data2d, sig, sig2d, model, M, tracesdict, chi2red_trace, vars, xlabels, modeltype


def fit_S61_TNS_continuous(datafile='/home/robert/science/ownpapers/lbv-paper/data-em-maps/em_map_s61-8GHz.fits',\
                           sigmafile='/home/robert/science/ownpapers/lbv-paper/data-em-maps/err_em_map_s61-8GHz.fits',\
                           nsample=500,nburn=100,physical_units=True,return_Mion=True,pickle=None,modeltype='TruncatedNormalShell'):

    # get image, flatten, normalize
    print "Reading data file"
    data = Data(fitsfile=datafile)
    data2d = data.data2d.copy()
    print "data.npix = ", data.npix

    # get uncertainties images, flatten, normalize same as image
    print "Reading uncertainties file"
    sig = Data(sigmafile)
    sig2d = sig.data2d.copy()  # CAUTION!

    co = (data.data2d.mask == False)
    nmeas = (data.data2d.mask[co]).size  # number of non-masked pixels = number of data points
    
    # model
    print "Defining model"
    npix = data.npix
    npix2 = npix/2
    px = 2/float(npix)

    # factor to translate from parsecs to pixels on cube with [-1,1] ranges
    factor = (data.px2pc*data.npix/2.)
             
    # ///// SET UP PRIORS
    eps = 0.001  # to avoid numerical armageddons
    r = pymc.Uniform('r',eps/factor,0.4/factor)  # r=mu of the Gaussian; values are in parsec, factor converts them to unit hypercube
    width = pymc.Uniform('width',eps/factor,(0.4/factor))#,value=0.05)  # width=sigma of the Gaussian
    clipa = pymc.Uniform('clipa',eps,r-eps)
    clipb = pymc.Uniform('clipb',r+eps,0.7/factor)
    # offsets drawn from narrow Gaussian, truncated at [-2,+2] pixels from (x,y)=(0,0)
    xoff = pymc.TruncatedNormal('xoff',0,1./0.05**2,-2*px,2*px)
    yoff = pymc.TruncatedNormal('yoff',0,1./0.05**2,-2*px,2*px)
    # ///// END OF SETTING UP PRIORS

#    r = pymc.Uniform('r',0.2/factor,0.35/factor,value=0.2/factor)  # r=mu of the Gaussian; values are in parsec, factor converts them to unit hypercube
#    width = pymc.Uniform('width',0.07/factor,(0.54/factor-r),value=0.07/factor)#,value=0.05)  # width=sigma of the Gaussian
#    clipa = pymc.Uniform('clipa',0.,r-eps,value=eps)
#    clipb = pymc.Uniform('clipb',r+eps,0.7/factor,value=r+eps)
#    # offsets drawn from narrow Gaussian, truncated at [-2,+2] pixels from (x,y)=(0,0)
#    xoff = pymc.TruncatedNormal('xoff',0,1./0.05**2,-2*px,2*px)
#    yoff = pymc.TruncatedNormal('yoff',0,1./0.05**2,-2*px,2*px)

    # model: truncated Gaussian shell
    mod = models.TruncatedNormalShell(npix,transform=Quad())  # Quad() will square ne before integration

    # MCMC sampling happens inside here
    @pymc.deterministic()
    def modeled_data(r=r,width=width,clipa=clipa,clipb=clipb,xoff=xoff,yoff=yoff):

#        print "In modeled_data; r,width,clipa,clipb,xoff,yoff = ", r,width,clipa,clipb,xoff,yoff
        mod(r,width,clipa,clipb,xoff,yoff,1,1.)
        image = N.sum(mod.transform(mod.rho),axis=-1)
        
#        print "image: ", image
#        print "(image == 0).all() ", (image == 0).all()
#        print "data2d: ", data2d

        image2d, dummy, scale = get_scale(image,data2d,sig2d, returnall=True)

        return image2d


    
    
    # MCMC model
    modely = pymc.Normal('modely',mu=modeled_data,tau=1./sig2d**2,value=data2d,observed=True)
    model = pymc.Model([r,width,clipa,clipb,xoff,yoff,modely])
    M = pymc.MCMC(model)

    
    # adaptive stepping methods, can lead to better convergence of MCMC chains
    for par in (r,width,clipa,clipb,xoff,yoff):
        M.use_step_method(pymc.AdaptiveMetropolis,[par],delay=100,shrink_if_necessary=True,greedy=False)

    # run the MCMC sampling
    M.sample(iter=nsample,burn=nburn,thin=1)

    # convenience
    from collections import OrderedDict
    params = ('r','width','clipa','clipb','xoff','yoff')
    tracesdict = OrderedDict([(par,M.trace(par).gettrace()) for par in params])
#    return M, tracesdict

    print "BEFORE: tracesdict['r'].min(), tracesdict['r'].max() = ", tracesdict['r'].min(), tracesdict['r'].max()
    chi2red_trace, idx, chi2red_min = get_logp_trace(data2d,sig2d,tracesdict,mod)

    if return_Mion is True:
        print "Computing Mion"
        Mion_trace = get_Mion_trace(data,sig2d,tracesdict,mod)
        tracesdict['Mion'] = Mion_trace

#    if physical_units is True:
    if physical_units is not None:
        print "Converting traces to physical units"
        for k,v in tracesdict.items():
            if k in params:
                print k
                v = v * factor  # now in pc
                tracesdict[k] = v

    print "AFTER: tracesdict['r'].min(), tracesdict['r'].max() = ", tracesdict['r'].min(), tracesdict['r'].max()

    # pickling means to "save a state" in Python; the pickle file will
    # contain all we need if we want to run the analysis, plotting,
    # etc. later/again.
    if pickle is not None:
        print "Pickling results to file %s" % pickle
        dopickle(pickle,(data,sig,tracesdict,chi2red_trace))

    
    # return these for the plotting routine
    vars = ['r','width','clipa','clipb','Mion']
    xlabels = [r'radius (pc)',r'width (pc)',r'lower clip radius (pc)',r'upper clip radius (pc)',r'M$_{\rm ion}$ (M$_{\odot}$)']

    return data, data2d, sig, sig2d, model, M, tracesdict, chi2red_trace, vars, xlabels, modeltype


def logp_trace(model):
    """
    return a trace of logp for model
    """

    #init
    db = model.db
    n_samples = db.trace('deviance').length()
    logp = N.empty(n_samples, N.double)

    #loop over all samples
    for i_sample in xrange(n_samples):
        #set the value of all stochastic to their 'i_sample' value
        for stochastic in model.stochastics:
            if stochastic.__name__ == 'clipb':

                print "stochastic.__name__ = ", stochastic.__name__
                
                try:
                    value = db.trace(stochastic.__name__)[i_sample]
                    print "value = ", value
                    stochastic.value = value

                except KeyError:
                    print "No trace available for %s. " % stochastic.__name__

                #get logp
                print "model.logp = ", model.logp
                logp[i_sample] = model.logp

    return logp


def get_logp_trace(data2d,sig2d,tracesdict,mod):

    traces = tracesdict.values()  # order is maintained, b/c tracesdict is OrderedDict
    
    nt = len(traces)
    n = traces[0][:].size
    chi2red_trace = N.zeros(n)

    Ndof = N.argwhere(~data2d.mask.flatten()).size - len(traces)

    for j in xrange(n):
        if j % 10 == 0:
            print "%d of %d" % (j,n)


        args = [traces[k][j] for k in xrange(nt)]
        args = tuple(args + [1.,1.])  # CDDC

        print "In get_logp_trace; args = ", args
        mod(*args)
        image = N.sum(mod.transform(mod.rho),axis=-1)
        image2d = get_scale(image,data2d,sig2d)

        chi2_ = chi2(data2d,sig2d,image2d)
        chi2red = chi2_ / float(Ndof)
        chi2red_trace[j] = chi2red

    idx = chi2red_trace.argmin()
    chi2red_min = chi2red_trace[idx]

    return chi2red_trace, idx, chi2red_min


def get_scale(image,data2d,sig2d, returnall=False):

    image_masked = ma.MaskedArray(image.copy(),data2d.mask)

#    print "image_masked.max(), image_masked.mean() = ", image_masked.max(), image_masked.mean() 

    scale = optimalscale(data2d,sig2d,image_masked)

#    print "scale = ", scale

    image_masked = image_masked * scale

#    print "image_masked.max(), image_masked.mean() = ", image_masked.max(), image_masked.mean() 

    image_nonmasked = image.copy() * scale

#    print "In get_scale; ", image * scale == 

    
    if returnall is False:
        return image_masked
    else:
        return image_masked, image_nonmasked, scale



def S61_get_best_model(data,sig,tracesdict,chi2trace,mod,theta=None):

    if theta is None:
        idx = N.argmin(chi2trace)
        theta = [t[idx] for t in tracesdict.values()[:-1]]

    print "theta = ", theta
    
    mod(*theta)
    image = N.sum(mod.transform(mod.rho),axis=-1)
    image2d, dummy, scale = get_scale(image,data.data2d,sig.data2d, returnall=True)

    return image2d, mod, scale, theta

