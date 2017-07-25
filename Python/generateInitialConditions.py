from galaxyParameters import generateGalaxyModel
from informativeGraphs import plotGalacticDensity
#from informativeGraphs import show_SemiNormalDistribution
from informativeGraphs import show_custom_VAR
from informativeGraphs import plotGCInitialPositionsAndVelocities
from informativeGraphs import demoVectorEvolutionGraph
from mpl_toolkits.mplot3d import Axes3D # do not remove even if some editor says that this package is not used
from scipy import stats
from scipy.integrate import simps
from scipy.spatial.distance import cdist
from util import addSecs
from util import barycenter
from util import criticalBadValue
from util import ensureLineVect
from util import estimateExecutionTime
from util import exportInitialConditionsToFile
from util import extractAnisotropyBeta
from util import extractAnisotropyComponents
from util import extractVelocitiesSigma
from util import g_GravConstant
from util import g_bigSquareFigSize
from util import g_scalingL
from util import g_scalingM
from util import g_scalingT
from util import g_maxSoftening
from util import g_outputFileLineSizeBytes
from util import g_scalingV
from util import g_squareFigSize
from util import getSigmaFaberJackson
from util import g_verbose
from util import generateVectorsRandomDirection
from util import normalise
import datetime
import matplotlib.pyplot as plt
import numpy as np

# Remarks. ####################################################
# [L] is usually 1 pc. [M] is usually 1 solar mass. [T] is usually 1 Myr. [V] is deduced. See util.py for units.
###############################################################

# Parameters. #################################################
# GC parameters. ##############
g_GCNStars=6000 # number of stars (maximum is 9000 for the moment)
g_GCConcentration=3 # GC concentration (in [L])
g_beta=0 # anisotropy (range : -inf<g_beta<1)
g_GCStarsMeanMass=1 # average mass of a star (in [M])
###############################
# Galaxy parameters, see     ##
# module galaxyParameters.py ##
# for more informations.     ##
#tmp_galaxyParameterName="isolated" # control setting
#tmp_galaxyParameterName="colongitudinal" # colongitudinal setting, meaning made to cross the disk far and twice
tmp_galaxyParameterName="radial" # quasi-radial setting, meaning made to pass very close to the bulge
#tmp_galaxyParameterName="brutDisk" # made to make the GC cross the disk vertically
#tmp_galaxyParameterName="brutBulge" # made to make the GC cross the bulge vertically
#tmp_galaxyParameterName="control" # control setting, meaning made to pass very far from galaxy and over it
g_galacticEnvironnement=1 # should a galactic field be used in the simulation (1 if yes, 0 if no) ?
###############################
# Simulation parameters. ######
g_customName="" # custom name to add to the simulation output
g_initialConditionsFile="zzz" # name of the file to which export data (without extension)
g_simulationDuration=1000 # simulation duration (in [T])
g_integrationTimeStep=0.001 # integration time step (in [T])
g_outputPrintInterval=50 # output interval to save bodies" states (in [T])
g_softening=g_maxSoftening # maximum softening (in [L])
g_technique="RK4" # should the simulator use the RK4 scheme (slower but more precise ?) ?
#g_technique="PECE" # should the simulator use the PECE scheme (faster but less precise ?) ?
#g_technique="PEC" # should the simulator use the PECE scheme (faster but less precise ?) ?
###############################
# Generation parameters. ######
g_printExplanationsInInputFile=False # should explanations on inputs be printed in the input file ?
show_durationEstimationFitPlot=False # should the fitting plot for the estimation of the duration be shown ?
show_galacticDensity=False # show a contour plot of the galactic density with starting position and velocity ?
###############################
# Informative graphs. #########
show_plotGCInitialPositionsAndVelocities=False
show_rho=False # show the radial distribution (as is) ?
show_rhoCustomRV=False # show the custom variable obtained from the radial distribution ?
show_3DPlot=False # show the 3D graph of the GC ?
show_demoVectorEvolutionGraph=False # show the demo of the vector evolution graph ?
###############################
# Automatic treatment of     ##
# parameters and computation ##
# of interesting quantities. ##
g_sigmaBar=getSigmaFaberJackson(g_GCNStars*g_GCStarsMeanMass) # theoretical velocities' modules' standard deviation
(g_GCPosition, g_GCVelocity, g_Mb, g_ab, g_Md, g_ad, g_hd, g_Vh, g_ah)=generateGalaxyModel(tmp_galaxyParameterName)
g_outputFilePath="./"+g_initialConditionsFile+"_"+g_customName+"_tech="+g_technique+"_NS="+str(g_GCNStars)+"_b="+str(g_GCConcentration)+"_s="+"{:0.2f}".format(g_sigmaBar)+"_Tm="+str(g_simulationDuration)+"_dt="+str(g_integrationTimeStep)+"_pInt="+str(g_outputPrintInterval)+".out" # output file path (relative to executable)
g_inputFilePath="./"+g_initialConditionsFile+".in" # input file path (relative to executable)
g_GCTotalMass=g_GCNStars*g_GCStarsMeanMass # GC total mass (in [M])
g_rhoIntegralOverR=g_GCTotalMass/(2*np.pi*g_GCConcentration**2) # value of the integral of rho from 0 to infty

g_Vh=g_Vh/g_scalingV # conversion of this parameter in the correct unit
g_galaxyParams=((g_Mb, g_ab), (g_Md, g_ad, g_hd), (g_Vh, g_ah))

g_crossingTime=g_GCConcentration/((3.15576/3.0856776)*1e-3*(g_sigmaBar*g_scalingV)) # estimation of the star crossing time within the GC (in [T])
###############################
# Functions. ##################
def rho(r) :
  # Defines the radial distribution function (not density function because it is not normalised) of positions of stars within the GC.
  # @param r wanted radius
  # @param g_GCTotalMass (global) GC total mass (in [M])
  # @param g_GCConcentration (global) GC concentration (in [L])
  # @return the spatial distribution function evaluated at radius r
  return(((3*g_GCTotalMass)/(4*np.pi*g_GCConcentration**3))*(1+(r**2)/(g_GCConcentration**2))**(-5/2))
def velocityDistribution(s, n) :
  # Generate a number of vector magnitudes along the velocity"s magnitude"s distribution function.
  # @param s variance parameter
  # @param n number of wanted magnitudes
  # @return n numbers following the velocity distribution
  law="genexpon"
  threshold=30
  if law=="genexpon" : # from fit, in [V]
    from scipy.stats import genexpon as LAW
    p=(0.0047873878960509909, 3.1479687030723507, 0.12425108925336203, 0.045146902566154105, 0.5349522410049703)
  elif law=="nakagami" : # from fit, in [V]
    from scipy.stats import nakagami as LAW
    p=(0.74238288775219075, 0.016279250245688111, 1.7390714560822418)
  elif law=="weibull" : # from fit, in [V]
    from scipy.stats import weibull_min as LAW
    p=(1.65912970300427, 0.025854142073003641, 1.6189880397056104)
  elif law=="alpha" : # from fit, in [V]
    from scipy.stats import alpha as LAW
    p=(1.4839568645145409, -0.86099270432631547, 3.9368475519967174)
  elif law=="fisher" : # from fit, in [V]
    from scipy.stats import f as LAW
    p=(19034.6624945916119032, 2.7920189586063682, -0.1535807572963293, 1.5971875341021544)
  rv=LAW.rvs(*p, size=n)
  rv[np.where(rv>threshold)]=np.mean(np.delete(rv, np.where(rv>threshold)))
  return(rv*g_scalingV) # return in units of m/s
def sigma(r) :
  # Defines the velocity"s distribution"s standard deviation at a certain radius.
  # @param r radius
  # @param g_sigmaBar (global) mean of velocity"s distribution"s standard deviation
  # @return velocity"s distribution"s standard deviation at radius r
  return(g_sigmaBar*np.ones(np.size(r)))
def sigmaGaussianForHalfGaussian(r) :
  # Given a standard deviation (here given by the sigma function) wanted for the semi-normal distribution, finds the standard deviation to use in the formula for the semi-normal distribution.
  # @param r radius
  # @param sigma (global, function) function that defines the velocity dispersion (the wanted variance**0.5)
  # @return the variance to use in the formula for the semi-normal distribution at radius r
  return(sigma(r)*(np.pi/(np.pi-2))**0.5)
###############################
###############################################################

# Define the random variable associated to the spatial    #####
# distribution function rho (normalise and create a SciPy #####
# random variable.                                        #####
pdf_threshold=10**(-9) # threshold to cut the normalised PDF (ordinate at which value the PDF stops being relevant)
pdf_right_cutter=g_GCConcentration*(((4*np.pi*g_GCConcentration**3*g_rhoIntegralOverR*pdf_threshold)/(3*g_GCTotalMass))**(-2/5)-1)**0.5 # abscissa where the PDF stops being relevant
class pdf_rho(stats.rv_continuous) :
  def _pdf(self, x) :
    return(rho(x)/g_rhoIntegralOverR) # spatial distribution normalised over its range (roughly)
customRandomVariableRho=pdf_rho(a=0, b=pdf_right_cutter, name="customRandomVariableRho")
###############################################################

# Generation of positions and tests. ##########################
def generate_radii(n) :
  # Generates a certain number of radii which density follows a wanted density given by a beforehand created random variable. It uses an inverse transform sampling.
  # @param n the number of wanted radii
  # @param customRandomVariableRho (global) the custom random variable
  # @return radii which density follows the wanted density
  us=np.random.uniform(0, 1, n) # generate n random numbers from the standard uniform distribution in the interval [0, 1]
  return(customRandomVariableRho.ppf(us)) # use the percent point function to find the values r such that rho(r)=u

def test_radii_distribution(r, func_rho) :
  # Confronts a radii distribution to the theoretical distribution.
  # @param r radii
  # @param func_rho (function) theoretical distribution
  # @param g_GCNStars (global) number of GC
  # @return none
  nbSamples=int(0.2*g_GCNStars) # number of samples to test
  choice_loglog=False # plot a log-log graph (True if yes, False if no) ?
  inf=0
  sup=5*g_GCConcentration
  
  hist=np.histogram(r, nbSamples, range=(inf, sup))[0] # create an histogram of the generated radii
  xHist=np.linspace(inf, sup, nbSamples) # create abscissas for the histogram
  mirroredRadii=np.concatenate((-np.flipud(r)[0:-1], r)) # mirror the data to be able to use a gaussian kernel fit
  kernel=stats.gaussian_kde(mirroredRadii) # get said kernel
  xFit=np.linspace(inf, sup, 1000) # create abscissas for the gaussian kernel fit (and the theoretical function)
  fig=plt.figure(figsize=g_squareFigSize)
  ax=fig.add_subplot(111)
  if choice_loglog :
    ax.loglog(xFit, rho(xFit)/simps(rho(xFit), xFit), "k")
    ax.loglog(xHist, hist/simps(hist, xHist), "b:")
    ax.loglog(xFit, kernel(xFit)/simps(kernel(xFit), xFit), "r")
  else :
    ax.plot(xFit, rho(xFit)/simps(rho(xFit), xFit), "k")
    ax.plot(xHist, hist/simps(hist, xHist), "b:")
    ax.plot(xFit, kernel(xFit)/simps(kernel(xFit), xFit), "r")
  ax.legend([r"Theoretical $\overline{\rho}$", "Histogram", "Gaussian Kernel Fit"], loc="best")
  ax.set_xlabel(r"$r$ ([L])")
  ax.set_ylabel(r"$\overline{\rho}(r)$")
  ax.set_title("Normalised theoretical and computed radial Densities\n(without spherical renormalisation of bins)")
  plt.savefig("initPlots/initialConditions/rho_exp_vs_rho_th")
###############################################################

# Generation of velocities and tests. #########################
def force_anisotropy(positions, velocities) :
  # Forces an anisotropy parameter on a set of velocities (and normalise them).
  # @param positions a set of positions (array 3 * g_GCNStars)
  # @param velocities a set of velocities (array 3 * g_GCNStars)
  # @param g_beta (global) the wanted anisotropy parameter
  # @return the velocities with wanted anisotropy and normalised
  (VR, VT)=extractAnisotropyComponents(X, velocities); # extracts the radial and tangential velocities
  return(normalise((1-g_beta)**(-0.5)*normalise(VR)+normalise(VT))) # force the ratio, reconstruct and normalise

def generate_velocity_module(r, func_distrib_velocities) :
  # Computes a velocity module following the velocity spatial distribution function for a cluster at radius r.
  # @param r radius
  # @param func_distrib_velocities (global, function) velocity"s magnitude"s distribution function
  # @return a velocity module
  return(func_distrib_velocities(sigmaGaussianForHalfGaussian(r), np.size(r, 0)))

def test_velocities_anisotropy(positions, velocities) :
  # Compare theoretical anisotropy and computed anisotropy.
  # @param positions a set of positions (array 3 * g_GCNStars)
  # @param velocities a set of velocities (array 3 * g_GCNStars)
  # @return none
  print("GC Self-Anisotropy :")
  print(" Wanted anisotropy :                "+"{:9.2e}".format(float(g_beta))+".")
  print(" Computed anistropy :               "+"{:9.2e}".format(extractAnisotropyBeta(positions, velocities))+".")

def test_velocities_dispersions(radii, velocities, nBins) :
  # Compare theoretical standard deviation of velocities" magnitudes and standard deviation of a set of velocities.
  # @param radii a set of radii (array g_GCNStars)
  # @param velocities a set of velocities (array 3 * g_GCNStars)
  dispersions=extractVelocitiesSigma(velocities, radii, nBins, retVals="all") # extract diverse informations (see method in util.py)
  xgrid=np.linspace(min(radii), max(radii), 1000)
  fig=plt.figure(figsize=g_squareFigSize)
  ax=fig.add_subplot(111)
  ax.set_xlabel(r"$r$ ([L])")
  ax.set_ylabel(r"$\sigma(r)$ (m/s)")
  ax.set_title("Theoretical velocity Distribution and computed Dispersions\n(difference : "+"{:1.2f}".format(100*np.mean(np.abs((sigma(xgrid)-dispersions[2]*np.ones(1000))/sigma(xgrid))))+" %)")
  ax.plot(xgrid, sigma(xgrid))
  ax.semilogx(dispersions[1], dispersions[0])
  ax.plot(xgrid, dispersions[2]*np.ones(1000))
  ax.legend(["wanted dispersion", "dispersion per bin", "global dispersion"], loc="best")
  plt.savefig("initPlots/initialConditions/dispersions")
###############################################################

# Main Program. ###############################################
plt.close("all")

# Informative pre-treatment ###
# graphs.                   ###
if show_rho :
  x=np.linspace(0, 2*g_GCConcentration, 1000)
  fig=plt.figure(figsize=g_squareFigSize)
  ax=fig.add_subplot(111)
  ax.set_xlabel(r"$r$ ([L])")
  ax.set_ylabel(r"$\rho(r)$")
  ax.set_title("Theoretical radial Distribution")
  ax.loglog(x, rho(x), "b")
  #ax.plot(x, rho(x), "b")
  #ax.plot(g_GCConcentration*np.ones([1000, 1]), np.linspace(0.95*min(rho(x)), 1.05*max(rho(x)), 1000), "g")
  ax.set_xlim([1e-2, max(x)])
  ax.set_ylim([1e-1, 1.25*rho(0)])
  ax.axvline(g_GCConcentration, linestyle="--", color="g")
  ax.axhline(rho(g_GCConcentration), linestyle=":", color="g")
  plt.legend([r"$\rho\left(r\right)$", "$b$", r"$\rho\left(b\right)$"], loc="best")
  plt.tight_layout()
  plt.savefig("initPlots/initialConditions/rho")

if show_rhoCustomRV :
  show_custom_VAR(customRandomVariableRho, "initPlots/initialConditions/custom_RV")

if show_plotGCInitialPositionsAndVelocities :
  plotGCInitialPositionsAndVelocities()

if show_galacticDensity and g_galacticEnvironnement==1 :
  plotGalacticDensity(6e3, 6e3, g_Mb, g_ab, g_Md, g_ad, g_hd, g_Vh, g_ah, plots="baryonic", initPos=True, initPosCoord=g_GCPosition, initVel=True, initVelCoord=g_GCVelocity)
  plt.tight_layout()
  #plt.savefig("plots/galactic_density")

if show_demoVectorEvolutionGraph :
  demoVectorEvolutionGraph()
###############################

# Positions. ##################
radii=generate_radii(g_GCNStars);
X=generateVectorsRandomDirection(radii)
###############################

# Velocities. #################
V=generateVectorsRandomDirection(np.ones(g_GCNStars)) # generate random velocities of norm 1
V=force_anisotropy(radii, V) # force the anisotropy parameter
V=V*generate_velocity_module(radii, velocityDistribution) # find the actual modules
###############################

# Visualise positions and #####
# velocities.             #####
if show_3DPlot :
  plot_window=3*g_GCConcentration
  quiv_ar_len_rat=0.25
  quiv_len=0.05
  fig=plt.figure(figsize=g_bigSquareFigSize)
  ax=fig.add_subplot(111, projection="3d")
  ax.set_xlim([-plot_window, plot_window])
  ax.set_ylim([-plot_window, plot_window])
  ax.set_zlim([-plot_window, plot_window])
  ax.set_xlabel(r"$x$ ([L])")
  ax.set_ylabel(r"$y$ ([L])")
  ax.set_zlabel(r"$z$ ([L])")
  plt.title("Globular Cluster Initial Positions and Velocities")
  
  ax.scatter(X[0, :], X[1, :], X[2, :], zdir="z", s=30, c="orange", depthshade=True, edgecolor="red")
  ax.scatter([0], [0], [0], zdir="z", s=20, c="k", depthshade=True) # show the theoretical GC center
  #ax.quiver(X[0, :], X[1, :], X[2, :], V[0, :], V[1, :], V[2, :], arrow_length_ratio=quiv_ar_len_rat, length=quiv_len, pivot="tail", color="g")
  ax.quiver(X[0, :], X[1, :], X[2, :], V[0, :], V[1, :], V[2, :], arrow_length_ratio=quiv_ar_len_rat, length=quiv_len, pivot="tail", color="k")
  plt.tight_layout(); plt.savefig("initPlots/initialConditions/3d_view")
  
  fig=plt.figure(figsize=g_bigSquareFigSize)
  ax=fig.add_subplot(111)
  
  plt.title(r"Globular Cluster Initial Positions and Velocities (XY plane)"); ax.set_xlabel(r"$x$ ([L])"); ax.set_ylabel(r"$y$ ([L])"); ax.set_xlim([-plot_window, plot_window]); ax.set_ylim([-plot_window, plot_window])
  ax.quiver(X[0, :], X[1, :], V[0, :], V[1, :], pivot="tail", color="k")
  ax.scatter(X[0, :], X[1, :], s=30, c="orange", edgecolor="red")
  plt.tight_layout(); plt.savefig("initPlots/initialConditions/3d_view_xOy"); ax.clear();
  
  plt.title(r"Globular Cluster Initial Positions and Velocities (XZ plane)"); ax.set_xlabel(r"$x$ ([L])"); ax.set_ylabel(r"$z$ ([L])"); ax.set_xlim([-plot_window, plot_window]); ax.set_ylim([-plot_window, plot_window])
  ax.quiver(X[0, :], X[2, :], V[0, :], V[2, :], pivot="tail", color="k")
  ax.scatter(X[0, :], X[2, :], s=30, c="orange", edgecolor="red")
  plt.tight_layout(); plt.savefig("initPlots/initialConditions/3d_view_xOz"); ax.clear()

  plt.title(r"Globular Cluster Initial Positions and Velocities (YZ plane)"); ax.set_xlabel(r"$y$ ([L])"); ax.set_ylabel(r"$z$ ([L])"); ax.set_xlim([-plot_window, plot_window]); ax.set_ylim([-plot_window, plot_window])
  ax.quiver(X[1, :], X[2, :], V[1, :], V[2, :], pivot="tail", color="k")
  ax.scatter(X[1, :], X[2, :], s=30, c="orange", edgecolor="red")
  plt.tight_layout(); plt.savefig("initPlots/initialConditions/3d_view_yOz")
###############################

# Define masses. ##############
M=ensureLineVect(np.ones(np.size(radii))*g_GCStarsMeanMass, np.size(radii))
###############################

# Scale quantities. ###########
M=M/g_scalingM
X=X/g_scalingL
V=V/g_scalingV
g_GCVelocity=g_GCVelocity/g_scalingV

d=cdist(X, X)
g_softening=np.min([g_softening, np.min(d[np.nonzero(d)])/100]) # update softening just in case
###############################

# Move to galactic position ###
# (if needed).              ###
if(g_galacticEnvironnement==1) :
  X+=g_GCPosition
  V+=g_GCVelocity
  print("A galaxy gravitationnal field will be used.")
else :
  print("No galaxy gravitationnal field will be used.")
###############################

# Tests and post-treatment ####
# graphs.                  ####
tmp_integrations=g_simulationDuration/g_integrationTimeStep
if(g_technique=="RK4") : tmp_evaluations=4
elif(g_technique=="PECE") : tmp_evaluations=2
elif(g_technique=="PEC") : tmp_evaluations=1
else : criticalBadValue("g_technique")
tmp_forcesComputations=tmp_evaluations*g_GCNStars**2
if g_galacticEnvironnement==1 :
  tmp_galacticForcesComputations=g_GCNStars
else :
  tmp_galacticForcesComputations=0
tmp_totalComputations=tmp_integrations*(tmp_forcesComputations+tmp_galacticForcesComputations)
tmp_expectedDuration=estimateExecutionTime(tmp_totalComputations, show_durationEstimationFitPlot); tmp_timeUnit="seconds"; tmp_expectedDurationSeconds=tmp_expectedDuration
if tmp_expectedDuration>=60 and tmp_expectedDuration<3600 : tmp_expectedDuration=tmp_expectedDuration/60; tmp_timeUnit="minutes"
if tmp_expectedDuration>=3600 : tmp_expectedDuration=tmp_expectedDuration/3600; tmp_timeUnit="hours"
tmp_printings=1+g_simulationDuration/g_outputPrintInterval
tmp_totalLinePrintings=tmp_printings*g_GCNStars
tmp_outputFileWeight=g_outputFileLineSizeBytes*tmp_totalLinePrintings; tmp_sizeUnit="o"
if tmp_outputFileWeight>=1024 and tmp_outputFileWeight<1048576 : tmp_outputFileWeight=tmp_outputFileWeight/1024; tmp_sizeUnit="ko"
if tmp_outputFileWeight>=1048576 : tmp_outputFileWeight=tmp_outputFileWeight/1048576; tmp_sizeUnit="Mo"
if tmp_outputFileWeight>=1073741824 : tmp_outputFileWeight=tmp_outputFileWeight/1073741824; tmp_sizeUnit="Go"

if g_GCNStars>=30 :
  test_radii_distribution(radii, rho) # test positions
  test_velocities_dispersions(radii, V-g_GCVelocity, int(g_GCNStars/10)) # test velocities dispersion of the GC alone
if(g_verbose>=2) :
  print("\n", end=""); print("Velocities :"); baryV=np.transpose(barycenter(V))[0];
  print(" Velocity barycenter : ", end=""); print("["+"{:9.2e}".format(baryV[0])+", "+"{:9.2e}".format(baryV[1])+", "+"{:9.2e}".format(baryV[2])+"]", end=""); print(".")

print("\n", end=""); test_velocities_anisotropy(X-g_GCPosition, V-g_GCVelocity) # test velocities anisotropy of the GC alone (against wanted anisotropy)

print("\n", end=""); print("GC Star Crossings :");
print(" GC Star Crossing time :            "+"{:5.2f}".format(g_crossingTime)+" Myr.\n"+
      " Estimated star crossings :         "+"{:5.2f}".format(g_simulationDuration/g_crossingTime)+".")

print("\n", end=""); print("Simulation :");
if(g_verbose>=2) :
  print(" Number of integrations :           "+"{:9.2e}".format(tmp_integrations)+".")
  print(" Evaluations of f :                  "+"{:1.0f}".format(tmp_evaluations)+".")
  print(" Interaction forces (p/i) :         "+"{:9.2e}".format(tmp_forcesComputations)+".")
  if g_galacticEnvironnement==1 :
    print(" Galatic forces (p/i) :             "+"{:9.2e}".format(tmp_galacticForcesComputations)+".")
print(" Forces to be calculated (total) :  "+"{:9.2e}".format(tmp_totalComputations)+".")
print(" Estimated expected duration :     "+"{:6.2f}".format(tmp_expectedDuration)+" "+tmp_timeUnit+".")
print(" Number of snapshots :              "+str(int(1+g_simulationDuration/g_outputPrintInterval))+".")
print(" Estimated output file size :      "+"{:6.2f}".format(tmp_outputFileWeight)+" "+tmp_sizeUnit+".")
print(" Simulation should finish on the "+addSecs(datetime.datetime.now(), tmp_expectedDurationSeconds).strftime("%d/%m at %H:%M")+".")
print("")
###############################

# Export to MiniNBody format. #
exportInitialConditionsToFile(g_inputFilePath,
                              X, V, M,
                              g_simulationDuration,
                              g_integrationTimeStep,
                              g_outputPrintInterval,
                              g_outputFilePath,
                              g_technique,
                              g_softening,
                              g_galacticEnvironnement,
                              g_printExplanationsInInputFile,
                              g_galaxyParams,
                              (g_GravConstant, g_scalingL, g_scalingM, g_scalingT, g_scalingV))
###############################
                              
###############################################################