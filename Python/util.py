from os import path
from scipy import stats
from sklearn.neighbors import KDTree
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Remarks. ####################################################
# Functions that are specific to the study of GCs have it mentionned in their description. The others are usable over any type of study.
###############################################################

# Modifiable parameters. ######################################
# These can be more or less modified to the convienience of the user.
g_bigSquareFigSize=(12, 12) # a figure size
g_fullpageFigSize=(9, 12) # a figure size
g_fullscreenFigSize=(18, 12) # a figure size
#g_maxNSearch=200
g_maxSoftening=1e-8 # maximum softening (in [L])
#g_smartSelectionMinPercentage=50 # smart selection : percentage of stars to absolutely keep
g_solarSystemOutputFolder="solarSystemOutputs"
g_squareFigSize=(10, 10) # a figure size
g_squishedSquareFigSize=(10, 6) # a figure size
g_verbose=1 # verbosity level of the generator/reader
g_diagnosticsFolder="diagnostics"
g_hideTitles=True
g_graphFontSize_Title=100
g_graphFontSize_XLabel=20
g_graphFontSize_YLabel=20
g_graphFontSize_Ticks=15
g_videoSizeInches=6
###############################################################

# Parameters. #################################################
# These should be left as they are.
# Constants. ##################
g_daysInYear=365.25 # self-explanatory
g_oneParsec=3.08567758149e16 # 1 pc, in m
g_oneSolarMass=1.9885e30 # 1 solar mass, in kg
g_GravConstant=4.302*10**(-3) # self-explanatory
###############################
# Data (for estimations). #####
g_executionDataNbInteg=np.array([
6.15e7, 4.86e8, 1.52e9, 6.02e9, 2.4e10, 5.4e10, 6e11, 2.4e12, 1e13, 1.21e13, 3.84e13
])
g_executionDataTimes=np.array([
1.35, 9.52, 22.6, 44.32, 60*2.64, 60*5.7, 3600*3.61, 3600*16.20, 3600*126.92, 3600*149.38, 3600*222.19
])/3600
tmp_totalOutputFileSizeBytes=156553216
tmp_outputFileNStars=5500
tmp_outputFileNSnapshots=202
###############################
# Solar system related. #######
solSysColors=(np.zeros([10, 3]), np.zeros([10, 3]))
solSysColors[0][0, :]=[1.00, 0.77, 0.40]; solSysColors[1][0, :]=[0.90, 0.67, 0.30]
solSysColors[0][1, :]=[0.69, 0.69, 0.69]; solSysColors[1][1, :]=[0.59, 0.59, 0.59]
solSysColors[0][2, :]=[0.79, 0.78, 0.74]; solSysColors[1][2, :]=[0.69, 0.68, 0.64]
solSysColors[0][3, :]=[0.63, 0.69, 0.85]; solSysColors[1][3, :]=[0.53, 0.59, 0.75]
solSysColors[0][4, :]=[0.91, 0.58, 0.33]; solSysColors[1][4, :]=[0.81, 0.58, 0.23]
solSysColors[0][5, :]=[0.91, 0.77, 0.51]; solSysColors[1][5, :]=[0.81, 0.67, 0.41]
solSysColors[0][6, :]=[0.90, 0.74, 0.62]; solSysColors[1][6, :]=[0.80, 0.64, 0.52]
solSysColors[0][7, :]=[0.75, 0.90, 0.91]; solSysColors[1][7, :]=[0.65, 0.80, 0.81]
solSysColors[0][8, :]=[0.30, 0.51, 0.99]; solSysColors[1][8, :]=[0.20, 0.41, 0.89]
solSysColors[0][9, :]=[0.87, 0.70, 0.54]; solSysColors[1][9, :]=[0.77, 0.60, 0.44]
###############################
# Scaling parameters. #########
g_scalingM=1 # mass unit : in solar masses (since we want to specify masses in terms of solar masses)
g_scalingT=1 # time unit : in Myr (since we want to specify wanted times in Myr)
g_scalingL=1 # distance unit : in pc (since we want to specify distances in parsecs)
# Automatic treatment of ######
# scaling units.         ######
g_oneMegaYear=60*60*24*g_daysInYear*1e6 # 1 Myr, in s
g_scalingAngMom=g_oneSolarMass*g_oneParsec**2*g_oneMegaYear**(-1) # angular momentum unit
g_scalingE=g_oneSolarMass*g_oneParsec**2*g_oneMegaYear**(-2) # energy unit
g_scalingV=g_oneParsec/g_oneMegaYear # velocity unit
g_scalingF=g_oneSolarMass*g_oneParsec**g_oneMegaYear**(-2) # force unit
###############################
# Automatic treatment of     ##
# parameters and computation ##
# of interesting quantities. ##
tmp_lM=5-1/138 # from graph
tmp_lsig=0.35 # from graph
g_aFaberJackson=(10**tmp_lsig)/((10**tmp_lM)**0.25) # a coefficient from law sigma=a*M**0.25 and Scarpa's sample and Lane's sample (Lane et al. 2009,  2010a,  b,  2011)
g_outputFileLineSizeBytes=tmp_totalOutputFileSizeBytes/(tmp_outputFileNStars*tmp_outputFileNSnapshots) # how a line in the output file should weight
g_theoreticalGravConstant=6.6738e-11/(g_oneParsec**3*g_oneSolarMass**(-1)*g_oneMegaYear**(-2)) # how G should be calculated (can't explain the difference with the value usually used in astrophysics)
###############################
# Titles of graphs. ###########
g_GCCentroidPositionGraphTitle="Position of the Centroid of the GC"
g_GCCentroidSpeedGraphTitle="Velocity of the GC"
g_angMomentaInertiaTensorSymbol=(r"$\Delta_{\vec{\mathcal{L}}", r"}(t)$")
g_angMomentaInertiaTensorGraphTitle="Rotation Factor "+g_angMomentaInertiaTensorSymbol[0]+g_angMomentaInertiaTensorSymbol[1]+"\n"+r"($1$ : no rotation axis privilegied, $>1$ : globally rotating in an axis)"
g_angMomScalingAxisTitle="([E][T])"
g_anisotropyGraphTitleLine1=r"Anisotropy $\beta(t)$"
g_anisotropyGraphTitleLine2=r"($\beta\rightarrow-\infty\Leftrightarrow$tangential velocities, $\beta\rightarrow1\Leftrightarrow$radial velocities)"
g_axisLabelTime=r"$t$ ([T])"
g_deviationFromSphericitySymbol=(r"$\Delta_{S", r"}(t)$")
g_deviationFromSphericityGraphTitle=r"Deviation from Sphericity "+g_deviationFromSphericitySymbol[0]+g_deviationFromSphericitySymbol[1]+"\n"+r"($1$ : perfectly spherical, $>1$ : stretched)"
g_energyScalingAxisTitle="([E])"
g_escapersGraphTitle="Evolution of the Percentage of Escapers"
g_fitVelocityGraphTitle="Fits of Velocity Density"
g_massEvolutionGraphTitle="Evolution of the total Mass of the GC"
g_specificVelocityDispersionGraphTitle=r"Specific Velocity Dispersion $\sigma_S(t)$"
g_tensorAnalysisMaxEigenGraphTitle="Maximum Eigenvector Evolution"
g_torqueScalingAxisTitle="([E])"
g_velocityDispersionGraphTitle=r"Velocity Dispersion $\sigma(t)$"
g_velocityKurtosisGraphTitle="Kurtosis of Velocities' Modules"
g_velocityMeanSymbol=r"$\langle\left\|\vec{V}\right\|_2\rangle$"
g_velocityMeanGraphTitle="Mean of Velocities' Modules "+g_velocityMeanSymbol
g_velocitySkewGraphTitle="Skewness of Velocities' Modules"
g_velocityTensorDeviationFromIsotropySymbol=(r"$\Delta_{I", r"}(t)$")
g_velocityTensorDeviationFromIsotropyGraphTitle=r"Velocity Deviation from Isotropy "+g_velocityTensorDeviationFromIsotropySymbol[0]+g_velocityTensorDeviationFromIsotropySymbol[1]+"\n"+r"($1$ : perfectly isotropic, $>1$ : stretched)"

g_tidalForcesGraphTitle=r"Tidal Forces Norms Evolution"
g_forceScalingAxisTitle="([F])"
###############################
###############################################################

# Functions. ##################################################
def addCrossingLines(doIt, ax, data, times) :
  # Add disk crossing lines (z=0) to a plot which has time on the x-axis.
  # This function is specific to a study where the galactic gravitationnal field has been added.
  # @param doIt allows conditions to be checked when called, encode if lines should be added
  # @param ax axes of the wanted plot (can be gotten via plt.gca() for a current plot)
  # @param data a set of positions (array 3 * N * T)
  # @param times a set of times (array T)
  # @return none
  if(doIt) :
    (cIDs, cTs)=getDiskCrossingTimes(data, times)
    if(cIDs==[]) and g_verbose>2 :
      warningError("No disk crossing time found, do nothing.")
    else :
      for t in cTs :
        ax.axvline(t, linestyle='k:')

def addCrossingLinesCPSaved(doIt, ax, cP, times) :
  # Add disk crossing lines (z=0) to a plot which has time on the x-axis.
  # This function is specific to a study where the galactic gravitationnal field has been added.
  # @param doIt allows conditions to be checked when called, encode if lines should be added (True if yes, False if no)
  # @param ax axes of the wanted plot (can be gotten via plt.gca() for a current plot)
  # @param cP the position of a centroid over time (array 3 * T)
  # @param times a set of times (array T)
  # @return none
  if(doIt) :
    (cIDs, cTs)=getDiskCrossingTimesCPSaved(cP, times)
    if(cIDs==[]) and g_verbose>2 :
      warningError("No disk crossing time found, do nothing.")
    else :
      for t in cTs :
        ax.axvline(t, linestyle=':', color="k")

def addCrossingLineInterface(cPSaved, doIt, ax, times, dat, savedCP) :
  # Provides an interface for the adding of the crossing lines in several cases.
  # This function is specific to a study where the galactic gravitationnal field has been added.
  # @param cPSaved whether or not the centroid position has been saved beforehand (True if yes, False if no)
  # @param doIt allows conditions to be checked when called, encode if lines should be added (True if yes, False if no)
  # @param ax axes of the wanted plot (can be gotten via plt.gca() for a current plot)
  # @param times a set of times (array T)
  # @param dat a set of positions (array 3 * N * T)
  # @param cP the position of a centroid over time (array 3 * T)
  # @return none
  if(cPSaved) : addCrossingLinesCPSaved(doIt, ax, savedCP, times)
  else : addCrossingLines(doIt, ax, dat, times)

def addSecs(tm, secs) :
  # Adds a certain amount of seconds to a time.
  # @param tm a time (for example, gotten with datetime.datetime.now())
  # @param sec a number of seconds
  # @return the time sec seconds after tm
  fulldate=datetime.datetime(tm.year, tm.month, tm.day, tm.hour, tm.minute, tm.second)
  fulldate=fulldate+datetime.timedelta(seconds=secs)
  return(fulldate)

def barycenter(dat, masses=-1) :
  # Returns the centroid of a set of positions, weighted by masses.
  # @param dat a set of positions (array 3 * N)
  # @param masses (optional) a set of masses (needs and array of size N, but if not specified masses will be supposed of same value)
  # @return the centroid of the set of positions (as an array 3 * 1, ie a column vector)
  if(type(masses)==np.ndarray and np.size(dat, axis=1)!=np.size(masses)) : criticalError("Sizes of arrays dat and masses are not the same.")
  if(type(masses)==int) : w=np.ones(np.size(dat, axis=1)) # masses of same value are assumed
  else : w=masses
  return(ensureColVect(np.average(dat, axis=1, weights=w)))

def changeReference(X, e1, e2, e3) :
  # Changes the reference in which the vector is expressed.
  # @param X the vector or set of vectors to express in the new reference (dimension 0 should be 3, dimension 1 can be any)
  # @param e1 1st new base vector, expressed in old reference
  # @param e2 2nd new base vector, expressed in old reference
  # @param e3 3rd new base vector, expressed in old reference
  # @return the vector, expressed in the new reference
  PInv=np.linalg.inv(np.hstack((ensureColVect(e1), ensureColVect(e2), ensureColVect(e3)))) # construct matrix to go from old to new base
  return(np.dot(PInv, X))

def centroidPosition(dat, masses=-1) :
  # Computes the position of the centroid of a set of positions over time.
  # This function is specific to the study of GCs as is because of the dependance on the function centroidPositionBetter, but decommenting the first line enables the use of it in any type of study.
  # @param dat a set of positions (array 3 * N * T)
  # @param masses (optional) a set of masses (needs and array of size N, but if not specified masses will be supposed of same value)
  # @return the position of the centroid over time (array 3 * T)
  if(type(masses)==np.ndarray and np.size(dat, axis=1)!=np.size(masses)) : criticalError("Sizes of arrays dat and masses are not the same.")
  #return(centroidPositionOld(dat, masses)) # decomment this for a use in a more general study
  if(g_useSmartSelection) : return(centroidPositionBetter(dat, masses))
  else : return(centroidPositionOld(dat, masses))

def centroidPositionBetter(dat, masses) :
  # Computes the position of the centroid of a set of positions over time with a better way.
  # This function is specific to the study of GCs because of the dependance of the function dataAutoSelection.
  # @param dat a set of positions (array 3 * N * T)
  # @param masses a set of masses (needs an array of size N or -1 to suppose masses of same value)
  # @return the position of the centroid over time (array 3 * T)
  T=np.size(dat, axis=2)
  centroidP=np.zeros([3, T])
  br=userInputGetNumber("[Computation of the COM] Enter the wanted GC radius ([L] dimension, >"+"{:1.2f}".format(0)+", <"+"{:1.2f}".format(10)+"). GC concentration works generally well.",
                        minAccepVal=1e-9,
                        maxAccepVal=np.inf,
                        specials=[])
  print("Computing COM position.")
  for t in range(0, T) :
    print("t="+str(t))
    #ns=max([int(g_maxNSearch*np.max(np.linalg.norm(dat[:, :, t], axis=0))/(3e5)), 20])
    centroidP[:, t]=ensureLineVect(dataAutoSelection1(dat[:, :, t], findWhat="centroid", bestRadius=br))
  return(centroidP)

def centroidPositionOld(dat, masses) :
  # Computes the position of the centroid of a set of positions over time in a classical way.
  # @param dat a set of positions (array 3 * N * T)
  # @param masses a set of masses (needs an array of size N or -1 to suppose masses of same value)
  # @return the position of the centroid over time (array 3 * T)
  T=np.size(dat, axis=2)
  centroidP=np.zeros([3, T])
  for t in range(0, T) :
    centroidP[:, t]=barycenter(dat[:, :, t], masses)[:, 0]
  return(centroidP)

#def coordinates_conversion_spherical2cartesian(r, theta, phi) :
#  # Calculates the cartesian coordinates of a point in spherical coordinates.
#  # @param r the radius
#  # @param theta the azimuth
#  # @param phi the polar angle
#  # return a column vector containing the cartesian coordinates
#  if(not(np.size(r)==np.size(theta)==np.size(phi))) : # coordinates do not have the same lengths
#    criticalError("Coordinates do not have the same lengths.")
#  X=np.zeros([3, np.size(r)])
#  X[0, :]=r*np.sin(phi)*np.cos(theta)
#  X[1, :]=r*np.sin(phi)*np.sin(theta)
#  X[2, :]=r*np.cos(phi)
#  return(X)

def criticalBadValue(parameterName, functionName=-1) :
  # Prints a message about a critical bad value in a function and stops the program.
  # @param parameterName the name of the bad valued parameter
  # @param functionName (optional) the name of the function where the error occured (default is -1 to not display it)
  # @return none
  if(functionName==-1) : string="Parameter \""+parameterName+"\" has a bad value."
  else : string="Parameter \""+parameterName+"\" (function \""+functionName+"\") has a bad value."
  print(string)
  raise ValueError

def criticalError(msg) :
  # Prints a message about a critical error and stops the program.
  # @param msg an error message
  # @return none
  print("> ERROR : "+msg+"\n")
  raise SystemExit(-1)

#def dataAutoSelection(dat, masses=-1, findWhat="recenteredData", nSearch=10, bestRadius=5) :
#  # @param dat (3 * N)
#  minimumGCRadius=1 # in [L]
#  maximumGCRadius=1e3 # in [L]
#  tree=KDTree(dat.T)
#  maxScoreCondFulfilled=False
#  while(not(maxScoreCondFulfilled)) :
#    ids=list()
#    scores=list()
#    maxUpdateRad=list()
#    for r in np.logspace(np.log10(minimumGCRadius), np.log10(maximumGCRadius), nSearch) :
#      q=tree.query_radius(dat.T, r, count_only=True)
#      qam=np.argmax(q)
#      if(qam in ids) :
#        savedId=ids.index(qam)
#        scores[savedId]+=1
#        maxUpdateRad[savedId]=r
#      else :
#        ids.append(qam)
#        scores.append(1)
#        maxUpdateRad.append(r)
#    scores=np.array(scores)
#    maxScoreCondFulfilled=((np.size(np.where(scores==max(scores))[0])<5 and max(scores)>4) or nSearch>200) # to update loop condition
#    nSearch=int(1.25*nSearch) # if loop condition not fulfilled, redo with more subdivisions
#  ids=np.array(ids)
#  maxUpdateRad=np.array(maxUpdateRad)
#  maxScores=scores[np.argsort(scores)] # sorted scores
#  maxScoresPond=maxScores-1 # remove null scores
#  maxRadPond=(maxUpdateRad[np.argsort(scores)]-bestRadius)**(-2) # ponderate last update radii by the square of the distance to the best radius (small is best)
#  bestSortedNumber=np.argsort(np.multiply(maxScoresPond, maxRadPond))[-1]
#  chosenOne=ids[np.argsort(scores)][bestSortedNumber]
##  maxScoreIds=ids[maxScore]
##  maxScoreMaxUpdateRad=maxUpdateRad[maxScore]
##  chosenOne=maxScoreIds[np.argsort(maxScoreMaxUpdateRad)[0]]
#  c=ensureColVect(dat[:, chosenOne])
#  if(findWhat=="recenteredData") : return(dat-c)
#  elif(findWhat=="centroid") : return(c)
#  else : criticalBadValue("findWhat"); return(-1)

#def dataAutoSelection0(dat, masses=-1, findWhat="recenteredData", minSelectionPercentage=-1) :
#  kde=stats.gaussian_kde(dat)
#  c=ensureColVect(dat[:, np.argmax(kde(dat))])
#  if(findWhat=="recenteredData") : return(dat-c)
#  elif(findWhat=="centroid") : return(c)
#  else : criticalBadValue("findWhat"); return(-1)

def dataAutoSelection1(dat, masses=-1, findWhat="recenteredData", bestRadius=5) :
  # Tries to find a centroid that fits as best as possible to a system of particles.
  # @param dat a set of positions (array 3 * N)
  # @param masses (optional) a set of masses (needs and array of size N, but if not specified masses will be supposed of same value)
  # @param findWhat (optional) what to return (can be either "recenteredData" to get the recentered data or "centroid" to get the centroid, default if "recenteredData")
  # @param bestRadius (optional) the best radius within which should be most of the particles (in [L], default is 5)
  # @return the recentered data by default of if parameter findWhat is set to "recenteredData" or the centroid if findWhat is set to "centroid"
  rs=np.linspace(0.5, 8, 5)*bestRadius # radii to test
  NBws=105 # number of bandwidths to generate test
  NSelBws=35 # number of bandwiths to test
  bws=np.logspace(-3, 1, NBws)[np.random.permutation(NBws)[0:NSelBws]] # bandwiths to test
  count=np.zeros((np.size(rs), np.size(bws)))
  tree=KDTree(dat.T)
  best=ensureColVect(dat[:, np.argmax(tree.kernel_density(dat.T, h=np.mean(bws), kernel='gaussian'))])
  bestScore=-np.inf
  bestBw=0
  for j, bw in enumerate(bws) :
    kde=tree.kernel_density(dat.T, h=bw, kernel='gaussian')
    c=ensureColVect(dat[:, np.argmax(kde)])
    for i, r in enumerate(rs) : # test radii, to choose least variating number
      count[i, j]=tree.query_radius(c.T, r, count_only=True)      
      if(np.var(count[:, j])>0) :
        score=np.mean(count[:, j])**(2)*np.var(count[:, j])**(-1)
        if(score>bestScore) :
          best=c
          bestScore=score
          bestBw=bws[j]
    # if not, continue
  c=best
  print("best bw : "+str(bestBw))
  if(findWhat=="recenteredData") : return(dat-c)
  elif(findWhat=="centroid") : return(c)
  else : criticalBadValue("findWhat"); return(-1)
  
#def dataAutoSelection(dat, masses=-1, findWhat="recenteredData", minSelectionPercentage=-1) :
#  n=1000
#  kdeX=stats.gaussian_kde(dat[0, :])
#  kdeY=stats.gaussian_kde(dat[1, :])
#  kdeZ=stats.gaussian_kde(dat[2, :])
#  x=np.linspace(min(dat[0, :]), max(dat[0, :]), n)
#  y=np.linspace(min(dat[1, :]), max(dat[1, :]), n)
#  z=np.linspace(min(dat[2, :]), max(dat[2, :]), n)
#  centroid=ensureColVect([x[np.argmax(kdeX(x))], y[np.argmax(kdeY(y))], z[np.argmax(kdeZ(z))]])
#  if(findWhat=="recenteredData") : return(dat-centroid)
#  elif(findWhat=="centroid") : return(centroid)
#  else : criticalBadValue("findWhat"); return(-1)

#def dataAutoSelection(dat, masses=-1, findWhat="recenteredData", minSelectionPercentage=-1) :
#  # Tries to find a centroid that fits as best as possible to the GC. That means, ignore stars that are way too far and try to center the centroid on the compact group of stars that should be in the center of the GC.
#  # This function is specific to the study of GCs.
#  # @param dat a set of positions (array 3 * N)
#  # @param masses (optional) a set of masses (needs and array of size N, but if not specified masses will be supposed of same value)
#  # @param findWhat (optional) what to return (can be either "recenteredData" to get the recentered data or "centroid" to get the centroid)
#  # @param minSelectionPercentage (optional) can be use to overwrite the global parameter g_smartSelectionMinPercentage
#  # @return the recentered data by default of if parameter findWhat is set to "recenteredData" or the centroid if findWhat is set to "centroid"
#  if(type(masses)==np.ndarray and np.size(dat, axis=1)!=np.size(masses)) : criticalError("Sizes of arrays dat and masses are not the same.")
#  maxIter=100 # maximum number of iterations to search for the best centroid
#  critCenterRadius=5 # in coordination with critCenterPercentage, specify the radius of the core of the GC
#  critCenterPercentage=80 # in coordination with critCenterRadius, specify what percentage of stars should be in the core of the GC
#  if(minSelectionPercentage==-1) : # use the percentage by default
#    minSelectionPercentage=g_smartSelectionMinPercentage
#  else :
#    if(minSelectionPercentage<0) :
#      criticalBadValue("minSelectionPercentage")
#  N=np.size(dat, axis=1) # get the number of stars at play
#  critCenterNumber=N*critCenterPercentage/100 # get the number of stars that should be in the core of the GC
#  i=0 # initialise
#  centroid=barycenter(dat, masses) # find global centroid
#  trueAbsoluteCentroid=centroid # in case true centroid is encountered at first step
#  bestTrueAbsoluteCentroid=trueAbsoluteCentroid # in case true centroid is encountered at first step
#  rData=dat-centroid # recenter first around global centroid
#  NAtCenter=0 # initialise at dummy value
#  NSelected=10*N # initialise at dummy value
#  prevBestNAtCenter=NAtCenter # initialise
#  s=1.25 # initialise the selecting range a little further than the maximum radii
#  while(NAtCenter<critCenterNumber and i<maxIter and NSelected>N*g_smartSelectionMinPercentage/100) :
#    # While these 3 conditions are all met : 1) we do not have enough stars in the core of the GC,
#    #                                        2) maximum number of iterations is not reached,
#    #                                        3) unselected number of stars is under maximum number of stars to be able to reject,
#    # continue to search.
#    NAtCenter=np.size(getIDsInsideRadius(rData, critCenterRadius, centroid)[0]) # evaluate the number of stars in the current core of the GC
#    if(NAtCenter>prevBestNAtCenter) : # if it is better than previous best, save this state (in case nothing else is found)
#      prevBestNAtCenter=NAtCenter
#      bestTrueAbsoluteCentroid=trueAbsoluteCentroid
#    radii=np.linalg.norm(rData, axis=0) # get the radii based on current recentered data
#    selIDs=getIDsInsideRadius(rData, s*np.max(radii))[0] # select some radii around current barycenter (reject stars further than a value specified by s)
#    centroid=barycenter(rData[:, selIDs], masses) # find new centroid (without rejected stars)
#    trueAbsoluteCentroid=trueAbsoluteCentroid+centroid # move the global centroid to current position
#    rData=rData-centroid # recenter
#    s=s/1.1 # prepare new selection radius, smaller than previous one to reject a little more stars if condition 1) is not yet met
#    NSelected=np.size(selIDs) # verify we have still enough stars
#    i=i+1
#  # This is the end of the loop, check if all went well and do adjustements if needed.
#  if(g_verbose>=2) :
#    errorMsg="Could not find a centroid where "+"{:1.0f}".format(critCenterPercentage)+"% of stars are present in a radius of "+"{:1.2f}".format(critCenterRadius)+" pc from the center"
#    if(NSelected<N*g_smartSelectionMinPercentage/100) :
#      print(errorMsg+" without removing less than "+"{:1.0f}".format(100-g_smartSelectionMinPercentage)+"% of the farthest stars."+
#      " Using a centroid ensuring "+"{:1.0f}".format(100*prevBestNAtCenter/N)+"% of stars are present in a radius of "+"{:1.2f}".format(critCenterRadius)+" [L] from the center.")
#      trueAbsoluteCentroid=bestTrueAbsoluteCentroid
#      rData=dat-trueAbsoluteCentroid
#    if(i==maxIter) :
#      print(errorMsg+" with "+"{:1.0f}".format(maxIter)+" iterations.")
#  # Return what is wanted.
#  if(findWhat=="recenteredData") : return(rData)
#  elif(findWhat=="centroid") : return(trueAbsoluteCentroid)
#  else : criticalBadValue("findWhat"); return(-1)

def deviationFromSphericity(dat, masses, IEig) :
  # Determines the deviation from sphericity arising from an inertia tensor.
  # @param dat a set of positions (array 3 * N)
  # @param masses a set of masses (array N)
  # @param IEig the diagonalised inertia tensor computed from positions given in dat and masses given in masses (via np.linalg.eig)
  # @return the deviation from sphericity
  if(np.size(dat, axis=1)!=np.size(masses)) : criticalError("[deviationFromSphericity] Sizes of arrays dat and masses are not the same.")
  e=np.sum(masses)**(-1)*np.max(np.linalg.norm(dat, axis=0))**(-2)*IEig[0]
  return(np.abs(np.max(e)/np.min(e)))

def ensureColVect(vect, length=3) :
  # Makes sure a vector is a column vector (array N * 1) and if not, transforms it.
  # @param vect the vector to check
  # @param length (optional) the length of the vector (default is 3)
  # @return the vector as a column vector (array N * 1)
  if(np.size(vect)!=length) : criticalError("Vector is not of size "+str(length)+", and thus cannot be reshaped properly.")
  if(np.shape(vect)!=(length, 1)) : return(np.array(vect).reshape(length, 1))
  else : return(np.array(vect))

def ensureLineVect(vect, length=3) :
  # Makes sure a vector is a line vector (array 1 * N) and if not, transforms it.
  # @param vect the vector to check
  # @param length (optional) the length of the vector (default is 3)
  # @return the vector as a line vector (array 1 * N)
  if(np.size(vect)!=length) : criticalError("Vector is not of size "+str(length)+", and thus cannot be reshaped properly.")
  if(np.shape(vect)!=(length, )) : return(np.array(vect).reshape(length, ))
  else : return(np.array(vect))

def estimate2DDensity(x, y, plotRange, n) :
  # Estimates a 2D density of points over a grid of given range and of given resolution.
  # @param x the 1st coordinate of points
  # @param x the 2nd coordinate of points
  # @param plotRange the plot range of the grid
  # @param n the resolution of the grid (how manys points on each dimension of the grid)
  # @return the two components of the mesh grid and the values of the density on the mesh grid
  (X, Y)=np.meshgrid(np.linspace(-plotRange, plotRange, n), np.linspace(-plotRange, plotRange, n))
  positions=np.vstack([X.ravel(), Y.ravel()])
  values=np.vstack((x, y))
  kernel=stats.gaussian_kde(values, 0.5)
  Z=np.reshape(kernel(positions).T, X.shape)
  return(X, Y, Z)

def estimateExecutionTime(nOfForcesToBeCalculated, showPlot) :
  # Estimate execution time based on a set of measures. This function depends on the global variables g_executionDataNbInteg and g_executionDataTimes which contains the data.
  # @param nOfForcesToBeCalculated the total number of forces to be computed by MiniNBody
  # @param showPlot encodes if the fitting plot for the estimation should be shown (True if yes, False if no)
  # @return an estimation in seconds of the execution time in seconds
  if(showPlot) : plt.figure(); plt.plot(g_executionDataNbInteg, g_executionDataTimes);
  return(3600*np.interp(nOfForcesToBeCalculated, g_executionDataNbInteg, g_executionDataTimes))
#  from scipy.optimize import curve_fit
#  par=curve_fit(estimateExecutionTimeF, g_executionDataNbInteg, 3600*g_executionDataTimes)[0]
#  est=estimateExecutionTimeF(nOfForcesToBeCalculated, *par)
#  if(showPlot) : plt.figure(); plt.plot(g_executionDataNbInteg, g_executionDataTimes); plt.plot(np.linspace(0, g_executionDataNbInteg[-1], 1000), estimateExecutionTimeF(np.linspace(0, g_executionDataNbInteg[-1], 1000), *par)/3600)
#  return(est)

#def estimateExecutionTimeF(x, a, b, c, d, e, f, g) :
#  # Defines the function to estimate the data contained in global variables g_executionDataNbInteg and g_executionDataTimes.
#  # @param x abscissa
#  # @param a 1st parameter of the function
#  # @param b 2nd parameter of the function
#  # @param c 3rd parameter of the function
#  # @return the estimation
#  return(a+b*x**2+c*x**3+d*x**4+e*x**5+f*x**g)

def extractAnisotropyComponents(X, V) :
  # Extracts the radial and tangential velocities (relative to the centroid) of a star.
  # @param X a set of positions, recentered around centroid (array 3 * N)
  # @param V a set of velocities (array 3 * N)
  # @return the radial and tangential velocities of the star (tuple of 2 vectors 3*g_GCNStars)
  VR=extractRadialVelocity(X, V) # extract radial velocities of the clusters
  VT=V-VR # extract tangential velocities of the clusters
  return(VR, VT)

def extractAnisotropyBeta(X, V) :
  # Extracts the anisotropy coefficient.
  # @param X a set of positions, recentered around centroid (array 3 * N)
  # @param V a set of velocities (array 3 * N)
  # @return the anisotropy coefficient
  N=np.size(X, axis=1)
  (VR, VT)=extractAnisotropyComponents(X, V) # extracts the radial and tangential velocities
  VR_mean2=np.sum(np.linalg.norm(VR, axis=0)**2)/N
  VT_mean2=np.sum(np.linalg.norm(VT, axis=0)**2)/N
  return(1-(VT_mean2/VR_mean2))

def extractData(NBodyOutputFileName) :
  # Extract data from a MiniNBody output file.
  # @param NBodyOutputFileName the file path to the MiniNBody output file
  # @return times (array T), masses (array N), positions over time (array 3 * N * T) and velocities over time (array 3 * N * T)
  # Detect N (the number of ###
  # particles) and their    ###
  # masses.                 ###
  seenIndexes=list()
  masses=list()
  with open(NBodyOutputFileName, "r") as f : # open the file
    for line in f : # start to look in the file
      if(line!="" and line!=" \n") : # ignore dummy lines
        current=line.split()[1] # get the ID present on the current line (see format of MiniNBody output)
        if(current in seenIndexes) : # we have seen all possible indexes because it starts to loop (save and quit loop)
          N=len(seenIndexes)
          break
        else :
          seenIndexes.append(current) # conserve see ID
          masses.append(float(line.split()[2])) # conserve masses
  masses=np.array(masses) # convert list into convenient array
  if(g_verbose>0) : print("[extractData] Detected "+str(N)+" particles.")
  #############################
  # Detect T (the number of ###
  # timestamps).            ###
  with open(NBodyOutputFileName, "r") as f : # open the file
    T=int(len(f.readlines())/N) # simply divide total number of lines by number of particles (int(...) in case of unexpected blank line at the end)
  # Note : if something fails here, the output file is probably badly formatted.
  if(g_verbose>0) : print("[extractData] Detected "+str(T)+" timesteps.")
  #############################
  # Extract data and times. ###
  if(g_verbose>0) : print("[extractData] Beginning extraction.")
  datX=np.zeros([3, N, T]) # data will be stored as a (coordinates, particle number, timestamp) array
  datV=np.zeros([3, N, T]) # data will be stored as a (velocities, particle number, timestamps) array
  timeIndex=0
  times=list()
  passed25=False; passed50=False; passed75=False
  with open(NBodyOutputFileName, "r") as f : # open the file
    for line in f : # start to look in the file
      if(line!="" and line!=" " and line!=" \n") : # ignore dummy lines
        ar=line.split() # separate data on line
        time=float(ar[0]) # get line time
        particleNumber=int(ar[1]) # get line particle number
        datX[0, particleNumber, timeIndex]=float(ar[3]) # get line x position
        datX[1, particleNumber, timeIndex]=float(ar[4]) # get line y position
        datX[2, particleNumber, timeIndex]=float(ar[5]) # get line z position
        datV[0, particleNumber, timeIndex]=float(ar[6]) # get line x velocity
        datV[1, particleNumber, timeIndex]=float(ar[7]) # get line y velocity
        datV[2, particleNumber, timeIndex]=float(ar[8]) # get line z velocity
        if(not float(time) in times) : times.append(float(time)) # if current has not been already saved, save it    
        if(particleNumber==N-1) :
          if(timeIndex==0 and g_verbose>0) : print("[extractData] Extraction :   0%.")
          if(timeIndex>0.25*T and g_verbose>0 and not(passed25)) : print("[extractData] Extraction :  25%."); passed25=True
          if(timeIndex>0.5*T and g_verbose>0 and not(passed50)) : print("[extractData] Extraction :  50%."); passed50=True
          if(timeIndex>0.75*T and g_verbose>0 and not(passed75)) : print("[extractData] Extraction :  75%."); passed75=True
          if(timeIndex==T-1 and g_verbose>0) : print("[extractData] Extraction : 100%.")
          timeIndex=timeIndex+1 # we reached the last particle, update timestamp
  times=np.array(times) # convert list into convenient array
  if(np.size(times)==np.size(datX, axis=2)-1) : # first two times are 0 and 0 (and printing step is probably too small) and the list is too short of one, add one 0 to the list of times
    times=np.concatenate(([0], times))
  elif(np.size(times)<np.size(datX, axis=2)) : # the time list is too short of strictly more than 1, error
    criticalError("Times list too short ("+str(np.size(times))+" instead of "+str(np.size(datX, axis=2))+" or "+str(np.size(datX, axis=2)-1)+"), check possible snapshots errors.")
  if(g_verbose>0) : print("[extractData] Extraction finished.\n")
  #############################
  return(times, masses, datX, datV)

def extractRadialVelocity(X, V) :
  # Extracts the radial velocity (relative to the centroid) of a star.
  # @param X a set of positions, recentered around centroid (array 3 * N)
  # @param V a set of velocities (array 3 * N)
  # @return the radial velocity of the cluster
  zerosIDs=np.where(np.linalg.norm(X, axis=0)==0)[0] # find points that are exactly at origin
  XNorm=X/np.linalg.norm(X, axis=0) # normalise X
  XNorm[:, zerosIDs]=ensureColVect([0, 0, 0])
  VR=np.zeros([3, np.size(X, 1)])
  for i in range(0, np.size(X, 1)):
    VR[:, i]=np.vdot(XNorm[:, i], V[:, i])*XNorm[:, i]
  return(VR)

def extractVelocitiesSigma(V, r=-1, nBins=-1, retVals="globalOnly") :
  # Extract the standard deviation of a set velocities modules.
  # @param V a set of velocities (array 3 * N)
  # @param r a set of radii (needs an array of size N if retVals is set to "all")
  # @param r nBins a number of bins for the evalation per bin (needs a positive integer if retVals is set to "all")
  # @param retVals either "all" if the user wants the global standard deviation as well as radial bins standard deviation or any other value if only the global variance is wanted
  # @return global standard deviation as well as radial bins standard deviation if retVals is set to "all", global variance only if retVals is set to any other value
  norms=np.linalg.norm(V, axis=0) # get the norms
  N=np.size(np.linalg.norm(V, axis=0), axis=0)
  if retVals=="all" :
    if(nBins<0 or len(r)==1) :
      criticalError("Variables r and nBins must be chosen if retVal has this value.")
    sortedIndexes=np.argsort(r) # get the sorted indexes of radii
    firstBinsWidth=int(N/nBins)
    lastBinWidth=N-(nBins-1)*firstBinsWidth
    variances=np.zeros(nBins)
    correspondingRadii=np.zeros(nBins)
    currentBinStartIndex=0
    for i in range(0, nBins-1) : # compute standard deviation per bin
      sortedRange=sortedIndexes[range(currentBinStartIndex, currentBinStartIndex+firstBinsWidth)]
      variances[i]=np.var(norms[sortedRange], ddof=1) # calculate the variance
      correspondingRadii[i]=0.5*(r[sortedRange[0]]+r[sortedRange[-1]]) # save the radius where the bin is computed
      currentBinStartIndex=currentBinStartIndex+firstBinsWidth
    sortedRange=sortedIndexes[range(currentBinStartIndex, currentBinStartIndex+lastBinWidth)]
    variances[nBins-1]=np.var(norms[sortedRange], ddof=1) # calculate the variance
    correspondingRadii[nBins-1]=0.5*(r[sortedRange[0]]+r[sortedRange[-1]]) # save the radius where the bin is computed
  globalVariance=np.var(norms, ddof=1) # get the global variance
  if retVals=="all" : return(variances**0.5, correspondingRadii, globalVariance**0.5)
  else : return(globalVariance**0.5)

def formatTime(t) :
  # Formats a timestamp into a nice LaTex format.
  # @param t a timestamp
  # @return the formatted timestamp
  return("$t="+'{:6.2f}'.format(t)+r"$ ([T])")

def generateVectorsRandomDirection(norms) :
  # Generates a set of 3D vectors, of random direction and of fixed norm. The size of the norm parameter array gives the number of vectors that will be generated.
  # @param norms the wanted norms
  # @return the cartesian coordinates of the vectors
  n=np.size(norms)
  res=np.random.normal(size=(3, n))
  return(res*norms/np.linalg.norm(res, axis=0))

def getAllAngMomenta(datX, momenta) :
  # Compute all particules angular momenta at a time.
  # @param datX a set of positions to use (recentered or not, array 3 * N)
  # @param momenta a set of momenta to use (array 3 * N)
  # @return all particles' angular momenta (array 3 * N)
  N=np.size(datX, axis=1)
  allAngMomenta=np.zeros((3, N))
  for i in range(0, N) : allAngMomenta[:, i]=np.cross(datX[:, i], momenta[:, i])
  return(allAngMomenta)

def getAllMomenta(datV, masses) :
  # Compute all particules momenta at a time.
  # @param datV a set of velocities to use (recentered or not, array 3 * N)
  # @param masses a set of masses to use (array N)
  # @return all particles' momenta (array 3 * N)
  N=np.size(masses)
  momenta=np.zeros(np.shape(datV))
  for i in range(0, N) : momenta[:, i]=masses[i]*datV[:, i]
  return(momenta)

def getAngMomenta(datX, datV, masses) :
  # Compute all particles' angular momenta over time.
  # @param datX a set of positions to use (recentered or not, array 3 * N * T)
  # @param datV a set of velocities to use (recentered or not, array 3 * N * T)
  # @param masses a set of masses to use (array N)
  # @return all particles' angular momenta over time (array 3 * N * T)
  N=np.size(masses)
  T=np.size(datX, axis=2)
  momenta=np.zeros(np.shape(datX))
  angMomenta=np.zeros([3, N, T])
  for i in range(0, N) : momenta[:, i, :]=masses[i]*datV[:, i, :]
  for t in range(0, T) : angMomenta[:, :, t]=getAllAngMomenta(datX[:, :, t], momenta[:, :, t])
  return(angMomenta)

def getCMapColorGradient(n, name="jet") :
  # Computes a set of color from a CMap from matplotlib.pyplot.
  # @param n the number of colors wanted
  # @param name (optional) the name of the CMap (default will be the "jet" CMap)
  cmap=plt.get_cmap(name)
  colors=cmap(np.linspace(0, 1.0, n))
  return(colors)

def getDiskCrossingTimes(dat, times) :
  # Computes all GC disk crossing times (z=0 plane crossing), based on its centroid position.
  # This function is specific to the study of GCs.
  # @param dat a set of positions (array 3 * N * T)
  # @param times a set of times (array T)
  # @return the IDs and timestamps of disk crossing times, in order of happening
  cP=centroidPosition(dat)
  return(getDiskCrossingTimesCPSaved(cP, times))

def getDiskCrossingTimesCPSaved(cP, times) :
  # Computes all GC disk crossing times (z=0 plane crossing), based on its centroid position.
  # This function is specific to the study of GCs.
  # @param cP the position of a centroid over time (array 3 * T)
  # @param times a set of times (array T)
  # @return the IDs and timestamps of disk crossing times, in order of happening
  cpz=cP[2, :]
  IDs=np.where(np.multiply(cpz[:-1], cpz[1:])<0)[0].tolist()
  cTs=list()
  for i in IDs : # interpolate crossing time
    cTs.append(times[i]+(times[i+1]-times[i+1])*(np.abs(cpz[i])/(np.abs(cpz[i])+np.abs(cpz[i+1]))))
  return((IDs, cTs))

def getIDsInsideRadius(pos, radius, center=-1) :
  # Computes the IDs of particles inside a sphere of given center and radius.
  # @param pos a set of positions (array 3 * N)
  # @param radius the radius of the wanted sphere
  # @param center (optional) the center of the wanted sphere (default will be the raw centroid of the set of positions)
  # @return the IDs of particles inside the sphere of given center and radius
  if(type(center)==int) : center=barycenter(pos)
  else : center=ensureColVect(center)
  return(np.where(np.linalg.norm(pos-center, axis=0)<radius))

def getIDsOutsideRadius(pos, radius, center=-1) :
  # Computes the IDs of particles outside a sphere of given center and radius.
  # @param pos a set of positions (array 3 * N)
  # @param radius the radius of the wanted sphere
  # @param center (optional) the center of the wanted sphere (default will be the raw centroid of the set of positions)
  # @return the IDs of particles outside the sphere of given center and radius
  if(type(center)==int) : center=barycenter(pos)
  else : center=ensureColVect(center)
  return(np.where(np.linalg.norm(pos-center, axis=0)>radius))

def getSigmaFaberJackson(mass, unit="[V]") :
  # Get a reference value of the velocities' modules' dispersion, given the mass of a cluster.
  # @param mass the mass of the cluster
  # @param unit (optional) the wanted unit of sigma (default is [V])
  # @return a reference value of the velocities' modules' dispersion deduced from the mass of the cluster
  val=g_aFaberJackson*mass**0.25 # value in km/s
  if(unit=="[V]") :
    val=val*1e3 # value in m/s
    val=val/g_scalingV # value in [V]
  else : criticalBadValue("unit", "getSigmaFaberJackson")
  return(val)

def graphInterface(fig, ax, title="", forceTitle=False) :
  graphManageTitle(ax, title, forceTitle)
  graphUpdateAxisLabels(ax)
  fig.tight_layout()

def graphManageTitle(ax, title, forceTitle=False) :
  if(g_hideTitles and not(forceTitle)) : ax.set_title("")
  else : ax.set_title(title)

def graphUpdateAxisLabels(ax, ignoreX=False, ignoreY=False) :
  if(not(ignoreX)) : ax.set_xlabel(ax.get_xlabel(), fontsize=g_graphFontSize_XLabel)
  if(not(ignoreY)) : ax.set_ylabel(ax.get_ylabel(), fontsize=g_graphFontSize_YLabel)
  ax.tick_params(labelsize=g_graphFontSize_Ticks)

def inertiaTensor(X, M, origin=[0, 0, 0], b1=[1, 0, 0], b2=[0, 1, 0], b3=[0, 0, 1]) :
  # Calculates the inertia tensor of a set of points.
  # @param X a set of positions (array 3 * N)
  # @param M a set of masses (array N)
  # @param origin (optional) origin of the wanted reference frame (default will be [0, 0, 0])
  # @param b1 1st (optional) vector of the wanted reference frame (default will be from canonical reference)
  # @param b2 2nd (optional) vector of the wanted reference frame (default will be from canonical reference)
  # @param b3 3rd (optional) vector of the wanted reference frame (default will be from canonical reference)
  # @return the inertia tensor of the set of points
  if(np.size(X, axis=1)!=np.size(M)) : criticalError("[inertiaTensor] Sizes of arrays X and M are not the same.")
  I=np.zeros([3, 3])
  P=changeReference(X-ensureColVect(origin), b1, b2, b3) # change to wanted reference frame
  I[0, 0]=np.sum(M*(P[1, :]**2+P[2, :]**2))
  I[1, 1]=np.sum(M*(P[0, :]**2+P[2, :]**2))
  I[2, 2]=np.sum(M*(P[0, :]**2+P[1, :]**2))
  I[1, 1]=np.sum(M*(P[1, :]**2+P[2, :]**2))
  I[0, 1]=np.sum(M*P[0, :]*P[1, :])
  I[1, 0]=I[0, 1]
  I[0, 2]=np.sum(M*P[0, :]*P[2, :])
  I[2, 0]=I[0, 2]
  I[1, 2]=np.sum(M*P[1, :]*P[2, :])
  I[2, 1]=I[1, 2]
  return(I)

def inertiaTensorAnalysis(Is, v, times, onlyDirection=False,
                          timeAxisTitle=g_axisLabelTime,
                          normTitle="",
                          normAxisAround=["", ""],
                          normAxisInside="",
                          directionTitle="",
                          normPlotType="plot") :
  # Plot all graphs necessary to a more or less complete and viewable analysis of an inertia tensor.
  # @param Is normally a set of diagonalised inertia tensors, but can be hacked into something else
  # @param v normally, the ratio np.abs(lambda_max/lambda_min) (where lambda_i are eigenvalues), but can be hacked into something else
  # @param times a set of times (array T)
  # @param onlyDirection (optional) encodes whether or not to plot the norm v (True if only directions of the maximum eigenvector should be plotted, False if all should be plotted, default is False)
  # @param timeAxisTitle (optional) the time axis title (default is g_axisLabelTime, from a global variable)
  # @param normTitle (optional) title of the plot showing v (default is empty, "")
  # @param normAxisAround (optional) exterior parts of y axis of the plot showing v (default is empty, ["", ""])
  # @param normAxisInside (optional) interior part of y axis of the plot showing v (default is empty, "")
  # @param directionTitle (optional) title of the plot showing the directions (default is empty, "")
  # @param normPlotType (optional) type of plot that should be done for v (default is normal plot)
  # @return handles on figures and axis
  # Prepare. ##################
  T=np.size(times)
  if(not(len(Is)==len(v)==T)) : criticalError("[inertiaTensorAnalysis] Number of saved inertia tensors is not equal to the number of times.")
  cmap=mpl.cm.jet # colorbar
  norm=mpl.colors.Normalize(vmin=times[0], vmax=times[-1]) # colorbar
  colors=getCMapColorGradient(T, cmap.name) # plots
  quiv_wid=0.003 # quivers
  quiv_head_wid=3 # quivers
  quiv_head_len=3 # quivers
  #############################
  
  # Initialise. ###############
  u=np.zeros([3, T])
  uxy_ratio=np.zeros(T)
  uxz_ratio=np.zeros(T)
  uyz_ratio=np.zeros(T)
  if(not(onlyDirection)) :
    fig=plt.figure(figsize=g_squishedSquareFigSize)
    ax=fig.add_subplot(111)
  figu=plt.figure(figsize=g_fullscreenFigSize)
  axuxy=plt.subplot2grid((9, 10), (0, 0), rowspan=3, colspan=6) # quivers
  axuxz=plt.subplot2grid((9, 10), (3, 0), rowspan=3, colspan=6) # quivers
  axuyz=plt.subplot2grid((9, 10), (6, 0), rowspan=3, colspan=6) # quivers
  rad=plt.subplot2grid((9, 10), (1, 6), rowspan=5, aspect="equal", colspan=4) # radar plot
  axcol=plt.subplot2grid((9, 10), (7, 6), colspan=4) # colorbar
  cb=mpl.colorbar.ColorbarBase(axcol, cmap=cmap, norm=norm, orientation='horizontal') # colorbar
  #############################
  
  # Prepare graphs. ###########
  rad=radarPlot3AddInSubplot(rad)
  axuxyt=axuxy.twinx(); axuxzt=axuxz.twinx(); axuyzt=axuyz.twinx() 
  axuxyt.get_yaxis().set_visible(False); axuxzt.get_yaxis().set_visible(False); axuyzt.get_yaxis().set_visible(False) # set the quiver axis invisible
  axuxy.set_zorder(axuxyt.get_zorder()+1); axuxz.set_zorder(axuxzt.get_zorder()+1); axuyz.set_zorder(axuyzt.get_zorder()+1) # make the curve draw over the quiver
  axuxy.patch.set_visible(False); axuxz.patch.set_visible(False); axuyz.patch.set_visible(False) # hide the canvas of the curve (to be able to see the quiver)
  axuxyt.patch.set_visible(True); axuxzt.patch.set_visible(True); axuyzt.patch.set_visible(True) # show the canvas of the quiver
  #############################
  
  # Start iterating times.
  for t in range(0, T) :
    # Select.
    I=Is[t]
    #############################
    
    #v[t]=max(I[0])/min(I[0])
    u[:, t]=I[1][:, np.argsort(I[0])[-1]]
    if(np.abs(u[0, t])+np.abs(u[1, t])==0) : uxy_ratio[t]=0
    else : uxy_ratio[t]=(1*np.abs(u[0, t])-1*np.abs(u[1, t]))/(np.abs(u[0, t])+np.abs(u[1, t]))
    if(np.abs(u[0, t])+np.abs(u[2, t])==0) : uxy_ratio[t]=0
    else : uxz_ratio[t]=(1*np.abs(u[0, t])-1*np.abs(u[2, t]))/(np.abs(u[0, t])+np.abs(u[2, t]))
    if(np.abs(u[1, t])+np.abs(u[2, t])==0) : uxy_ratio[t]=0
    else : uyz_ratio[t]=(1*np.abs(u[1, t])-1*np.abs(u[2, t]))/(np.abs(u[1, t])+np.abs(u[2, t]))
    
    # Graph (1st step). #######
    axuxyt.quiver(times[t], [0], u[0, t], u[1, t], color=colors[t], pivot="mid", width=quiv_wid, headwidth=quiv_head_wid, headlength=quiv_head_len)
    axuxzt.quiver(times[t], [0], u[0, t], u[2, t], color=colors[t], pivot="mid", width=quiv_wid, headwidth=quiv_head_wid, headlength=quiv_head_len)
    axuyzt.quiver(times[t], [0], u[1, t], u[2, t], color=colors[t], pivot="mid", width=quiv_wid, headwidth=quiv_head_wid, headlength=quiv_head_len)
    radarPlot3AddPoint(rad, radarPlot3GetCoordsOfValue(u[:, t]), style=".", color=colors[t])
    ###########################

  # Graph (2nd step). #########
  if(not(onlyDirection)) :
    if(normPlotType=="semilogy") :
      ax.semilogy(times, v)
    else :
      ax.plot(times, v)
  axuxy.plot(times, uxy_ratio, 'k'); axuxy.set_ylim([-1, 1])
  axuxz.plot(times, uxz_ratio, 'k'); axuxz.set_ylim([-1, 1])
  axuyz.plot(times, uyz_ratio, 'k'); axuyz.set_ylim([-1, 1])
  #############################
  
  # Set titles. ###############
  graphManageTitle(axuxy, directionTitle+" - XY Plane ")
  graphManageTitle(axuxy, directionTitle+" - XZ Plane ")
  graphManageTitle(axuxy, directionTitle+" - YZ Plane ")
  #############################
  
  # Set axis labels. ##########
  if(not(onlyDirection)) : ax.set_xlabel(timeAxisTitle)
  axuxy.set_xlabel(timeAxisTitle); axuxz.set_xlabel(timeAxisTitle); axuyz.set_xlabel(timeAxisTitle); cb.set_label(timeAxisTitle)
  if(not(onlyDirection)) : ax.set_ylabel(normAxisAround[0]+normAxisInside+normAxisAround[1])
  axuxy.set_ylabel(r"$\leftarrow$ mainly $y$ | mainly $x$ $\rightarrow$")
  axuxz.set_ylabel(r"$\leftarrow$ mainly $z$ | mainly $x$ $\rightarrow$")
  axuyz.set_ylabel(r"$\leftarrow$ mainly $z$ | mainly $y$ $\rightarrow$")
  #############################
  
  # Set axis limits. ##########
  if(not(onlyDirection)) : ax.set_xlim([times[0], times[-1]])
  xAxisSpan=[times[0]-0.1*(times[-1]-times[0]), times[-1]+0.1*(times[-1]-times[0])]
  axuxy.set_xlim(xAxisSpan); axuxz.set_xlim(xAxisSpan); axuyz.set_xlim(xAxisSpan)
  #############################

  # Finalise. #################
  if(not(onlyDirection)) : graphInterface(fig, ax, normTitle)
  axuxy.get_yaxis().set_ticks([]); axuxz.get_yaxis().set_ticks([]); axuyz.get_yaxis().set_ticks([]) # remove ticks
  graphUpdateAxisLabels(axuxy, ignoreY=True)
  graphUpdateAxisLabels(axuxz, ignoreY=True)
  graphUpdateAxisLabels(axuyz, ignoreY=True)
  graphUpdateAxisLabels(axuxyt)
  graphUpdateAxisLabels(axuxzt)
  graphUpdateAxisLabels(axuyzt)
  if(not(onlyDirection)) : return((fig, ax), (figu, (axuxy, axuxyt), (axuxz, axuxzt), (axuyz, axuyzt), rad, cb))
  else : return(figu, (axuxy, axuxyt), (axuxz, axuxzt), (axuyz, axuyzt), rad, cb)
  #############################

def isNumber(s) :
  # Checks if a string can be understood as a number.
  # @param s a string
  # @return True if the string can be understood as a number, False if not
  try :
    float(s)
    return True
  except ValueError :
    return False

def loadGPEFromFile(filepath, T) :
  # Load the graviational potential energy from a file where it has previously been stored.
  # @param filepath the path to the file containing the GPE
  # @param T the number of timestamps to retrieve
  # @return the GPE over the timestamps
  GPEL=np.zeros(T)
  i=0
  with open(filepath, "r") as f :
    for line in f :
      GPEL[i]=float(line)
      i=i+1
  return(GPEL)

def mergeOutputFiles(fF, sF, oF) :
  # Merges two MiniNBody output files, in order, into one.
  # @param fF the path to the first file.
  # @param sF the path to the second file.
  # @param oF the path to the output file.
  # @return none
  if(path.isfile(fF) and path.isfile(sF)) :
    (t1, m1, d1X, d1V)=extractData(fF)
    (t2, m2, d2X, d2V)=extractData(sF)
    if(np.max(np.abs(m1-m2))==0) :
      if(np.max(np.abs(d1X[:, :, -1]-d2X[:, :, 0]))==0) :
        if(np.max(np.abs(d1V[:, :, -1]-d2V[:, :, 0]))==0) :
          t=np.hstack((t1, t2[1:]+t1[-1]))
          m=m1
          X=np.concatenate((d1X, d2X[:, :, 1:]), axis=2)
          V=np.concatenate((d1V, d2V[:, :, 1:]), axis=2)
          T=np.size(t)
          N=np.size(m)
          f=open(oF, "w")
          for i in range(0, T) :
            for j in range(0, N) :
              f.write("{:22.16e}".format(t[i])+" "+
                      "{:5d}".format(j)+" "+
                      "{:12.6e}".format(m[j])+" "+
                      "{:16.9e}".format(X[0, j, i])+" "+
                      "{:16.9e}".format(X[1, j, i])+" "+
                      "{:16.9e}".format(X[2, j, i])+" "+
                      "{:16.9e}".format(V[0, j, i])+" "+
                      "{:16.9e}".format(V[1, j, i])+" "+
                      "{:16.9e}".format(V[2, j, i])+"\n")
          f.close()
        else : criticalError("Last velocities of first file do not correspond to first velocities of second file.")
      else : criticalError("Last positions of first file do not correspond to first positions of second file.")
    else : criticalError("Masses do not correspond.")
  else : criticalError("One or both of the files do not exist.")

def normalise(A) :
  # Normalises a vector.
  # @param A a vector
  # @return the vector, normalised (with norm 2 equalling 1)
  return(A/np.linalg.norm(A, axis=0))

def plotPointsOutsideRadius(rdat, radius) :
  # Plots over the 3 canonic planes the selected/rejected particles from a selection of given radius.
  # @param rdat a set of positions recentered around their centroid (array 3 * N * T)
  # @param radius the limit radius
  # @return the IDs of particles outside the sphere of given radius
  mks=2
  mkc="orange"
  mkec="red"
  circLineColor="k"
  circLineStyle="dotted"
  IDs=getIDsOutsideRadius(rdat, radius, [0, 0, 0])
  cir1=plt.Circle((0, 0), radius, color=circLineColor, linestyle=circLineStyle, fill=False); cir2=plt.Circle((0, 0), radius, color=circLineColor, linestyle=circLineStyle, fill=False); cir3=plt.Circle((0, 0), radius, color=circLineColor, linestyle=circLineStyle, fill=False); cir4=plt.Circle((0, 0), radius, color=circLineColor, linestyle=circLineStyle, fill=False); cir5=plt.Circle((0, 0), radius, color=circLineColor, linestyle=circLineStyle, fill=False); cir6=plt.Circle((0, 0), radius, color=circLineColor, linestyle=circLineStyle, fill=False)
  fig=plt.figure(figsize=(18, 18))
  ax1=fig.add_subplot(231); ax2=fig.add_subplot(232); ax3=fig.add_subplot(233)
  ax4=fig.add_subplot(234); ax5=fig.add_subplot(235); ax6=fig.add_subplot(236)
  ax1.plot(rdat[0, :], rdat[1, :], "o", c=mkc, markeredgecolor=mkec, markersize=mks); ax2.plot(rdat[0, :], rdat[2, :], "o", c=mkc, markeredgecolor=mkec, markersize=mks); ax3.plot(rdat[1, :], rdat[2, :], "o", c=mkc, markeredgecolor=mkec, markersize=mks)
  ax1.plot(rdat[0, IDs], rdat[1, IDs], "kx", markersize=2*mks); ax2.plot(rdat[0, IDs], rdat[2, IDs], "kx", markersize=2*mks); ax3.plot(rdat[1, IDs], rdat[2, IDs], "kx", markersize=2*mks)
  ax4.plot(rdat[0, :], rdat[1, :], "o", c=mkc, markeredgecolor=mkec, markersize=mks); ax5.plot(rdat[0, :], rdat[2, :], "o", c=mkc, markeredgecolor=mkec, markersize=mks); ax6.plot(rdat[1, :], rdat[2, :], "o", c=mkc, markeredgecolor=mkec, markersize=mks)
  ax4.plot(rdat[0, IDs], rdat[1, IDs], "kx", markersize=2*mks); ax5.plot(rdat[0, IDs], rdat[2, IDs], "kx", markersize=2*mks); ax6.plot(rdat[1, IDs], rdat[2, IDs], "kx", markersize=2*mks)
  ax1.add_artist(cir1); ax2.add_artist(cir2); ax3.add_artist(cir3); ax4.add_artist(cir4); ax5.add_artist(cir5); ax6.add_artist(cir6)
  ax1.set_title("XY Plane"); ax2.set_title("XZ Plane"); ax3.set_title("YZ Plane"); ax4.set_title("XY Plane"); ax5.set_title("XZ Plane"); ax6.set_title("YZ Plane")
  plt.tight_layout()
  ax1.axis('equal'); ax2.axis('equal'); ax3.axis('equal'); ax4.axis('equal'); ax5.axis('equal'); ax6.axis('equal')
  ax4.set_xlim([-1.1*radius, 1.1*radius]); ax5.set_xlim([-1.1*radius, 1.1*radius]); ax6.set_xlim([-1.1*radius, 1.1*radius])
  ax4.set_ylim([-1.1*radius, 1.1*radius]); ax5.set_ylim([-1.1*radius, 1.1*radius]); ax6.set_ylim([-1.1*radius, 1.1*radius])
  return(IDs)

def plotSolarSystem(dat, projPlane) :
  # Plots the solar system with pretty colors.
  # @param dat a set of positions of the solar system (array 3 * 10 * T)
  # @param projPlane a string encoding on which plane to project the plot
  # @return none
  if projPlane=="XY" :
    fC=0; sC=1
  elif projPlane=="XZ" :
    fC=0; sC=2
  elif projPlane=="YZ" :
    fC=1; sC=2
  for i in range(0, 10) :
    plt.plot(dat[fC, i], dat[sC, i], 'o', color=solSysColors[0][i], markeredgecolor=solSysColors[1][i])

def plotSolarSystem3D(dat, ax) :
  # Plots the solar system with pretty colors.
  # @param dat a set of positions of the solar system (array 3 * 10 * T)
  # @param ax 3D-projection plot figure axes
  # @return none
  for i in range(0, 10) :
    ax.scatter(dat[0, i], dat[1, i], dat[2, i], c=solSysColors[0][i], edgecolor=solSysColors[1][i])

def radialHistogram(radii, nbSamples, rangeInf, rangeSup, binsShape="auto") :
  # Plays the role of the function histogram in NumPy, except with bins of fixed size and normalised by the volume of the sphere shell contained by each radial bin.
  # @param radii a set of radial data (array N)
  # @param nbSamples a number of samples
  # @param low boundary for histogram
  # @param high boundary for histogram
  # @return the histogram (same shape as NumPy classical histogram) but which values are normalised
  tmpHist=np.zeros(nbSamples)
  if(binsShape=="uniform") : completeHist=np.histogram(radii, bins=np.linspace(rangeInf, rangeSup, nbSamples+1), range=(rangeInf, rangeSup))
  elif(binsShape=="log") : completeHist=np.histogram(radii, bins=np.hstack(([0], np.logspace(np.log10(0.1), np.log10(rangeSup), nbSamples))), range=(rangeInf, rangeSup))
  else : completeHist=np.histogram(radii, bins=nbSamples, range=(rangeInf, rangeSup))
  hist=completeHist[0]
  for j in range(0, nbSamples) : tmpHist[j]=hist[j]/volumeShellOfSphere(completeHist[1][j], completeHist[1][j+1]) # renormalise by volume of slice of sphere between said radii
  return(tmpHist, completeHist[1])

def recenterDataAroundCentroid(dat, masses=-1) :
  # Recenters a set of positions around its centroid over time.
  # This function is specific to the study of GCs as is because of the dependance on the function recenterDataAroundCentroidBetter, but decommenting the first line enables the use of it in any type of study.
  # @param dat a set of positions (array 3 * N * T)
  # @param masses (optional) a set of masses (needs and array of size N, but if not specified masses will be supposed of same value)
  # @return the set of positions recentered around its centroid over time (array 3 * N * T)
  if(type(masses)==np.ndarray and np.size(dat, axis=1)!=np.size(masses)) : criticalError("Sizes of arrays dat and masses are not the same.")
  #return(recenterDataAroundCentroidOld(dat, masses)) # decomment this for a use in a more general study
  if(g_useSmartSelection) : return(recenterDataAroundCentroidBetter(dat, masses))
  else : return(recenterDataAroundCentroidOld(dat, masses))

def recenterDataAroundCentroidBetter(dat, masses) :
  # Recenters a set of positions around its centroid over time with a better way.
  # @param dat a set of positions (array 3 * N * T)
  # @param masses a set of masses (needs an array of size N or -1 to suppose masses of same value)
  # @return the set of positions recentered around its centroid over time (array 3 * N * T)
  new=np.zeros(np.shape(dat))
  for i in range(0, np.size(dat, axis=2)) :
    new[:, :, i]=dataAutoSelection1(dat[:, :, i], masses, findWhat="recenteredData")
  return(new)

def recenterDataAroundCentroidOld(dat, masses) :
  # Recenters a set of positions around its centroid over time in a classical way.
  # @param dat a set of positions (array 3 * N * T)
  # @param masses a set of masses (needs an array of size N or -1 to suppose masses of same value)
  # @return the set of positions recentered around its centroid over time (array 3 * N * T)
  new=np.zeros(np.shape(dat))
  for i in range(0, np.size(dat, axis=2)) :
    new[:, :, i]=dat[:, :, i]-barycenter(dat[:, :, i], masses)
  return(new)

def removeGlobalSpeed(datV) :
  # Remove the global component of a set of speeds (so that the system remains motionless).
  # @param datV a set of velocities (array 3 * N * T)
  # @return the set of velocities without its global component
  return(recenterDataAroundCentroidOld(datV, -1))

def selectLimitationRadius(rdat, showPoints=False) :
  # Provides an interface to select a radius to reject some stars that are too far from the system's centroid.
  # @param datFull a set of positions (array 3 * N * T)
  # @param rdat a set of positions recentered around their centroid (array 3 * N)
  # @param showPointsOutside (optional) encodes whether or not to plot selected/rejected particles (True if yes, False if no, default if False)
  # @return the indexes of stars that are kept
  N=np.size(rdat, axis=1)
  sure=False
  rad=np.linalg.norm(rdat, axis=0)
  while(not(sure)) :
    limitRadius=userInputGetNumber("Limitation radius or -1 for maximum range ([L], >="+"{:5.2f}".format(1)+", <="+"{:5.2f}".format(np.max(rad)-1)+") ?",
                                   1,
                                   np.max(rad)-1,
                                   ["-1"])
    if(limitRadius=="-1") :
      limitRadius=np.max(rad)+1
    if(showPoints) :
      selIDs=np.setdiff1d(np.array(range(0, N)), plotPointsOutsideRadius(rdat, limitRadius))
    else :
      selIDs=np.setdiff1d(np.array(range(0, N)), getIDsOutsideRadius(rdat, limitRadius, [0, 0, 0]))
    sureInput=0
    if(np.size(selIDs)==0) :
      print("0 stars remain from this selection. Please enter a greater limitation radius.\n")
    else :
      while(not(sureInput in ["0", "1"])) :
        sureInput=userInput("{:6.2f}".format(100*(1-np.size(selIDs)/N))+" % of the stars have been removed. Are you sure you want to continue (0 to enter a new limitation radius, 1 to continue) ?")
        if(sureInput=="1") : sure=True
        if(not(sureInput in ["0", "1"])) : print("Please enter \"0\" or \"1\".\n")
  return(selIDs)

#def semiNormalDistributionRVariate(s, n) :
#  # Generate n numbers following the semi-normal distribution.
#  # @param s the standard deviation of the wanted semi-normal distribution
#  # @return n numbers following the semi-normal distribution
#  return(abs(np.random.normal(0, s, n)))

def testFilePath(filepath) :
  if(path.isfile(filepath)) : userInputOverwrite(filepath)

#def update_xlabels(ax):
#    xlabels=[format(label, '1.0e') for label in ax.get_xticks()]
#    ax.set_xticklabels(xlabels)

def userInput(question) :
  # Ask the user a question and waits for its input.
  # @param question a question
  # @return the user's input
  print("> UserInput Interface. --------<")
  print("| "+question, end="")
  res=input("| > ")
  print(">------------------------------<\n")
  return(res)

def userInput0Or1(question, if0, if1) :
  res=-1
  while(not(res in ["q", "0", "1"])) :
    res=userInput(question)
    if res=="0" :
      print(" "+if0+"\n")
      return(0)
    elif res=="1" :
      print(" "+if1+"\n")
      return(1)
    elif res=="q" :
      criticalError("Execution aborted.")
    else :
      print("Please enter 0 or 1.\n")

def userInputAskRecenter() :
  # Ask the user if he wants to recenter the data.
  # @param none
  # @return the user's as a True/False answer (if answer can't be parsed, an error is raised)
  rcntr=-1
  while(not(rcntr in ["0", "1"])) :
    rcntr=userInput("Recenter around GC centroid (0 for no, 1 for yes) ?")
    if rcntr=="0" :
      recenter=False
      print(" Data will not be recentered around centroid.\n")
    elif rcntr=="1" :
      recenter=True
      print(" Data will be recentered around centroid.\n")
    else :
      print("Please enter 0 or 1.")
  return(recenter)

def userInputGetWantedProjection(restriction="none") :
  # Ask the user if which projection plane he wants.
  # @param none
  # @return the user's as a "XY", "XZ", "YZ" or "XYZ" (if answer can't be parsed, an error is raised)
  if(restriction=="none") :
    proj=userInput("Graph axis (XY, XZ, YZ or XYZ) ?").upper()
    if proj!="XY" and proj!="XZ" and proj!="YZ" and proj!="XYZ" : # wanted plane not correctly asked
      criticalError("Please select a correct set of axis onto which draw.")
  elif(restriction=="2D") :
    proj=userInput("Graph axis (XY, XZ, YZ) ?").upper()
    if proj!="XY" and proj!="XZ" and proj!="YZ" : # wanted plane not correctly asked
      criticalError("Please select a correct set of axis onto which draw.")
  return(proj)

def userInputGetNumber(usrQ, minAccepVal, maxAccepVal, specials=[]) :
  # Tests if a user input is a number between some values.
  # @param usrIn a user's raw input (generally done with the function userInput)
  # @param minAccepVal low acceptance bound
  # @param minAccepVal high acceptance bound
  # @return the user's input converted to int if inside specified boundaries, nothing and raises an error if unparseable or outsied boundaries
  usrIn=np.nan
  while( not( (usrIn in specials) or (isNumber(usrIn) and float(usrIn)>=minAccepVal and float(usrIn)<=maxAccepVal) ) ) :
    usrIn=userInput(usrQ)
    if usrIn=="q" : criticalError("Execution aborted.")
    else :
      if usrIn in specials : return(usrIn)
      else :
        if isNumber(usrIn) :
          if float(usrIn)>=minAccepVal and float(usrIn)<=maxAccepVal : return(float(usrIn))
          else : print("Entered value is outside specified boundaries.\n")
        else : print("Entered value is not a number.")

def userInputOverwrite(filePath) :
  # Asks the user if he wants to overwrite a file.
  # @param filePath the path to the file to overwrite
  # @return True if the answer is positive, False if not
  answ=-1
  while(answ!="0" and answ!="1") :
    answ=userInput("File \""+filePath+"\" already exists. Overwrite (1) or abort (0) ?")
    if(answ=="0") :
      criticalError("Execution aborted due to existing file (\""+filePath+"\").")
      return(False)
    elif(answ=="1") : return(True)
    else : print("Please enter 0 or 1.\n")

def userInputTestIfDigit(usrIn, minAccepVal, maxAccepVal) :
  # Tests if a user input is an integer between some values.
  # @param usrIn a user's raw input (generally done with the function userInput)
  # @param minAccepVal low acceptance bound
  # @param minAccepVal high acceptance bound
  # @return the user's input converted to int if inside specified boundaries, nothing and raises an error if unparseable or outsied boundaries
  if usrIn.isdigit() :
    if int(usrIn)>=minAccepVal and int(usrIn)<=maxAccepVal :
      return(int(usrIn))
    else :
      criticalError("Entered value is outside specified boundaries.")
  else :
    criticalError("Entered value is not a digit.")

def userInputTestIfNumber(usrIn, minAccepVal, maxAccepVal) :
  # Tests if a user input is a number between some values.
  # @param usrIn a user's raw input (generally done with the function userInput)
  # @param minAccepVal low acceptance bound
  # @param minAccepVal high acceptance bound
  # @return the user's input converted to int if inside specified boundaries, nothing and raises an error if unparseable or outsied boundaries
  if isNumber(usrIn) :
    if float(usrIn)>=minAccepVal and float(usrIn)<=maxAccepVal : return(float(usrIn))
    else : criticalError("Entered value is outside specified boundaries.")
  else : criticalError("Entered value is not a number.")

def vectorEvolution(vect, times, onlyDir=True, v=[-1],
                    timeAxisTitle=g_axisLabelTime,
                    normTitle="",
                    normAxisAround=["", ""],
                    normAxisInside="",
                    directionTitle="") :
  # Plots the evolution in time of a three-dimensional vector.
  # @param vect the vector over time (array 3 * T)
  # @param times the time steps
  # @param onlyDir (optional) encodes whether or not to plot the norm  of the vector over time (True if only directions of the vector should be plotted, False if all should be plotted, default is False)
  # @param timeAxisTitle (optional) the time axis title (default is g_axisLabelTime, from a global variable)
  # @param normTitle (optional) title of the plot showing v (default is empty, "")
  # @param normAxisAround (optional) exterior parts of y axis of the plot showing v (default is empty, ["", ""])
  # @param normAxisInside (optional) interior part of y axis of the plot showing v (default is empty, "")
  # @param directionTitle (optional) title of the plot showing the directions (default is empty, "")
  # @return handles on figures and axis
  N=np.size(vect, axis=1)
  Is=list()
  dummyEigVals=np.array([-np.inf, -np.inf, 1])
  for i in range(0, N) :
    eigVects=np.array(np.vstack(([0, 0, 0], [0, 0, 0], vect[:, i]))).T
    Is.append((dummyEigVals, eigVects))
  if(onlyDir) : v=np.zeros(N)
  else : v=v
  r=inertiaTensorAnalysis(Is, v, times, onlyDirection=onlyDir,
                          timeAxisTitle=timeAxisTitle,
                          normTitle=normTitle,
                          normAxisAround=normAxisAround,
                          normAxisInside=normAxisInside,
                          directionTitle=directionTitle)
  return(r)

def volumeShellOfSphere(R1, R2) :
  # Computes the volume of the shell between a sphere of a given radius R1 and a sphere of another given radius.
  # @param R1 the small sphere's radius
  # @param R1 the big sphere's radius
  # @return the volume of the shell between the sphere of radius R1 and the sphere of radius R2
  if(R1>R2) :
    criticalError("Spheres have incompatible radii.")
    return(-1)
  else :
    return((4/3)*np.pi*(R2**3-R1**3))

def warningError(msg) :
  # Prints a message about a not critical error.
  # @param msg an error message
  # @return none
  print("> Warning : "+msg+"\n")

def writeGPEToFile(GPE, T, filepath) :
  # Store the graviational potential energy to a file.
  # GPE the GPE over the timestamps
  # @param T the number of timestamps to store
  # @param filepath the path to the file
  # @return none
  print(filepath)
  with open(filepath, "w") as f :
    for i in range(0, T) : f.write("{:1.16e}".format(GPE[i])+"\n")

def exportInitialConditionsToFile(inputFilePath,
                                  X, V, M,
                                  simulationDuration,
                                  integrationTimeStep,
                                  outputTimeInterval,
                                  outputFilePath,
                                  technique,
                                  softening,
                                  galacticEnvironnement,
                                  explanations,
                                  galaxyParams,
                                  scalings) :
  # Create the input file for MiniNBody.
  # @param inputFilePath the input file path, relative to the script
  # @param X a set of positions (array 3 * N)
  # @param V a set of velocities (array 3 * N)
  # @param M a set of masses (array N)
  # @param simulationDuration the duration of the simulation
  # @param outputTimeInterval the interval between each printing
  # @param outputFilePath the output file path, relative to executable
  # @param technique the integration technique to use
  # @param softening the softening to use
  # @param galacticEnvironnement encodes if a galactic environnement should be used (0 if no, 1 if yes)
  # @param explanations encodes if explanations should be printed (True if yes, False if no)
  # @param galaxyParam a set of galaxy parameters (unused if galacticEnvironnement is set to 0)
  # @param scalings a set of scalings
  # @return none
  testFilePath(inputFilePath)
  N=np.size(X, axis=1)
  if(np.size(V, axis=1)!=N or np.size(M)!=N) :
    criticalError("Sizes of arrays X, V and M are not the same.")
  else :
    f=open(inputFilePath, "w")
  
    f.write("# Constants and units used here :\n")
    f.write("# - G  ="+'{:19.12e}'.format(scalings[0])+" [L^{3}][M^{-1}][T^{-2}],\n")
    f.write("# - [L]="+'{:19.12e}'.format(scalings[1])+" pc,\n")
    f.write("# - [M]="+'{:19.12e}'.format(scalings[2])+" solar mass(es),\n")
    f.write("# - [T]="+'{:19.12e}'.format(scalings[3])+" Myr,\n")
    f.write("# - [V]="+'{:19.12e}'.format(scalings[4])+" m.s^{-1},\n")
    
    f.write("\n")
  
    if(explanations) :
      f.write("# Describes the gravitationnal constant to use in the simulation. This fully characterise the units used in the simulation.\n")
      f.write("# 6.67408e-11 is to use when using SI units ([L] in meters, [M] in kilograms and [T] in seconds).\n")
      f.write("# 4.302e-3  is to use when using astrophysical units ([L] in parsecs, [M] in solar masses and [T] in megayears).\n")
      f.write("# Example :\n")
      f.write("# gravitational_constant 4.302e-3\n")
    f.write("gravitational_constant "+'{:19.12e}'.format(scalings[0])+"\n")
    
    f.write("\n")
  
    if(explanations) :
      f.write("# Describes the softening to use in the simulation (not to consider particles too close to each other, in [L]).\n")
      f.write("# One can use 0 to disable softening use.\n")
      f.write("# Example :\n")
      f.write("# softening 1e-6\n")
    f.write("softening "+'{:19.12e}'.format(softening)+"\n")
    
    f.write("\n")
  
    if(explanations) :
      f.write("# Describes the number of bodies in the simulation.\n")
      f.write("# Example :\n")
      f.write("# nbody 3\n")
    f.write("nbody "+str(N)+"\n")
    
    f.write("\n")
    
    if(explanations) :
      f.write("# Describes the duration of the simulation (in [T]).\n")
      f.write("# Example :\n")
      f.write("# duration 20\n")
    f.write("duration "+'{:19.12e}'.format(simulationDuration)+"\n")
    
    f.write("\n")
    
    if(explanations) :
      f.write("# Describes the integration time step (in [T]).\n")
      f.write("# Example :\n")
      f.write("# timestep  1.0e-3\n")
    f.write("timestep "+'{:19.12e}'.format(integrationTimeStep)+"\n")
    
    f.write("\n")
    
    if(explanations) :
      f.write("# Decribes the interval between printing positions of each body in the output file (in [T]).\n")
      f.write("# Example :\n")
      f.write("# print_interval 1\n")
    f.write("print_interval "+'{:19.12e}'.format(outputTimeInterval)+"\n")
    
    f.write("\n")
    
    if(explanations) :
      f.write("# Should the simulator determine the center-of-mass velocity of the system from initial conditions and modify all velocities so that the overall system center-of-mass velocity is zero ?\n")
      f.write("# Should be either 'yes' or 'no'.\n")
      f.write("# Example :\n")
      f.write("# recenter yes\n")
    f.write("recenter no\n")
    
    f.write("\n")
    
    if(explanations) :
      f.write("# Should the simulator use a galactic model ? The galaxy is supposed to be centered on the origin of the reference (O=[0.0, 0.0, 0.0]) and its model is given by a spherical Plummer bulge, a Miyamoto-Nagai disk and a dark matter halo.\n")
      f.write("# Should be either 'yes' or 'no'.\n")
      f.write("# Should be followed by galactic parameters.\n")
      f.write("# Example :\n")
      f.write("# galactic yes\n")
    if(galacticEnvironnement==1) :
      f.write("galactic yes\n")
    elif(galacticEnvironnement==0) :
      f.write("galactic no\n")
    else :
      criticalBadValue("galacticEnvironnement")
    
    f.write("\n")
  
    if (galacticEnvironnement==1) :
      if(explanations) :
        f.write("# Galactic parameters (only used if \"galactic\" is set to \"yes\" just before). Note that units should be respected here.\n")
        f.write("# Format :\n")
        f.write("# galParam mb ab md ad hd vh ah\n")
        f.write("# Where :\n"+
                "# - mb is the bulge total mass (in [M]),\n"+
                "# - ab is a bulge parameter (in [L]),\n"+
                "# - md is the disk total mass (in [M]),\n"+
                "# - ad is a disk parameter (in [L]),\n"+
                "# - hd is a disk shape parameter (in [L]),\n"+
                "# - vh is a halo parameter (in [L][T^{-1}]),\n"+
                "# - ah is a halo parameter (in [L]).\n")
        f.write("# Example (for a classical model, using astrophysical units) :\n")
        f.write("# galParam 9.9e9 0.25e3 72e9 3.5e3 0.4e3 2.2e5 20e3\n")
      f.write("galParam "+'{:19.12e}'.format(galaxyParams[0][0])+" "+
                          '{:19.12e}'.format(galaxyParams[0][1])+" "+
                          '{:19.12e}'.format(galaxyParams[1][0])+" "+
                          '{:19.12e}'.format(galaxyParams[1][1])+" "+
                          '{:19.12e}'.format(galaxyParams[1][2])+" "+
                          '{:19.12e}'.format(galaxyParams[2][0])+" "+
                          '{:19.12e}'.format(galaxyParams[2][1])+"\n")
  
      f.write("\n")
    
    if(explanations) :
      f.write("# Describes the integration technique to use. The first three should not be used, as they have not been fully tested, but they are not very efficient anyways. Please prefer the three following ones.\n")
      f.write("# - \"E1\"  is Euler's method (first order) for both velocity and acceleration,\n")      
      f.write("# - \"E1a\" is Euler's method (first order) but where the new velocity is used to advance old position,\n")
      f.write("# - \"E2\"  is Euler's method (second order) also known as Heun's method.\n")
      f.write("# - \"RK4\" is fourth-order Runge-Kutta's method (order 4, better stability, longer execution (4 evaluations)).\n")
      f.write("# - \"PECE\" is a PECE schema with Adams-Bashforth(3+1) prediction, Adams-Moulton(3+1) correction and RK4 initialisations (order 5, slightly worse stability, moderate execution time (2 evaluations)).\n")
      f.write("# - \"PEC\" is a PEC schema (PECE without last evaluation) with Adams-Bashforth(3+1) prediction, Adams-Moulton(3+1) correction and RK4 initialisations (order 5, slightly worse stability, faster execution (1 evaluation)).\n")
      f.write("# Example :\n")
      f.write("# technique RK4\n")
    f.write("technique "+technique+"\n")
    
    f.write("\n")
    
    if(explanations) :
      f.write("# Describes the name of file into which to write output.\n")
      f.write("# One can use \"-\" to write to stdout.\n")
      f.write("# Example :\n")
      f.write("# outfile -\n")
    f.write("outfile "+outputFilePath+"\n")
    
    f.write("\n")
    
    if(explanations) :
      f.write("# Description of bodies : one line per body. Note that units should be respected here.\n")
      f.write("# Format :\n")
      f.write("# body index name mass px py pz vx vy vz \n")
      f.write("# Where :\n"+
              "# - body     is the string \"body\",\n"+
              "# - index    is an index, starting at 0, and unique,\n"+
              "# - name     is the name of the object, unique,\n"+
              "# - mass     is mass (in [M]),\n"+
              "# - px py pz are starting positions (in [L]),\n"+
              "# - vx vy vz are starting velocities (in [V]).\n")
      f.write("# Example :\n")
      f.write("# body 0 sun 1.9885e+30 0.0 0.0 0.0 0.0 0.0 0.0\n")
    for i in range(0, N) :
      f.write("body"+" "+str(i)+" "+"body"+str(i)+" "+
              '{:19.12e}'.format(M[i])+" "+
              '{:19.12e}'.format(X[0, i])+" "+'{:19.12e}'.format(X[1, i])+" "+'{:19.12e}'.format(X[2, i])+" "+
              '{:19.12e}'.format(V[0, i])+" "+'{:19.12e}'.format(V[1, i])+" "+'{:19.12e}'.format(V[2, i])+"\n")
    f.close()
###############################################################

# Three-edged radar plot functions. ###########################
def radarPlot3(titles=[r"$x$", r"$y$", r"$z$"], titleShifts=[[-0.02, 0.03], [-0.07, -0.02], [0.03, -0.02]]) :
  # Create a three-edged radar plot in a new figure.
  # @param titles the titles of the axis
  # @param titleshifts the shifts from base positions for the titles (to adjust according the texts)
  # @return handles on the figure and axis
  fig=plt.figure(figsize=g_squareFigSize)
  (fig, ax)=radarPlot3AddAsSubplot(fig, 111, titles, titleShifts)
  plt.axis("equal")
  return(fig, ax)

def radarPlot3AddAsSubplot(fig, subplotCode, titles=[r"$x$", r"$y$", r"$z$"], titleShifts=[[-0.02, 0.03], [-0.07, -0.02], [0.03, -0.02]]) :
  # Add a three-edged radar plot in an existing figure as a subplot which code is given.
  # @param fig a handle on the fig
  # @param subplotCode the code (three digits) for the subplot
  # @param titles the titles of the axis
  # @param titleshifts the shifts from base positions for the titles (to adjust according the texts)
  # @return handles on the figure and axis
  ax=fig.add_subplot(subplotCode, aspect="equal")
  ax=radarPlot3AddInSubplot(ax, titles, titleShifts)
  return(fig, ax)

def radarPlot3AddCurve(ax, coords, style="-", color="r") :
  # Add a curve in an existing three-edged radar plot.
  # @param ax a handle on the axis of an existing three-edged radar plot
  # @param coords an array of coordinates in the radar plot to plot (array 2 * N)
  # @param style (optional) the wanted linestyle for the curve (default is "-", solid line)
  # @param color (optional) the wanted color for the curve (default is "r", red)
  # @return none
  ax.plot(coords[0, :], coords[1, :], style, color=color)

def radarPlot3AddInSubplot(ax, titles=[r"$x$", r"$y$", r"$z$"], titleShifts=[[-0.02, 0.03], [-0.07, -0.02], [0.03, -0.02]]) :
  # Create a three-edged radar plot in an existing subplot which handle on axis is given.
  # @param ax a handle on the axis of the subplot
  # @param titles the titles of the axis
  # @param titleshifts the shifts from base positions for the titles (to adjust according the texts)
  # @return handle on the axis
  border="k"
  filling="white"
  top=[0, 1]
  botleft=[np.cos(7*np.pi/6), np.sin(7*np.pi/6)]
  botright=[np.cos(-np.pi/6), np.sin(-np.pi/6)]
  ax.set_xlim([botleft[0], botright[0]])
  ax.set_ylim([botleft[1]-0.1, top[1]])
  center=[0, 0]
  trianglePoints=[top, botleft, botright]
  triangle=plt.Polygon(trianglePoints, facecolor=filling, edgecolor=border)
  ax.add_patch(triangle)
  ax.axis('off')
  ax.patch.set_visible(False)
  ax.plot([center[0], top[0]], [center[1], top[1]], "k:")
  ax.plot([center[0], botleft[0]], [center[1], botleft[1]], "k:")
  ax.plot([center[0], botright[0]], [center[1], botright[1]], "k:")
  ax.text(top[0]+titleShifts[0][0], top[1]+titleShifts[0][1], titles[0], fontsize=20)
  ax.text(botleft[0]+titleShifts[1][0], botleft[1]+titleShifts[1][1], titles[1], fontsize=20)
  ax.text(botright[0]+titleShifts[2][0], botright[1]+titleShifts[2][1], titles[2], fontsize=20)
  return(ax)

def radarPlot3AddPoint(ax, coord, style="o", color="b", markeredgecolor="none") :
  # Add a point in an existing three-edged radar plot.
  # @param ax a handle on the axis of an existing three-edged radar plot
  # @param coords coordinates in the radar plot to plot (array 2)
  # @param style (optional) the wanted linestyle for the marker (default is "o", circle)
  # @param color (optional) the wanted color for the marker (default is "b", blue)
  # @param markeredgecolor (optional) the wanted color for the edge of the marker (default is "none", none)
  # @return none
  ax.plot(coord[0], coord[1], style, color=color, markeredgecolor=markeredgecolor)

def radarPlot3GetCoordsOfValue(val) :
  # For a radar plot representing a three-dimensional vector, find the two coordinates in the radar plot corresponding to the vector.
  # @param val the vector (array 3)
  # @param the two coordinates in the radar plot corresponding to the vector
  if(np.size(val)!=3) : criticalError("[radarPlot3GetCoordsOfValue] val must be of size 3.")
  val=ensureLineVect(val, 3)
  norm100PercentVal=np.linalg.norm(val, 1)
  if(norm100PercentVal)==0 :
    val=val
  else :
    val=val/norm100PercentVal
  fv=ensureLineVect([0, 1], 2)
  sv=ensureLineVect([np.cos(7*np.pi/6), np.sin(7*np.pi/6)], 2)
  tv=ensureLineVect([np.cos(-np.pi/6), np.sin(-np.pi/6)], 2)
  return(np.abs(val[0])*fv+np.abs(val[1])*sv+np.abs(val[2])*tv)
###############################################################

# DO NOT CHANGE. ##############################################
g_useSmartSelection=True # smart selection : use it
###############################################################

#a=np.array([
#6.5727935629870002e-09, 1.916709328522e-05, -1.543169497787e-04, 1.097382718166e-05, 5.483495730494e+00, -4.121213370272e-01, -1.563849632196e+00
#])
##a[0]=a[0]/g_oneSolarMass; a[1:4]=a[1:4]*1e3/g_oneParsec; a[4:7]*1e3/g_scalingV; # convert JPL -> astro
#a[0]=a[0]*g_oneSolarMass; a[1:4]=a[1:4]*g_oneParsec; a[4:7]*g_scalingV; # convert astro -> SI
#for f in a :
#  print('{:19.12e}'.format(f), end=" ")

# example dataX
# X=np.array([[[1, 2, 3, 4, 5], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], [[1, 2, 3, 4, 5], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], [[1, 2, 3, 4, 5], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]])