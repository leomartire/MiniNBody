from mpl_toolkits.mplot3d import Axes3D # do not remove even if some editor says that this package is not used
from scipy.integrate import simps
from scipy.spatial.distance import cdist
from util import barycenter
from util import centroidPosition
from util import criticalBadValue
from util import criticalError
from util import getAngMomenta
from util import deviationFromSphericity
from util import ensureLineVect
from util import estimate2DDensity
from util import exportInitialConditionsToFile
from util import g_videoSizeInches
from util import generateVectorsRandomDirection
from util import ensureColVect
from util import g_tidalForcesGraphTitle
from util import g_forceScalingAxisTitle
from util import graphInterface
from util import extractAnisotropyBeta
from util import extractVelocitiesSigma
from util import formatTime
from util import g_GCCentroidPositionGraphTitle
from util import g_GravConstant
from util import g_angMomScalingAxisTitle
from util import g_angMomentaInertiaTensorGraphTitle
from util import g_angMomentaInertiaTensorSymbol
from util import g_hideTitles
from util import g_anisotropyGraphTitleLine1
from util import g_anisotropyGraphTitleLine2
from util import g_axisLabelTime
from util import g_deviationFromSphericityGraphTitle
from util import g_deviationFromSphericitySymbol
from util import g_energyScalingAxisTitle
from util import g_escapersGraphTitle
from util import g_fitVelocityGraphTitle
from util import g_fullscreenFigSize
from util import g_massEvolutionGraphTitle
from util import g_maxSoftening
from util import g_scalingL
from util import g_scalingM
from util import g_scalingT
from util import g_scalingV
from util import g_specificVelocityDispersionGraphTitle
from util import g_squareFigSize
from util import g_tensorAnalysisMaxEigenGraphTitle
from util import g_torqueScalingAxisTitle
from util import g_velocityDispersionGraphTitle
from util import g_velocityKurtosisGraphTitle
from util import g_velocityMeanGraphTitle
from util import g_velocityMeanSymbol
from util import g_velocitySkewGraphTitle
from util import g_velocityTensorDeviationFromIsotropyGraphTitle
from util import g_velocityTensorDeviationFromIsotropySymbol
from util import getCMapColorGradient
from util import getIDsInsideRadius
from util import getIDsOutsideRadius
from util import getSigmaFaberJackson
from util import inertiaTensor
from util import inertiaTensorAnalysis
from util import g_squishedSquareFigSize
from util import plotSolarSystem
from util import plotSolarSystem3D
from util import radialHistogram
from util import removeGlobalSpeed
from util import selectLimitationRadius
from util import userInput
from util import userInput0Or1
from util import userInputAskRecenter
from util import userInputGetNumber
from galaxyParameters import getGalaxyParameters
from util import userInputGetWantedProjection
from util import userInputTestIfDigit
from util import userInputTestIfNumber
from util import vectorEvolution
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

def angularMomentum(datX, datV, masses, times, rdatX, which="angularMomentum") :
  # Compute a particle system's total angular momenta over time.
  # @param datX a set of positions to use (recentered or not, array 3 * N * T)
  # @param datV a set of velocities to use (recentered or not, array 3 * N * T)
  # @param masses a set of masses to use (array N)
  # @param times a set of times (array T)
  # @param rdatX a set of positions recentered around their centroid (array 3 * N * T)
  # @param which encodes (optional) which information to get (can be either "angularMomentum" or "torque", default is "angularMomentum")
  # @return a code for figure saving and handles on figures and axis
  if(not(which in ["angularMomentum", "torque"])) : criticalBadValue("which", "angularMomentum")
  T=np.size(datX, axis=2)
  angMomenta=np.zeros([3, T])
  limitingRadius=userInputGetNumber("Enter the wanted range of selection or -1 for maximum range ([L] dimension, >"+"{:1.2f}".format(0)+").",
                                    minAccepVal=1e-9,
                                    maxAccepVal=np.inf,
                                    specials=["-1"])
  if(limitingRadius=="-1") : # take all particles into account
    tmp_normAxisInside=""
    directionTitleAdd=""
    returnText="_range_max"
  else :
    tmp_normAxisInside=r"R<"+str(float(limitingRadius))
    directionTitleAdd=r"(stars taken at $R<"+str(float(limitingRadius))+"$ [L])"
    returnText="_range_"+str(float(limitingRadius)).replace(".", ",")
  angMomenta=getAngMomenta(datX, datV, masses)
  totAngMomentum=np.zeros([3, T])
  for t in range(0, T) :
    if(limitingRadius=="-1") : selIDs=np.arange(0, np.size(masses))
    else : selIDs=getIDsInsideRadius(rdatX[:, :, t], limitingRadius, [0, 0, 0])[0]
    totAngMomentum[:, t]=[0, 0, 0]
    for i in selIDs :
      totAngMomentum[:, t]+=angMomenta[:, i, t]
  
  if(which=="angularMomentum") :
    ((fig, ax),
     (figu,
      (axuxy, axuxyt),
      (axuxz, axuxzt),
      (axuyz, axuyzt),
      rad,
      cb))=vectorEvolution(totAngMomentum, times,
                      onlyDir=False, v=np.linalg.norm(totAngMomentum, axis=0),
                      timeAxisTitle=g_axisLabelTime,
                      normTitle=r"Angular Momentum Norm $\left\|\vec{\mathcal{L}}_{"+tmp_normAxisInside+r"}\right\|_2$",
                      normAxisAround=[r"$\left\|\vec{\mathcal{L}}_{", r"}\right\|_2$ "+g_angMomScalingAxisTitle],
                      normAxisInside=tmp_normAxisInside,
                      directionTitle=r"Angular Momentum $\vec{\mathcal{L}}_{"+tmp_normAxisInside+r"}$ "+directionTitleAdd)
    return(returnText, (fig, ax), (figu, (axuxy, axuxyt), (axuxz, axuxzt), (axuyz, axuyzt), rad, cb))
  elif(which=="torque") :
    torque=np.zeros((3, T-1)) # differentiate angular momentum
    torque=(totAngMomentum[:, 1:]-totAngMomentum[:, :-1])/np.diff(times)
    torque=torque[:, 1:] # exclude first one because it bugs when times start at 0 twice
    ((fig, ax),
     (figu,
      (axuxy, axuxyt),
      (axuxz, axuxzt),
      (axuyz, axuyzt),
      rad,
      cb))=vectorEvolution(torque, times[T-np.size(torque, axis=1):],
                      onlyDir=False, v=np.linalg.norm(torque, axis=0),
                      timeAxisTitle=g_axisLabelTime,
                      normTitle=r"Torque Norm $\left\|\vec{\tau}_{"+tmp_normAxisInside+r"}\right\|_2$",
                      normAxisAround=[r"$\left\|\vec{\tau}_{", r"}\right\|_2$ "+g_torqueScalingAxisTitle],
                      normAxisInside=tmp_normAxisInside,
                      directionTitle=r"Torque $\vec{\tau}_{"+tmp_normAxisInside+r"}$ "+directionTitleAdd)
    return(returnText, (fig, ax), (figu, (axuxy, axuxyt), (axuxz, axuxzt), (axuyz, axuyzt), rad, cb))

def centroidPositionInterface(dat, times) :
  # Provides an interface for the plotting of the position of the centroid of the system.
  # @param dat a set of positions (array 3 * N * T)
  # @param times a set of times (array T)
  # @return a code for figure saving
  cP=centroidPosition(dat)
  return(centroidPositionInterfaceCPSaved(cP, times))

def centroidPositionInterfaceCPSaved(cP, times) :
  # Plots the centroid position.
  # @param cP the position of a centroid over time (array 3 * T)
  # @param times a set of times (array T)
  # @return a code for figure saving
  plotWindow=np.max(np.abs(cP))
  projPlane=userInputGetWantedProjection()
  marker=userInput("Enter the wanted marker (example : '.', '-', ':', ...).") 
  fig=plt.figure(figsize=g_squareFigSize)
  if projPlane=="XYZ" : ax=fig.add_subplot(111, projection='3d')
  else : ax=fig.add_subplot(111)
  ax.set_xlim([-plotWindow, plotWindow]); ax.set_ylim([-plotWindow, plotWindow])
  if projPlane=="XYZ" :
    ax.set_zlim([-plotWindow, plotWindow])
    ax.plot(cP[0, :], cP[1, :], cP[2, :], marker); ax.plot([0], [0], [0], 'k.', markersize=10)
    ax.set_xlabel(r'$x$ ([L])'); ax.set_ylabel(r'$y$ ([L])'); ax.set_zlabel(r'$z$ ([L])')
  elif projPlane=="XY" :
    ax.plot(cP[0, :], cP[1, :], marker); ax.plot([0], [0], 'k.', markersize=10)
    ax.set_xlabel(r'$x$ ([L])'); ax.set_ylabel(r'$y$ ([L])')
  elif projPlane=="XZ" :
    ax.plot(cP[0, :], cP[2, :], marker); ax.plot([0], [0], 'k.', markersize=10)
    ax.set_xlabel(r'$x$ ([L])'); ax.set_ylabel(r'$z$ ([L])')
  elif projPlane=="YZ" :
    ax.plot(cP[1, :], cP[2, :], marker); ax.plot([0], [0], 'k.', markersize=10)
    ax.set_xlabel(r'$y$ ([L])'); ax.set_ylabel(r'$z$ ([L])')
  graphInterface(fig=fig, ax=ax, title=g_GCCentroidPositionGraphTitle)
  if marker=="." :
    marker="DOTS"
  elif marker==":" :
    marker="DDOTS"
  return(marker+"_"+projPlane)

def computeGPEnergies(dat, masses) :
  # Computes the graviational potential energy of the system over time.
  # @param dat a set of positions (array 3 * N * T)
  # @param masses a set of masses (array N)
  # @return the graviational potential energy of the system over time (array T)
  N=np.size(dat, axis=1)
  T=np.size(dat, axis=2)
  GPE=np.zeros(T)
  for t in range(0, T) :
    for i in range(0, N) :
      for j in range (i+1, N) :
        d=np.linalg.norm(dat[:, i, t]-dat[:, j, t]); gpe=-(g_GravConstant*masses[i]*masses[j])/d; GPE[t]+=gpe
  return(GPE)

def computeKineticEnergies(datV, masses, rdatX, verbose=False,
                           indexes=[-1], plotEachKE=False,
                           plotKEContributions=False,
                           plotEachV=False, recenterVelocities=True) :
  # Computes the kinetic energy of the system over time. Does also some other things that are more of debugging.
  # @param datV a set of velocities (array 3 * N * T)
  # @param masses a set of masses (array N)
  # @param rdatX a set of positions recentered around their centroid (array 3 * N * T)
  # @param verbose (optionnal) verbosity toggle (True if yes, False if no, default is False)
  # @param indexes (optionnal) a set of timestamps indexes to consider (mainly for debug, default is [-1] to take all timestamps)
  # @param plotEachKE (optionnal) toggle to plot for every selected timestep (see parameter indexes) each star's kinetic energy (mainly for debug, default is False)
  # @param plotKEContributions (optionnal) toggle to plot  for each timestep (see parameter indexes) each star's kinetic energy one over the other to see contributions (mainly for debug, default is False)
  # @param plotEachV (optionnal) toggle to plot for every selected timestep (see parameter indexes) each star's velocity module (mainly for debug, default is False)
  # @param recenterVelocities (optionnal) toggle to remove the global component of velocities (True if yes, False if no, default is True)
  # @return the kinetic energy of the system over time (array T)
  if(recenterVelocities) : dataV=removeGlobalSpeed(datV)
  else : dataV=datV
  N=np.size(dataV, axis=1)
  T=np.size(dataV, axis=2)
  limitingRadius=userInputGetNumber("Enter the wanted range of selection or -1 for maximum range ([L] dimension, >"+"{:1.2f}".format(0)+").",
                               minAccepVal=1e-9,
                               maxAccepVal=np.inf,
                               specials=["-1"])
  if(limitingRadius=="-1") : # take all particles into account
    #axisLabeltext=""
    titleText=""
    approxTitleText=""
    returnText="_range_max"
  else :
    #axisLabeltext=r", R<"+str(float(limitingRadius))
    titleText="\n"+r"(stars taken at $R<"+str(float(limitingRadius))+"$ [L])"
    approxTitleText="\n"+r"(approximation : stars for $E_c$ taken at $R<"+str(float(limitingRadius))+"$ [L])"
    returnText="_range_"+str(float(limitingRadius)).replace(".", ",")
  if indexes==[-1] : timeRange=range(0, T)
  else : timeRange=indexes # a test to check if given array has bad values would be nice here
  KE=np.zeros(len(timeRange))
  
  if plotEachKE : fig=plt.figure(figsize=(18, 5)); ax=fig.add_subplot(111); tmpKEBuffer=np.zeros(N); legends=list(); tit="KE of each Star"
  if plotEachV : figV=plt.figure(figsize=(18, 5)); axV=figV.add_subplot(111); tmpVBuffer=np.zeros(N); legendsV=list(); titV=r"$\left\|\vec{V}\right\|_2$ of each Star ([V])"
  if plotKEContributions : figKEC=plt.figure(figsize=(5, 10)); axKEC=figKEC.add_subplot(111); tmpKECBuffer=np.zeros(N); colors=getCMapColorGradient(N); titKEC="Contributions to KE"
  
  c=0
  for t in timeRange :
    if(limitingRadius=="-1") : selIDs=np.arange(0, np.size(masses))
    else : selIDs=getIDsInsideRadius(rdatX[:, :, t], limitingRadius, [0, 0, 0])[0]
    for i in range(0, N) :
      if(i in selIDs) :
        modV=np.linalg.norm(dataV[:, i, t], axis=0)
        ke=0.5*masses[i]*modV**2;
        if verbose : print("Star "+str(i)+", timestamp "+str(t)+", v**2="+'{:10.4e}'.format(modV)+", m="+'{:10.4e}'.format(masses[i])+", KE="+'{:10.4e}'.format(ke)+" [E].")
        if plotEachKE : tmpKEBuffer[i]=ke
        if plotEachV : tmpVBuffer[i]=modV
        if plotKEContributions :
          tmpKECBuffer[i]=ke
          axKEC.plot(t, tmpKECBuffer.cumsum()[i], 'o', color=colors[i], markersize=4, markeredgecolor="none")
          axKEC.set_ylabel("cumulative sum")
        KE[c]+=ke
    c+=1
    if plotEachKE :
      ax.plot(tmpKEBuffer, 'o', markersize=4, markeredgecolor="none")
      ax.set_xlabel("n° of star"); ax.set_ylabel("energy "+g_energyScalingAxisTitle)
      legends.append("t "+str(t).zfill(4))
    if plotEachV :
      axV.plot(tmpVBuffer, 'o', markersize=4, markeredgecolor="none")
      axV.set_xlabel("n° of star"); axV.set_ylabel(r"$\left\|\vec{V}\right\|_2$ ([V])")
      legendsV.append("t "+str(t).zfill(4))
    if plotKEContributions : axKEC.set_xlim([indexes[0]-1, indexes[-1]+1])
  if plotEachKE : ax.legend(legends, loc="best"); graphInterface(fig=fig, ax=ax, title=tit)
  if plotEachV : axV.legend(legendsV, loc="best"); graphInterface(fig=figV, ax=axV, title=titV)
  if plotKEContributions : graphInterface(fig=figKEC, ax=axKEC, title=titKEC)
  return(KE, (returnText, titleText, approxTitleText))

def createVideo(dat, rdat, times, tmpImagesPath, diagnosticsFolder, solarSystem=False) :
  # Create a video from a given set of snapshots.
  # @param dat a set of positions (array 3 * N * T)
  # @param rdat a set of positions recentered around their centroid (array 3 * N * T)
  # @param time a set of times (array T)
  # @param tmpImagesPath a folder path to store temporary images
  # @param diagnosticsFolder a folder path to put the final video
  # @param solarSystem a boolean encoding if the that is a simulation of the solar system that should be plotted (True if yes, False if no)
  # @return none
  # Ask the user what he ######
  # wants to plot.       ######
  progressionPercentages=5 # how many percentages of generation to print ?
  recenter=userInputAskRecenter()
  if recenter : data=rdat
  else : data=dat
  rmImages=userInput0Or1("Conserve temporary images (0 for no, 1 for yes) ?",
                         if0="Temporary images will be deleted.",
                         if1="Temporary images will be conserved.")
  if(rmImages==0) : rmImages=True
  else : rmImages=False
  projPlane=userInputGetWantedProjection()
  plotRange=userInputGetNumber("Enter the wanted range of the graph or -1 for maximum range ([L] dimension, >"+"{:1.2f}".format(0)+").",
                               minAccepVal=1e-9,
                               maxAccepVal=np.inf,
                               specials=["-1"])
  if plotRange=="-1" : maximiseRange=True
  else : maximiseRange=False; plotRange=float(plotRange)
  
  T=np.size(data, axis=2)
  userT=userInput("A total of "+str(T)+" frames have been detected, plot how many of them ("+str(T)+" frames = "+'{:1.0f}'.format(times[0])+" to "+'{:1.0f}'.format(times[-1])+" [T]) ?")
  if userT.isdigit() and int(userT)>0 :
    if int(userT)>=T :
      userT=T-1
    elif int(userT)<progressionPercentages :
      userT=progressionPercentages
    else :
      userT=int(userT)
  else :
    criticalError("Bad number of frames. Please retry.")
  
  data=np.delete(data, np.s_[userT:], 2) # delete unwanted timestamps (by default : last ones)
  #############################
  
  # Prepare the data to be ####
  # plotted and the plot   ####
  # window.                ####
  if "X" in projPlane :
    xmax=data[0, :, :].max()
    xmin=data[0, :, :].min()
    if not recenter :
      xmax=np.max([xmax, 0])
      xmin=np.min([xmin, 0])
  if "Y" in projPlane :
    ymin=data[1, :, :].min()
    ymax=data[1, :, :].max()
    if not recenter :
      ymax=np.max([ymax, 0])
      ymin=np.min([ymin, 0])
  if "Z" in projPlane :
    zmax=data[2, :, :].max()
    zmin=data[2, :, :].min()
    if not recenter :
      zmax=np.max([zmax, 0])
      zmin=np.min([zmin, 0])

  fig=plt.figure()
  if projPlane=="XYZ" :
    ax=fig.add_subplot(111, projection='3d')
  else :
    ax=fig.add_subplot(111)
  
  fig.set_size_inches([g_videoSizeInches, g_videoSizeInches])
  #############################
  
  # Start plotting and ########
  # saving.            ########
  numFrames=np.size(data, 2)
  for i in range(0, numFrames) :
    ax.clear()
    if projPlane=="XYZ" :
      if(solarSystem) : plotSolarSystem3D(data[:, :, i], ax)
      else :
        ax.scatter(data[0, :, i], data[1, :, i], data[2, :, i], s=30, c='orange', depthshade=True, edgecolor='red')
        if not recenter : ax.plot([0], [0], [0], 'k.', markersize=10) # if the data is recentered, do not plot galatic center (or centroid)
      ax.set_xlabel(r'$x$ ([L])'); ax.set_ylabel(r'$y$ ([L])'); ax.set_zlabel(r'$z$ ([L])')
      if maximiseRange : ax.set_xlim([xmin, xmax]); ax.set_ylim([ymin, ymax]); ax.set_zlim([zmin, zmax])
      else : ax.set_xlim([-plotRange, plotRange]); ax.set_ylim([-plotRange, plotRange]); ax.set_zlim([-plotRange, plotRange])
    elif projPlane=="XY" :
      if maximiseRange : ax.set_xlim([xmin, xmax]); ax.set_ylim([ymin, ymax])
      else : ax.set_xlim([-plotRange, plotRange]); ax.set_ylim([-plotRange, plotRange])
      if(solarSystem) : plotSolarSystem(data[:, :, i], projPlane)
      else :
        ax.plot(data[0, :, i], data[1, :, i], 'o', c='orange', markeredgecolor='red')
        if not recenter : ax.plot([0], [0], 'k.', markersize=10) # if the data is recentered, do not plot galatic center (or centroid)
      ax.set_xlabel(r'$x$ ([L])'); ax.set_ylabel(r'$y$ ([L])')
    elif projPlane=="XZ" :
      if maximiseRange : ax.set_xlim([xmin, xmax]); ax.set_ylim([zmin, zmax])
      else : ax.set_xlim([-plotRange, plotRange]); ax.set_ylim([-plotRange, plotRange])
      if(solarSystem) : plotSolarSystem(data[:, :, i], projPlane)
      else :
        ax.plot(data[0, :, i], data[2, :, i], 'o', c='orange', markeredgecolor='red')
        if not recenter : ax.plot([0], [0], 'k.', markersize=10) # if the data is recentered, do not plot galatic center (or centroid)
      ax.set_xlabel(r'$x$ ([L])'); ax.set_ylabel(r'$z$ ([L])')
    elif projPlane=="YZ" :
      if maximiseRange : ax.set_xlim([ymin, ymax]); ax.set_ylim([zmin, zmax])
      else : ax.set_xlim([-plotRange, plotRange]); ax.set_ylim([-plotRange, plotRange])
      if(solarSystem) : plotSolarSystem(data[:, :, i], projPlane)
      else :
        ax.plot(data[1, :, i], data[2, :, i], 'o', c='orange', markeredgecolor='red')
        if not recenter : ax.plot([0], [0], 'k.', markersize=10) # if the data is recentered, do not plot galatic center (or centroid)
      ax.set_xlabel(r'$y$ ([L])'); ax.set_ylabel(r'$z$ ([L])')
    #ax.set_axis_off()
    graphInterface(fig=fig, ax=ax, title=r'$t='+'{:6.2f}'.format(times[i])+"$ ([T])", forceTitle=True)
    plt.savefig(tmpImagesPath+'/img_'+str(i).zfill(10))
    if i%(int(numFrames/(progressionPercentages-1)))==0 :
      print(" Video generation : "+'{:6.2f}'.format(100*i/numFrames)+" %.")
  print(" Video generation : "+'{:6.2f}'.format(100)+" %.")
  #############################
  # Make video and clean up. ##
  diagnosticsFolder=diagnosticsFolder.replace("/", "\\") # convert to filepath usable by .bat
  videoOutputName=diagnosticsFolder+"\output_"+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  os.system("makeMovie.bat \""+tmpImagesPath+"\" "+videoOutputName)
  print(" Video created : \""+videoOutputName+"\".")
  if rmImages : shutil.rmtree(tmpImagesPath) # remove temporary directory
  else : print(" Temporary images saved to : \""+tmpImagesPath+"\".")
  #############################

def escapersDetect(dat, radius) :
  # Detects escapers.
  # @param dat a set of positions (array 3 * N * T)
  # @param radius a radius beyond which a star is considered escaped
  # @return the number of stars outside the escape radius over time (array T)
  T=np.size(dat, axis=2)
  escapers=np.zeros(T)
  for t in range(0, T) : escapers[t]=np.size(getIDsOutsideRadius(dat[:, :, t], radius, [0, 0, 0]))
  return(escapers)

def escapersPlot(rdat, times, param="auto", escRadii=[8, 12, 16, 20, 24, 28]) :
  # Detects escapers for different radii and plots the percentage of escapers as a function of time.
  # @param rdat a set of positions recentered around their centroid (array 3 * N * T)
  # @param times a set of times (array T)
  # @param param (optionnal) toggles if radii should be chosen or asked to the user (can be either "auto" or "manual", default is "auto")
  # @return a code for figure saving
  if(not(param in ["auto", "manual"])) : criticalBadValue("param")
  if(param=="manual") :
    escRadius=userInputGetNumber("Enter the wanted sphere of evaluation radius ([L] dimension, >"+"{:1.2f}".format(0)+").",
                      minAccepVal=1e-9,
                      maxAccepVal=np.inf,
                      specials=[])
    retText="_"+str(escRadius).replace(".", ",")
    escRadii=[escRadius]
  else : retText=""
  N=np.size(rdat, axis=1)
  cmap=plt.get_cmap('jet'); colors=cmap(np.linspace(0, 1.0, len(escRadii)))  
  legends=list()
  fig=plt.figure(figsize=g_squareFigSize)
  ax=fig.add_subplot(111)
  for i in range(0, len(escRadii)) :
    esc=escapersDetect(rdat, escRadii[i])
    plt.plot(times, 100*esc/N, color=colors[i])
    legends.append(r"$r="+'{:6.2f}'.format(escRadii[i])+"$ ([L])")
  ax.legend(legends, loc='best')
  ax.set_xlabel(g_axisLabelTime)
  ax.set_ylabel('escapers (% of total number of stars)')
  ax.set_xlim([times[0], times[-1]])
  graphInterface(fig=fig, ax=ax, title=g_escapersGraphTitle)
  return(retText)

#def evolutionInertiaTensor(rdat, masses, times, snaps, plotType) :
#  # Plots the evolution of the inertia tensor of a system of particles.
#  # @param rdat a set of positions recentered around their centroid (array 3 * N * T)
#  # @param masses a set of masses (array N)
#  # @param times a set of times (array T)
#  # @param snaps a set of timestamps indexes
#  # @param plotType encodes which information from the inertia tensor to plot (can be either "3D" for the inertia tensor itself or "deviationFromSphericity" to plot the deviation from sphericity)
#  # @return none
#  if(not(plotType=="3D" or plotType=="deviationFromSphericity")) : criticalBadValue("plotType")
#  nbSnapshots=np.size(rdat, axis=2)
#  choosenSnapshots=(np.array(snaps)*nbSnapshots).astype(int)
#  plotRange=userInputGetNumber("Enter the wanted range of the graph or -1 for maximum range ([L] dimension, >"+"{:1.2f}".format(0)+").",
#                               minAccepVal=1e-9,
#                               maxAccepVal=np.inf,
#                               specials=["-1"])
#  if plotRange=="-1" : # take all particles into account
#    axisLabeltext=""
#    titleText=""
#    returnText="range_max"
#  else :
#    axisLabeltext=r", $R<"+str(float(plotRange))+r"$"
#    titleText=r"($R<"+str(float(plotRange))+"$)"
#    returnText="range_"+str(float(plotRange)).replace(".", ",")
#  if plotType=="3D" :
#    quiv_ar_len_rat=0.2
#    nbL=int(np.floor(len(choosenSnapshots)**0.5))
#    nbC=int(np.ceil(len(choosenSnapshots)/nbL))
#    fig=plt.figure(figsize=g_fullscreenFigSize)
#    plt.suptitle("Inertia Tensor", fontsize="large")
#  elif plotType=="deviationFromSphericity" :
#    fig=plt.figure(figsize=g_squareFigSize)
#    ax=fig.add_subplot(111)
#    deviationBuffer=np.zeros(len(choosenSnapshots))
#  for i in range(1, len(choosenSnapshots)+1) :
#    if(plotRange=="-1") : selIDs=np.arange(0, np.size(masses))
#    else : selIDs=getIDsInsideRadius(rdat[:, :, i-1], plotRange, [0, 0, 0])[0]
#    I=inertiaTensor(rdat[:, selIDs, i-1], masses[selIDs])
#    IEig=np.linalg.eig(I)
#    sortedMagnitudesIndexes=np.argsort(IEig[0])[-1::-1]
#    if plotType=="3D" :
#      ax=fig.add_subplot(nbL, nbC, i, projection='3d')
#      IEigScaled=IEig[0]*IEig[1]/np.max(IEig[0]) # scale : major axis is of norm 1 and others are scaled
#      majorAxis=IEigScaled[:, sortedMagnitudesIndexes[0]]
#      mediumAxis=IEigScaled[:, sortedMagnitudesIndexes[1]]
#      minorAxis=IEigScaled[:, sortedMagnitudesIndexes[2]]
#      ax.set_xlim([-1.1, 1.1])
#      ax.set_ylim([-1.1, 1.1])
#      ax.set_zlim([-1.1, 1.1])
#      ax.quiver([0], [0], [0], majorAxis[0], majorAxis[1], majorAxis[2], color='r', pivot='tail', arrow_length_ratio=quiv_ar_len_rat, length=np.linalg.norm(majorAxis))
#      ax.quiver([0], [0], [0], mediumAxis[0], mediumAxis[1], mediumAxis[2], color='b', pivot='tail', arrow_length_ratio=quiv_ar_len_rat, length=np.linalg.norm(mediumAxis))
#      ax.quiver([0], [0], [0], minorAxis[0], minorAxis[1], minorAxis[2], color='g', pivot='tail', arrow_length_ratio=quiv_ar_len_rat, length=np.linalg.norm(minorAxis))
#      ax.legend([r"$I_1$", r"$I_2$", r"$I_3$"], loc="best")
#      ax.set_title(r"$t="+'{:6.2f}'.format(times[choosenSnapshots[i-1]])+"$ [T]")
#      ax.set_xlabel(r'$x$ axis')
#      ax.set_ylabel(r'$y$ axis')
#      ax.set_zlabel(r'$z$ axis')
#    elif plotType=="deviationFromSphericity" :
#      deviationBuffer[i-1]=deviationFromSphericity(rdat[:, selIDs, i-1], masses[selIDs], IEig)
#  if plotType=="deviationFromSphericity" :
#    ax.plot(times[choosenSnapshots], deviationBuffer)
#    ax.set_title(g_deviationFromSphericityGraphTitle+" "+titleText)
#    ax.set_xlabel(g_axisLabelTime)
#    ax.set_ylabel("$\Delta_S(t)$"+axisLabeltext)
#  return(returnText)

def evolutionTidalForces(cP, times, NSamples=300) :
  # Compute the evolution of tidal forces induced by a galaxy on the GC.
  # This function is specific to the implementation of the galactic gravitational field of this project.
  # @param cP a centroid position over time (array 3 * T)
  # @param times a set of times (array T)
  # @param NSamples (optionnal) the number of samples to take around the centroid (default is 300)
  # @return a code for figure saving and the values of the lowest, mean and highest norms of tidal forces at each time stamp (each, array T)
  rt=userInputGetNumber("Enter the wanted sphere of evaluation radius ([L] dimension, >"+"{:1.2f}".format(0)+").",
                        minAccepVal=1e-9,
                        maxAccepVal=np.inf,
                        specials=[])
  returnText="range_"+str(float(rt)).replace(".", ",")
  norms=rt*np.ones(NSamples)
  testPos=generateVectorsRandomDirection(norms)
  T=np.size(times)
  mean=np.zeros(T)
  highest=np.zeros(T)
  lowest=np.zeros(T)
  for t in range(0, T) :
    COM=ensureColVect(cP[:, t])
    testPosGlobal=testPos+COM
    tidalForcesNorms=np.zeros(NSamples)
    for i in range(0, NSamples) :
      tidalForcesNorms[i]=np.linalg.norm(getTidalForce(testPosGlobal[:, i], COM))
    mean[t]=np.mean(tidalForcesNorms)
    highest[t]=np.max(tidalForcesNorms)
    lowest[t]=np.min(tidalForcesNorms)
  fig=plt.figure()
  ax=fig.add_subplot(111)
  ax.plot(times, highest, "r:")
  ax.plot(times, mean, "g")
  ax.plot(times, lowest, "b:")
  ax.set_xlim([times[0], times[-1]])
  ax.legend(["highest", "mean", "lowest"], loc="best")
  ax.set_xlabel(g_axisLabelTime)
  ax.set_ylabel(r"$\left\|\vec{F_T}\right\|_2(t)$ "+g_forceScalingAxisTitle)
  graphInterface(fig=fig, ax=ax, title=g_tidalForcesGraphTitle)
  return(returnText, lowest, highest, mean)

def extractAnisotropy(rdatX, datV, times, radiusForMagnitude=-1) :
  # Plots the the anisotropy of a system.
  # @param rdatX a set of positions recentered around their centroid (array 3 * N * T)
  # @param rdatV a set of velocities without global velocity (array 3 * N * T)
  # @param times a set of times (array T)
  # @return none
  if radiusForMagnitude==-1 : # take all particles into account when approximating magnitude by total mass
    axisLabeltext=""
    titleText=""
    returnText="range_max"
  else :
    axisLabeltext=r"R<"+str(float(radiusForMagnitude))
    titleText="\n"+r"(stars taken at $R<"+str(float(radiusForMagnitude))+"$ [L])"
    returnText="range_"+str(float(radiusForMagnitude)).replace(".", ",")
  anisotropies=np.zeros(np.size(rdatX, axis=2))
  for i in range(0, np.size(rdatX, axis=2)) :
    if(radiusForMagnitude==-1) :
      anisotropies[i]=extractAnisotropyBeta(rdatX[:, :, i], datV[:, :, i])
    else :
      selectedIDs=getIDsInsideRadius(rdatX[:, :, i], radiusForMagnitude, [0, 0, 0])[0]
      anisotropies[i]=extractAnisotropyBeta(rdatX[:, selectedIDs, i], datV[:, selectedIDs, i])
  fig=plt.figure(figsize=g_squareFigSize)
  ax=fig.add_subplot(111)
  ax.plot(times, anisotropies)
  ax.set_xlabel(g_axisLabelTime)
  ax.set_ylabel(r"$\beta_{"+axisLabeltext+r"}$")
  ax.set_xlim([times[0], times[-1]])
  graphInterface(fig=fig, ax=ax, title=g_anisotropyGraphTitleLine1+" "+titleText+"\n"+g_anisotropyGraphTitleLine2)
  return(returnText)

#def fromMInf1ToM11(xgr) :
#  T=np.size(xgr)
#  res=np.zeros(T)
#  subZero=np.where(xgr<0)[0]
#  supZero=np.where(xgr>=0)[0]
#  res[subZero]=-1+10**xgr[subZero]
#  res[supZero]=xgr[supZero]
#  return(res)

def extractStat(rdatX, rdatV, times, masses, which="dispersion", radiusForMagnitude=-1) :
  # Extracts the specific dispersion of a system.
  # @param rdatX a set of positions recentered around their centroid (array 3 * N * T)
  # @param rdatV a set of velocities without global velocity (array 3 * N * T)
  # @param times a set of times (array T)
  # @param masses a set of masses (array N)
  # @param which (optional) a code to specify which statistic to get (can be either of the following : "mean", "dispersion", "specificDispersion", "kurtosis" or "skewness", default is "dispersion")
  # @param radiusForMagnitude (optionnal) radius below which particles are taken into account (default is -1, to take all particles into account)
  # @return a code for figure saving
  if(not(which in ["mean", "dispersion", "specificDispersion", "kurtosis", "skewness"])) : criticalBadValue("which", "extractDispersion")
  if radiusForMagnitude==-1 : # take all particles into account when approximating magnitude by total mass
    axisLabeltext=""
    titleText=""
    returnText="range_max"
  else :
    axisLabeltext=r", R<"+str(float(radiusForMagnitude))
    titleText="\n"+r"(stars taken at $R<"+str(float(radiusForMagnitude))+"$ [L])"
    returnText="range_"+str(float(radiusForMagnitude)).replace(".", ",")
  val=np.zeros(np.size(rdatX, axis=2))
  thDisp=np.zeros(np.size(rdatX, axis=2))
  if(which=="kurtosis") : from scipy.stats import kurtosis
  elif(which=="skewness") : from scipy.stats import skew
  for t in range(0, np.size(rdatX, axis=2)) :
    if(radiusForMagnitude==-1) : selIDs=np.arange(0, np.size(masses))
    else : selIDs=getIDsInsideRadius(rdatX[:, :, t], radiusForMagnitude, [0, 0, 0])[0]
    if(which in ["dispersion", "specificDispersion"]) :
      dev=extractVelocitiesSigma(rdatV[:, selIDs, t], retVals="deviation")
      if(which=="specificDispersion") :
        val[t]=dev/np.sum(masses[selIDs])
      elif(which=="dispersion") :
        val[t]=dev
        thDisp[t]=getSigmaFaberJackson(np.sum(masses[selIDs]))
    elif(which in ["mean", "kurtosis", "skewness"]) :
      rad=np.linalg.norm(rdatV[:, selIDs, t], axis=0)
      if(which=="mean") : val[t]=np.mean(rad)
      elif(which=="kurtosis") : val[t]=kurtosis(rad)
      elif(which=="skewness") : val[t]=skew(rad)
        
  fig=plt.figure(figsize=g_squishedSquareFigSize)
  ax=fig.add_subplot(111)
  if(which in ["dispersion", "specificDispersion", "skewness", "kurtosis"]) :
    ax.plot(times, val)
  elif(which in ["mean"]) :
    ax.semilogy(times, val)
  ax.set_xlabel(g_axisLabelTime)
  if(which=="specificDispersion") :
    ax.set_ylabel(r"$\sigma_S(t)=\frac{\sigma(t)}{M_{T"+axisLabeltext+"}}$ ([V]/[M])")
    tit=g_specificVelocityDispersionGraphTitle+titleText
  elif(which=="dispersion") :
    ax.plot(times, thDisp)
    ax.set_ylabel(r"$\sigma(t)$ ([V])")
    tit=g_velocityDispersionGraphTitle+titleText
    ax.legend(["computed dispersion", "theoretical dispersion"], loc="best")
  elif(which=="mean") :
    ax.set_ylabel(g_velocityMeanSymbol+" ([V])")
    tit=g_velocityMeanGraphTitle+titleText
  elif(which=="skewness") :
    ax.set_ylabel("skewness of velocities' modules ([V])")
    tit=g_velocitySkewGraphTitle+titleText
  elif(which=="kurtosis") :
    ax.set_ylabel("kurtosis of velocities' modules ([V])")
    tit=g_velocityKurtosisGraphTitle+titleText
  ax.set_xlim([times[0], times[-1]])
  graphInterface(fig=fig, ax=ax, title=tit)
  return(returnText)

def extractVelocityInformationsInterface(information, rdatX, datV, masses, times) :
  # Provides an interface for the extraction of specific dispersion and the plotting of anisotropy.
  # @param information encodes which diagnostic to run (can be either "specificDispersion" or "anisotropy")
  # @param rdatX a set of positions recentered around their centroid (array 3 * N * T)
  # @param datV a set of velocities (array 3 * N * T)
  # @param masses a set of masses (array N)
  # @param times a set of times (array T)
  # @return a code for figure saving
  if(not(information in ["mean", "dispersion", "specificDispersion", "kurtosis", "skewness", "anisotropy"])) : criticalBadValue("information", "extractVelocityInformationsInterface")
  datV=removeGlobalSpeed(datV) # subtract global velocity of the system
  plotRange=userInputGetNumber("Enter the wanted range of selection or -1 for maximum range ([L] dimension, >="+"{:1.2f}".format(np.min(np.linalg.norm(datV, axis=0)))+").",
                               minAccepVal=np.min(np.linalg.norm(datV, axis=0)),
                               maxAccepVal=np.inf,
                               specials=["-1"])
  magnitudeRadius=float(plotRange)
  if(information in ["mean", "dispersion", "specificDispersion", "kurtosis", "skewness"]) :
    ref=extractStat(rdatX, datV, times, masses, information, magnitudeRadius)
    return(ref)
  elif(information=="anisotropy") :
    ref=extractAnisotropy(rdatX, datV, times, magnitudeRadius)
    return(ref)

def fitVelocityDistribution(datV, time) :
  # Tries to fit a known distribution to a computed velocity module distribution.
  # @param datV a set of velocities (array 3 * N)
  # @param time a time
  # @return a code for figure saving
  datV=datV-barycenter(datV) # subtract global velocity of the system
  nbSamples=int(0.05*np.size(datV, axis=1))
  modV=np.linalg.norm(datV, axis=0)
  
  plotRange=userInputGetNumber("Enter the wanted range of the graph or -1 for maximum range ([V] dimension, >="+"{:1.2f}".format(np.min(modV))+").",
                               minAccepVal=np.min(modV),
                               maxAccepVal=np.inf,
                               specials=["-1"])
  if plotRange=="-1" :
    rightCut=np.max(modV)
    returnText="range_max"
  else :
    rightCut=float(plotRange)
    returnText="range_"+str(plotRange).replace(".", ",")
  # Evaluate the data.
  hist=np.histogram(modV, nbSamples, range=(np.min(modV), rightCut))[0]
  xHist=np.linspace(0, rightCut, nbSamples)
  # Do the different fits.
  x=np.linspace(0, rightCut, 1000)
  histInter=np.interp(x, xHist, hist/simps(hist, xHist))
  #laws=["beta", "chi2", "f", "gamma", "levy", "lognorm", "weibull_min"]
  #lawsFancyNames=["Beta", "Chi-Squared", "Fisher-Snedecor", "Gamma", "Lévy", "Log-Normal", "Weibull"]
  laws=["alpha", "anglit", "arcsine", "beta", "betaprime", "bradford", "burr", "burr12", "cauchy", "chi", "chi2", "cosine", "dgamma", "dweibull", "erlang", "expon", "exponnorm", "exponweib", "exponpow", "f", "fatiguelife", "fisk", "foldcauchy", "foldnorm", "frechet_r", "frechet_l", "genlogistic", "gennorm", "genpareto", "genexpon", "genextreme", "gausshyper", "gamma", "gengamma", "genhalflogistic", "gilbrat", "gompertz", "gumbel_r", "gumbel_l", "halfcauchy", "halflogistic", "halfnorm", "halfgennorm", "hypsecant", "invgamma", "invgauss", "invweibull", "johnsonsb", "johnsonsu", "kappa4", "kappa3", "ksone", "kstwobign", "laplace", "levy", "levy_l", "levy_stable", "logistic", "loggamma", "loglaplace", "lognorm", "lomax", "maxwell", "mielke", "nakagami", "ncx2", "ncf", "nct", "norm", "pareto", "pearson3", "powerlaw", "powerlognorm", "powernorm", "rdist", "reciprocal", "rayleigh", "rice", "recipinvgauss", "semicircular", "skewnorm", "t", "trapz", "triang", "truncexpon", "truncnorm", "tukeylambda", "uniform", "vonmises", "vonmises_line", "wald", "weibull_min", "weibull_max", "wrapcauchy"]
#  laws=["alpha", "anglit", "arcsine", "beta", "betaprime", "bradford", "burr", "cauchy", "chi", "chi2", "cosine", "dgamma", "dweibull", "erlang", "expon", "exponnorm", "exponweib", "exponpow", "f", "fatiguelife", "fisk", "foldcauchy", "foldnorm", "frechet_r", "frechet_l", "genlogistic", "gennorm", "genpareto", "genexpon", "genextreme", "gausshyper", "gamma", "gengamma", "genhalflogistic", "gilbrat", "gompertz", "gumbel_r", "gumbel_l", "halfcauchy", "halflogistic", "halfnorm", "halfgennorm", "hypsecant", "invgamma", "invgauss", "invweibull", "johnsonsb", "johnsonsu", "ksone", "kstwobign", "laplace", "levy", "levy_l", "logistic", "loggamma", "loglaplace", "lognorm", "lomax", "maxwell", "mielke", "nakagami", "ncx2", "ncf", "nct", "norm", "pareto", "pearson3", "powerlaw", "powerlognorm", "powernorm", "rdist", "reciprocal", "rayleigh", "rice", "recipinvgauss", "semicircular", "t", "trapz", "triang", "truncexpon", "truncnorm", "tukeylambda", "uniform", "vonmises", "vonmises_line", "wald", "weibull_min", "weibull_max", "wrapcauchy"]
#  laws=["t", "trapz", "triang", "truncexpon", "truncnorm", "tukeylambda", "uniform", "vonmises", "vonmises_line", "wald", "weibull_min", "weibull_max", "wrapcauchy"]
#  laws=["levy_stable"]
  lawsFancyNames=laws
  errs=list()
  params=list()
  for law in laws :
    try :
      curLaw=getattr(__import__('scipy.stats', fromlist=[law]), law)
    except AttributeError :
      print(curLaw.name+" : law not known, skipping.")
      errs.append(np.inf) # to keep the errs list at correct size
      params.append("") # to keep the params list at correct size
      continue
    try :
      param=curLaw.fit(modV)
    except NotImplementedError :
      print(curLaw.name+" : fit not implemented, skipping.")
      errs.append(np.inf) # to keep the errs list at correct size
      params.append("") # to keep the params list at correct size
      continue
    params.append(param)
    fit=curLaw.pdf(x, *param)/simps(curLaw.pdf(x, *param), x)
    errs.append(np.sum((fit-histInter)**2))
  sortedErrs=np.argsort(errs)
  # Plot the two fits with lowest error.
  fig=plt.figure(figsize=g_fullscreenFigSize)
  ax=fig.add_subplot(111)
  plt.plot(xHist, hist/simps(hist, xHist))
  law1=getattr(__import__('scipy.stats', fromlist=[laws[sortedErrs[0]]]), laws[sortedErrs[0]])
  law2=getattr(__import__('scipy.stats', fromlist=[laws[sortedErrs[1]]]), laws[sortedErrs[1]])
  law3=getattr(__import__('scipy.stats', fromlist=[laws[sortedErrs[2]]]), laws[sortedErrs[2]])
  plt.plot(x, law3.pdf(x, *params[sortedErrs[2]])/simps(law3.pdf(x, *params[sortedErrs[2]]), x))
  plt.plot(x, law2.pdf(x, *params[sortedErrs[1]])/simps(law2.pdf(x, *params[sortedErrs[1]]), x))
  plt.plot(x, law1.pdf(x, *params[sortedErrs[0]])/simps(law1.pdf(x, *params[sortedErrs[0]]), x))
  plt.legend(["computed data density", "Third best : "+lawsFancyNames[sortedErrs[2]], "Second best : "+lawsFancyNames[sortedErrs[1]], "Best : "+lawsFancyNames[sortedErrs[0]]], loc='best')
  ax.set_xlabel(r"$\left\|\vec{V}\right\|_2$ ([V])")
  ax.set_ylabel('densities')
  graphInterface(fig=fig, ax=ax, title=g_fitVelocityGraphTitle+" at "+formatTime(time))
  # Give information about the two best fits in console.
  print("Best fit ("+lawsFancyNames[sortedErrs[0]]+", error="+"{:1.2e}".format(errs[sortedErrs[0]])+") parameters :"); print(params[sortedErrs[0]]);
  print("Second best fit ("+lawsFancyNames[sortedErrs[1]]+", error="+"{:1.2e}".format(errs[sortedErrs[1]])+") parameters :"); print(params[sortedErrs[1]]);
  print("Third best fit ("+lawsFancyNames[sortedErrs[2]]+", error="+"{:1.2e}".format(errs[sortedErrs[2]])+") parameters :"); print(params[sortedErrs[2]]);
  return(returnText)
  
  # alpha PDF : exp(-0.5*(a-1/x)^2)/(x^2*(0.5*(1+erf(x/(sqrt(2)))))*sqrt(2*pi))

def fitVelocityDistributionInterface(datV, rdatX, times) :
  # Provides an interface for the fitting of a known distribution to a computed velocity module distribution.
  # @param datV a set of velocities (array 3 * N * T)
  # @param times a set of times (array T)
  # @return a code for figure saving
  snapshotID=userInputTestIfDigit(userInput("Which snapshot to fit (>=0, <="+str(np.size(datV, axis=2)-1)+") ?"), 0, np.size(datV, axis=2)-1)
  limitingRadius=userInputGetNumber("Enter the wanted range of selection or -1 for maximum range ([L] dimension, >"+"{:1.2f}".format(0)+").",
                                    minAccepVal=1e-9,
                                    maxAccepVal=np.inf,
                                    specials=["-1"])
  if(limitingRadius=="-1") : selectedIDs=np.arange(0, np.size(datV, axis=1))
  else : selectedIDs=getIDsInsideRadius(rdatX[:, :, snapshotID], limitingRadius, [0, 0, 0])[0]
  rangeText=fitVelocityDistribution(datV[:, selectedIDs, snapshotID], times[snapshotID])
  return("_t_"+str(snapshotID)+"_"+rangeText)

def getGravitationalForce(pos, mass) :
  # Compute the gravitationnal force on a body.
  # This function is specific to the implementation of the galactic gravitational field of this project.
  # @param pos a three-dimensional position (array 3)
  # @param mass a mass
  # @return the galactic gravitational force a the given position for a particle of given mass
  tmpX=pos[0]
  tmpY=pos[1]
  tmpZ=pos[2]
  tmpRCSquared=tmpX*tmpX+tmpY*tmpY;
  tmpZSquared=tmpZ*tmpZ;
  tmpRSquared=tmpRCSquared+tmpZSquared;
  F=np.zeros(3)
  (g_mb, g_ab, g_md, g_ad, g_hd, g_vh, g_ah)=getGalaxyParameters()
  # x
  tmpPhiB=g_GravConstant*g_mb*pow(tmpRSquared+g_ab*g_ab, -1.5)*tmpX;
  tmpPhiD=g_GravConstant*g_md*pow(tmpRCSquared+pow(g_ad+pow(tmpZSquared+g_hd*g_hd, 0.5), 2.0), -1.5)*tmpX;
  tmpPhiH=-(g_vh*g_vh)/(tmpRSquared+g_ah*g_ah)*tmpX;
  F[0]=tmpPhiB+tmpPhiD+tmpPhiH;
  # y
  tmpPhiB*=(tmpY/tmpX);
  tmpPhiD*=(tmpY/tmpX);
  tmpPhiH*=(tmpY/tmpX);
  F[1]=tmpPhiB+tmpPhiD+tmpPhiH;
  # z
  tmpPhiB*=(tmpZ/tmpY);
  tmpPhiD*=(tmpZ/tmpY);
  tmpPhiH*=(tmpZ/tmpY);
  tmpPhiD*=((g_ad+pow(tmpZSquared+g_hd*g_hd, 0.5))/(pow(tmpZSquared+g_hd*g_hd, 0.5)));
  F[2]=tmpPhiB+tmpPhiD+tmpPhiH;
  F[0]*=-mass;
  F[1]*=-mass;
  F[2]*=-mass;
  return(F)

def getTidalForce(pos, COM) :
  # Get the tidal force at a given position, given the position of the COM of the GC.
  # This function is specific to the implementation of the galactic gravitational field of this project.
  # @param pos the wanted position where to compute the tidal force (array 3)
  # @param COM the position of the COM of the GC
  # @return the tidal force at the given position
  FS=getGravitationalForce(pos, 1)
  FGC=getGravitationalForce(COM, 1)
  return(FS-FGC)

def inertiaTensorWrapper(dat, masses, times, rdatX, dataType="velocity") :
  # Wraps an inertia tensor study and graphing.
  # @param dat a set of data of which compute inertia tensors (array 3 * N * T, can be either velocities or positions for example)
  # @param masses a set of masses (array N)
  # @param times a set of times (array T)
  # @param rdatX a set of positions recentered around their centroid (array 3 * N * T)
  # @para dataType (optional) a string to specify some special treatments, titles and other things (can be either of the following : "positions", "velocity" or "angularMomentum", default is "velocity")
  # @return a code for figure saving and handles on figures and axis
  if(not(dataType in ["positions", "velocity", "angularMomentum"])) : criticalBadValue("dataType", "inertiaTensorWrapper")
  T=np.size(times)
  if(not(np.size(dat, axis=0)==np.size(rdatX, axis=0)
         and np.size(dat, axis=1)==np.size(masses)==np.size(rdatX, axis=1)
         and np.size(dat, axis=2)==np.size(times)==np.size(rdatX, axis=2))) :
    criticalError("[inertiaTensorWrapper] Arrays do not have corresponding dimensions.")
  limitingRadius=userInputGetNumber("Enter the wanted range of selection or -1 for maximum range ([L] dimension, >"+"{:1.2f}".format(0)+").",
                               minAccepVal=1e-9,
                               maxAccepVal=np.inf,
                               specials=["-1"])
  if(limitingRadius=="-1") : # take all particles into account
    tmp_normAxisInside=""
    tmp_directionTitleAdd=""
    tmp_normTitleAdd=""
    returnText="range_max"
  else :
    tmp_normAxisInside=r", R<"+str(float(limitingRadius))
    tmp_directionTitleAdd=r"(stars taken at $R<"+str(float(limitingRadius))+"$ [L])"
    tmp_normTitleAdd="\n"+tmp_directionTitleAdd
    returnText="range_"+str(float(limitingRadius)).replace(".", ",")
  inertiaTensors=list()
  v=list()
  for t in range(0, T) :
    if(limitingRadius=="-1") : selIDs=np.arange(0, np.size(masses))
    else : selIDs=getIDsInsideRadius(rdatX[:, :, t], limitingRadius, [0, 0, 0])[0]
    IEig=np.linalg.eig(inertiaTensor(dat[:, selIDs, t], masses[selIDs]))
    inertiaTensors.append(IEig)
    v.append(deviationFromSphericity(dat[:, selIDs, t], masses[selIDs], IEig))
  
  if(dataType=="velocity") :
    tmp_normTitle=g_velocityTensorDeviationFromIsotropyGraphTitle
    tmp_normAxisAround=g_velocityTensorDeviationFromIsotropySymbol
    tmp_directionTitle=g_tensorAnalysisMaxEigenGraphTitle+" "+tmp_directionTitleAdd
    tmp_normPlotType="semilogy"
  elif(dataType=="positions") :
    tmp_normTitle=g_deviationFromSphericityGraphTitle
    tmp_normAxisAround=g_deviationFromSphericitySymbol
    tmp_directionTitle=g_tensorAnalysisMaxEigenGraphTitle+" "+tmp_directionTitleAdd
    tmp_normPlotType="plot"
  elif(dataType=="angularMomentum") :
    tmp_normTitle=g_angMomentaInertiaTensorGraphTitle
    tmp_normAxisAround=g_angMomentaInertiaTensorSymbol
    tmp_directionTitle="Privilegied Rotation Axis"+" "+tmp_directionTitleAdd
    tmp_normPlotType="plot"
    
  ((fig, ax),
   (figu,
    (axuxy, axuxyt),
    (axuxz, axuxzt),
    (axuyz, axuyzt),
    rad,
    cb))=inertiaTensorAnalysis(inertiaTensors, v, times,
                               timeAxisTitle=g_axisLabelTime,
                               normTitle=tmp_normTitle+tmp_normTitleAdd,
                               normAxisAround=tmp_normAxisAround,
                               normAxisInside=tmp_normAxisInside,
                               directionTitle=tmp_directionTitle,
                               normPlotType=tmp_normPlotType)
  return(returnText, (fig, ax), (figu, (axuxy, axuxyt), (axuxz, axuxzt), (axuyz, axuyzt), rad, cb), v)

def plotDensityContourLines(rdat, times, snaps) :
  # Plots as contour lines the density of particles on a chosen plane.
  # @param rdat a set of positions recentered around their centroid (array 3 * N * T)
  # @param times a set of times (array T)
  # @param snaps a set of timestamps indexes
  # @return a code for figure saving
  nbSnapshots=np.size(rdat, axis=2)
  choosenSnapshots=(np.array(snaps)*nbSnapshots).astype(int)
  nbL=int(np.floor(len(choosenSnapshots)**0.5))
  nbC=int(np.ceil(len(choosenSnapshots)/nbL))
  nbLines=6
  resolution=100
  cols=getCMapColorGradient(nbLines)
  projPlane=userInputGetWantedProjection(restriction="2D")
  if projPlane=="XY" : fC=0; sC=1
  elif projPlane=="XZ" : fC=0; sC=2
  elif projPlane=="YZ" : fC=1; sC=2
  
  plotRange=userInputGetNumber("Enter the wanted range of the graph or -1 for maximum range ([L] dimension, >"+"{:1.2f}".format(0)+").",
                               minAccepVal=1e-9,
                               maxAccepVal=np.inf,
                               specials=["-1"])
  if plotRange=="-1" :
    plotRange=np.max([np.max(np.abs(rdat[fC, :, :])), np.max(np.abs(rdat[sC, :, :]))])
    returnText="range_max"
  else :
    plotRange=float(plotRange)
    returnText="range_"+str(int(plotRange))
  fig=plt.figure(figsize=g_fullscreenFigSize)
  for i in range(1, len(choosenSnapshots)+1) :
    ax=fig.add_subplot(nbL, nbC, i)
    ax.set_xlim([-plotRange, plotRange])
    ax.set_ylim([-plotRange, plotRange])
    ids=np.where((np.abs(rdat[fC, :, choosenSnapshots[i-1]])<plotRange) & (np.abs(rdat[sC, :, choosenSnapshots[i-1]])<plotRange))
    x=rdat[fC, ids, choosenSnapshots[i-1]]
    y=rdat[sC, ids, choosenSnapshots[i-1]]
    (X, Y, Z)=estimate2DDensity(x, y, plotRange, resolution)
    ax.plot(rdat[fC, :, choosenSnapshots[i-1]], rdat[sC, :, choosenSnapshots[i-1]], 'k.', markersize=1)
    levs=np.logspace(np.log10(np.max(Z)/5000), np.log10(np.max(Z)), nbLines)
    #levs=np.linspace(np.max(Z)/100, np.max(Z), nbLines)
    ax.contour(X, Y, Z, levels=levs, colors=cols)
    #ax.contour(X, Y, Z, colors=cols)
    ax.set_title(r"$t="+'{:6.2f}'.format(times[choosenSnapshots[i-1]])+"$ [T]")
  if(g_hideTitles) : plt.tight_layout()
  else : plt.suptitle("GC Density Logarithmic Contour Lines ("+projPlane+" plane)", fontsize="large")
  return("_"+projPlane+"_"+returnText)

def plotMassEvolution(rdatX, masses, times) :
  # Plot the evolution of masses inside a certain radius of the GC.
  # @param rdatX a set of positions recentered around their centroid (array 3 * N * T)
  # @param masses a set of masses (array N)
  # @param times a set of times (array T)
  # @return a code for figure saving
  limitingRadius=userInputGetNumber("Enter the wanted range of selection or -1 for maximum range ([L] dimension, >"+"{:1.2f}".format(0)+").",
                               minAccepVal=1e-9,
                               maxAccepVal=np.inf,
                               specials=["-1"])
  if(limitingRadius=="-1") : # take all particles into account
    axisLabeltext=""
    titleText=""
    returnText="range_max"
  else :
    axisLabeltext=r"R<"+str(float(limitingRadius))
    titleText="\n"+r"(stars taken at $R<"+str(float(limitingRadius))+"$ [L])"
    returnText="range_"+str(float(limitingRadius)).replace(".", ",")
  T=np.size(times)
  totMasses=np.zeros(T)
  for t in range(0, T) :
    if(limitingRadius=="-1") :
      selectedIDs=np.arange(0, np.size(masses))
    else :
      selectedIDs=getIDsInsideRadius(rdatX[:, :, t], limitingRadius, [0, 0, 0])[0]
    totMasses[t]=np.sum(masses[selectedIDs])
  fig=plt.figure(figsize=g_squareFigSize)
  ax=fig.add_subplot(111)
  ax.plot(times, totMasses)
  ax.set_xlabel(g_axisLabelTime)
  ax.set_ylabel(r"$M_{"+axisLabeltext+"}(t)$ ([M])")
  graphInterface(fig=fig, ax=ax, title=g_massEvolutionGraphTitle+titleText)
  return(returnText)

def plotRadialDistributionEvolution(rdatX, datV, times, title, snaps) :
  # Plots radial or velocities distributions.
  # @param rdatX a set of positions recentered around their centroid (array 3 * N * T)
  # @param datV a set of velocities (array 3 * N * T)
  # @param times a set of times (array T)
  # @param title a parameter toggling the type of plot (can be either "radial" or "velocities")
  # @param snaps a set of timestamps indexes
  # @return a code for figure saving
  if not (title=="radial" or title=="velocity") : criticalBadValue("title")
  nbSnapshots=np.size(rdatX, axis=2)
  choosenSnapshots=(np.array(snaps)*nbSnapshots).astype(int)
  nbSamples=int(0.15*np.size(rdatX, axis=1))
  colors=getCMapColorGradient(len(choosenSnapshots))
  nbSelectedSnapshots=len(choosenSnapshots)
  if title=="radial" : selectedData=rdatX[:, :, choosenSnapshots]
  elif title=="velocity" : selectedData=removeGlobalSpeed(datV)[:, :, choosenSnapshots] # remember to remove the global speed of the cluster
  radii=np.linalg.norm(selectedData, axis=0)
  
  if title=="radial" : unit="[L]"
  elif title=="velocity" : unit="[V]"
  plotRange=userInputGetNumber("Enter the wanted range of the graph or -1 for maximum range ("+unit+" dimension, >="+"{:1.2f}".format(np.min(radii))+").",
                               minAccepVal=np.min(radii),
                               maxAccepVal=np.inf,
                               specials=["-1"])
  if plotRange=="-1" :
    rightCut=np.max(radii)
    returnText="range_max"
    titleText=""
  else :
    rightCut=float(plotRange)
    returnText="range_"+str(plotRange).replace(".", ",")
    if(title=="radial") :
      titleText=r", and cut at $r="+str(plotRange)+"$ [L]"
    else :
      titleText=r", and cut at $\left\|\vec{V}\right\|_2="+str(plotRange)+"$ [V]"
  
  legends=(list(), list())
  fig=plt.figure(figsize=g_squareFigSize)
  ax=fig.add_subplot(111)
  for i in range(0, nbSelectedSnapshots) :
    if title=="radial" :
      bshape="auto"
      completeRadialHist=radialHistogram(radii[:, i], nbSamples, np.min(radii), rightCut, binsShape=bshape)
    elif title=="velocity" :
      completeRadialHist=np.histogram(radii[:, i], bins=np.linspace(min(radii[:, i]), rightCut, nbSamples+1), range=(np.min(radii), rightCut))
    #xHist=np.linspace(completeRadialHist[1][0], completeRadialHist[1][-1], nbSamples)
    if title=="radial" :
      # Do a linear spline aDjustement to make it pretty.
      zerosIDs=np.where(completeRadialHist[0]<1e-5)[0] # find where histogram is 0
      notConsideredIDs=zerosIDs[np.where(zerosIDs<0.5*nbSamples)[0]] # IDs not considered are where histogram is 0 too close to r=0
      consideredIDs=np.setdiff1d(np.arange(0, nbSamples), notConsideredIDs) # considered IDs are complementary set
      
#      highWeightsIDs=zerosIDs[np.where(zerosIDs>0.75*nbSamples)[0]] # high weights should be put at last values, more important
#      w=np.ones(np.size(completeRadialHist[0])) # rest of weights are ones
#      w[highWeightsIDs]=np.linspace(w[highWeightsIDs[0]-1], 1e9, np.size(highWeightsIDs)) # high weights should be put at last values, more important
      xValues=np.hstack(([0], 0.5*(completeRadialHist[1][consideredIDs]+completeRadialHist[1][consideredIDs+1])[1:])) # better xvalues for histogram
#      #spline=splrep(completeRadialHist[1][consideredIDs], completeRadialHist[0][consideredIDs], w[consideredIDs], k=1, s=1e4) # compute spline
#      spline=splrep(xValues, completeRadialHist[0][consideredIDs], w[consideredIDs], k=1, s=0) # compute spline
#      xPlot=np.hstack(([0], np.logspace(np.log10(0.1), np.log10(rightCut), 1000))) # plotting axis
#      yPlot=splev(xPlot, spline)
      
      p, =ax.plot(xValues, completeRadialHist[0][consideredIDs], color=colors[i], markersize=2) # plot values as dots
      #ax.plot(xValues, completeRadialHist[0][consideredIDs], '.', color=colors[i], markersize=2) # plot values as dots
      #p, =ax.plot(xPlot, yPlot, color=colors[i]) # plot normalised spline
      ax.set_xscale('symlog')
      ax.set_yscale('symlog')
    elif title=="velocity" :
      #xValues=0.5*(completeRadialHist[1][1:]+completeRadialHist[1][:-1])
      xValues=completeRadialHist[1][:-1]
      p, =ax.plot(xValues, completeRadialHist[0]/simps(completeRadialHist[0], xValues), color=colors[i])
    legends[0].append(r'$t='+'{:6.2f}'.format(times[choosenSnapshots[i]])+'$ ([T])')
    legends[1].append(p)
  ax.legend(legends[1], legends[0], loc='best')
  if title=="radial" : ax.set_xlabel(r'$r$ ([L])'); ax.set_ylabel("computed "+title+r" density (stars per [L$^3$])")
  elif title=="velocity" : ax.set_xlabel(r"$\left\|\vec{V}\right\|_2$ ([V])"); ax.set_ylabel("computed "+title+" density")
  plt.xlim([np.min(radii), rightCut])
  graphInterface(fig=fig, ax=ax, title="Evolution of the Normalised computed "+title+" Densities\n(recentered around centroid"+titleText+")")
  return(returnText)

def rewriteInput(datX, rdatX, datV, masses, outputFileName, showPointsOutside=False) :
  # Rewrite input by rejecting particles that are too far.
  # @param datX a set of positions (array 3 * N * T)
  # @param rdatX a set of positions recentered around their centroid (array 3 * N * T)
  # @param datV a set of velocities (array 3 * N * T)
  # @param masses a set of masses (array N)
  # @param outputFileName the current output file name
  # @param showPointsOutside (optionnal) encodes whether or not to plot selected/rejected particles (True if yes, False if no, default if False)
  N=np.size(datX, axis=1)
  snapshotID=userInputTestIfDigit(userInput("Which snapshot to rewrite (>=0, <="+str(np.size(datX, axis=2)-1)+") ?"), 0, np.size(datX, axis=2)-1)
  initialState=datX[:, :, snapshotID]
  initialVels=datV[:, :, snapshotID]
  initialStateRecentered=rdatX[:, :, snapshotID]
  selectedIDs=selectLimitationRadius(initialStateRecentered, showPointsOutside)
  X=initialState[:, selectedIDs]
  V=initialVels[:, selectedIDs]
  M=ensureLineVect(masses[selectedIDs], np.size(selectedIDs))
  print("Current output file name : "+outputFileName+".")
  newInputFileName=userInput("New input file name (without \".in\") ?")
  simulationDuration=userInputTestIfNumber(userInput("Simulation duration (>="+"{:5.2f}".format(1)+", <=inf) ?"),
                                           1,
                                           np.inf)
  integrationTimeStep=userInputTestIfNumber(userInput("Integration timestep (>="+"{:7.1e}".format(1e-8)+", <="+"{:7.1e}".format(1e-2)+") ?"),
                                            1e-8,
                                            1e-2)
  outputPrintInterval=userInputTestIfNumber(userInput("Output printing interval (>="+"{:5.2f}".format(0.1)+", <="+"{:5.2f}".format(simulationDuration/2)+") ?"),
                                            0.1,
                                            simulationDuration/2)
  technique=userInput("Technique to use (RK4, PECE or PEC) ?")
  if(not(technique in ["RK4", "PECE", "PEC"])) : criticalBadValue("technique")
  outputFilePath="./"+newInputFileName+"_rewritten_NS="+str(N)+"_TMax="+str(simulationDuration)+"_dt="+str(integrationTimeStep)+"_printInt="+str(outputPrintInterval)+".out"
  from scipy.spatial.distance import cdist
  tmp_d=cdist(np.transpose(X), np.transpose(X))
  softening=np.min([g_maxSoftening, np.min(tmp_d[np.nonzero(tmp_d)])/100]) # update softening just in case
  
  printExplanationsInInputFile=False
  from galaxyParameters import g_galaxyModelNames, generateGalaxyModel
  (GCPos, GCVel, g_Mb, g_ab, g_Md, g_ad, g_hd, g_Vh, g_ah)=generateGalaxyModel(userInput("Galaxy model ("+g_galaxyModelNames+") ?"))
  if(GCPos==-1 or GCVel==-1) : galacticEnvironnement=0
  else : galacticEnvironnement=1
  galaxyParams=((g_Mb, g_ab), (g_Md, g_ad, g_hd), (g_Vh, g_ah))
  exportInitialConditionsToFile("./"+newInputFileName+".in",
                                X, V, M,
                                simulationDuration,
                                integrationTimeStep,
                                outputPrintInterval,
                                outputFilePath,
                                technique,
                                softening,
                                galacticEnvironnement,
                                printExplanationsInInputFile,
                                galaxyParams,
                                (g_GravConstant, g_scalingL, g_scalingM, g_scalingT, g_scalingV))

def selectLimitationRadiusInterface(rdat) :
  # Provides an interface for the user to find the limiting radius he wants.
  # @param rdat a set of positions recentered around their centroid (array 3 * N * T)
  # @return none
  snapshotID=userInputTestIfDigit(userInput("Which snapshot to use (>=0, <="+str(np.size(rdat, axis=2)-1)+") ?"), 0, np.size(rdat, axis=2)-1)
  selectLimitationRadius(rdat[:, :, snapshotID])

# Debug. ######################################################
def computeSmallestDistanceBetweenHighVelocityVariatingStars(datX, datV, timestamp=1, number=2) :
  # Compute the distance to the nearest star for stars with highest velocity module variation at a certain time.
  # @param datX a set of positions (array 3 * N * T)
  # @param datV a set of velocities (array 3 * N * T)
  # @param timestamp (optionnal) a timestamp index (should be >0, and less than the maximum number of timestamps) at which detect high velocities variations (default is 1)
  # @param number (optionnal) the number of stars to find (default is 2)
  # @return the distance to the nearest star for stars with highest velocity module variation at a certain time
  if timestamp<=0 or timestamp>=np.size(datV, axis=2) : criticalBadValue("timestamp")
  datV=removeGlobalSpeed(datV) # subtract global velocity of the system
  sIndexes=findExtremeDeltaSpeedStarsIDs(datV, timestamp, number)[0]
  selectedPositions=datX[:, sIndexes, timestamp-1]
  return(np.min(cdist(np.transpose(datX[:, :, timestamp]), np.transpose(selectedPositions)), axis=0))

def findExtremeDeltaSpeedStarsIDs(datV, timestamp=1, numberOfExtremesToReturn=2) :
  # Detects indexes of stars that have the biggest variation of speed at a given timestamp.
  # @param datV a set of velocities (array 3 * N * T)
  # @param timestamp (optionnal) a timestamp index (should be >0, and less than the maximum number of timestamps) at which detect high velocities variations (default is 1)
  # @param numberOfExtremesToReturn (optionnal) the number of indexes to return (default is 2)
  # @return indexes of the stars having the biggest variation of speed at the given timestamp and their velocities variations
  if timestamp<=0 or timestamp>=np.size(datV, axis=2) : criticalBadValue("timestamp")
  N=np.size(datV, axis=1)
  VMods=np.zeros([2, N])
  for i in range(0, N) :
    VMods[0, i]=np.linalg.norm(datV[:, i, timestamp-1], axis=0)
    VMods[1, i]=np.linalg.norm(datV[:, i, timestamp], axis=0)
  dV=(VMods[1, :]-VMods[0, :])/VMods[0, :]
  sortedIndexes=np.argsort(dV)
  selectedSortedIndexes=sortedIndexes[-numberOfExtremesToReturn:len(sortedIndexes)]
  return(selectedSortedIndexes, dV[selectedSortedIndexes])

def findExtremeDeltaSpeedStarsIDsInterface(datV) :
  # Provides an interface for the finding of stars that have the biggest variation of speed.
  # @param datV a set of velocities (array 3 * N * T)
  # @return indexes of the stars having the biggest variation of speed at the chosen timestamp and their velocities variations
  tim=userInputTestIfDigit(userInput("Wanted timestamp (>0, <"+str(np.size(datV, axis=2))+") ?"),
                           1,
                           np.size(datV, axis=2)-1)
  num=userInputTestIfDigit(userInput("Wanted number of extreme speed stars (>0, <"+str(np.size(datV, axis=1))+") ?"),
                           1,
                           np.size(datV, axis=1)-1)
  return(findExtremeDeltaSpeedStarsIDs(datV, tim, num))
###############################################################