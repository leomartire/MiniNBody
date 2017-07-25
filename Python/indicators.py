from util import deviationFromSphericity
from util import g_fullpageFigSize
from util import getAllAngMomenta
from util import getAllMomenta
from util import getCMapColorGradient
from util import getIDsInsideRadius
from util import inertiaTensor
from util import g_verbose
from util import graphUpdateAxisLabels
from util import extractAnisotropyBeta
from util import g_axisLabelTime
from util import removeGlobalSpeed
import matplotlib.pyplot as plt
from util import getDiskCrossingTimesCPSaved
import numpy as np
import pickle
from util import g_diagnosticsFolder
import os
import time

g_sigmaSecurity=3

# Indicators. #################################################
def velocityIsotropyDeviation(X, V, masses) :
  # Compute velocities' modules' deviation from isotropy.
  # @param X a set of positions recentered on their centroid (array 3 * N, in [L])
  # @param V a set of velocities recentered on their centroid (array 3 * N, in [V])
  # @param M a set of masses (array N, in [M])
  # @return the velocities' modules' deviation from isotropy
  dat=V # (velocities of stars 3 * N)
  IEig=np.linalg.eig(inertiaTensor(dat, masses))
  val=deviationFromSphericity(dat, masses, IEig)
  return(val)
def velocityMean(X, V, masses) :
  # Compute velocities' modules' mean.
  # @param X a set of positions recentered on their centroid (array 3 * N, in [L])
  # @param V a set of velocities recentered on their centroid (array 3 * N, in [V])
  # @param M a set of masses (array N, in [M])
  # @return the velocities' modules' mean
  val=np.mean(np.linalg.norm(V, axis=0))
  return(val)
def velocityDispersion(X, V, masses) :
  # Compute velocities' modules' dispersion.
  # @param X a set of positions recentered on their centroid (array 3 * N, in [L])
  # @param V a set of velocities recentered on their centroid (array 3 * N, in [V])
  # @param M a set of masses (array N, in [M])
  # @return the velocities' modules' dispersion
  val=np.var(np.linalg.norm(V, axis=0), ddof=1)**0.5
  return(val)
def positionSphericityDeviation(X, V, masses) :
  # Compute positions' deviation from sphericity.
  # @param X a set of positions recentered on their centroid (array 3 * N, in [L])
  # @param V a set of velocities recentered on their centroid (array 3 * N, in [V])
  # @param M a set of masses (array N, in [M])
  # @return the positions' deviation from sphericity
  dat=X # (positions of stars 3 * N)
  IEig=np.linalg.eig(inertiaTensor(dat, masses))
  val=deviationFromSphericity(dat, masses, IEig)
  return(val)
#def rotationFactor(X, V, masses) :
#  # Compute the rotation factor ("deviation from sphericity" when using angular momenta instead of positions).
#  # @param X a set of positions recentered on their centroid (array 3 * N, in [L])
#  # @param V a set of velocities recentered on their centroid (array 3 * N, in [V])
#  # @param M a set of masses (array N, in [M])
#  # @return the velocities' modules' deviation from isotropy
#  dat=getAllAngMomenta(X, getAllMomenta(V, masses)) # (angular momentum of stars 3 * N)
#  IEig=np.linalg.eig(inertiaTensor(dat, masses))
#  val=deviationFromSphericity(dat, masses, IEig)
#  return(val)
def angularMomentum(X, V, masses) :
  # Compute the system's total angular momentum.
  # @param X a set of positions recentered on their centroid (array 3 * N, in [L])
  # @param V a set of velocities recentered on their centroid (array 3 * N, in [V])
  # @param M a set of masses (array N, in [M])
  # @return the system's total angular momentum
  N=np.size(masses)
  angMomenta=getAllAngMomenta(X, getAllMomenta(V, masses))
  totAngMomentum=np.zeros(3)
  for i in range(0, N) :
    totAngMomentum+=angMomenta[:, i]
  val=np.linalg.norm(totAngMomentum)
  return(val)
def anisotropy(X, V, masses) :
  val=extractAnisotropyBeta(X, V)
  return(val)
###############################################################

# Diagnostic functions. #######################################
def testActive(val, mean, var, sec=g_sigmaSecurity) :
  # Test if an indicator is active or not.
  # @param val the value to test
  # @param mean the mean value of the indicator for the isolated simulation
  # @param var the standard deviation value of the indicator for the isolated simulation
  # @param sec the number of times sigma to each side of the mean should be considered for the confidence interval
  # @return False if the indicator is not active, the location in terms of sigma (as a string) of the value if the indicator is active
  if(val>mean+sec*var) : return("+"+"{:6.2f}".format((val-mean)/var)+"*s")
  elif(val<mean-sec*var) : return("-"+"{:6.2f}".format(-(val-mean)/var)+"*s")
  else : return(False)
def callIndicators(X, V, M, verbose=True) :
  # Test the indicators on a system state.
  # @param X a set of positions recentered on their centroid (array 3 * N, in [L])
  # @param V a set of velocities recentered on their centroid (array 3 * N, in [V])
  # @param M a set of masses (array N, in [M])
  # @param verbose should the informations be printed ? (True if yes, False if no, default is True)
  # @param g_indicators (global) the list of indicators and their needed informations
  # @return all indicators values
  indVals=np.zeros(len(g_indicators))
  for i, ind in enumerate(g_indicators) :
    indVals[i]=ind[0](X, V, M)
  if(verbose) :
    for i, indV in enumerate(indVals) :
      m=g_indicators[i][1][0]
      v=g_indicators[i][1][1]
      active=testActive(indV, m, v)
      if(not(active)) :
        act="not active        "
      else :
        
        act="active @ "+active
      print("  I"+str(i+1).zfill(2)+"  = "+
              "{:10.3e}".format(indV)+" "+
              "["+act+"] "+
              "("+g_indicators[i][-2]+")")
  return(indVals)
###############################################################

# Setting of the global variable g_indicators. ################
# function, (mean, var), description
vIsoDev=(velocityIsotropyDeviation,
         #(1.1919293173113601, 0.022450129498805909), # taken on [150, 200]
         #(1.1376586215871367, 0.064530195750148442), # taken on [150, 455]
         (1.1215146296800433, 0.059884048871380831), # taken on [200, 521]
         "log",
         "deviation from velocity isotropy",
         "Deviation from Velocity Isotropy")
vMean  =(velocityMean,
         #(1.5544580558217849, 0.031413687354570216), # taken on [150, 200]
         #(1.3781015649872461, 0.10611110722295641), # taken on [150, 455]
         (1.3178370409361477, 0.095599175250410506), # taken on [200, 521]
         "log",
         "velocity distribution mean",
         "Velocity Distribution Mean")
angMom =(angularMomentum,
         #(177.03354306775262, 79.693180058382481), # taken on [0, 200]
         #(174.45184796478489, 61.400605664300294), # taken on [150, 455]
         (161.13406968945358, 62.026570417839466), # taken on [200, 521]
         "log",
         "angular momentum norm",
         "Angular Momentum Norm") # in [M][L^{2}][T^{-1}]
#rFact  =(rotationFactor,
#         #(1.4059280380338171, 0.031081000248521444), # taken on [150, 200]
#         (1.2931629205986104, 0.082524764292573966), # taken on [150, 455]
#         "normal",
#         "rotation factor",
#         "Rotation Factor")
anisot =(anisotropy,
         #(-0.14895431243687762, 0.082694042018675459), # taken on [150, 455]
         (-0.16589417931066958, 0.080639061420233457), # taken on [200, 521]
         "normal",
         "anisotropy parameter",
         "Anisotropy parameter")
vDisp  =(velocityDispersion,
         #(0.82781767929661276, 0.031114037661443099), # taken on [150, 200]
         #(0.73769885496617271, 0.089472966713063096), # taken on [150, 455]
         (0.69764351853593176, 0.085096518204197641), # taken on [200, 521]
         "normal",
         "velocity distribution dispersion",
         "Velocity Distribution Dispersion")
pSphDev=(positionSphericityDeviation,
         #(1.4953314437983218, 0.032808856856127909), # taken on [150, 200]
         #(1.3974614128627858, 0.10911903364214594), # taken on [150, 455]
         #(1, 0.10911903364214594), # taken on [150, 455], cheated
         (1.3563403805588237, 0.08974131538620389), # taken on [200, 521]
         #(1, 0.08974131538620389), # taken on [200, 521], cheated
         "normal",
         "deviation from sphericity",
         "Deviation from Sphericity") # in [V]
g_indicators=(vIsoDev, vMean, angMom, anisot, vDisp, pSphDev)
###############################################################

# Normal usage. ###############################################
#X=0 # get X data
#V=0 # get V data
#M=0 # get M data
#p=(X, V, M)
#callIndicators(*p)
###############################################################

# Specific for tests on simulations. ##########################
#def getIdealCrossInd(crossingTimes, times, forwardTimeSpan=3, backwardsTimeSpan=1) :
#  T=np.size(times)
#  idealCrossInd=np.zeros(T)
#  crossIds=list()
#  tmpFwSets=list()
#  tmpBwSets=list()
#  for ct in crossingTimes :
#    tid=np.argmin(np.abs(times-ct))
#    crossIds.append(tid)
#    tmpFwSets.append(np.intersect1d(np.where(times>times[tid])[0], np.where(times<times[tid]+forwardTimeSpan)[0]))
#    tmpBwSets.append(np.intersect1d(np.where(times<times[tid])[0], np.where(times>times[tid]-backwardsTimeSpan)[0]))
#  for i in range(0, len(crossingTimes)) :
#    croId=np.union1d(np.union1d(tmpBwSets[i], [crossIds[i]]), tmpFwSets[i])
#    cro=np.zeros(len(croId))
#    cro[0:len(tmpBwSets[i])+1]=np.linspace(0, 1, len(tmpBwSets[i])+1)
#    cro[len(tmpBwSets[i]):]=np.linspace(1, 0, len(tmpFwSets[i])+1)
#    idealCrossInd[croId]+=cro
#  idealCrossInd[np.where(idealCrossInd>1)[0]]=1
#  idealCrossInd=2*idealCrossInd
#  return(idealCrossInd)

def callIndicatorsSimulation(g_outputFileName, rdataX, dataV, masses, times, timestep) :
  # Call the function callIndicators on a certain timestep of a simulation.
  # @param g_outputFileName the output file name read
  # @param rdataX the whole set of positions recentered on centroid of the simulation (array 3 * N * T)
  # @param dataV the whole set of velocities of the simulation (array 3 * N * T)
  # @param masses the whole set of masses of the simulation (array N)
  # @param timestep the wanted timestep
  # @return none
  (sim, crossingTimes, rL)=initialise(g_outputFileName, times)
  sel=getIDsInsideRadius(rdataX[:, :, timestep], rL, [0, 0, 0])[0]
  callIndicators(rdataX[:, sel, timestep], removeGlobalSpeed(dataV)[:, sel, timestep], masses[sel])

def getMeanAndVar(indValsOverTime, i, times, t0=150, t1=200) :
  # Compute the mean and standard deviation of an indicator on a specified time period.
  # Remark : to be used only on the control simulation.
  # @param indValsOverTime the computed indicators' values over the whole control simulation
  # @param i the index of the wanted indicator
  # @param times the whole set of times of the control simulation
  # @param t0 the lower boundary of the time period
  # @param t1 the higher boundary of the time period
  # @return the mean and standard deviation of the indicator on the specified time period
  t0id=np.argmin(np.abs(times-t0))
  t1id=np.argmin(np.abs(times-t1))
  sample=indValsOverTime[i, t0id:t1id]
  return(np.mean(sample), np.var(sample, ddof=1)**0.5)

def getMeansAndVarsPostComputation(indValsOverTime, times, t0=150, t1=200) :
  # Print the mean and standard deviation of all indicators on a specified time period, when their values are already computed.
  # Remark : to be used only on the control simulation.
  # @param indValsOverTime the computed indicators' values over the whole control simulation
  # @param times the whole set of times of the control simulation
  # @param t0 the lower boundary of the time period
  # @param t1 the higher boundary of the time period
  # @return none
  for i in range(0, np.size(indValsOverTime, axis=0)) :
    print("I"+str(i+1).zfill(2)+" : ", end="")
    print(getMeanAndVar(indValsOverTime, i, times, t0, t1), end="")
    print(" ("+g_indicators[i][-2]+")")

def getMeansAndVars(g_outputFileName, rdataX, dataV, masses, times, t0=150, t1=200) :
  # Print the mean and standard deviation of all indicators on a specified time period, when their values are not already computed.
  # Remark : to be used only on the control simulation.
  # @param g_outputFileName the output file name read
  # @param rdataX the whole set of positions recentered on centroid of the simulation (array 3 * N * T)
  # @param dataV the whole set of velocities of the simulation (array 3 * N * T)
  # @param masses the whole set of masses of the simulation (array N)
  # @param times the whole set of times of the control simulation
  # @param t0 the lower boundary of the time period
  # @param t1 the higher boundary of the time period
  # @return none
  (sim, crossingTimes, rL)=initialise(g_outputFileName, times)
  if(not("0" in sim)) :
    print("Please use control simulation.")
    raise ValueError
  if(g_verbose>0) : print("Calculating confidence intervals for indicators (between t="+str(t0)+" and t="+str(t1)+").")
  indValsOverTime=getIndicatorsOverTime(g_outputFileName, rdataX, dataV, masses, times)
  getMeansAndVarsPostComputation(indValsOverTime, times, t0, t1)

def getIndicatorsOverTime(g_outputFileName, rdataX, dataV, masses, times) :
  # Compute all indicators' values over time on a simulation.
  # @param g_outputFileName the output file name read
  # @param rdataX the whole set of positions recentered on centroid of the simulation (array 3 * N * T)
  # @param dataV the whole set of velocities of the simulation (array 3 * N * T)
  # @param masses the whole set of masses of the simulation (array N)
  # @param times the whole set of times of the control simulation
  # @return all indicators' values over time on the simulation
  (sim, crossingTimes, rL)=initialise(g_outputFileName, times)
  T=np.size(times)
  indValsOverTime=np.zeros((len(g_indicators), T))
  print("Calculating "+str(len(g_indicators))+" indicators over "+str(T)+" timesteps.")
  for t in range(0, T) :
    sel=getIDsInsideRadius(rdataX[:, :, t], rL, [0, 0, 0])[0]
    testX=rdataX[:, sel, t]
    testV=removeGlobalSpeed(dataV)[:, sel, t]
    testM=masses[sel]
    p=(testX, testV, testM)
    indValues=callIndicators(*p, verbose=False)
    for i, indV in enumerate(indValues) :
      indValsOverTime[i, t]=indV
  return(indValsOverTime)

def initialise(g_outputFileName, times) :
  # Extract simulation name and crossing times from an output file name. Also specify a selecting radius.
  # @param g_outputFileName the output file name read
  # @return the simulation's name, its crossing times and a limiting radius
  r=25 # in [L]
  sim=-1
  #cPFile=("./"+g_diagnosticsFolder+"/"+time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(os.path.getmtime(g_outputFileName)))+"("+g_outputFileName.split("/")[-1]+")/centroidPositionData.pckl").replace("=", "")
  if("S0" in g_outputFileName) : sim="S0"
  elif("S1" in g_outputFileName) : sim="S1"
  elif("S2" in g_outputFileName) : sim="S2"
  elif("S3" in g_outputFileName) : sim="S3"
  elif("S4" in g_outputFileName) : sim="S4"
  if("Long" in g_outputFileName) : sim=sim+"L"
  if(sim in ["S1", "S1S", "S1L", "S2", "S2S", "S3", "S4"]) :
    crossingTimes=getDiskCrossingTimesCPSaved(pickle.load(open(("./"+g_diagnosticsFolder+"/"+time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(os.path.getmtime(g_outputFileName)))+"("+g_outputFileName.split("/")[-1]+")/centroidPositionData.pckl").replace("=", ""), "rb")), times)[1]
#  if(sim=="S1") : crossingTimes=[43, 123]
#  elif(sim=="S1L") : crossingTimes=[43, 123, 360]
#  elif(sim=="S2") : crossingTimes=[24, 52, 156, 173]
#  elif(sim=="S3") : crossingTimes=[45.00]
#  elif(sim=="S4") : crossingTimes=[19.13]
  else : crossingTimes=[]
  return(sim, crossingTimes, r)

def plotIndicators(g_outputFileName, times, indValsOverTime, separatePlots=False, titles=False, ylabels=True) :
  # Plot indicators values over time of a simulation.
  # @param g_outputFileName the output file name read
  # @param times the whole set of times of the control simulation
  # @param indValsOverTime the computed indicators' values over the simulation
  # @return none
  (sim, crossingTimes, rL)=initialise(g_outputFileName, times)
  colors=getCMapColorGradient(len(g_indicators)+1, name="brg")
  if(not(separatePlots)) : fig=plt.figure(figsize=g_fullpageFigSize)
  nCols=2
  nRows=int(np.ceil(len(g_indicators)/nCols))
  for i in range(0, len(g_indicators)) :
    if(separatePlots) :
      fig=plt.figure(figsize=g_fullpageFigSize)
      ax=fig.add_subplot(111)
    else :
      ax=fig.add_subplot(nRows, nCols, i+1)
    if(g_indicators[i][2]=="log") : # prefer semilogy plot
      ax.semilogy(times, indValsOverTime[i, :], color=colors[i])
    else :
      ax.plot(times, indValsOverTime[i, :], color=colors[i])
    ax.plot(times, indValsOverTime[i, :], color=colors[i])
    if(ylabels) : ax.set_ylabel(r"$\mathbb{I}_"+str(i+1)+r"$")
    ax.set_xlabel(g_axisLabelTime)
    ax.axhline(g_indicators[i][1][0], linestyle="--", color=colors[i])
    ax.axhline(g_indicators[i][1][0]+g_sigmaSecurity*g_indicators[i][1][1], linestyle=":", color=colors[i])
    ax.axhline(g_indicators[i][1][0]-g_sigmaSecurity*g_indicators[i][1][1], linestyle=":", color=colors[i])
    ax.set_xlim([times[0], times[-1]])
    for ct in crossingTimes : ax.axvline(ct, linestyle=':', color="k")
    graphUpdateAxisLabels(ax)
    if(titles) : ax.set_title(g_indicators[i][-1])
    if(separatePlots) : fig.tight_layout()
  if(not(separatePlots)): fig.tight_layout()

def testIndicatorsInterface(g_outputFileName, rdataX, dataV, masses, times) :
  # Provide a testing interface to this indicators' module.
  # @param g_outputFileName the output file name read
  # @param rdataX the whole set of positions recentered on centroid of the simulation (array 3 * N * T)
  # @param dataV the whole set of velocities of the simulation (array 3 * N * T)
  # @param masses the whole set of masses of the simulation (array N)
  # @param times the whole set of times of the control simulation
  # @return a code for figure saving
  (sim, crossingTimes, rL)=initialise(g_outputFileName, times)
  plotIndicators(g_outputFileName,
                 times,
                 getIndicatorsOverTime(g_outputFileName,
                                       rdataX,
                                       dataV,
                                       masses,
                                       times))
  for i, ct in enumerate(crossingTimes) :
    tid=np.argmin(np.abs(times-ct))
    sel=getIDsInsideRadius(rdataX[:, :, tid], rL, [0, 0, 0])[0]
    testX=rdataX[:, sel, tid]
    testV=removeGlobalSpeed(dataV)[:, sel, tid]
    testM=masses[sel]
    print("Crossing n°"+str(i+1)+" (t="+"{:7.2f}".format(ct)+" [T]) : call to indicator function.")
    callIndicators(testX, testV, testM)
    print("")
  return("_range_"+str(float(rL)).replace(".", ","))
###############################################################