from JPLData import JPLX
from util import criticalBadValue
from util import criticalError
from util import g_GravConstant
from util import g_oneParsec
from util import g_squareFigSize
from util import g_scalingV
from util import normalise
from util import vectorEvolution
from util import g_solarSystemOutputFolder
from util import g_axisLabelTime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

def show_custom_VAR(cv, figname="") :
  # Plots the PDF, CDF and  PPF of a beforehand declared random variable (subclass of stats.rv_continuous).
  # @param cv the custom random variable (subclass of stats.rv_continuous)
  # @param figname a figure name to save the obtained plot (facultative)
  # @return none
  x=np.linspace(0, 8, 1000)
  fig=plt.figure(figsize=g_squareFigSize)
  ax=fig.add_subplot(111)
  ax.set_xlim([-0.1, 4.1])
  ax.set_ylim([-0.1, 2.1])
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  plt.title("Custom random Variable obtained from the radial Distribution")
  plt.plot(x, cv.pdf(x)) # show the probability distribution function
  plt.plot(x, cv.cdf(x)) # show the cumulative distribution function
  plt.plot(x, cv.ppf(x)) # show the percent point function (inverse of cdf)
  ax.legend(["PDF", "CDF", "PPF"], loc="best")
  plt.tight_layout()
  if figname!="" :
    plt.savefig(figname)

#def show_SemiNormalDistribution(s, figname="") :
#  # Plots an illustration of the PDF of the semi-normal distribution.
#  # @param s the variance of the semi-normal distribution
#  # @param figname a figure name to save the obtained plot (facultative)
#  # @return none
#  x=np.linspace(-1, 4, 1000)
#  pdf=((2**0.5)/(s*np.pi**0.5))*np.exp(-(x**2)/(2*s**2))*(x>0)
#  fig=plt.figure(figsize=g_squareFigSize)
#  ax=fig.add_subplot(111)
#  ax.set_ylim([-0.1, 1])
#  plt.title("Semi-Normal Distribution")
#  plt.plot(x, pdf, "k") # show the probability distribution function
#  plt.plot(s*np.ones(2), np.linspace(-0.05, 0.95, 2), "g")
#  plt.plot(s*(2/(np.pi-2))**0.5*np.ones(2), np.linspace(-0.05, 0.95, 2), "r")
#  plt.plot(2*s*np.ones(2), np.linspace(-0.05, 0.95, 2), ":b")
#  plt.plot(3*s*np.ones(2), np.linspace(-0.05, 0.95, 2), ":c")
#  plt.legend(["PDF", r"$\sigma$", r"$\mathbb{E}(X)$", r"$2\sigma$", r"$3\sigma$"], loc="best")
#  if figname!="" :
#    plt.savefig(figname)

def PlummerPotential(X, Z, m, a) :
  # Compute the value of a Plummer type potential.
  # @param X radial coordinate
  # @param Z altitude coordinate
  # @param m first parameter of the Plummer type potential
  # @param a second parameter of the Plummer type potential
  # @return the value of the Plummer type potential
  P=-g_GravConstant*m*(X**2+Z**2+a**2)**(-0.5)
  return(P)

def MiyamotoNagaiPotential(X, Z, m, a, h) :
  # Compute the value of a Miyamoto-Nagai type potential.
  # @param X radial coordinate
  # @param Z altitude coordinate
  # @param m first parameter of the Miyamoto-Nagai type potential
  # @param a second parameter of the Miyamoto-Nagai type potential
  # @param h second parameter of the Miyamoto-Nagai type potential
  # @return the value of the Miyamoto-Nagai type potential
  P=-g_GravConstant*m*(X**2+(a+(Z**2+h**2)**0.5)**2)**(-0.5)
  return(P)

def DarkMatterHaloPotential(X, Z, v, a) :
  # Compute the value of a dark matter halo potential.
  # @param X radial coordinate
  # @param Z altitude coordinate
  # @param v first parameter of the dark matter halo potential
  # @param a second parameter of the dark matter halo potential
  # @return the value of the dark matter halo potential
  P=-0.5*v*np.log(X**2+Z**2+a**2)
  return(P)

def plotGalacticPotentials(xrange, yrange, mb, ab, md, ad, hd, vh, ah) :
  # Plot all galactic potentials (Plummer type for the bulge, Miyamoto-Nagai for the disk and a dark matter halo).
  # @param xrange plotting x-axis (radial) range
  # @param yrange plotting y-axis (altitude) range
  # @param mb first parameter for the bulge (Plummer)
  # @param ab second parameter for the bulge (Plummer)
  # @param md first parameter for the disk (Miyamoto-Nagai)
  # @param ad second parameter for the disk (Miyamoto-Nagai)
  # @param hd third parameter for the disk (Miyamoto-Nagai)
  # @param vh first parameter for the dark matter halo
  # @param ah second parameter for the dark matter halo
  # @return none
  showAxis=True
  (X, Z)=np.meshgrid(np.linspace(-xrange, xrange, 500), np.linspace(-yrange, yrange, 500))
  PB=PlummerPotential(X, Z, mb, ab)
  PD=MiyamotoNagaiPotential(X, Z, md, ad, hd)
  PH=DarkMatterHaloPotential(X, Z, vh, ah)
  fig=plt.figure(figsize=(8, 8))
  ax=fig.add_subplot(111)
  matplotlib.rcParams["contour.negative_linestyle"]="solid"
  CH=plt.contour(X, Z, PH, 8, colors="grey")
  CD=plt.contour(X, Z, PD, 8, colors="r")
  CB=plt.contour(X, Z, PB, 8, colors="b")
  plt.clabel(CH, inline=1, fontsize=8)
  plt.clabel(CD, inline=1, fontsize=8)
  plt.clabel(CB, inline=1, fontsize=8)
  lines=[CB.collections[1], CD.collections[1], CH.collections[1]]
  labels=["bulge", "disk", "dark matter"]
  plt.legend(lines, labels)
  plt.title("Galactic Potentials")
  ax.get_xaxis().set_visible(showAxis)
  ax.get_yaxis().set_visible(showAxis)

def PlummerDensity(X, Z, m, a) :
  # Compute the value of a Plummer type density (derived from the Plummer type potential with Poisson"s equation).
  # @param X radial coordinate
  # @param Z altitude coordinate
  # @param m first parameter of the Plummer type density
  # @param a second parameter of the Plummer type density
  # @return the value of the Plummer type density
  P=((3*m)/(4*np.pi*a**3))*(1+(X**2+Z**2)/(a**2))**(-5/2)
  return(P)

def MiyamotoNagaiDensity(X, Z, m, a, h) :
  # Compute the value of a Miyamoto-Nagai type density (derived from the Miyamoto-Nagai type potential with Poisson"s equation).
  # @param X radial coordinate
  # @param Z altitude coordinate
  # @param m first parameter of the Miyamoto-Nagai type density
  # @param a second parameter of the Miyamoto-Nagai type density
  # @param h second parameter of the Miyamoto-Nagai type density
  # @return the value of the Miyamoto-Nagai type density
  R2=X**2+Z**2
  P=((m*h**2)/(4*np.pi))*(a*R2+(a+3*(Z**2+h**2)**0.5)*(a+(a+(Z**2+h**2)**0.5))**2)/((Z**2+h**2)**1.5*(R2+(a+(Z**2+h**2)**0.5)**2)**(2.5))
  return(P)

def DarkMatterHaloDensity(X, Z, v, a) :
  # Compute the value of a dark matter halo density (derived from its potential with Poisson"s equation).
  # @param X radial coordinate
  # @param Z altitude coordinate
  # @param v first parameter of the dark matter halo density
  # @param a second parameter of the dark matter halo density
  # @return the value of the dark matter halo density
  R2=X**2+Z**2
  P=-((v**2)/(4*np.pi*g_GravConstant))*(3*a**2+R2)/(a**2+R2)
  return(P)

def plotPlummerDensity(m, a) :
  # Plot the Plummer type density.
  # @param m first parameter of the Plummer type density
  # @param a second parameter of the Plummer type density
  # @return none
  showAxis=True
  xrange=20*1e3
  yrange=xrange
  (X, Z)=np.meshgrid(np.linspace(-xrange, xrange, 500), np.linspace(-yrange, yrange, 500))
  P=PlummerDensity(X, Z, m, a)
  fig=plt.figure(figsize=(6, 6))
  ax=fig.add_subplot(111)
  matplotlib.rcParams["contour.negative_linestyle"]="solid"
  plt.contour(X, Z, P, 8, colors="k")
  plt.title("Plummer Density\n"+r"($a_b$="+"{:3.1f}".format(a/1000)+" kpc, contour lines)")
  ax.get_xaxis().set_visible(showAxis)
  ax.get_yaxis().set_visible(showAxis)

def plotMiyamotoNagaiDensity(m, a, h) :
  # Plot the Miyamoto-Nagai type density.
  # @param m first parameter of the Miyamoto-Nagai type density
  # @param a second parameter of the Miyamoto-Nagai type density
  # @param h second parameter of the Miyamoto-Nagai type density
  # @return none
  showAxis=True
  xrange=20*1e3
  yrange=xrange
  (X, Z)=np.meshgrid(np.linspace(-xrange, xrange, 500), np.linspace(-yrange, yrange, 500))
  P=MiyamotoNagaiDensity(X, Z, m, a, h)
  fig=plt.figure(figsize=(6, 6))
  ax=fig.add_subplot(111)
  matplotlib.rcParams["contour.negative_linestyle"]="solid"
  plt.contour(X, Z, P, 8, colors="k")
  plt.title("Miyamoto-Nagai Density\n"+r"($a_d$="+"{:3.1f}".format(a/1000)+" kpc, $h_d$="+"{:3.1f}".format(h/1000)+" kpc, contour lines)")
  ax.get_xaxis().set_visible(showAxis)
  ax.get_yaxis().set_visible(showAxis)

def plotDarkMatterHaloDensity(v, a) :
  # Plot the dark matter halo density.
  # @param v first parameter of the dark matter halo density
  # @param a second parameter of the dark matter halo density
  # @return none
  showAxis=True
  xrange=20*1e3
  yrange=xrange
  (X, Z)=np.meshgrid(np.linspace(-xrange, xrange, 500), np.linspace(-yrange, yrange, 500))
  P=DarkMatterHaloDensity(X, Z, v, a)
  fig=plt.figure(figsize=g_squareFigSize)
  ax=fig.add_subplot(111)
  matplotlib.rcParams["contour.negative_linestyle"]="solid"
  plt.contour(X, Z, P, 8, colors="k")
  plt.title("Dark Matter Halo Density\n"+r"($V_h$="+"{:3.1f}".format(v/1000)+"km/s, $a_h$="+"{:3.1f}".format(a/1000)+" kpc, contour lines)")
  ax.get_xaxis().set_visible(showAxis)
  ax.get_yaxis().set_visible(showAxis)

def plotGalacticDensity(xrange, yrange, mb, ab, md, ad, hd, vh, ah, plots="all", initPos=False, initPosCoord=-1, initVel=False, initVelCoord=-1) :
  # Plot all galactic densities (Plummer type for the bulge, Miyamoto-Nagai for the disk and a dark matter halo).
  # @param xrange plotting x-axis (radial) range
  # @param yrange plotting y-axis (altitude) range
  # @param mb first parameter for the bulge (Plummer)
  # @param ab second parameter for the bulge (Plummer)
  # @param md first parameter for the disk (Miyamoto-Nagai)
  # @param ad second parameter for the disk (Miyamoto-Nagai)
  # @param hd third parameter for the disk (Miyamoto-Nagai)
  # @param vh first parameter for the dark matter halo
  # @param ah second parameter for the dark matter halo
  # @param plots parameter to choose which densities to plot ("baryonic" for baryonic densities only, "all" for all)
  # @param initPos should a position (of a particle or GC) be plotted too (True if yes, False if no) ?
  # @param initPosCoord position to plot (makes sense only if initPos is set to True)
  # @param initVel should a velocity (of a particle or GC) be plotted too (True if yes, False if no) ?
  # @param initVelCoord velocity to plot (makes sense only if initVel is set to True)
  # @return none
  showAxis=True
  if initPos :
    xrange=np.max([xrange, 1.1*initPosCoord[0, 0], 1.1*initPosCoord[2, 0]])
    yrange=xrange
  if initVel and not initPos :
    criticalError("In galactic density plotting, cannot want initial velocity without initial position.")
  if initVel :
    xrange=np.max([xrange, 1.1*(initPosCoord[0, 0]+normalise(initVelCoord)[0, 0]), 1.1*(initPosCoord[2, 0]+normalise(initVelCoord)[2, 0])])
    yrange=xrange
  (X, Z)=np.meshgrid(np.linspace(-xrange, xrange, 500), np.linspace(-yrange, yrange, 500))
  PB=PlummerDensity(X, Z, mb, ab)
  PD=MiyamotoNagaiDensity(X, Z, md, ad, hd)
  PH=DarkMatterHaloDensity(X, Z, vh, ah)
  fig=plt.figure(figsize=g_squareFigSize)
  #fig=plt.figure(figsize=(24, 6))
  ax=fig.add_subplot(111)
  matplotlib.rcParams["contour.negative_linestyle"]="solid"
  if plots=="baryonic" :
    CB=plt.contour(X, Z, PB+PD, 8, colors="g", levels=np.logspace(np.log10(np.max([np.min(PB+PD), 1e-1])), np.log10(np.max(PB+PD)), 8))
    plt.clabel(CB, inline=1, fontsize=10)
    plt.title("Galactic Baryonic Matter Density\n(contour plot, logarithmic levels)")
  elif plots=="all" :
    CH=plt.contour(X, Z, PH, 8, colors="grey")
    CD=plt.contour(X, Z, PD, 8, colors="r", levels=np.logspace(np.log10(np.max([np.min(PD), 1e-1])), np.log10(np.max(PD)), 8))
    CB=plt.contour(X, Z, PB, 8, colors="b", levels=np.logspace(np.log10(np.max([np.min(PB), 1e-1])), np.log10(np.max(PB)), 8))
    plt.clabel(CH, inline=1, fontsize=8)
    plt.clabel(CD, inline=1, fontsize=8)
    plt.clabel(CB, inline=1, fontsize=8)
    lines=[CB.collections[1], CD.collections[1], CH.collections[1]]
    labels=["bulge", "disk", "dark matter"]
    plt.legend(lines, labels)
    plt.title("Galactic Densities\n(contour plot, logarithmic levels)")
  if initVel :
    plt.quiver(initPosCoord[0, 0], initPosCoord[2, 0], normalise(initVelCoord)[0, 0], normalise(initVelCoord)[2, 0], pivot="tail", color="r", width=0.004)
  if initPos :
    plt.plot(initPosCoord[0, 0], initPosCoord[2, 0], "k*", markersize=8, markeredgecolor="none")  
  ax.set_xlabel(r"$x$ ([L])")
  ax.set_ylabel(r"$z$ ([L])")
  ax.set_xlim([-xrange, xrange])
  ax.set_ylim([-yrange, yrange])
  ax.get_xaxis().set_visible(showAxis)
  ax.get_yaxis().set_visible(showAxis)
  plt.tight_layout()

def diagnosticSolarSystem(param="", N=5) :
  if(param!="" and param!="SI") :
    criticalBadValue("param")
  f=open(g_solarSystemOutputFolder+"/solarSystem"+param+"RK4DataX.pckl", "rb")
  solarSystemRK4dataX=pickle.load(f)
  f.close()
  f=open(g_solarSystemOutputFolder+"/solarSystem"+param+"PECEDataX.pckl", "rb")
  solarSystemPECEdataX=pickle.load(f)
  f.close()
  f=open(g_solarSystemOutputFolder+"/solarSystem"+param+"PECDataX.pckl", "rb")
  solarSystemPECdataX=pickle.load(f)
  f.close()
  from scipy.spatial.distance import cdist
  T=np.size(solarSystemRK4dataX, axis=2)
  
  if(param=="SI") :
    JPL=JPLX*g_oneParsec
  else :
    JPL=JPLX
  errRK4JPLX=np.linalg.norm(np.abs(solarSystemRK4dataX[:, :, -1]-JPL), axis=0)
  errPECEJPLX=np.linalg.norm(np.abs(solarSystemPECEdataX[:, :, -1]-JPL), axis=0)
  errPECJPLX=np.linalg.norm(np.abs(solarSystemPECdataX[:, :, -1]-JPL), axis=0)
  errRK4PECE=np.linalg.norm(np.abs(solarSystemRK4dataX[:, :, -1]-solarSystemPECEdataX[:, :, -1]), axis=0)
  errPECEPEC=np.linalg.norm(np.abs(solarSystemPECEdataX[:, :, -1]-solarSystemPECdataX[:, :, -1]), axis=0)
  tmp_d=cdist(np.transpose(JPL), np.transpose(JPL))
  minDist=np.min(tmp_d[np.nonzero(tmp_d)])
  radii=np.linalg.norm(JPL, axis=0)
  if(param=="SI") : par=" (SI units)"
  else : par=param
  for (err, t) in ((errRK4JPLX, "JPLH v RK4"), (errPECEJPLX, "JPLH v PECE"), (errPECJPLX, "JPLH v PEC"), (errRK4PECE, "RK4 v PECE"), (errPECEPEC, "PECE v PEC")) :
    print(t+par+" :")
    print(" Maximum error :                 "+"{:17.9e}".format(max(err)))
    print(" Minimum error :                 "+"{:17.9e}".format(min(err)))
    print(" Mean error :                    "+"{:17.9e}".format(np.mean(err)))
    print(" Mean error / minimum distance : "+"{:6.2f}".format(100*np.mean(err)/minDist)+" %")
    print("")
  
  plt.figure(figsize=g_squareFigSize)
  for i in range(0, N) :
    plt.plot(solarSystemRK4dataX[0, i, -8:T], solarSystemRK4dataX[1, i, -8:T], "r:")
    plt.plot(solarSystemPECEdataX[0, i, -8:T], solarSystemPECEdataX[1, i, -8:T], "b:")
    plt.plot(solarSystemPECdataX[0, i, -8:T], solarSystemPECdataX[1, i, -8:T], "g:")
  pJPL, =plt.plot(JPL[0, 0:N], JPL[1, 0:N], "k.")
  pRK4, =plt.plot(solarSystemRK4dataX[0, 0:N, -1], solarSystemRK4dataX[1, 0:N, -1], "r.")
  pPECE, =plt.plot(solarSystemPECEdataX[0, 0:N, -1], solarSystemPECEdataX[1, 0:N, -1], "b.")
  pPEC, =plt.plot(solarSystemPECdataX[0, 0:N, -1], solarSystemPECdataX[1, 0:N, -1], "g.")
  plt.legend([pJPL, pRK4, pPECE, pPEC], ["JPL Horizons Data", "RK4 Run", "PECE Run", "PEC Run"], loc="best")
  plt.title("Solar System MiniNBody simulations"+par)
  plt.tight_layout()
  plt.axis("equal")
  plt.savefig("plots/solarSystemTests"+param)
  
  plt.figure(figsize=g_squareFigSize)
  plt.plot(errRK4JPLX/radii)
  plt.plot(errPECEJPLX/radii)
  plt.plot(errRK4PECE/radii)
  plt.legend(["JPLH v RK4", "JPLH v PECE", "RK4 v PECE"])
  plt.title("Relative Errors to JPL Data")
  plt.tight_layout()

from galaxyParameters import generateGalaxyModel

def plotGCInitialPositionsAndVelocities() :
  models=["colongitudinal", "radial"]
  modelsNames=["S1", "S2"]
  for i in range(0, len(models)) :
    (g_GCPosition, g_GCVelocity, g_Mb, g_ab, g_Md, g_ad, g_hd, g_Vh, g_ah)=generateGalaxyModel(models[i])
    plotGalacticDensity(6e3, 6e3, g_Mb, g_ab, g_Md, g_ad, g_hd, g_Vh, g_ah, plots="baryonic", initPos=True, initPosCoord=g_GCPosition, initVel=True, initVelCoord=g_GCVelocity)
    plt.tight_layout()
    plt.savefig("initPlots/initialGCPosAndSpeed/"+modelsNames[i])

def demoVectorEvolutionGraph() :
  # TODO :
  N=900
  t=np.linspace(0, 3, N)
  t1=np.arange(0, N/3).astype(int)
  t2=np.arange(N/3, 2*N/3).astype(int)
  t3=np.arange(2*N/3, N).astype(int)
  x=np.zeros(N)
  y=np.zeros(N)
  z=np.zeros(N)
  x[t1]=np.linspace(1, 0, N/3)
  y[t1]=np.linspace(0, 1, N/3)
  y[t2]=np.linspace(1, 0, N/3)
  z[t2]=np.linspace(0, 1, N/3)
  x[t3]=np.linspace(0, 1, N/3)
  y[t3]=np.linspace(0, 1, N/3)
  z[t3]=np.linspace(1, 0, N/3)
  P=np.vstack((x, y, z))
  P=P/np.linalg.norm(P, axis=0)
  
  P=np.delete(P, [0, 299, 300, 599, 600], axis=1) # remove vectors where 2 components are zero
  t=np.delete(t, [0, 299, 300, 599, 600])
  N=N-5 # update N
  
  t=t/3
  r=vectorEvolution(P, t,
                    timeAxisTitle=g_axisLabelTime,
                    directionTitle="Vector")
  plt.figure(r[0].number); plt.tight_layout()
  plt.savefig("initPlots/tools/vectorAnalysisDemo")

plt.close("all")
#plotPlummerDensity(9.9*1e9, 0.25*1e3)
#plotMiyamotoNagaiDensity(72*1e9, 3.5*1e3, 0.4*1e3)
#plotDarkMatterHaloDensity(220*1e3, 10*1e3)
#plotGalacticDensity(7e3, 7e3, 9.9*1e9, 0.25*1e3, 72*1e9, 3.5*1e3, 0.4*1e3, 220*1e3/g_scalingV, 20*1e3, "baryonic")
#plotGalacticDensity(10e3, 10e3, 9.9*1e9, 0.25*1e3, 72*1e9, 3.5*1e3, 0.4*1e3, 220*1e3/g_scalingV, 20*1e3, "all")
#plotGalacticPotentials(1e4, 1e4, 9.9*1e9, 0.25*1e3, 72*1e9, 3.5*1e3, 0.4*1e3, 220*1e3/g_scalingV, 20*1e3)
#plotGalacticDensity(6e3, 6e3, 19.8*1e9, 0.5*1e3, 72*1e9, 6*1e3, 1*1e3, 190*1e3/g_scalingV, 10*1e3, "all")