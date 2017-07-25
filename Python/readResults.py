from os.path import isfile
from outputDataGetting import centroidPositionInterface
from outputDataGetting import centroidPositionInterfaceCPSaved
from outputDataGetting import computeGPEnergies
from outputDataGetting import computeKineticEnergies
from outputDataGetting import createVideo
from outputDataGetting import escapersPlot
from outputDataGetting import angularMomentum
from indicators import testIndicatorsInterface
from indicators import getMeansAndVars
from util import extractData
from outputDataGetting import evolutionTidalForces
from util import getAngMomenta
from outputDataGetting import extractVelocityInformationsInterface
from outputDataGetting import fitVelocityDistributionInterface
from outputDataGetting import plotDensityContourLines
from outputDataGetting import rewriteInput
from outputDataGetting import plotMassEvolution
from outputDataGetting import inertiaTensorWrapper
from outputDataGetting import plotRadialDistributionEvolution
from outputDataGetting import selectLimitationRadiusInterface
from util import addCrossingLineInterface
from util import centroidPosition
from util import g_energyScalingAxisTitle
from util import g_squareFigSize
from util import g_axisLabelTime
from util import g_hideTitles
from util import g_GCCentroidSpeedGraphTitle
from util import getDiskCrossingTimes
from util import graphInterface
from util import getDiskCrossingTimesCPSaved
from util import removeGlobalSpeed
from util import recenterDataAroundCentroid
from util import userInput0Or1
from util import centroidPositionOld
from util import loadGPEFromFile
from util import warningError
from util import g_diagnosticsFolder
from util import writeGPEToFile
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path
import pickle
import time

# S0 merging. #################################################
# Uncomment to do the merging. Comment to do basic results reading.
#from util import mergeOutputFiles
#g_heavyOutputFilesFolder="./HeavyDiagnostics/"
#firstFile=g_heavyOutputFilesFolder+"mnbodyS0Cont_sim0Control_tech=PECE_NS=5000_b=3_sBar=150_Tm=200_dt=0.001_pInt=1.out"
#secondFile=g_heavyOutputFilesFolder+"mnbodyS0ContRestart_rewritten_NS=5000_TMax=400.0_dt=0.001_printInt=1.0.out"
#output=g_heavyOutputFilesFolder+"mnbodyS0Long_NS=5000_dt=0.001_printInt=1.0.out"
#mergeOutputFiles(firstFile, secondFile, output)
#raise ValueError
###############################################################

# Parameters and units. #######################################
g_heavyOutputFilesFolder="./HeavyDiagnostics/"
g_usingWindows=False # set this to false when using linux, as the temporary folder, ffmpeg, and some functions won't work

#g_outputFileName=g_heavyOutputFilesFolder+"mnbodyS0Cont_sim0Control_tech=PECE_NS=5000_b=3_sBar=150_Tm=200_dt=0.001_pInt=1.out"

#g_outputFileName=g_heavyOutputFilesFolder+"mnbodyS0Long_NS=5000_dt=0.001_printInt=1.0.out"
#g_outputFileName=g_heavyOutputFilesFolder+"mnbodyS1Col_sim1Colongitudinal_tech=PECE_NS=5500_b=3_sBar=150_Tm=200_dt=0.001_pInt=1.out"
#g_outputFileName=g_heavyOutputFilesFolder+"mnbodyS2Rad_sim2Radial_tech=PECE_NS=5500_b=3_sBar=150_Tm=200_dt=0.001_pInt=1.out"

#g_outputFileName=g_heavyOutputFilesFolder+"mnbodyS0ContSmall_sim0Control_tech=PECE_NS=2500_b=3_sBar=150_Tm=200_dt=0.001_pInt=1.out"
#g_outputFileName=g_heavyOutputFilesFolder+"mnbodyS1ColSmall_sim1Colongitudinal_tech=PECE_NS=2500_b=3_sBar=150_Tm=200_dt=0.001_pInt=1.out"
#g_outputFileName=g_heavyOutputFilesFolder+"mnbodyS1ColSmallLong_sim1Colongitudinal_tech=PECE_NS=2500_b=3_sBar=150_Tm=400_dt=0.001_pInt=1.out"
#g_outputFileName=g_heavyOutputFilesFolder+"mnbodyS2RadSmall_sim2Radial_tech=PECE_NS=2500_b=3_sBar=150_Tm=200_dt=0.001_pInt=1.out"

#g_outputFileName=g_heavyOutputFilesFolder+"mnbodyS3BrutDisk_sim3BrutDisk_tech=PECE_NS=3000_b=3_s=0.96_Tm=75_dt=0.001_pInt=0.375.out"
g_outputFileName=g_heavyOutputFilesFolder+"mnbodyS4BrutBulge_sim4BrutBulge_tech=PECE_NS=3000_b=3_s=0.96_Tm=75_dt=0.001_pInt=0.375.out"
###############################################################

# Prepare the execution. ######################################
plt.close("all")
g_fileID=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(os.path.getmtime(g_outputFileName)))+"("+g_outputFileName.split("/")[-1]+")" # create pseudo-unique file ID (based on file's last modification date)
if(g_usingWindows) :
  tmpDirPath=os.environ.get("TEMP").replace("\\", "/") # find the TEMP folder (Windows)
  if(tmpDirPath[-1]!="/") : tmpDirPath=tmpDirPath+"/" # add final "/" if not present
  tmpImagesPath=tmpDirPath+"MiniNBody/"+g_fileID # name a subfolder in TEMP folder for images
  if(not os.path.exists(tmpImagesPath)) : os.makedirs(tmpImagesPath) # create subfolder in TEMP folder for images
g_currentDiagnosticsFolder=g_diagnosticsFolder+"/"+g_fileID # name a folder for current snapshots' diagnostics
g_currentDiagnosticsFolder=g_currentDiagnosticsFolder.replace("=", "") # remove eventual accidental "=" that can mess with the video creation script
if(not os.path.exists(g_currentDiagnosticsFolder)) : os.makedirs(g_currentDiagnosticsFolder) # create folder for current snapshots' diagnostics
if("solarSystem" in g_outputFileName) :
  g_plotDiskCrossings=False
  g_solarSystemCase=True
else :
  g_plotDiskCrossings=True
  g_solarSystemCase=False
def diagnosticPath(filename) :
  # Add the path to the current diagnostics' folder before a filename to make it easy to save plots.
  # @param filename the wanted filename
  # @param g_currentDiagnosticsFolder (global) the folder for current snapshots' diagnostics
  # @return the path to the filename within current diagnostics' folder
  return("./"+g_currentDiagnosticsFolder+"/"+filename)

def inertiaTensorPostTreatment(t1, t2, r, doIt, plotDC, times, dataX, savedCentroidPosition) :
  addCrossingLineInterface(doIt, plotDC, r[1][1], times, dataX, savedCentroidPosition)
  addCrossingLineInterface(doIt, plotDC, r[2][1][0], times, dataX, savedCentroidPosition)
  addCrossingLineInterface(doIt, plotDC, r[2][2][0], times, dataX, savedCentroidPosition)
  addCrossingLineInterface(doIt, plotDC, r[2][3][0], times, dataX, savedCentroidPosition)
  if(plotDC) :
    ticks=[0]
    for i in CIDs :
      ticks.append(i)
    ticks.append(T-1)
    r[2][5].set_ticks(times[ticks])
  plt.figure(r[1][0].number); plt.tight_layout()
  plt.savefig(diagnosticPath(t1+r[0]))
  plt.figure(r[2][0].number); plt.tight_layout()
  plt.savefig(diagnosticPath(t2+r[0]))
###############################################################

print("MiniNBody output file to be read : \""+g_outputFileName+"\".\n")
print("ID (creation date) : "+str(g_fileID)+".\n")
#print("Smart selection will be used.")
if(g_hideTitles) : print("Graph titles will be hidden.")

# Switches according to if the output file is the one from control simulation or not.
if("S0" in g_outputFileName) :
  tidalOption=""
  controlSimuMeansAndVarsText="101 : Get control simulation confidence interval values for indicators.\n"
else :
  tidalOption="  7 : Plot tidal forces evolution (as function of time).\n"
  controlSimuMeansAndVarsText=""

choice=""
dataImported=False
dataRecentered=False
g_centroidPositionSaved=False
plotDiskCrossingDecided=False or not(g_plotDiskCrossings)
while choice!="q" :
  print("\n"+
        "  q : Exit.\n"+
        "  0 : Get data and exit.\n", end="")
  if(g_usingWindows) : print("  1 : Make a video from the data.\n", end="")
  print("  2 : Graph the position of COM of the GC.\n"+
        "  3 : Print disk crossing times.\n"+
        "  4 : Compute and save centroid position (for further uses).\n"+
        "  5 : Launch module to find a limiting radius.\n"+
        "  6 : Plot GC global speed (as function of time).\n"+
        tidalOption+
        "\n"+
        "  9 : Graph radial distributions at different times (automatic set).\n"+
        " 10 : Graph velocities' modules' distributions at different times (automatic set).\n"+
        " 11 : Launch module to fit a known law to a velocity distribution.\n"+
        "\n"+
        " 12 : Graph the number of escapers (chosen radius).\n"+
        " 13 : Graph the number of escapers (automatic set of radii).\n"+
        "\n"+
        " 20 : Draw isodensities of the GC.\n"+
        " 21 : Graph the evolution of the total mass of the GC.\n"+
        "\n"+
        " 30 : Graph the deviation from sphericity (as function of time).\n"+
        "\n"+
        " 40 : Graph the anisotropy parameter (as function of time).\n"+
        " 41 : Graph the velocities' deviation from isotropy (as function of time).\n"+
        " 42 : Graph the velocities' modules' mean (as function of time).\n"+
        " 43 : Graph the velocities' modules' dispersion (as function of time).\n"+
#        " 44 : Graph the specific velocity dispersion (as function of time).\n"+
#        " 45 : Graph the velocities' modules' kurtosis (as function of time).\n"+
#        " 46 : Graph the velocities' modules' skewness (as function of time).\n"+
        "\n"+
        " 50 : Do 53 and 56 (see below).\n"+
        " 51 : Compute E_c (as function of time) and graph.\n"+
        " 52 : Compute E_p (as function of time) and graph.\n"+
        " 53 : Do 51, 52, graph E and graph virial.\n"+
        " 54 : Compute E_c variations (as function of time) and graph.\n"+
        " 55 : Compute E_p variations (as function of time) and graph.\n"+
        " 56 : Do 54, 55, and graph E variations.\n"+
        "\n"+
        " 60 : Graph the self angular momentum (as function of time).\n"+
#        " 61 : Graph the self torque (as function of time).\n"+
#        " 62 : Graph the global angular momentum (as function of time).\n"+
#        " 63 : Graph the global torque (as function of time).\n"+
        " 64 : Graph the rotation factor (as function of time).\n"+
        "\n"+
        "100 : Launch indicators' module.\n"+
        controlSimuMeansAndVarsText+
#        "100 : Graph first E_c of each star.\n"+
#        "101 : Graph first velocity modules of each star.\n"+
#        "102 : Compute previous distances between extreme velocities at a certain time.\n"+
#        "103 : Graph first E_c contributions of each star.\n"+
        "\n"+
        "200 : Rewrite input file from output.\n", end="")
  choice=input("> ")
  print("\n", end="")
  if(choice==7 and tidalOption=="") : continue # won't work ?
  
  # Pre-treatments. #############
  if(not choice.isdigit()) :
    warningError("Please enter a correct digit.")
    continue
  choice=int(choice)
  
  # If data has not already been imported from input file, do so.
  # This prevents the data to be imported too many times on a single run of diagnostics.
  if(not dataImported) :
    (times, masses, dataX, dataV)=extractData(g_outputFileName)
    dataImported=True
  
  N=np.size(masses)
  T=np.size(dataX, axis=2)
  uniformRepartition4=np.linspace(0.05, 0.999999, 4)
  uniformRepartition8=np.linspace(0.05, 0.999999, 8)
  uniformRepartition10=np.linspace(0.05, 0.999999, 10)
  uniformRepartition12=np.linspace(0.05, 0.999999, 12)
  fullRepartition=np.linspace(0, 0.999999, T)
  
  # If centroid position has already been saved, use it.
  if(isfile(diagnosticPath("centroidPositionData")+".pckl")) :
    g_centroidPositionSaved=True
    savedCentroidPosition=pickle.load(open(diagnosticPath("centroidPositionData")+".pckl", "rb"))
  else :
    warningError("Centroid position is not saved, computations can be very long !")
  
  # If position data has not already been recentered, do so.
  # Except if choice==0.
  # This prevents the data to be recentered too many times on a single run of diagnostics.
  if(not dataRecentered and choice!=0) :
    if(g_centroidPositionSaved) : # use the saved centroid position if possible
      rdataX=np.zeros(np.shape(dataX))
      for i in range(0, N) :
        rdataX[:, i, :]=dataX[:, i, :]-savedCentroidPosition[:, :]
    else :
      if(choice!=4) :
        warningError("Centroid position is not saved, computations can be very long !")
        rdataX=recenterDataAroundCentroid(dataX)
    dataRecentered=True
  
  # Ask if the disk crossing times should be plotted on the graphs that are function of time.
  # For isolated simulations, it makes no sense to try to plot them.
  if(not(plotDiskCrossingDecided) and not(choice in [0, 1, 2, 3, 4, 5, 20, 100, 101, 200])) :
    res=userInput0Or1("Plot disk crossings (0 for no, 1 for yes) ?",
                      if0="Disk crossings will not be plotted.",
                      if1="Disk crossings will be plotted.")
    if(res==0) : g_plotDiskCrossings=False
    else : g_plotDiskCrossings=True
    plotDiskCrossingDecided=True
  
  print("Number of stars involved : "+str(N)+".")
  print("Number of snapshots :      "+str(T)+".")
  print("Final time :               "+str(times[-1])+" [T].")
  print("\n", end="")
  ###############################
  
  # Generic diagnostics. ########
  if(choice==0) :
    raise SystemExit(0)
  if(choice==1 and g_usingWindows) :
    createVideo(dataX, rdataX, times, tmpImagesPath, g_currentDiagnosticsFolder, solarSystem=g_solarSystemCase)
  if(choice==2) :
    if(g_centroidPositionSaved) : ref=centroidPositionInterfaceCPSaved(savedCentroidPosition, times)
    else : ref=centroidPositionInterface(dataX, times)
    plt.tight_layout(); plt.savefig(diagnosticPath("centroid_positions_"+ref))
  if(choice in [0, 1, 3, 30, 43, 60, 61, 62, 63, 64, 101, 200]) :
    if(g_centroidPositionSaved) : (CIDs, CTs)=getDiskCrossingTimesCPSaved(savedCentroidPosition, times)
    else : (CIDs, CTs)=getDiskCrossingTimes(dataX, times)
  if(choice==3) :
    print("IDs :", end="")
    for i in CIDs : print(" "+str(i), end="")
    print(".\nTimes :", end="")
    for t in CTs : print(" "+"{:6.2f}".format(t), end="")
    print(" ([T]).")
  if(choice==4) :
    centroidPos=centroidPosition(dataX)
    pickle.dump(centroidPos, open(diagnosticPath("centroidPositionData")+".pckl", "wb"))
  if(choice==5) :
    selectLimitationRadiusInterface(rdataX)
  if(choice==6) :
    fig=plt.figure(figsize=g_squareFigSize); ax=fig.add_subplot(111); ax.set_xlabel(g_axisLabelTime)
    ax.plot(times, np.linalg.norm(centroidPositionOld(dataV, masses), axis=0))
    ax.set_xlabel(g_axisLabelTime); ax.set_ylabel(r"$\left\|\vec{V}_{GC}\right\|_2$ ([V])")
    ax.set_xlim([times[0], times[-1]]);
    addCrossingLineInterface(g_centroidPositionSaved, g_plotDiskCrossings, plt.gca(), times, dataX, savedCentroidPosition)
    graphInterface(fig=fig,
                   ax=ax,
                   title=g_GCCentroidSpeedGraphTitle)
    plt.savefig(diagnosticPath("centroid_speed"))
  if(choice==7 and tidalOption!="") :
    (ref, _unused, _unused, _unused)=evolutionTidalForces(savedCentroidPosition, times)
    addCrossingLineInterface(g_centroidPositionSaved, g_plotDiskCrossings, plt.gca(), times, dataX, savedCentroidPosition)
    plt.savefig(diagnosticPath("tidalForces_"+ref))
  ###############################

  # Distributions diagnostics. ##
  if(choice==9) :
    ref=plotRadialDistributionEvolution(rdataX, dataV, times, "radial", uniformRepartition8)
    plt.savefig(diagnosticPath("positions_radiiDistributionEvolution_"+ref))
  if(choice==10) :
    ref=plotRadialDistributionEvolution(rdataX, dataV, times, "velocity", uniformRepartition8)
    plt.savefig(diagnosticPath("velocities_distributionEvolution_"+ref))  
  if(choice==11) :
    ref=fitVelocityDistributionInterface(dataV, rdataX, times)
    plt.savefig(diagnosticPath("velocities_fit"+ref))
  ###############################
  
  # Escapers diagnostics. #######
  if(choice==12) :
    ref=escapersPlot(rdataX, times, param="manual")
    addCrossingLineInterface(g_centroidPositionSaved, g_plotDiskCrossings, plt.gca(), times, dataX, savedCentroidPosition)
    plt.savefig(diagnosticPath("escapers_manual"+ref))
  if(choice==13) :
    ref=escapersPlot(rdataX, times)
    addCrossingLineInterface(g_centroidPositionSaved, g_plotDiskCrossings, plt.gca(), times, dataX, savedCentroidPosition)
    plt.savefig(diagnosticPath("escapers_automatic"+ref))
  ###############################

  # Contour plot diagnostics. ###
  if(choice==20) :
    ref=plotDensityContourLines(rdataX, times, uniformRepartition12)
    plt.savefig(diagnosticPath("contourDensities"+ref))
  if(choice==21) :
    ref=plotMassEvolution(rdataX, masses, times)
    addCrossingLineInterface(g_centroidPositionSaved, g_plotDiskCrossings, plt.gca(), times, dataX, savedCentroidPosition)
    plt.savefig(diagnosticPath("massEvolution_"+ref))
  ###############################

  # Positions diagnostics. ######
#  if(choice==30) :
#    ref=evolutionInertiaTensor(rdataX, masses, times, uniformRepartition12, "3D")
#    plt.savefig(diagnosticPath("inertiaTensor_directions_"+ref))
#  if(choice==31) :
#    ref=evolutionInertiaTensor(rdataX, masses, times, fullRepartition, "deviationFromSphericity")
#    addCrossingLineInterface(g_centroidPositionSaved, g_plotDiskCrossings, plt.gca(), times, dataX, savedCentroidPosition)
#    plt.tight_layout(); plt.savefig(diagnosticPath("inertiaTensor_deviation_"+ref))
#  if(choice==32) :
  if(choice==30) :
    r=inertiaTensorWrapper(rdataX, masses, times, rdataX, "positions")
    inertiaTensorPostTreatment("positions_tensor_", "positions_tensor_maxEigen_", r, g_centroidPositionSaved, g_plotDiskCrossings, times, dataX, savedCentroidPosition)
  ###############################
  
  # Velocities diagnostics. #####
  if(choice==40) :
    ref=extractVelocityInformationsInterface("anisotropy", rdataX, dataV, masses, times)
    addCrossingLineInterface(g_centroidPositionSaved, g_plotDiskCrossings, plt.gca(), times, dataX, savedCentroidPosition)
    plt.savefig(diagnosticPath("velocities_anisotropy_"+ref))
  if(choice==41) :
    r=inertiaTensorWrapper(removeGlobalSpeed(dataV), masses, times, rdataX, "velocity")
    inertiaTensorPostTreatment("velocities_tensor_", "velocities_tensor_maxEigen_", r, g_centroidPositionSaved, g_plotDiskCrossings, times, dataX, savedCentroidPosition)
  if(choice==42) :
    ref=extractVelocityInformationsInterface("mean", rdataX, dataV, masses, times)
    addCrossingLineInterface(g_centroidPositionSaved, g_plotDiskCrossings, plt.gca(), times, dataX, savedCentroidPosition)
    plt.savefig(diagnosticPath("velocities_mean_"+ref))
  if(choice==43) :
    ref=extractVelocityInformationsInterface("dispersion", rdataX, dataV, masses, times)
    addCrossingLineInterface(g_centroidPositionSaved, g_plotDiskCrossings, plt.gca(), times, dataX, savedCentroidPosition)
    plt.savefig(diagnosticPath("velocities_dispersion_"+ref))
  if(choice==44) :
    ref=extractVelocityInformationsInterface("specificDispersion", rdataX, dataV, masses, times)
    addCrossingLineInterface(g_centroidPositionSaved, g_plotDiskCrossings, plt.gca(), times, dataX, savedCentroidPosition)
    plt.savefig(diagnosticPath("velocities_specificDispersion_"+ref))
#  if(choice==45) :
#    ref=extractVelocityInformationsInterface("kurtosis", rdataX, dataV, masses, times)
#    addCrossingLineInterface(g_centroidPositionSaved, g_plotDiskCrossings, plt.gca(), times, dataX, savedCentroidPosition)
#    plt.savefig(diagnosticPath("velocities_kurtosis_"+ref))
#  if(choice==46) :
#    ref=extractVelocityInformationsInterface("skewness", rdataX, dataV, masses, times)
#    addCrossingLineInterface(g_centroidPositionSaved, g_plotDiskCrossings, plt.gca(), times, dataX, savedCentroidPosition)
#    plt.savefig(diagnosticPath("velocities_skewness_"+ref))
  ###############################

  # Energies diagnostics. #######

  (rText, titText, apTitText)=("", "", "")
  if(choice in [50, 51, 53, 54, 56]) :
    (KE, (rText, titText, apTitText))=computeKineticEnergies(dataV, masses, rdataX, recenterVelocities=not(g_solarSystemCase))
  if(choice in [50, 52, 53, 55, 56]) :
    GPEFilePath=diagnosticPath("GPE")+".gpe"
    if(isfile(GPEFilePath)) :
      print(" Loading GPE from pre-calculated file.")
      GPE=loadGPEFromFile(GPEFilePath, T)
    else :
      print(" Launching calculation of GPE.")
      GPE=computeGPEnergies(dataX, masses)
      print(" Saving GPE.")
      writeGPEToFile(GPE, T, GPEFilePath)
  if(choice in [50, 53, 56]) :
    E=KE+GPE
    V=2*KE+GPE
    #V[np.where(V<0)[0]]=0 # remove negative values that come from the approximation
  if(choice in [50, 54, 56]) :
    KEVar=100*(KE-KE[0])/np.abs(KE[0])
  if(choice in [50, 55, 56]) :
    GPEVar=100*(GPE-GPE[0])/np.abs(GPE[0])
  if(choice in [50, 56]) :
    EVar=100*(E-E[0])/np.abs(E[0])
  
  KETitle="Kinetic Energy of the System"+titText
  GPETitle="Potential Energy of the System"
  ETitle="Energy of the System"+apTitText
  VTitle="Virial of the System"+apTitText
  deltaKETitle="Kinetic Energy of the system - Signed Variation"+titText
  deltaGPETitle="Potential Energy of the system - Signed Variation"
  deltaETitle="Energy of the GC- Signed Variation"+apTitText
  KEYLabel=r"$E_c$ "+g_energyScalingAxisTitle
  GPEYLabel=r"$E_p$ "+g_energyScalingAxisTitle
  EYLabel=r"$E$ "+g_energyScalingAxisTitle
  VYLabel=r"$\Upsilon$ "+g_energyScalingAxisTitle
  deltaKEYLabel=r"$\Delta E_c$ regarding $t=0$ (%)"
  deltaGPEYLabel=r"$\Delta E_p$ regarding $t=0$ (%)"
  deltaEYLabel=r"$\Delta E$ regarding $t=0$ (%)"
  KEFigName="energies_KE"+rText
  GPEFigName="energies_GPE"
  EFigName="energies_E"+rText
  VFigName="energies_V"+rText
  deltaKEFigName="energiesVariations_KE"+rText
  deltaGPEFigName="energiesVariations_GPE"+rText
  deltaEFigName="energiesVariations_E"+rText
  KEColor="r"; GPEColor="g"; EColor="b"; VColor="m"
  VPlotType="plot"
  
  if(choice in [51, 52, 54, 55]) :
    fig=plt.figure(figsize=g_squareFigSize); ax=fig.add_subplot(111); ax.set_xlabel(g_axisLabelTime)
  if(choice in [50, 53]) :
    figKE=plt.figure(figsize=g_squareFigSize); axKE=figKE.add_subplot(111); axKE.set_xlabel(g_axisLabelTime); axKE.set_ylabel(KEYLabel)
    figGPE=plt.figure(figsize=g_squareFigSize); axGPE=figGPE.add_subplot(111); axGPE.set_xlabel(g_axisLabelTime); axGPE.set_ylabel(GPEYLabel)
    figE=plt.figure(figsize=g_squareFigSize); axE=figE.add_subplot(111); axE.set_xlabel(g_axisLabelTime); axE.set_ylabel(EYLabel)
    figV=plt.figure(figsize=g_squareFigSize); axV=figV.add_subplot(111); axV.set_xlabel(g_axisLabelTime); axV.set_ylabel(VYLabel); axV.axhline(0, linestyle=':')
  if(choice in [50, 56]) :
    figVarKE=plt.figure(figsize=g_squareFigSize); axVarKE=figVarKE.add_subplot(111); axVarKE.set_xlabel(g_axisLabelTime); axVarKE.set_ylabel(deltaKEYLabel)
    figVarGPE=plt.figure(figsize=g_squareFigSize); axVarGPE=figVarGPE.add_subplot(111); axVarGPE.set_xlabel(g_axisLabelTime); axVarGPE.set_ylabel(deltaGPEYLabel)
    figVarE=plt.figure(figsize=g_squareFigSize); axVarE=figVarE.add_subplot(111); axVarE.set_xlabel(g_axisLabelTime); axVarE.set_ylabel(deltaEYLabel)
  if(choice==51) : ax.set_ylabel(KEYLabel)
  if(choice==52) : ax.set_ylabel(GPEYLabel)
  if(choice==54) : ax.set_ylabel(deltaKEYLabel)
  if(choice==55) : ax.set_ylabel(deltaGPEYLabel)
  
  if(choice in [50, 53]) :
    axKE.plot(times, KE, KEColor); axKE.set_xlim([times[0], times[-1]]); addCrossingLineInterface(g_centroidPositionSaved, g_plotDiskCrossings, axKE, times, dataX, savedCentroidPosition)
    graphInterface(fig=figKE,
                   ax=axKE,
                   title=KETitle)
    plt.figure(figKE.number); plt.savefig(diagnosticPath(KEFigName))
    axGPE.plot(times, GPE, GPEColor); axGPE.set_xlim([times[0], times[-1]]); addCrossingLineInterface(g_centroidPositionSaved, g_plotDiskCrossings, axGPE, times, dataX, savedCentroidPosition)
    graphInterface(fig=figGPE,
                   ax=axGPE,
                   title=GPETitle)
    plt.figure(figGPE.number); plt.savefig(diagnosticPath(GPEFigName))
    axE.plot(times, E, EColor); axE.set_xlim([times[0], times[-1]]); addCrossingLineInterface(g_centroidPositionSaved, g_plotDiskCrossings, axE, times, dataX, savedCentroidPosition)
    graphInterface(fig=figE,
                   ax=axE,
                   title=ETitle)
    plt.figure(figE.number); plt.savefig(diagnosticPath(EFigName))
    if(VPlotType=="plot") : axV.plot(times, V, VColor);
    elif(VPlotType=="semilogy") : axV.semilogy(times, V, VColor);
    axV.set_xlim([times[0], times[-1]]); addCrossingLineInterface(g_centroidPositionSaved, g_plotDiskCrossings, axV, times, dataX, savedCentroidPosition)
    graphInterface(fig=figV,
                   ax=axV,
                   title=VTitle)
    plt.figure(figV.number); plt.savefig(diagnosticPath(VFigName))
  if(choice in [50, 56]) :
    axVarKE.plot(times, KEVar, KEColor); axVarKE.set_xlim([times[0], times[-1]]); addCrossingLineInterface(g_centroidPositionSaved, g_plotDiskCrossings, axVarKE, times, dataX, savedCentroidPosition)
    graphInterface(fig=figVarKE,
                   ax=axVarKE,
                   title=deltaKETitle)
    plt.figure(figVarKE.number); plt.savefig(diagnosticPath(deltaKEFigName))
    axVarGPE.plot(times, GPEVar, GPEColor); axVarGPE.set_xlim([times[0], times[-1]]); addCrossingLineInterface(g_centroidPositionSaved, g_plotDiskCrossings, axVarGPE, times, dataX, savedCentroidPosition)
    graphInterface(fig=figVarGPE,
                   ax=axVarGPE,
                   title=deltaGPETitle)
    plt.figure(figVarGPE.number); plt.savefig(diagnosticPath(deltaGPEFigName))
    axVarE.plot(times, EVar, EColor); axVarE.set_xlim([times[0], times[-1]]); addCrossingLineInterface(g_centroidPositionSaved, g_plotDiskCrossings, axVarE, times, dataX, savedCentroidPosition)
    graphInterface(fig=figVarE,
                   ax=axVarE,
                   title=deltaETitle)
    plt.figure(figVarE.number); plt.savefig(diagnosticPath(deltaEFigName))
  if(choice==51) :
    ax.plot(times, KE, KEColor); ax.set_xlim([times[0], times[-1]]); addCrossingLineInterface(g_centroidPositionSaved, g_plotDiskCrossings, ax, times, dataX, savedCentroidPosition)
    graphInterface(fig=fig,
                   ax=ax,
                   title=KETitle)
    plt.savefig(diagnosticPath(KEFigName))
  if(choice==52) :
    ax.plot(times, GPE, GPEColor); ax.set_xlim([times[0], times[-1]]); addCrossingLineInterface(g_centroidPositionSaved, g_plotDiskCrossings, ax, times, dataX, savedCentroidPosition)
    graphInterface(fig=fig,
                   ax=ax,
                   title=GPETitle)
    plt.savefig(diagnosticPath(GPEFigName))
  if(choice==54) :
    ax.plot(times, KEVar, KEColor); ax.set_xlim([times[0], times[-1]]); addCrossingLineInterface(g_centroidPositionSaved, g_plotDiskCrossings, ax, times, dataX, savedCentroidPosition)
    graphInterface(fig=fig,
                   ax=ax,
                   title=deltaKETitle)
    plt.savefig(diagnosticPath(deltaKEFigName))
  if(choice==55) :
    ax.plot(times, GPEVar, GPEColor); ax.set_xlim([times[0], times[-1]]); addCrossingLineInterface(g_centroidPositionSaved, g_plotDiskCrossings, ax, times, dataX, savedCentroidPosition)
    graphInterface(fig=fig,
                   ax=ax,
                   title=deltaGPETitle)
    plt.savefig(diagnosticPath(deltaGPEFigName))  
  ###############################

  #
  if(choice==60) :
    r=angularMomentum(rdataX, removeGlobalSpeed(dataV), masses, times, rdataX)
    inertiaTensorPostTreatment("angularMomentum_norm", "angular(Momentum_direction", r, g_centroidPositionSaved, g_plotDiskCrossings, times, dataX, savedCentroidPosition)
#  if(choice==61) :
#    r=angularMomentum(rdataX, removeGlobalSpeed(dataV), masses, times, rdataX, which="torque")
#    inertiaTensorPostTreatment("torque_norm", "torque_direction", r, g_centroidPositionSaved, g_plotDiskCrossings, times, dataX, savedCentroidPosition)
#  if(choice==62) :
#    r=angularMomentum(dataX, dataV, masses, times, rdataX)
#    inertiaTensorPostTreatment("angularMomentumGlobal_norm", "angularMomentumGlobal_direction", r, g_centroidPositionSaved, g_plotDiskCrossings, times, dataX, savedCentroidPosition)
#  if(choice==63) :
#    r=angularMomentum(dataX, dataV, masses, times, rdataX, which="torque")
#    inertiaTensorPostTreatment("torqueGlobal_norm", "torqueGlobal_direction", r, g_centroidPositionSaved, g_plotDiskCrossings, times, dataX, savedCentroidPosition)
  if(choice==64) :
    r=inertiaTensorWrapper(getAngMomenta(rdataX, removeGlobalSpeed(dataV), masses), masses, times, rdataX, "angularMomentum")
    inertiaTensorPostTreatment("angMom_tensor_", "angMom_tensor_maxEigen_", r, g_centroidPositionSaved, g_plotDiskCrossings, times, dataX, savedCentroidPosition)
  ###############################

  # Debug diagnostics. ##########
  if(choice==100) :
    ref=testIndicatorsInterface(g_outputFileName, rdataX, dataV, masses, times)
    plt.savefig(diagnosticPath("test_indicators"+ref))
  if(choice==101 and controlSimuMeansAndVarsText!="") :
    getMeansAndVars(g_outputFileName, rdataX, dataV, masses, times, t0=200, t1=times[-1])
#  if(choice==100) :
#    computeKineticEnergies(dataV, masses, plotEachKE=True, indexes=[0, 1, 2])
#    plt.tight_layout(); plt.savefig(diagnosticPath("debug_firstKE_eachStar"))
#  if(choice==101) :
#    computeKineticEnergies(dataV, masses, plotEachV=True, indexes=[0, 1, 2])
#    plt.tight_layout(); plt.savefig(diagnosticPath("debug_firstV_eachStar"))
#  if(choice==102) :
#    eVSD=computeSmallestDistanceBetweenHighVelocityVariatingStars(dataX, dataV)
#    print(eVSD[0])
#    print(eVSD[1])
#  if(choice==103) :
#    computeKineticEnergies(dataV, masses, plotKEContributions=True, indexes=[0, 1, 2])
#    plt.tight_layout(); plt.savefig(diagnosticPath("debug_KEContributions"))
  ###############################
    
  # Correction tentatives. ######
  if(choice==200) :
    rewriteInput(dataX, rdataX, dataV, masses, g_outputFileName, showPointsOutside=False)
  ###############################

#g_outputFileName="nbody.out"
#g_outputFileName="nbody_NStars=200_b=1_sigmaBar=100_beta=0_TMax=100_dt=0.005_printInt=2.out" # basic data for tests
#g_outputFileName="nbody_NStars=1000_b=1_sigmaBar=100_beta=0_TMax=100_dt=0.005_printInt=3.out"
#g_outputFileName="nbody_GCNStars=1000_b=1_sigmaBar=150_beta=0_TMax=100_dt=0.001_printInt=0.5.out"
#g_outputFileName="nbody_GCNStars=500_b=1_sigmaBar=150_beta=0_TMax=4_dt=0.001_printInt=0.033333.out"
#g_outputFileName="nbody_testEnergy-detectBinaries_NS=400_b=1_sBar=150_beta=0_TMax=100_dt=0.005_printInt=1.out"
#g_outputFileName="nbody_testEnergy-detectBinaries_NS=400_b=1_sBar=150_beta=0_TMax=20_dt=0.005_printInt=0.66.out"
#g_outputFileName="nbody_testConcentration&dt_NS=400_b=1_sBar=150_beta=0_TMax=60_dt=0.0001_printInt=2.out"
#g_outputFileName="nbody_testConcentration&dt_NS=300_b=8_sBar=200_beta=0_TMax=150_dt=0.001_printInt=3.out"
#g_outputFileName="nbody_testConcentrationAnddt_NS=300_b=8_sBar=150_beta=0_TMax=150_dt=0.001_printInt=1.out"
#g_outputFileName="nbody_testSmallNumberOfStars_NS=5_b=8_sBar=150_beta=0_TMax=45_dt=0.001_printInt=1.out"
#g_outputFileName="nbody_testSmallNumberOfStars_NS=5_b=8_sBar=150_beta=0_TMax=45_dt=0.001_printInt=1.out"
#g_outputFileName="nbodyT01_testBench0721_NS=5_b=8_sBar=100_beta=0_TMax=40_dt=0.01_printInt=1.out"
#g_outputFileName="nbodyT02_testBench0721_NS=5_b=8_sBar=100_beta=0_TMax=40_dt=0.001_printInt=1.out"
#g_outputFileName="nbodyT03_testBench0721_NS=5_b=8_sBar=100_beta=0_TMax=40_dt=0.0001_printInt=1.out"
#g_outputFileName="nbodyT04_testBench0721_NS=300_b=1_sBar=200_beta=0_TMax=40_dt=0.001_printInt=1.out"
#g_outputFileName="nbodyT05_testBench0721_NS=300_b=1_sBar=200_beta=0_TMax=40_dt=0.0001_printInt=1.out"
#g_outputFileName="nbodyT06_testBench0721_NS=300_b=8_sBar=200_beta=0_TMax=40_dt=0.001_printInt=1.out"
#g_outputFileName="nbodyT07_testBench0721_NS=300_b=1_sBar=200_beta=0_TMax=200_dt=0.00001_printInt=1.out"
#g_outputFileName="nbodyT08_testBench0722_NS=9000_b=3_sBar=1000.0_beta=0_TMax=20_dt=0.001_printInt=0.5.out"
#g_outputFileName="nbodyT11_testGalacticWithoutDM_NS=1_b=3_sBar=1000.0_beta=0_TMax=1000_dt=0.001_printInt=10.out"
#g_outputFileName="nbodyT11_testGalacticWithDM_NS=1_b=3_sBar=1000.0_beta=0_TMax=1000_dt=0.001_printInt=10.out"
#g_outputFileName="nbodyT11_testGalacticWithDM_NS=1_b=3_sBar=1000.0_beta=0_TMax=30_dt=0.001_printInt=0.1.out"
#g_outputFileName="nbodyT11_testGalacticWithDM_NS=1_b=3_sBar=1000.0_beta=0_TMax=100_dt=0.001_printInt=0.1.out"
#g_outputFileName="nbodyT11_testGalacticWithDMAndCloser_NS=1_b=3_sBar=1000.0_beta=0_TMax=100_dt=0.001_printInt=0.1.out"
#g_outputFileName="nbodyT11_testGalacticMultiParticle_NS=4_b=3_sBar=250_beta=0_TMax=50_dt=0.0001_printInt=0.6.out"
#g_outputFileName="nbodyT11_testGalacticMultiParticle_NS=10_b=3_sBar=250_beta=0_TMax=50_dt=0.0001_printInt=0.6.out"
#g_outputFileName="nbodyT11_testGalacticMultiParticle_NS=100_b=3_sBar=250_beta=0_TMax=40_dt=0.0001_printInt=0.4.out"
#g_outputFileName="nbodyT12_testGMPIndicators_NS=250_b=3_sBar=300_beta=0_TMax=50_dt=0.0001_printInt=0.5.out"
#g_outputFileName="nbodyT12_testGMPIndicatorsTheoreticalPosition_NS=200_b=3_sBar=300_beta=0_TMax=50_dt=0.0001_printInt=0.5.out"
#g_outputFileName="nbodyT14_testPrepareTrajectories_NS=1_b=3_sBar=300_beta=0_TMax=500_dt=0.0001_printInt=1.1.out"
#g_outputFileName="nbodyT14_testPrepareTrajectories_NS=1_b=3_sBar=300_beta=0_TMax=100_dt=0.0001_printInt=1.1.out"
#g_outputFileName="nbodyT14_testPrepareTrajectories_NS=1_b=3_sBar=300_beta=0_TMax=200_dt=0.0001_printInt=1.1.out"
#g_outputFileName="nbodyT14_testPrepareTrajectories_NS=1_b=3_sBar=300_beta=0_TMax=300_dt=0.0001_printInt=1.1.out"
#g_outputFileName="nbodyT14_testPrepareTrajectories_NS=1_b=3_sBar=300_beta=0_TMax=150_dt=0.0001_printInt=1.1.out"
#g_outputFileName="nbodyT09_testBigInitialKE_NS=300_b=3_sBar=1000.0_beta=0_TMax=20_dt=0.001_printInt=0.5.out"
#g_outputFileName="nbodyT13_testGMPIndicatorsBigN_NS=2500_b=3_sBar=300_beta=0_TMax=100_dt=0.0001_printInt=0.666.out"
#g_outputFileName="nbodyT16_testGlobal_NS=600_b=3_sBar=300_beta=0_TMax=250_dt=0.001_printInt=0.8.out"
#g_outputFileName="nbodyT16_testGlobalControl_NS=600_b=3_sBar=300_beta=0_TMax=150_dt=0.0001_printInt=0.4.out"
#g_outputFileName=g_heavyOutputFilesFolder+"nbodyT20_testControlMassive_NS=5500_b=3_sBar=150_beta=0_TMax=150_dt=0.001_printInt=0.6.out"
#g_outputFileName="solarSystemRK4.out"
#g_outputFileName="solarSystemPECE.out"
#g_outputFileName="solarSystemPEC.out"
#g_outputFileName="solarSystemSIRK4.out"
#g_outputFileName="solarSystemSIPECE.out"
#g_outputFileName="solarSystemSIPEC.out"
