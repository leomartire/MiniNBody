from util import criticalBadValue
import numpy as np

g_galaxyModelNames="\"isolated\", \"control\", \"colongitudinal\", \"radial\", \"brutDisk\" or \"brutBulge\""

g_Mb=9.9*1e9 # bulge, total mass (in [M] = 1 solar mass)
g_ab=0.25*1e3 # bulge, radial scale (in [L] = 1 pc)
g_Md=72*1e9 # disk, total mass (in [M] = 1 solar mass)
g_ad=3.5*1e3 # disk, radial scale (in [L] = 1 pc)
g_hd=0.4*1e3 # disk, vertical scale (in [L] = 1 pc)
g_Vh=2.2e5 # halo, rotation curve (constant value) (in m/s, converted just below)
g_ah=20*1e3 # halo, radial scale (in [L] = 1 pc)

def getGalaxyParameters() :
  return(g_Mb, g_ab, g_Md, g_ad, g_hd, g_Vh, g_ah)

def generateGalaxyModel(param) :
  if param=="colongitudinal" :
    g_GCPositionRadial=7e3 # cylindrical radial position of the GC (in [L] = 1 pc)
    g_GCPositionAltitude=6e3 # cylindrical altitude of the GC (in [L] = 1 pc)
    g_GCVelocityModule=100e3 # GC global speed (in m/s, to be converted later)
    g_GCVelocityR=-2**0.5 # radial component of the GC global velocity
    g_GCVelocityCol=13**0.5 # cologitudinal component of the GC global velocity (u_theta, along polar direction or colatitude)
    g_GCVelocityLon=0 # logitudinal component of the GC global velocity (u_phi, along longitude)
    print("Suggered simulation duration : 500 Myr (minimum for one disk crossing : 50 Myr).")
  elif param=="radial" :
    g_GCPositionRadial=2.5e3 # cylindrical radial position of the GC (in [L] = 1 pc)
    g_GCPositionAltitude=5e3 # cylindrical altitude of the GC (in [L] = 1 pc)
    g_GCVelocityModule=100e3 # GC global speed (in m/s, to be converted later)
    g_GCVelocityR=-5000 # radial component of the GC global velocity
    g_GCVelocityCol=5000*np.tan(np.arcsin(0.1)) # cologitudinal component of the GC global velocity (u_theta, along polar direction or colatitude)
    g_GCVelocityLon=0 # logitudinal component of the GC global velocity (u_phi, along longitude)
    print("Suggered simulation duration : 200 Myr (minimum for one disk crossing : 30 Myr).")
  elif param=="control" :
    g_GCPositionRadial=-20e3 # cylindrical radial position of the GC (in [L] = 1 pc)
    g_GCPositionAltitude=20e3 # cylindrical altitude of the GC (in [L] = 1 pc)
    g_GCVelocityModule=300e3 # GC global speed (in m/s, to be converted later)
    # see velocity below
    print("Suggered simulation duration : 150 Myr.")
  elif param=="isolated" :
    # see below
    print("Suggered simulation duration : whatever.")
  elif param=="brutDisk" :
    g_GCPositionRadial=6e3 # cylindrical radial position of the GC (in [L] = 1 pc)
    g_GCPositionAltitude=7e3 # cylindrical altitude of the GC (in [L] = 1 pc)
    g_GCVelocityModule=100e3 # GC global speed (in m/s, to be converted later)
    # see velocity below
    print("Suggered simulation duration : 75 Myr.")
  elif param=="brutBulge" :
    g_GCPositionRadial=0 # cylindrical radial position of the GC (in [L] = 1 pc)
    g_GCPositionAltitude=5e3 # cylindrical altitude of the GC (in [L] = 1 pc)
    g_GCVelocityModule=150e3 # GC global speed (in m/s, to be converted later)
   # see velocity below
    print("Suggered simulation duration : 75 Myr.")
  else :
    criticalBadValue("param", "generateGalaxyModel")
  
  if param=="isolated" :
    g_GCPosition=-1
  else :
    g_GCPosition=np.array([[g_GCPositionRadial],[0],[g_GCPositionAltitude]]) # GC is glued to the y=0 plane
  
  if param=="control" :
    g_GCVelocity=np.array([[g_GCVelocityModule], [0], [0]])
  elif param=="isolated" :
    g_GCVelocity=-1
  elif param in ["brutDisk", "brutBulge"] :
    g_GCVelocity=np.array([[0], [0], [-g_GCVelocityModule]])
  else :
    tmp_normX=np.linalg.norm(g_GCPosition)
    tmp_GCTVel=g_GCVelocityCol*np.array([[g_GCPosition[2, 0]], [0], [-g_GCPosition[0, 0]]])/tmp_normX
    tmp_GCRVel=g_GCVelocityR*g_GCPosition/tmp_normX
    tmp_GCLVel=np.array([[0], [g_GCVelocityLon], [0]])
    g_GCVelocity=tmp_GCTVel+tmp_GCRVel+tmp_GCLVel
    g_GCVelocity=g_GCVelocityModule*g_GCVelocity/np.linalg.norm(g_GCVelocity)
  
  return(g_GCPosition, g_GCVelocity, g_Mb, g_ab, g_Md, g_ad, g_hd, g_Vh, g_ah)