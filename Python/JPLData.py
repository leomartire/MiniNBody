import numpy as np

# Positions of the solar system planets on the 01/01/2016 at 12:00:00, according to JPL Horizons.
JPLX=np.zeros([3, 10])
JPLV=np.zeros([3, 10])
JPLX[:, 0]=[ 0.0,                 0.0,                 0.0]
JPLX[:, 1]=[ 1.310223587392e-06,  8.580816694232e-07, -5.009134244873e-08]
JPLX[:, 2]=[-3.471239529760e-06, -3.331553484641e-07,  1.957445583004e-07]
JPLX[:, 3]=[-8.502260838489e-07,  4.690786870465e-06, -1.671248900225e-10]
JPLX[:, 4]=[-7.993827965524e-06,  7.934218034735e-07,  2.128152384722e-07]
JPLX[:, 5]=[-2.509962943043e-05,  7.692685025476e-06,  5.296908268160e-07]
JPLX[:, 6]=[-1.799598626698e-05, -4.504897582105e-05,  1.499439565382e-06]
JPLX[:, 7]=[ 9.147303018797e-05,  3.178045910651e-05, -1.066449688087e-06]
JPLX[:, 8]=[ 1.355122209249e-04, -5.224330007674e-05, -2.046939872848e-06]
JPLX[:, 9]=[ 4.136911818921e-05, -1.545498672466e-04,  4.566263859251e-06]
JPLV[:, 0]=[ 0.0,                 0.0,                 0.0]
JPLV[:, 1]=[-3.623081834546e+01,  4.287575030099e+01,  6.827364904599e+00]
JPLV[:, 2]=[ 3.131619220556e+00, -3.501604272439e+01, -6.608043301519e-01]
JPLV[:, 3]=[-2.979634053627e+01, -5.413765322989e+00, -7.004340617254e-04]
JPLV[:, 4]=[-1.489256286581e+00, -2.204169894656e+01, -4.253352685762e-01]
JPLV[:, 5]=[-3.986538591082e+00, -1.188510556980e+01,  1.386795199719e-01]
JPLV[:, 6]=[ 8.440083434199e+00, -3.622627391282e+00, -2.726474518813e-01]
JPLV[:, 7]=[-2.286324663824e+00,  6.103400164144e+00,  5.208249810334e-02]
JPLV[:, 8]=[ 1.917310382482e+00,  5.092683927276e+00, -1.487752960809e-01]
JPLV[:, 9]=[ 5.376692264797e+00,  2.870885353150e-01, -1.593839800048e+00]