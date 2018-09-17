# MiniNBody
A simple N-Body simulator, able to support a galactic environnement (under the form of a gravitational field).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1420237.svg)](https://doi.org/10.5281/zenodo.1420237)

## Description
This is a simple N-Body simulator, based on simpler one by Michael Richmond in 2009.

It supports gravitational simulations with or without a galactic environnement.

Maximum number of particles is for the moment 9000, somehow limited by the compiler (to be corrected in the future).

The numerical schemes implemented are (from the slowest and more accurate to the fastest and least accurate) :
- 4th order Runge-Kutta (RK4),
- 5th order PECE ("Prediction-Evaluation-Correction-Evaluation", Adams-Bashforth for prediction, Adams-Moulton for correction and RK4 for initialisations),
- 5th order PEC ("Prediction-Evaluation-Correction", Adams-Bashforth for prediction, Adams-Moulton for correction and RK4 for initialisations),
- 2nd order Euler,
- 1st order symplectic Euler,
- and 1st order Euler.

The first 3 (RK4, PECE, PEC) were extensively tested and should be used. PEC is about twice faster than PECE, which is itself twice faster than RK4. More information about the schemes are disponible in the PDF report.

The galactic environnement is composed of (see below for the parametrisation) :
- a Plummer type bulb,
- a Miyamoto-Nagai type disk,
- a dark matter halo.

Initial positions, velocities and masses should be given in the input file. Furthermore, the value of the gravitational constant should be given too, as it encodes the dimensions used. For example :
- If units are those of SI (International System of Units, ie kilogram / meter / second), G = 6.67e-11 should be used.
- If units are astrophycal units (solar mass / parsec / megayear), G = 4.302e-3 should be used.
- And so forth with other units.

The gravitational field, if activated in input file, is implemented to represent a galaxy located at O = [0.0, 0.0, 0.0] and which disk is in the plane z = 0. Initial positions of stars should take this into account.

The name of the output file is also to be specified in the input file.

The Python scripts can be used to generate initial conditions and read and analyse outputs. That being said, note they were used in a study on globular clusters and because of that might be quite specific to it. But it can be used as a starting point for different studies.

## Usage
To compile the executable :
1) Use the following command line :
```
make mininbody
```

To run a simulation :
1) Have a correctely formatted input file, let us say "`input.in`".
2) Use the following command line :
```
./mininbody input=input.in [verbose]
```

`[verbose]` can be :
`verbose`   to set verbosity level to 1 (should print useful informations during the run)
`verbose=#` where # is a number to set verbosity level to this number (the greater the number, the more informations are printed)

If verbose is absent, it is set to the default value in the source code.

See the example input file, "`solarSystem.in`", for more informations about the parametrisation of the simulator.

## Further Details
Plummer type potential :
- G * M_b * ( r**2 + a_b**2 )**(-0.5)

Miyamoto-Nagai type potential :
- G * M_d * ( r_c**2 + ( a_d + ( z**2 + h_d**2 )**(0.5) )**2 )**(-0.5)

Dark matter halo potential :
- 0.5 * V_h**2 * log( r**2 + a_h**2 )

G is the gravitational constant.
M_b, a_b, M_d, a_d, h_d, V_h, a_h are parameters.
r is the spherical radius from the center of the galaxy.
r_c is the cylindrical radius from the center of the galaxy.
z is the cylindrical altitude from the disk of the galaxy.

As an order of magnitude, a 5000 particles PECE (2 calls to evaluation function) simulation over 150 Myr and with a timestep of 1e-3 Myr lasts in roughly 5 days with 22 CPU threads.
