/*-------------------------------------------------------------
 * Title       : MiniNBody.c
 * Author      : Léo Martire, Michael Richmond.
 * Date        : 2016.
 * Description :
 Implementation of a simple N-Body simulation. This program is
 amply based on Michael Richmond's NBody program. Tweaks,
 optimisations, more features and advanced schemes have been
 added in order to fulfil a study on globular clusters.
 * Usage :
 1) Compile using the provided Makefile.
 2) As command line :
	    nbody input=inputfile [verbose=]
	  where :
	    - inputfile is an ASCII text file with description of
                  each body, plus some controlling parameters
                  (see provided input file solarSystem.in for
                  further explanations)
	    - verbose   if present, set verbosity level to 1
                  if not, set verbosity level to 0
	    - verbose=N if present, set verbosity level to N
-------------------------------------------------------------*/

/* Includes. ------------------------------------------------*/
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
/*-----------------------------------------------------------*/

/* Defines. -------------------------------------------------*/
#undef DEBUG
#define COMMENT_CHAR '#' // any line starting with this in input file is ignored
#define LINELEN 1024 // max length of lines in input file
#define MAX_NBODY 9000 // maximum number of bodies we can handle (change this according to the type of simulation : no need to allocate 9000 spaces to simulate 10 particles)
#define MAX_TECHNIQUE 10 // maximum number of intergration techniques we can handle
#define NAMELEN 30 // max length of names of objects
#define NBUF 1024
/*-----------------------------------------------------------*/

/* Type definitions. ----------------------------------------*/
typedef int(*PFI)(double, double *); // pointer to a function returning an integer
typedef struct s_vector { // vector
  double val[3];
}
VECTOR;
typedef struct s_body { // body (state)
  int index;
  char name[NAMELEN];
  double mass;
  double pos[3];
  double vel[3];
}
BODY;
typedef struct s_dbody { // body derivative (state derivative)
  double dp[3];
  double dv[3];
}
DBODY;
typedef struct s_technique { // integration technique
  char name[NAMELEN];
  PFI func;
}
TECHNIQUE;
/*-----------------------------------------------------------*/

/* Macros. --------------------------------------------------*/
/* 
 * set the given VECTOR elements to have the difference in 
 * positions between bodies i and j, in the sense
 *     g_body_array[i].pos[0]-g_body_array[j].pos[0]
 * etc.
 */
#define SET_POS_DIFF(vec, i, j)\
vec.val[0] = g_body_array[i].pos[0] - g_body_array[j].pos[0];\
vec.val[1] = g_body_array[i].pos[1] - g_body_array[j].pos[1];\
vec.val[2] = g_body_array[i].pos[2] - g_body_array[j].pos[2];

/* the magnitude of a VECTOR */
#define VECTOR_MAG(vec)\
  (sqrt(vec.val[0] * vec.val[0] + vec.val[1] * vec.val[1] + \
    vec.val[2] * vec.val[2]))

/* the square of the magnitude of a VECTOR */
#define VECTOR_MAG_SQUARED(vec)\
  (vec.val[0] * vec.val[0] + vec.val[1] * vec.val[1] + \
    vec.val[2] * vec.val[2])

/* the SQUARE of distance between two bodies */
#define DIST_SQUARED(i, j)\
  (((g_body_array[i].pos[0] - g_body_array[j].pos[0]) * \
      (g_body_array[i].pos[0] - g_body_array[j].pos[0])) + \
    ((g_body_array[i].pos[1] - g_body_array[j].pos[1]) * \
      (g_body_array[i].pos[1] - g_body_array[j].pos[1])) + \
    ((g_body_array[i].pos[2] - g_body_array[j].pos[2]) * \
      (g_body_array[i].pos[2] - g_body_array[j].pos[2])))

/* the difference of two vectors */
#define VECTOR_SUBTRACT(a, b, difference)\
difference.val[0] = a.val[0] - b.val[0];\
difference.val[1] = a.val[1] - b.val[1];\
difference.val[2] = a.val[2] - b.val[2];

/* the cross product of two vectors */
#define VECTOR_CROSS(a, b, product)\
product.val[0] = a.val[1] * b.val[2] - a.val[2] * b.val[1];\
product.val[1] = a.val[2] * b.val[0] - a.val[0] * b.val[2];\
product.val[2] = a.val[0] * b.val[1] - a.val[1] * b.val[0];

#define max(a,b) \
   ({ __typeof__ (a) _a=(a); \
       __typeof__ (b) _b=(b); \
     _a>_b ? _a : _b; })

#define min(a,b) \
   ({ __typeof__ (a) _a=(a); \
       __typeof__ (b) _b=(b); \
     _a<_b ? _a : _b; })
  /*-----------------------------------------------------------*/

/* Global variables. ----------------------------------------*/
BODY g_body_array[MAX_NBODY]; // allocate space for bodies
BODY pyn1[MAX_NBODY]; // PECE : predicted state
BODY yn1[MAX_NBODY]; // PECE : new state
DBODY fn1[MAX_NBODY]; // PECE : new derivative state
DBODY fp0[MAX_NBODY]; // PECE : current derivative state
DBODY fp1[MAX_NBODY]; // PECE : 1 step backwards derivative state
DBODY fp2[MAX_NBODY]; // PECE : 2 steps backwards derivative state
DBODY fp3[MAX_NBODY]; // PECE : 3 steps backwards derivative state
DBODY pfn1[MAX_NBODY]; // PECE : predicted derivative state
FILE *g_outfile_fp; // FILE pointer for output
PFI g_integration_func; // PFI pointer for techniques
TECHNIQUE g_technique_array[MAX_TECHNIQUE]; // allocate space for techniques
VECTOR g_cur_forces[MAX_NBODY][MAX_NBODY]; // variable made global to make sure enough memory space is allocated and to enable faster access for simulations with high number of bodies.
VECTOR tmpForcesForDerivation[MAX_NBODY]; // PECE : temporary vector for derivative state calculation
double g_AB_b03 =   55.0 /  24.0; // PECE : coefficient for Adams-Bashforth scheme with 3+1 steps
double g_AB_b13 =  -59.0 /  24.0; // PECE : coefficient for Adams-Bashforth scheme with 3+1 steps
double g_AB_b23 =   37.0 /  24.0; // PECE : coefficient for Adams-Bashforth scheme with 3+1 steps
double g_AB_b33 =   -9.0 /  24.0; // PECE : coefficient for Adams-Bashforth scheme with 3+1 steps
double g_AM_b03 =  646.0 / 720.0; // PECE : coefficient for Adams-Moulton scheme with 3+1 steps
double g_AM_b13 = -264.0 / 720.0; // PECE : coefficient for Adams-Moulton scheme with 3+1 steps
double g_AM_b23 =  106.0 / 720.0; // PECE : coefficient for Adams-Moulton scheme with 3+1 steps
double g_AM_b33 =  -19.0 / 720.0; // PECE : coefficient for Adams-Moulton scheme with 3+1 steps
double g_AM_bp3 =  251.0 / 720.0; // PECE : coefficient for Adams-Moulton scheme with 3+1 steps
double g_G = -1; // gravitational constant (in [L^{3}][M^{-1}][T^{-2}], (to be set by reading the input file))
double g_ab = -1; // galactic parameter (to be set by reading the input file)
double g_ad = -1; // galactic parameter (to be set by reading the input file)
double g_ah = -1; // galactic parameter (to be set by reading the input file)
double g_duration = -1; // (to be set by reading the input file)
double g_hd = -1; // galactic parameter (to be set by reading the input file)
double g_mb = -1; // galactic parameter (to be set by reading the input file)
double g_md = -1; // galactic parameter (to be set by reading the input file)
double g_print_interval = -1; // (to be set by reading the input file)
double g_softening = -1; // softening (in [L], to be set by reading the input file)
double g_timestep = -1; // (to be set by reading the input file)
double g_vh = -1; // galactic parameter (to be set by reading the input file)
int g_body_number = -1; // (to be set by reading the input file)
int g_minimumRemTimePrintInt = 5; // minimum interval (in seconds) between printing of remaining time (to prevent flooding on stdout)
int g_galactic_flag = -1; // (to be set by reading the input file)
int g_nbStarsPerThread = 250; // wanted number of stars treated by thread
int g_recenter_flag = -1; // (to be set by reading the input file)
int g_technique_number = 0; // (to be set by reading the input file)
int g_usePECE_flag = -1; // PECE : use PECE (0 if no, 1 if yes) ?
int g_usePEC_flag = -1; // PEC : use PEC (0 if no, 1 if yes) ?
int g_verbose_flag = 0; // if set to 1, print messages as we execute (modifiable by passing arguments when executing)
/*-----------------------------------------------------------*/

/* Declarations of functions. -------------------------------*/
static double calc_total_gpe(void);
static double calc_total_ke(void);
static int AdamsBashforth(BODY *outState, double step, BODY *state0, DBODY *dStateB0, DBODY *dStateB1, DBODY *dStateB2, DBODY *dStateB3);
static int AdamsMoulton(BODY *outState, double step, BODY *state0, DBODY *predictedDState, DBODY *dStateB0, DBODY *dStateB1, DBODY *dStateB2, DBODY *dStateB3);
static int add_galactic_effect(VECTOR *forces);
static int calc_grav_forces(BODY *body_array, VECTOR *forces);
static int calc_total_angmom(VECTOR *result);
static int computeTotalForceOnBodies(VECTOR *forces);
static int print_positions(FILE *outfile_fp, double currentSimulationTime);
static int read_input(char *filename);
static int recenter_velocities(void);
static int getGravitationalForce(BODY body, VECTOR *tmpGravitationalForce);
static int shiftTemporaryStates(DBODY *dStateB3, DBODY *dStateB2, DBODY *dStateB1, DBODY *dStateB0, DBODY *newDState, BODY *state0, BODY *newState);
static int stateToDerivativeState(BODY *b, DBODY *db);
static int tech_euler_1(double suggested_timestep, double *actual_timestep);
static int tech_euler_1a(double suggested_timestep, double *actual_timestep);
static int tech_euler_2(double suggested_timestep, double *actual_timestep);
static int tech_rk4(double suggested_timestep, double *actual_timestep);
static void copy_bodies(int num_bodies, BODY *from_array, BODY *to_array);
static void printError(char *funcName, char *msg);
void kek(double suggested_timestep, double *actual_timestep);
/*-----------------------------------------------------------*/

int main(int argc, char *argv[]) {
  /* Initialize stuff. --------------------------------------*/
  VECTOR initial_ang_mom, final_ang_mom;
  char inputfile_name[NBUF];
  char timeUnit[8];
  double actual_step, currentSimulationTime, energyTimeTaken1, energyTimeTaken2, final_gpe, final_ke, final_tot_e, initial_gpe, initial_ke, initial_tot_e, mainLoopClock, mainLoopTimeTaken, printing_clock, remainingRealTime, suggested_step, previousInfoClock;
  int i, n, retval, remainingTimePrintModulo;

  inputfile_name[0] = '\0';
  strcpy(timeUnit, "seconds");
  for (i = 0; i < MAX_NBODY; i++) {
    g_body_array[i].index = -1;
    strcpy(g_body_array[i].name, "");
    g_body_array[i].mass = -1;
  }
  g_outfile_fp = (FILE *) NULL;
  strcpy(g_technique_array[g_technique_number].name, "E1");
  g_technique_array[g_technique_number++].func = tech_euler_1;
  strcpy(g_technique_array[g_technique_number].name, "E1a");
  g_technique_array[g_technique_number++].func = tech_euler_1a;
  strcpy(g_technique_array[g_technique_number].name, "E2");
  g_technique_array[g_technique_number++].func = tech_euler_2;
  strcpy(g_technique_array[g_technique_number].name, "RK4");
  g_technique_array[g_technique_number++].func = tech_rk4;
  /*---------------------------------------------------------*/

  /* Parse arguments. ---------------------------------------*/
  if (argc < 2) {
    fprintf(stderr, "usage: nbody input=  [verbose=] \n");
    exit(1);
  }
  for (i = 1; i < argc; i++) {
    if (g_verbose_flag > 0) {
      printf(" arg %d is ..%s.. \n", i, argv[i]);
    }
    /* read input file name */
    if (strncmp(argv[i], "input=", 6) == 0) {
      if (strlen(argv[i]) >= NBUF) {
        fprintf(stderr, "inputfile name ..%s.. is too long, max %d chars. \n", argv[i], (int) strlen(argv[i]));
        exit(1);
      }
      if (sscanf(argv[i] + 6, "%s", inputfile_name) != 1) {
        fprintf(stderr, "can't read inputfile_name from ..%s.. \n", argv[i] + 6);
        exit(1);
      }
      if (g_verbose_flag > 0) {
        printf(" read input file name ..%s.. \n", inputfile_name);
      }
    }
    if (strcmp(argv[i], "verbose") == 0) {
      g_verbose_flag = 1;
      printf(" set verbose level to ..%d.. \n", g_verbose_flag);
    }
    if (strncmp(argv[i], "verbose=", 8) == 0) {
      if (sscanf(argv[i] + 8, "%d", &g_verbose_flag) != 1) {
        fprintf(stderr, "can't read verbosity level from ..%s.. \n", argv[i] + 8);
        exit(1);
      }
      if (g_verbose_flag > 0) {
        printf(" set verbose level to ..%d.. \n", g_verbose_flag);
      }
    }
  }
  if (inputfile_name[0] == '\0') {
    fprintf(stderr, "no input= argument provided ?! \n");
    exit(1);
  } // Check to make sure we read all required arguments.
  printf("\n> Starting input file reading.\n");
  if (read_input(inputfile_name) != 0) {
    fprintf(stderr, "read_input fails \n");
    exit(1);
  } /* read all input information */
  if (g_recenter_flag == 1) {
    if (recenter_velocities() != 0) {
      fprintf(stderr, "recenter_velocities returns with error \n");
      exit(1);
    }
  } // If desired, add a constant velocity to all bodies so that the center of mass of the system is motionless.
  /*---------------------------------------------------------*/

  /* Compute initial quantities. ----------------------------*/
  printf("\n");
  double energyClock = clock(); // check time before
  initial_ke = calc_total_ke();
  if (g_verbose_flag > 0) {
    printf("> Initial KE : %12.5e.\n", initial_ke);
  }
  initial_gpe = calc_total_gpe();
  if (g_verbose_flag > 0) {
    printf("> Initial GPE : %12.5e.\n", initial_gpe);
  }
  initial_tot_e = initial_ke + initial_gpe;
  if (g_verbose_flag > 0) {
    printf("> Initial E   : %12.5e.\n", initial_tot_e);
  }
  if (calc_total_angmom( & initial_ang_mom) != 0) {
    fprintf(stderr, "calc_total_angmom fails for initial \n");
    exit(1);
  }
  if (g_verbose_flag > 0) {
    printf("> Initial angular momentum : [%12.5e, %12.5e, %12.5e].\n",
      initial_ang_mom.val[0],
      initial_ang_mom.val[1],
      initial_ang_mom.val[2]);
  }
  energyClock = clock() - energyClock; // check time after
  energyTimeTaken1 = ((double) energyClock) / CLOCKS_PER_SEC; // time taken in seconds
  /*---------------------------------------------------------*/

  /* Print initial states to output. ------------------------*/
  currentSimulationTime = 0;
  if (print_positions(g_outfile_fp, currentSimulationTime) != 0) {
    fprintf(stderr, "Iteration %16d, time %9.4e [T].\n", 0, currentSimulationTime);
    printError("main", "Function print_positions failed.");
    exit(1);
  }
  /*---------------------------------------------------------*/

  /* Enter main loop. ---------------------------------------*/
  printf("\n");
  printf("> Available threads : %d.\n", omp_get_max_threads());
  omp_set_num_threads(min(max((int)(floor(g_body_number / g_nbStarsPerThread)), 1), omp_get_max_threads()));
  //omp_set_num_threads(3);
  #pragma omp parallel
  printf("> Launching simulation with %d thread(s) (i.e., roughly, %d star(s) per thread).\n", omp_get_num_threads(), (int)(g_body_number / omp_get_num_threads()));

  printing_clock = 0.0; // initialise printing clock
  mainLoopClock = clock(); // save timer before entering the main loop
  n = 0; // to follow number of iterations
  if (g_verbose_flag > 0) {
    remainingTimePrintModulo = 20000;
    previousInfoClock = clock();
  } // every how many iterations print informations about remaining time and dummy value for first previousRemainingRealTime
  printf("\n> Entering main integration loop.\n");
  while (currentSimulationTime <= g_duration) {
    /* Call the integration -*/
    /* function.            -*/
    if (g_verbose_flag > 1) {
      printf(" >> About to enter the integration function (t=%9.4e).\n", currentSimulationTime);
    }
    suggested_step = g_timestep;
    actual_step = g_timestep; // safety, in case it does not get allocated in the upcoming calls
    if (g_usePECE_flag == 0 && g_usePEC_flag == 0) { // PECE and PEC are desactivated : just use RK4 all the way.
      retval = (*(g_integration_func))(suggested_step, &actual_step);
      if (retval != 0) {
        printError("main", "Integration function failed.");
        return (1);
      }
      g_timestep = actual_step;
    } else { // PECE or PEC is activated.
      if (n <= 3) { // For the first 4 steps of PECE, call RK4.
        retval = (*(g_integration_func))(suggested_step, &actual_step);
        if (retval != 0) {
          printError("main", "Integration function failed.");
          return (1);
        }
        g_timestep = actual_step;

        if (n == 0) {
          stateToDerivativeState(g_body_array, fp3);
        }
        if (n == 1) {
          stateToDerivativeState(g_body_array, fp2);
        }
        if (n == 2) {
          stateToDerivativeState(g_body_array, fp1);
        }
        if (n == 3) {
          stateToDerivativeState(g_body_array, fp0);
        }
      } else { // For n>=4, call PECE/PEC AB AM.
        AdamsBashforth(pyn1, g_timestep, g_body_array, fp0, fp1, fp2, fp3); // prediction (Adams Bashforth)
        //printf("n=%d, AB, position : [%6.2f, %6.2f, %6.2f], velocity : [%6.2f, %6.2f, %6.2f].\n", n, g_body_array[1].pos[0], g_body_array[1].pos[1], g_body_array[1].pos[2], g_body_array[1].vel[0], g_body_array[1].vel[1], g_body_array[1].vel[2]);
        stateToDerivativeState(pyn1, pfn1); // evaluation (with forces computed using predicted state)
        //printf("n=%d, evaluation : pyn1 : [%6.2f, %6.2f, %6.2f][%6.2f, %6.2f, %6.2f], pfn1 : [%6.2f, %6.2f, %6.2f][%6.2f, %6.2f, %6.2f]\n", n, pyn1[1].pos[0], pyn1[1].pos[1], pyn1[1].pos[2], pyn1[1].vel[0], pyn1[1].vel[1], pyn1[1].vel[2], pfn1[1].dp[0], pfn1[1].dp[1], pfn1[1].dp[2], pfn1[1].dv[0], pfn1[1].dv[1], pfn1[1].dv[2]);
        AdamsMoulton(yn1, g_timestep, g_body_array, pfn1, fp0, fp1, fp2, fp3); // correction (Adams Moulton)
        //printf("n=%d, AM, position : [%6.2f, %6.2f, %6.2f], velocity : [%6.2f, %6.2f, %6.2f].\n", n, g_body_array[1].pos[0], g_body_array[1].pos[1], g_body_array[1].pos[2], g_body_array[1].vel[0], g_body_array[1].vel[1], g_body_array[1].vel[2]);
        if (g_usePEC_flag == 1) { // If only PEC, use pfn1 as fn1.
          shiftTemporaryStates(fp3, fp2, fp1, fp0, pfn1, g_body_array, yn1);
        } else { // If PECE, evaluate (with forces computed using corrected state).
          stateToDerivativeState(yn1, fn1);
          shiftTemporaryStates(fp3, fp2, fp1, fp0, fn1, g_body_array, yn1);
        }
      }
    }
    currentSimulationTime += g_timestep;
    n += 1;
    /*-----------------------*/
    /* Update printing    ---*/
    /* clock and print to ---*/
    /* output file.       ---*/
    printing_clock += actual_step;
    if (printing_clock >= g_print_interval) {
      /* print the new positions, etc. */
      if (print_positions(g_outfile_fp, currentSimulationTime) != 0) {
        fprintf(stderr, "Iteration %16d, time %9.4e [T].\n", n, currentSimulationTime);
        printError("main", "Function print_positions failed.");
        exit(1);
      }
      printing_clock = 0.0;
    }
    /*-----------------------*/
    /* Print information ----*/
    /* about simulation run. */
    if (g_verbose_flag > 0) {
      if (n == 100 || n == 1000 || n == 2000 || n % remainingTimePrintModulo == 0) { // after first 100, first 1000, first 2000 and then every 20000 integrations, estimate the remaining time
        remainingRealTime = (((clock() - mainLoopClock) / currentSimulationTime) * (g_duration - currentSimulationTime)) / CLOCKS_PER_SEC; // estimated remaining time in seconds
        if ((n > 2000) && ((clock() - previousInfoClock) < g_minimumRemTimePrintInt * CLOCKS_PER_SEC)) {
          remainingTimePrintModulo *= 2;
        } // if informations prints too fast, raise interval in terms of iterations
        if (remainingRealTime < 60) {
          strcpy(timeUnit, "seconds");
        }
        if (remainingRealTime >= 60 && remainingRealTime < 3600) {
          strcpy(timeUnit, "minutes");
          remainingRealTime /= 60;
        }
        if (remainingRealTime >= 3600) {
          strcpy(timeUnit, "hours");
          remainingRealTime /= 3600;
        }
        printf("> Iteration %16d, time %9.4e [T], estimated remaining time : %6.2f %s.\n", n, currentSimulationTime, remainingRealTime, timeUnit);
        previousInfoClock = clock();
      }
    }
    /*-----------------------*/
  }
  printf("> Main integration loop finished.\n");
  mainLoopTimeTaken = ((double)(clock() - mainLoopClock) / CLOCKS_PER_SEC); // time taken by main loop (in seconds)
  /*---------------------------------------------------------*/

  /* Print final states to output. --------------------------*/
  if (print_positions(g_outfile_fp, currentSimulationTime) != 0) {
    fprintf(stderr, "Iteration %16d, time %9.4e [T].\n", n, currentSimulationTime);
    printError("main", "Function print_positions failed.");
    exit(1);
  }
  /*---------------------------------------------------------*/

  /* Compute final quantities. ------------------------------*/
  printf("\n");
  energyClock = clock(); // check time before
  final_ke = calc_total_ke();
  if (g_verbose_flag > 0) {
    printf("> Final KE  : %12.5e.\n", final_ke);
  }
  final_gpe = calc_total_gpe();
  if (g_verbose_flag > 0) {
    printf("> Final GPE : %12.5e.\n", final_gpe);
  }
  final_tot_e = final_ke + final_gpe;
  if (g_verbose_flag > 0) {
    printf("> Final E   : %12.5e.\n", final_tot_e);
  }
  if (calc_total_angmom( & final_ang_mom) != 0) {
    fprintf(stderr, "calc_total_angmom fails for final \n");
    exit(1);
  }
  if (g_verbose_flag > 0) {
    printf("> Final angular momentum : [%12.5e, %12.5e, %12.5e].\n",
      final_ang_mom.val[0],
      final_ang_mom.val[1],
      final_ang_mom.val[2]);
  }
  energyClock = clock() - energyClock; // check time after
  energyTimeTaken2 = ((double) energyClock) / CLOCKS_PER_SEC; // time taken in seconds
  /*---------------------------------------------------------*/

  /* Compute change in quantities. --------------------------*/
  {
    double delta_e, fraction_delta_e;
    double start_angmom_mag, end_angmom_mag, delta_angmom_mag;
    double fraction_delta_angmom_mag;

    delta_e = final_tot_e - initial_tot_e;
    fraction_delta_e = delta_e / fabs(initial_tot_e);
    if (g_verbose_flag > 0) {
      printf("> Delta_E     =%12.5e,\n  relative difference=%12.5e.\n",
        delta_e, fraction_delta_e);
    }

    start_angmom_mag = VECTOR_MAG(initial_ang_mom);
    end_angmom_mag = VECTOR_MAG(final_ang_mom);
    delta_angmom_mag = end_angmom_mag - start_angmom_mag;
    fraction_delta_angmom_mag = delta_angmom_mag / fabs(start_angmom_mag);
    if (g_verbose_flag > 0) {
      printf("> Delta_angmom=%12.5e,\n  relative difference=%12.5e.\n",
        delta_angmom_mag, fraction_delta_angmom_mag);
    }
  }
  /*---------------------------------------------------------*/

  /* Prepare to exit. ---------------------------------------*/
  if (g_verbose_flag > 0) {
    if (mainLoopTimeTaken < 60) {
      strcpy(timeUnit, "seconds");
    }
    if (mainLoopTimeTaken >= 60 && mainLoopTimeTaken < 3600) {
      strcpy(timeUnit, "minutes");
      mainLoopTimeTaken /= 60;
    }
    if (mainLoopTimeTaken >= 3600) {
      strcpy(timeUnit, "hours");
      mainLoopTimeTaken /= 3600;
    }
    printf("\n> All done.\n");
    printf(">  Main loop was              %12d iterations (integrations) long.\n", n);
    printf(">  Main loop took             %12.2f %s to execute.\n", mainLoopTimeTaken, timeUnit);
    printf(">  Energies computations took %12.2f seconds to execute.\n", energyTimeTaken1 + energyTimeTaken2);
  }
  // Close file opened in read_input routine.
  fclose(g_outfile_fp);
  exit(0);
  /*---------------------------------------------------------*/
}

static int stateToDerivativeState(BODY * b, DBODY * db) {
  // Get the state derivative of a state. In other terms, for the PDE Y'=f(Y), apply f to a Y to get Y'.
  // @param *b array of bodies (in other terms, Y)
  // @param *db (output) derivative state (in other terms, Y')
  int i, j, c;
  VECTOR dist, dist_frac, tmpGravitationalForce;
  double dist_tot;
  double dist_squared;
  double top, bottom, force_mag;
  #pragma omp for
  for (i = 0; i < g_body_number; i++) { // on body i, ...
    for (c = 0; c < 3; c++) { // initialise force at 0, ...
      tmpForcesForDerivation[i].val[c] = 0.0;
    }
    for (j = i; j < g_body_number; j++) { // compute effect of body j, ...
      if (i == j) {
        g_cur_forces[i][j].val[0] = 0.0;
        g_cur_forces[i][j].val[1] = 0.0;
        g_cur_forces[i][j].val[2] = 0.0;
        continue;
      }
      dist.val[0] = b[i].pos[0] - b[j].pos[0];
      dist.val[1] = b[i].pos[1] - b[j].pos[1];
      dist.val[2] = b[i].pos[2] - b[j].pos[2];
      dist_squared = VECTOR_MAG_SQUARED(dist);
      dist_tot = sqrt(dist_squared);
      /* if(dist_tot<=0.0){
      	fprintf(stderr, 
      			" calc_grav_forces: distance of %le -- quitting \n", dist_tot);
      	return(1);
      } */
      top = g_G * b[i].mass * b[j].mass;
      bottom = (dist_tot + g_softening) * (dist_tot + g_softening);
      force_mag = top / bottom;
      for (c = 0; c < 3; c++) { // and for each coordinate, prepare the value and add it as contribution
        dist_frac.val[c] = dist.val[c] / dist_tot;
        g_cur_forces[i][j].val[c] = 0.0 - (force_mag * dist_frac.val[c]);
        g_cur_forces[j][i].val[c] = 0.0 - g_cur_forces[i][j].val[c];

        tmpForcesForDerivation[i].val[c] += -(force_mag * dist_frac.val[c]);
        tmpForcesForDerivation[j].val[c] += (force_mag * dist_frac.val[c]);
      }
    }
    for (j = 0; j < g_body_number; j++) { // add contribution of each body j on body i
      for (c = 0; c < 3; c++) {
        tmpForcesForDerivation[i].val[c] += g_cur_forces[i][j].val[c];
      }
    }
    if (g_galactic_flag == 1) { // if the effect of a galaxy is wanted, add it to the current force on body i
      getGravitationalForce(g_body_array[i], &tmpGravitationalForce);
      for (c = 0; c < 3; c++) {
        tmpForcesForDerivation[i].val[c] += tmpGravitationalForce.val[c];
      }
    }
    for (c = 0; c < 3; c++) { // when all contributions have been added, for body i, apply the derivation
      db[i].dp[c] = b[i].vel[c]; // dp=v
      db[i].dv[c] = tmpForcesForDerivation[i].val[c] / b[i].mass; // dv=a/m
    }
  }
  return (0);
}

static int AdamsBashforth(BODY * outState, double step, BODY * state0, DBODY * dStateB0, DBODY * dStateB1, DBODY * dStateB2, DBODY * dStateB3) {
  // Apply an iteration of the Adams-Bashforth algorithm.
  // @param *outState (output) the system state 1 iteration after (BODY array)
  // @param step the step
  // @param *state0 the system state at current iteration (BODY array)
  // @param *dStateB0 the system state's derivative at current iteration (DBODY array)
  // @param *dStateB1 the system state's derivative 1 iteration before (DBODY array)
  // @param *dStateB2 the system state's derivative 2 iterations before (DBODY array)
  // @param *dStateB3 the system state's derivative 3 iterations before (DBODY array)
  int i, c;
  #pragma omp for
  for (i = 0; i < g_body_number; i++) {
    for (c = 0; c < 3; c++) {
      outState[i].pos[c] = state0[i].pos[c] + step * (g_AB_b03 * dStateB0[i].dp[c] + g_AB_b13 * dStateB1[i].dp[c] + g_AB_b23 * dStateB2[i].dp[c] + g_AB_b33 * dStateB3[i].dp[c]); // p=f(y)
      outState[i].vel[c] = state0[i].vel[c] + step * (g_AB_b03 * dStateB0[i].dv[c] + g_AB_b13 * dStateB1[i].dv[c] + g_AB_b23 * dStateB2[i].dv[c] + g_AB_b33 * dStateB3[i].dv[c]); // v=f(y)
    }
    outState[i].mass = state0[i].mass; // m=f(y)
  }
  return (0);
}

static int AdamsMoulton(BODY * outState, double step, BODY * state0, DBODY * predictedDState, DBODY * dStateB0, DBODY * dStateB1, DBODY * dStateB2, DBODY * dStateB3) {
  // Apply an iteration of the Adams-Moulton algorithm.
  // @param *outState (output) the system state 1 iteration after (BODY array)
  // @param step the step
  // @param *state0 the system state at current iteration (BODY array)
  // @param *predictedDState the predicted system state's derivative 1 iteration after (DBODY array)
  // @param *dStateB0 the system state's derivative at current iteration (DBODY array)
  // @param *dStateB1 the system state's derivative 1 iteration before (DBODY array)
  // @param *dStateB2 the system state's derivative 2 iterations before (DBODY array)
  // @param *dStateB3 the system state's derivative 3 iterations before (DBODY array)
  int i, c;
  #pragma omp for
  for (i = 0; i < g_body_number; i++) {
    for (c = 0; c < 3; c++) {
      outState[i].pos[c] = state0[i].pos[c] + step * (g_AM_bp3 * predictedDState[i].dp[c] + g_AM_b03 * dStateB0[i].dp[c] + g_AM_b13 * dStateB1[i].dp[c] + g_AM_b23 * dStateB2[i].dp[c] + g_AM_b33 * dStateB3[i].dp[c]);
      outState[i].vel[c] = state0[i].vel[c] + step * (g_AM_bp3 * predictedDState[i].dv[c] + g_AM_b03 * dStateB0[i].dv[c] + g_AM_b13 * dStateB1[i].dv[c] + g_AM_b23 * dStateB2[i].dv[c] + g_AM_b33 * dStateB3[i].dv[c]);
    }
    outState[i].mass = state0[i].mass; // m=f(y)
  }
  return (0);
}

static int shiftTemporaryStates(DBODY * dStateB3, DBODY * dStateB2, DBODY * dStateB1, DBODY * dStateB0, DBODY * newDState, BODY * state0, BODY * newState) {
  // For the PECE scheme, shifts needed temporary states as number of iteration goes from n to n+1.
  // @param *dStateB3 (output) the system state's derivative 3 iterations before (DBODY array)
  // @param *dStateB2 (input/output) the system state's derivative 2 iterations before (DBODY array)
  // @param *dStateB1 (input/output) the system state's derivative 1 iteration before (DBODY array)
  // @param *dStateB0 (input/output)the system state's derivative at current iteration (DBODY array)
  // @param *newDState the system state's derivative 1 iteration after (DBODY array)
  // @param *state0 (output) the system state at current iteration (BODY array)
  // @param *newState the system state 1 iteration after (BODY array)
  int i, c;
  #pragma omp for
  for (i = 0; i < g_body_number; i++) {
    for (c = 0; c < 3; c++) {
      dStateB3[i].dp[c] = dStateB2[i].dp[c];
      dStateB3[i].dv[c] = dStateB2[i].dv[c];
      dStateB2[i].dp[c] = dStateB1[i].dp[c];
      dStateB2[i].dv[c] = dStateB1[i].dv[c];
      dStateB1[i].dp[c] = dStateB0[i].dp[c];
      dStateB1[i].dv[c] = dStateB0[i].dv[c];
      dStateB0[i].dp[c] = newDState[i].dp[c];
      dStateB0[i].dv[c] = newDState[i].dv[c];
      state0[i].pos[c] = newState[i].pos[c];
      state0[i].vel[c] = newState[i].vel[c];
    }
  }
  return (0);
}

static int tech_rk4(double suggested_timestep, double *actual_timestep) {
  // Advance particle system one timestep using Runge-Kutta fourth-order method.
  // @param suggested_timestep suggested timestep
  // @param *actual_timestep actual timestep used (for now, this function does not try to change the timestep)
  int i, c;
  double timestep;
  double half_step, sixth_step;
  VECTOR v1[g_body_number];
  VECTOR v2[g_body_number];
  VECTOR v3[g_body_number];
  VECTOR v4[g_body_number];
  VECTOR a1[g_body_number];
  VECTOR a2[g_body_number];
  VECTOR a3[g_body_number];
  VECTOR a4[g_body_number];
  BODY step2_array[g_body_number];
  BODY step3_array[g_body_number];
  BODY step4_array[g_body_number];
  
  timestep = suggested_timestep;
  half_step = timestep * 0.5;
  sixth_step = timestep / 6.0;
  
  copy_bodies(g_body_number, g_body_array, step2_array); // For S3.
  copy_bodies(g_body_number, g_body_array, step3_array); // For S6.
  copy_bodies(g_body_number, g_body_array, step4_array); // For S9.
  
  // S1: use current positions to compute accelerations. Call these "a1".
  if (calc_grav_forces(step2_array, a1) != 0) {
    printError("tech_rk4", "Function calc_grav_forces fails in step a1.");
    return (1);
  }
  #pragma omp for
  for (i = 0; i < g_body_number; i++) {
    for (c = 0; c < 3; c++) {
      a1[i].val[c] /= step2_array[i].mass; // Convert those forces to accelerations.
      v1[i].val[c] = g_body_array[i].vel[c]; // S2 : copy the current velocities into an array for later use. Call these "v1".
      step2_array[i].pos[c] += half_step * v1[i].val[c]; // S3 : use the "v1" velocities to predict positions of objects one-half a step into the future. We'll place the positions into the "step2_array[]."
      v2[i].val[c] = v1[i].val[c] + half_step * a1[i].val[c]; // S4 : use the "a1" accelerations to predict future velocities at half a step into the future. Call these "v2".
    }
  }
  // S5: use the "step2_array" positions (at half a step in future) to compute forces on the bodies in the future. Convert these forces to accelerations and place into the "a2" array.
  if (calc_grav_forces(step2_array, a2) != 0) {
    printError("tech_rk4", "Function calc_grav_forces fails in step a2.");
    return (1);
  }
  #pragma omp for
  for (i = 0; i < g_body_number; i++) {
    for (c = 0; c < 3; c++) {
      a2[i].val[c] /= step2_array[i].mass; // Convert those forces to accelerations.
      v3[i].val[c] = v1[i].val[c] + half_step * a2[i].val[c]; // S6: use the "a2" accelerations to predict future velocities at half a step into the future. Call these "v3".
      step3_array[i].pos[c] += half_step * v2[i].val[c]; // S6: use the "v2" velocities to predict positions of objects one-half a step into the future. We'll place the positions into the "step3_array[]."
    }
  }
  // S7: use the "step3_array" positions (at half a step in future) to compute forces on the bodies in the future. Convert these forces to accelerations and place into the "a3" array.
  if (calc_grav_forces(step3_array, a3) != 0) {
    printError("tech_rk4", "Function calc_grav_forces fails in step a3.");
    return (1);
  }
  #pragma omp for
  for (i = 0; i < g_body_number; i++) {
    for (c = 0; c < 3; c++) {
      a3[i].val[c] /= step3_array[i].mass; // Convert those forces to accelerations.
      v4[i].val[c] = v1[i].val[c] + timestep * a3[i].val[c]; // S8: use the "a3" accelerations to predict future velocities at one FULL step into the future. Call these "v4".
      step4_array[i].pos[c] += timestep * v3[i].val[c]; // S9: use the "v3" velocities to predict positions of objects one FULL step into the future.  We'll place the positions into the "step4_array[]."
    }
  }
  // S10: use the "step4_array" positions (at one FULL step in future) to compute forces on the bodies in the future. Convert these forces to accelerations and place into the "a4" array.
  if (calc_grav_forces(step4_array, a4) != 0) {
    printError("tech_rk4", "Function calc_grav_forces fails in step a4.");
    return (1);
  }
  #pragma omp for
  for (i = 0; i < g_body_number; i++) {
    for (c = 0; c < 3; c++) {
      a4[i].val[c] /= step4_array[i].mass; // Convert those forces to accelerations.

      // See note below.
      double delta_vel, delta_pos;
      delta_vel = sixth_step * (a1[i].val[c] + 2 * a2[i].val[c] + 2 * a3[i].val[c] + a4[i].val[c]);
      delta_pos = sixth_step * (v1[i].val[c] + 2 * v2[i].val[c] + 2 * v3[i].val[c] + v4[i].val[c]);
      g_body_array[i].vel[c] += delta_vel;
      g_body_array[i].pos[c] += delta_pos;
    }
  }
  /* Note :
   * At this point, we have 4 velocities and 4 accelerations for each object :
   * - v1     current velocity,
   * - v2     velocity predicted half a step in future,
   * - v3     improved velocity predicted half a step in future,
   * - v4     velocity predicted one FULL step in future,
   * - a1     current acceleration,
   * - a2     acceleration predicted half a step in future,
   * - a3     improved acceleration predicted half a step in future,
   * - a4     acceleration predicted one FULL step in future.
   * We can now combine these 4 measurements to produce one good value of velocity (or acceleration) one full step into the future.
   */
  *actual_timestep = timestep;
  return (0);
}

static int calc_grav_forces(BODY * body_array, VECTOR * forces) {
  // Given an array of bodies, compute the gravitational forces between each pair. The forces are then placed into a vector array.  
  // @param *body_array array with pos and mass of all bodies (BODY array)
  // @param *forces (output) forces on each body (VECTOR array)
  int i, j, c;

  // Compute the forces between all objects using the current positions and store this in the pre-allocated global variable g_cur_forces.
  
  #pragma omp for
  for (i = 0; i < g_body_number; i++) {
    for (j = i; j < g_body_number; j++) {
      VECTOR dist, dist_frac;
      double dist_tot;
      double dist_squared;
      double top, bottom, force_mag;

      if (i == j) {
        g_cur_forces[i][j].val[0] = 0.0;
        g_cur_forces[i][j].val[1] = 0.0;
        g_cur_forces[i][j].val[2] = 0.0;
        continue;
      }

      dist.val[0] = body_array[i].pos[0] - body_array[j].pos[0];
      dist.val[1] = body_array[i].pos[1] - body_array[j].pos[1];
      dist.val[2] = body_array[i].pos[2] - body_array[j].pos[2];
      dist_squared = VECTOR_MAG_SQUARED(dist);
      dist_tot = sqrt(dist_squared);
      /* if(dist_tot<=0.0){
      	fprintf(stderr, 
      			" calc_grav_forces: distance of %le -- quitting \n", dist_tot);
      	return(1);
      } */

      top = g_G * body_array[i].mass * body_array[j].mass;
      bottom = (dist_tot + g_softening) * (dist_tot + g_softening);
      force_mag = top / bottom;

      for (c = 0; c < 3; c++) {
        dist_frac.val[c] = dist.val[c] / dist_tot;
        g_cur_forces[i][j].val[c] = 0.0 - (force_mag * dist_frac.val[c]);
        g_cur_forces[j][i].val[c] = 0.0 - g_cur_forces[i][j].val[c];
      }
    }
  }
  computeTotalForceOnBodies(forces); // convert forces between each pair (stored in g_cur_forces) to a set of forces on each body
  if (g_galactic_flag == 1) {
    add_galactic_effect(forces);
  } // if the effect of a galaxy is wanted, add it to the current forces
  return (0);
}

static int computeTotalForceOnBodies(VECTOR * forces) {
  // Convert forces between each pair (stored in g_cur_forces) to a set of forces on each body.
  // @param *forces (output) forces on each body (VECTOR array)
  int i, j, c;
  #pragma omp for
  for (i = 0; i < g_body_number; i++) {
    for (c = 0; c < 3; c++) {
      forces[i].val[c] = 0.0;
      for (j = 0; j < g_body_number; j++) {
        forces[i].val[c] += g_cur_forces[i][j].val[c];
      }
    }
  }
  return (0);
}

static int add_galactic_effect(VECTOR * forces) {
  // Adds to an array of forces the effect of a galactic field.
  // The galaxy is supposed to be centered on the origin of the
  // reference (O=[0.0, 0.0, 0.0]) and its model is given by :
  //-a Plummer type bulge,
  //-a Miyamoto-Nagai disk,
  //-a dark matter halo.
  // @param *forces (input/output) forces on each body (VECTOR array)
  int i;
  VECTOR tmpGravitationalForce;
  #pragma omp for
  for (i = 0; i < g_body_number; i++) {
    getGravitationalForce(g_body_array[i], &tmpGravitationalForce);
    forces[i].val[0] += tmpGravitationalForce.val[0];
    forces[i].val[1] += tmpGravitationalForce.val[1];
    forces[i].val[2] += tmpGravitationalForce.val[2];
  }
  return (0);
}

static int getGravitationalForce(BODY body, VECTOR * tmpGravitationalForce) {
  // Compute the graviational force on a body.
  // @param body the considered body
  // @param *tmpGravitationalForce (output) the graviational force (pointer to a VECTOR)
  double tmpX, tmpY, tmpZ;
  double tmpRSquared, tmpRCSquared, tmpZSquared; // temporary variables to store the spherical radius, cylindrical radius and cylindrical altitude of a body
  double tmpPhiB, tmpPhiD, tmpPhiH; // temporary variables to store the bulge, disk and halo components
  tmpX = body.pos[0];
  tmpY = body.pos[1];
  tmpZ = body.pos[2];
  tmpRCSquared = tmpX * tmpX + tmpY * tmpY;
  tmpZSquared = tmpZ * tmpZ;
  tmpRSquared = tmpRCSquared + tmpZSquared;
  // x
  tmpPhiB = g_G * g_mb * pow(tmpRSquared + g_ab * g_ab, -1.5) * tmpX;
  tmpPhiD = g_G * g_md * pow(tmpRCSquared + pow(g_ad + pow(tmpZSquared + g_hd * g_hd, 0.5), 2.0), -1.5) * tmpX;
  tmpPhiH = -(g_vh * g_vh) / (tmpRSquared + g_ah * g_ah) * tmpX;
  tmpGravitationalForce -> val[0] = tmpPhiB + tmpPhiD + tmpPhiH;
  // y
  tmpPhiB *= (tmpY / tmpX);
  tmpPhiD *= (tmpY / tmpX);
  tmpPhiH *= (tmpY / tmpX);
  tmpGravitationalForce -> val[1] = tmpPhiB + tmpPhiD + tmpPhiH;
  // z
  tmpPhiB *= (tmpZ / tmpY);
  tmpPhiD *= (tmpZ / tmpY);
  tmpPhiH *= (tmpZ / tmpY);
  tmpPhiD *= ((g_ad + pow(tmpZSquared + g_hd * g_hd, 0.5)) / (pow(tmpZSquared + g_hd * g_hd, 0.5)));
  tmpGravitationalForce -> val[2] = tmpPhiB + tmpPhiD + tmpPhiH;
  tmpGravitationalForce -> val[0] *= -body.mass;
  tmpGravitationalForce -> val[1] *= -body.mass;
  tmpGravitationalForce -> val[2] *= -body.mass;
  return (0);
}

static void copy_bodies(int N, BODY *in, BODY *out) {
  // Copy an array of BODY structures into a second array in such a manner that we can then modify the copies and leave the originals untouched.
  // @param N number of bodies to be copied
  // @param in copy from this array
  // @param out (output) copy to this array
  int i, c;
  BODY *from_body, *to_body;
  #pragma omp for
  for (i = 0; i < N; i++) {
    from_body = &(in [i]);
    to_body = &(out[i]);
    to_body->index = from_body->index;
    strcpy(from_body->name, to_body->name);
    to_body->mass = from_body->mass;
    for (c = 0; c < 3; c++) {
      to_body->pos[c] = from_body->pos[c];
      to_body->vel[c] = from_body->vel[c];
    }
  }
}

static int recenter_velocities(void) {
  // Compute the global velocity of the system. Then, subtract that value from all bodies, so that the center of mass of the system should remain motionless as time passes.
  // @param none
  int i, c;
  double total_mass;
  VECTOR momentum, com_velocity;
  total_mass = 0.0;
  for (c = 0; c < 3; c++) {
    momentum.val[c] = 0.0;
  }
  for (i = 0; i < g_body_number; i++) {
    total_mass += g_body_array[i].mass;
    for (c = 0; c < 3; c++) {
      momentum.val[c] += g_body_array[i].mass * g_body_array[i].vel[c];
    }
  }
  if (total_mass == 0.0) {
    return (0);
  } // If the total mass is zero, just return now (why would that be ever the case anyways ?).
  for (c = 0; c < 3; c++) {
    com_velocity.val[c] = momentum.val[c] / total_mass;
  } // Compute the velocity of the center of mass.
  
  #pragma omp for
  for (i = 0; i < g_body_number; i++) {
    for (c = 0; c < 3; c++) {
      g_body_array[i].vel[c] -= com_velocity.val[c];
    }
  } // Subtract this velocity from each body.
  return (0);
}

static double calc_total_ke(void) {
  // Compute the total KE of all bodies. The result will have [E]=[M]*[L^2]*[T^{-2}] as unit.
  // @param none
  // @return the total KE
  int i;
  double vSquared, ke, total_ke;
  total_ke = 0;
  for (i = 0; i < g_body_number; i++) {
    vSquared = g_body_array[i].vel[0] * g_body_array[i].vel[0] + g_body_array[i].vel[1] * g_body_array[i].vel[1] + g_body_array[i].vel[2] * g_body_array[i].vel[2];
    ke = 0.5 * g_body_array[i].mass * vSquared;
    total_ke += ke;
    if (g_verbose_flag > 1) {
      printf(">> [calc_total_ke] : Body %5d has v**2=%9.4e and m=%9.4e, thus %9.4e of KE (total updated to %9.4e).\n", i, vSquared, g_body_array[i].mass, ke, total_ke);
    }
  }
  return (total_ke);
}

static double calc_total_gpe(void) {
  // Compute the total GPE of all bodies. The result will have [E]=[M]*[L^2]*[T^{-2}] as unit.
  // @param none
  // @return the total GPE
  int i, j;
  double d, d_squared, gpe, total_gpe;
  total_gpe = 0.0;
  for (i = 0; i < g_body_number - 1; i++) {
    for (j = i + 1; j < g_body_number; j++) {
      d_squared = DIST_SQUARED(i, j);
      d = sqrt(d_squared);
      if (d <= 0) {
        fprintf(stderr, "Distance %9.4le (between bodies %d and %d).\n", d, i, j);
        printError("calc_total_gpe", "Distance is invalid.");
        exit(1);
      }
      gpe = 0.0 - ((g_G * g_body_array[i].mass * g_body_array[j].mass) / d);
      total_gpe += gpe;
      if (g_verbose_flag > 1) {
        printf(">> [calc_total_gpe] Bodies %5d and %5d have %9.4e of GPE (total GPE updated to %9.4e).\n", i, j, gpe, total_gpe);
      }
    }
  }
  return (total_gpe);
}

static int calc_total_angmom(VECTOR * result) {
  // Compute the total angular momentum of all the bodies, as a three-D vector, computed around the center of mass of the system. The result will have [M]*[L]*[T^{-1}] as unit.
  // *result total angular momentum of system
  int i, c;
  double total_mass;
  VECTOR body_pos, body_vel, body_ang_mom, com, com_vel, total_ang_mom;
  total_mass = 0.0;
  for (c = 0; c < 3; c++) {
    com.val[c] = 0.0;
    com_vel.val[c] = 0.0;
    total_ang_mom.val[c] = 0.0;
  }
  // Compute the center of mass of the system and the velocity of the center of mass.
  for (i = 0; i < g_body_number; i++) {
    total_mass += g_body_array[i].mass;
    for (c = 0; c < 3; c++) {
      com.val[c] += g_body_array[i].mass * g_body_array[i].pos[c];
      com_vel.val[c] += g_body_array[i].mass * g_body_array[i].vel[c];
    }
  }
  for (c = 0; c < 3; c++) {
    com.val[c] /= total_mass;
    com_vel.val[c] /= total_mass;
  }
  // Compute the angular momentum of each body around this point.
  for (i = 0; i < g_body_number; i++) {
    // Compute the position vector from center of mass to body i.
    body_pos.val[0] = g_body_array[i].pos[0] - com.val[0];
    body_pos.val[1] = g_body_array[i].pos[1] - com.val[1];
    body_pos.val[2] = g_body_array[i].pos[2] - com.val[2];
    // Compute the velocity difference between center of mass and body i, and weight it by the mass of the body.
    body_vel.val[0] = g_body_array[i].mass * (g_body_array[i].vel[0] - com_vel.val[0]);
    body_vel.val[1] = g_body_array[i].mass * (g_body_array[i].vel[1] - com_vel.val[1]);
    body_vel.val[2] = g_body_array[i].mass * (g_body_array[i].vel[2] - com_vel.val[2]);
    // Now calculate the angular momentum of body i around center of mass.
    VECTOR_CROSS(body_pos, body_vel, body_ang_mom);
    // Add this body's angular momentum to the total.
    for (c = 0; c < 3; c++) {
      total_ang_mom.val[c] += body_ang_mom.val[c];
    }
  }
  for (c = 0; c < 3; c++) {
    result -> val[c] = total_ang_mom.val[c];
  } // Copy the total angular momentum to output argument.
  return (0);
}

static void printError(char * funcName, char * msg) {
  // Prints on the standard error output an error message.
  // @param *funcName the name of the function from where this method is called
  // @param *msg the message to print
  fprintf(stderr, "ERROR [%s] : %s\n", funcName, msg);
}

static int print_positions(FILE * outfile_fp, double currentSimulationTime) {
  // Print out information on each body, one body per line. Each line has format :
  //   time index mass px py pz vx vy vz 
  // @param *outfile_fp ASCII text file where to write output information (FILE, already opened)
  // @param currentSimulationTime current simulation time
  int i;
  if (outfile_fp == NULL) {
    printError("print_positions", "File was given a NULL pointer.");
    return (1);
  }
  // Note : do not use OpenMP optimisations here, as it will mess up the output file.
  for (i = 0; i < g_body_number; i++) {
    fprintf(outfile_fp, "%1.16e %5d %1.6e %16.9e %16.9e %16.9e %16.9e %16.9e %16.9e\n",
      currentSimulationTime,
      g_body_array[i].index,
      g_body_array[i].mass,
      g_body_array[i].pos[0],
      g_body_array[i].pos[1],
      g_body_array[i].pos[2],
      g_body_array[i].vel[0],
      g_body_array[i].vel[1],
      g_body_array[i].vel[2]);
  }
  return (0);
}

static int read_input(char * filename) {
  // Given a file name, open the file and read information from it. The number of objects, their initial positions, parameters values, should be initialised by this function.
  // @param *filename name of ASCII text file with input information
  char line[LINELEN];
  int j, nbody;
  FILE * fp;

  /* Open file (and see if it exists). ----------------------*/
  if ((fp = fopen(filename, "r")) == NULL) {
    fprintf(stderr, "File : \"%s\".\n", filename);
    printError("read_input", "Can't open file.");
    return (1);
  }
  /*---------------------------------------------------------*/

  /* Start to read file. ------------------------------------*/
  rewind(fp);
  while (fgets(line, LINELEN, fp) != NULL) {
    if (g_verbose_flag > 1) {
      printf(">> Next line in input file is :\n%s", line);
    }
    if (line[0] == COMMENT_CHAR) {
      if (g_verbose_flag > 1) {
        printf(">> Skipping comment line.\n");
      }
      continue;
    }
    if (line[0] == '\n') {
      if (g_verbose_flag > 1) {
        printf(">> Skipping empty line.\n");
      }
      continue;
    }

    /*
    Read the number of bodies in the simulation.
    Line format :
      nbody 10
    */
    nbody = -1;
    if (strncmp(line, "nbody", 5) == 0) {
      if (sscanf(line + 5, "%d", &nbody) != 1) {
        fprintf(stderr, "ERROR [read_input] : Bad \"nbody\" parameter value in line \"%s\".\n", line);
        return (1);
      }
      if (nbody < 1) {
        fprintf(stderr, "ERROR [read_input] : Parameter \"nbody\" must be>=1.\n");
        return (1);
      }
      g_body_number = nbody;
      if (g_verbose_flag > 0) {
        printf(">> Number of bodies is %d.\n", nbody);
      }
    }

    /*
    Read the duration of the simulation (in units of [T],
    default being [T]=1 Myr).
    Line format :
      duration 100
    */
    double duration;
    if (strncmp(line, "duration", 8) == 0) {
      if (sscanf(line + 8, "%lf", &duration) != 1) {
        fprintf(stderr, "bad duration value in line ..%s.. \n", line);
        return (1);
      }
      if (duration < 0) {
        fprintf(stderr, "duration must be>=0 \n");
        return (1);
      }
      g_duration = duration;
      if (g_verbose_flag > 0) {
        printf(">> Simulation duration is %9.4e [T].\n", duration);
      }
    }

    /*
    Read the (initial) timestep of the simulation (in units of
    [T], default being [T]=1 Myr).
    Line format :
      timestep 1
    */
    double timestep;
    if (strncmp(line, "timestep", 8) == 0) {
      if (sscanf(line + 8, "%lf", &timestep) != 1) {
        fprintf(stderr, "bad timestep value in line ..%s.. \n", line);
        return (1);
      }
      if (timestep <= 0) {
        fprintf(stderr, "timestep must be>0 \n");
        return (1);
      }
      g_timestep = timestep;
      if (g_verbose_flag > 0) {
        printf(">> Time step is %9.4e [T].\n", g_timestep);
      }
    }

    /*
    Read the interval we should allow to elapse between printing
    positions to the output file (in units of [T], default being
    [T]=1 Myr).
    Line format :
      print_interval 1e-4
    */
    double interval;
    if (strncmp(line, "print_interval", 14) == 0) {
      if (sscanf(line + 14, "%lf", &interval) != 1) {
        fprintf(stderr, "bad print_interval value in line ..%s.. \n", line);
        return (1);
      }
      if (interval <= 0) {
        fprintf(stderr, "print_interval must be>0 \n");
        return (1);
      }
      g_print_interval = interval;
      if (g_verbose_flag > 0) {
        printf(">> Printing interval is %9.4e [T].\n", g_print_interval);
      }
    }

    /*
    Check for a line which tells us :
   -should we leave initial velocities as provided by the user
    or
   -should we add a constant value to force the center-of-mass
    velocity to be zero
    ?
    Line format :
      recenter yes
      recenter no
    */
    char recenter_value[NAMELEN];
    if (strncmp(line, "recenter", 8) == 0) {
      if (sscanf(line + 8, "%s", recenter_value) != 1) {
        fprintf(stderr, "bad recenter value in line ..%s.. \n", line);
        return (1);
      }
      if (strcmp(recenter_value, "yes") == 0) {
        g_recenter_flag = 1;
      }
      if (strcmp(recenter_value, "no") == 0) {
        g_recenter_flag = 0;
      }
      if (g_verbose_flag > 0) {
        printf(">> Flag \"g_recenter_flag\" is set to %d.\n", g_recenter_flag);
      }
    }

    /*
    Read the technique we should use to perform the numerical
    integration during each timestep. The input file contains
    a code, which this section of the program must match to one
    of the existing functions.
    Line format :
      technique E1
      technique E1a
      technique E2
      technique RK4
    Near the top of the main() routine is a block of code which assigns functions to codes.
    */
    char technique[NAMELEN];
    if (strncmp(line, "technique", 9) == 0) {
      if (sscanf(line + 9, "%s", technique) != 1) {
        fprintf(stderr, "ERROR [read_input] : Bad technique value in line \"%s\".\n", line);
        return (1);
      }

      if (strcmp(technique, "PECE") == 0) { // PECE is activated
        g_usePECE_flag = 1;
        g_usePEC_flag = 0;
        strcpy(technique, "RK4"); // save RK4 as technique for first iterations
      } else {
        if (strcmp(technique, "PEC") == 0) { // PEC is activated
          g_usePECE_flag = 0;
          g_usePEC_flag = 1;
          strcpy(technique, "RK4"); // save RK4 as technique for first iterations
        } else { // it's something else (that has been read)
          g_usePECE_flag = 0;
          g_usePEC_flag = 0;
        }
      }
      // Make sure that the technique matches one of the known entries.
      g_integration_func = (PFI) NULL;
      for (j = 0; j < g_technique_number; j++) {
        if (strcmp(technique, g_technique_array[j].name) == 0) {
          g_integration_func = g_technique_array[j].func;
          break;
        }
      }
      if (g_integration_func == NULL) {
        fprintf(stderr, "ERROR [read_input] : Can't find technique called \"%s\".\n", technique);
        return (1);
      }
      if (g_verbose_flag > 0) {
        if (g_usePECE_flag == 1) {
          if (g_usePEC_flag == 1) {
            printf(">> Technique PEC (initialisations with %s) will be used.\n", technique);
          } else {
            printf(">> Technique PECE (initialisations with %s) will be used.\n", technique);
          }
        } else {
          printf(">> Technique %s will be used.\n", technique);
        }
      }
    }

    /*
    Read the name of the file into which to write the output. If
    the name is "-", then write to stdout.
    Line format :
      outfile -
      outfile ./2Stars.out
    */
    char outfile_name[NAMELEN];
    if (strncmp(line, "outfile", 7) == 0) {
      if (sscanf(line + 7, "%s", outfile_name) != 1) {
        fprintf(stderr, "bad output filename in line ..%s.. \n", line);
        return (1);
      }
      /* 
       * open the file for output, set the global variable
       *    g_outfile_fp to point to the opened file
       */
      if (strcmp(outfile_name, "-") == 0) {
        g_outfile_fp = stdout;
      } else {
        if ((g_outfile_fp = fopen(outfile_name, "w")) == NULL) {
          fprintf(stderr, "ERROR [read_input] : Can't open file %s for output.\n",
            outfile_name);
          return (1);
        }
      }
    }

    /*
    Read the initial information for each body, one at a time.
    Make sure that the number of bodies with information is the
    same as the "nbody" line indicated.
    Line format :
      body index name mass px py pz vx vy vz
    */
    if (strncmp(line, "body ", 5) == 0) {
      int this_index;
      char this_name[LINELEN];
      double this_mass;
      double this_px, this_py, this_pz;
      double this_vx, this_vy, this_vz;
      if (g_verbose_flag > 1) {
        printf(" >> About to scan for a body's initial info...\n");
      }
      if (sscanf(line + 4, " %d %s %lf %lf %lf %lf %lf %lf %lf", &this_index, this_name, &this_mass, &this_px, &this_py, &this_pz, &this_vx, &this_vy, &this_vz) != 9) {
        fprintf(stderr, "ERROR [read_input] : Bad body in line ..%s..\n", line);
        return (1);
      }
      if (strlen(this_name) >= NAMELEN) {
        fprintf(stderr, "ERROR [read_input] : Body with name \"%s\" has name longer than %d chars.\n", this_name, NAMELEN);
        return (1);
      }
      if ((this_index < 0) || (this_index >= g_body_number)) {
        fprintf(stderr, "ERROR [read_input] : Body with name \"%s\" has invalid index %d.\n", this_name, this_index);
        return (1);
      }
      if (this_mass <= 0) {
        fprintf(stderr, "ERROR [read_input] : Body with name \"%s\" has invalid mass %lf.\n", this_name, this_mass);
        return (1);
      }
      // Now try to set entries in the appropriate g_body_array[].
      if (g_body_array[this_index].index != -1) {
        fprintf(stderr, "ERROR [read_input] : Body with name \"%s\" has repeated index %d.\n", this_name, this_index);
        return (1);
      } else {
        g_body_array[this_index].index = this_index;
        strcpy(g_body_array[this_index].name, this_name);
        g_body_array[this_index].mass = this_mass;
        g_body_array[this_index].pos[0] = this_px;
        g_body_array[this_index].pos[1] = this_py;
        g_body_array[this_index].pos[2] = this_pz;
        g_body_array[this_index].vel[0] = this_vx;
        g_body_array[this_index].vel[1] = this_vy;
        g_body_array[this_index].vel[2] = this_vz;
      }
      if (g_verbose_flag > 1) {
        printf(" >> Body %d information : \n", this_index);
        printf("   -name : \"%s\",\n", g_body_array[this_index].name);
        printf("   -mass : %12.5le,\n", g_body_array[this_index].mass);
        printf("   -position : [%12.5le, %12.5le, %12.5le],\n", g_body_array[this_index].pos[0], g_body_array[this_index].pos[1], g_body_array[this_index].pos[2]);
        printf("   -velocity : [%12.5le, %12.5le, %12.5le].\n", g_body_array[this_index].vel[0], g_body_array[this_index].vel[1], g_body_array[this_index].vel[2]);
      }
    }

    /*
    Read the softening parameter.
    Line format :
      softening 1e-3
    */
    double softening;
    if (strncmp(line, "softening", 9) == 0) {
      if (sscanf(line + 9, "%lf", &softening) != 1) {
        fprintf(stderr, "ERROR [read_input] : Bad softening value in line :\n%s\n", line);
        return (1);
      }
      if (softening < 0) {
        fprintf(stderr, "ERROR [read_input] : Softening must be>=0.\n");
        return (1);
      }
      g_softening = softening;
      if (g_verbose_flag > 0) {
        printf(">> Softening is %9.4e [L].\n", g_softening);
      }
    }

    /*
    Read the constant G to use.
    Line format :
      gravitational_constant 4.302e-3
    */
    double gravC;
    if (strncmp(line, "gravitational_constant", 22) == 0) {
      if (sscanf(line + 22, "%lf", &gravC) != 1) {
        fprintf(stderr, "ERROR [read_input] : Bad gravitational constant value in line :\n%s\n", line);
        return (1);
      }
      if (gravC <= 0) {
        fprintf(stderr, "ERROR [read_input] : Gravitational constant must be>0.\n");
        return (1);
      }
      g_G = gravC;
      if (g_verbose_flag > 0) {
        printf(">> Gravitational constant is %9.4e [L^{3}][M^{-1}][T^{-2}].\n", g_G);
      }
    }

    /*
    Check for a line which tells us if a galactic field is to be considered.
    Line format :
      galactic yes
      galactic no
    */
    char galactic[NAMELEN];
    if (strncmp(line, "galactic", 8) == 0) {
      if (sscanf(line + 8, "%s", galactic) != 1) {
        fprintf(stderr, "bad galactic value in line ..%s.. \n", line);
        return (1);
      }
      if (strcmp(galactic, "yes") == 0) {
        g_galactic_flag = 1;
      }
      if (strcmp(galactic, "no") == 0) {
        g_galactic_flag = 0;
      }
      if (g_verbose_flag > 0) {
        printf(">> Flag \"g_galactic_flag\" is set to %d.\n", g_galactic_flag);
      }
    }

    /*
    Read the galactic parameters
    Line format :
      galParam_bulge_mb
    */
    double mb, ab, md, ad, hd, vh, ah;
    if (strncmp(line, "galParam", 8) == 0) {
      if (sscanf(line + 8, " %lf %lf %lf %lf %lf %lf %lf", &mb, &ab, &md, &ad, &hd, &vh, &ah) != 7) {
        fprintf(stderr, "ERROR [read_input] : Bad galaxy parameter line :\n%s\n", line);
        return (1);
      }
      if (mb <= 0 || ab <= 0 || md <= 0 || ad <= 0 || hd <= 0 || vh <= 0 || ah <= 0) {
        fprintf(stderr, "ERROR [read_input] : Galaxy parameters must all be>0.\n");
        return (1);
      }
      g_mb = mb;
      g_ab = ab;
      g_md = md;
      g_ad = ad;
      g_hd = hd;
      g_vh = vh;
      g_ah = ah;
      if (g_verbose_flag > 0) {
        printf(">> Galaxy parameters are :\n   g_mb=%9.4e,\n   g_ab=%9.4e,\n   g_md=%9.4e,\n   g_ad=%9.4e,\n   g_hd=%9.4e,\n   g_vh=%9.4e,\n   g_ah=%9.4e.\n", g_mb, g_ab, g_md, g_ad, g_hd, g_vh, g_ah);
      }
    }
  }
  /*---------------------------------------------------------*/

  /* Check read information. --------------------------------*/
  // Check number of bodies in the simulation.
  if ((g_body_number < 1) || (g_body_number >= MAX_NBODY)) {
    fprintf(stderr, "ERROR [read_input] : nbody %d is<1 or>MAX_NBODY (%d).\n", g_body_number, MAX_NBODY);
    return (1);
  }
  // Check duration of the simulation.
  if (g_duration < 0) {
    fprintf(stderr, "ERROR [read_input] : g_duration %lf is<0 (probably not correctly set).\n", g_duration);
    return (1);
  }
  // Check timestep of the simulation.
  if (g_timestep <= 0) {
    fprintf(stderr, "ERROR [read_input] : g_timestep %lf is<0 (probably not correctly set).\n", g_timestep);
    return (1);
  }
  // Check printing interval.
  if (g_print_interval <= 0) {
    fprintf(stderr, "ERROR [read_input] : g_print_interval %lf is<0 (probably not correctly set).\n", g_print_interval);
    return (1);
  }
  // Check recenter flag.
  if (g_recenter_flag < 0) {
    fprintf(stderr, "ERROR [read_input] : g_recenter_flag not set.\n");
    return (1);
  }
  // Check technique.
  if (g_integration_func == NULL) {
    fprintf(stderr, "ERROR [read_input] : No technique provided in %s.\n", filename);
    return (1);
  }
  // Check output file.
  if (g_outfile_fp == NULL) {
    fprintf(stderr, "ERROR [read_input] : No valid outfile supplied.\n");
    return (1);
  }
  // Check softening.
  if (g_softening == -1) {
    fprintf(stderr, "ERROR [read_input] : g_softening %lf is<0 (probably not correctly set).\n", g_softening);
    return (1);
  }
  // Check gravitational constant.
  if (g_G == -1) {
    fprintf(stderr, "ERROR [read_input] : g_G %lf is<0 (probably not correctly set).\n", g_G);
    return (1);
  }
  // Check galactic flag.
  if (g_galactic_flag < 0) {
    fprintf(stderr, "ERROR [read_input] : g_galactic_flag not set.\n");
    return (1);
  }
  // Check galaxy parameters. If a galaxy is wanted and parameters are not set, raise an error. If a galaxy is not wanted and parameters are not set, it is not important.
  if (g_galactic_flag == 1 && (g_mb <= 0 || g_ab <= 0 || g_md <= 0 || g_ad <= 0 || g_hd <= 0 || g_vh <= 0 || g_ah <= 0)) {
    fprintf(stderr, "ERROR [read_input] : Galaxy parameters must all be>0.\n");
    return (1);
  }
  /*---------------------------------------------------------*/

  /* Finish input reading. ----------------------------------*/
  fclose(fp);
  return (0);
  /*---------------------------------------------------------*/
}

/* Other integration functions. -----------------------------*/
/* Note : most of the optimisation work has been done using
 * the RK4 scheme and later on with it combined to a PECE
 * algorithm. Using the following schemes is not
 * recommanded because they have not been thoroughly
 * tested.                                                  -*/

/********************************************************************
 * PROCEDURE: tech_euler_1
 *
 * DESCRIPTION: Advance particles one timestep 
 *              using Euler's method to first order.
 *
 *              This method does not (yet) attempt to change
 *              the timestep.
 *
 * RETURNS:
 *              0       if all goes well
 *              1       if an error occurs
 */
static int tech_euler_1(double suggested_timestep, double *actual_timestep) {
    int i, j, c;
    double timestep;
    VECTOR new_pos[g_body_number];
    VECTOR new_vel[g_body_number];
    timestep = suggested_timestep;
    /* 
     * compute the forces between all objects
     *   using the current positions 
     */
    for (i = 0; i < g_body_number; i++) {
      for (j = i; j < g_body_number; j++) {
        VECTOR dist, dist_frac;
        double dist_tot;
        double dist_squared;
        double top, bottom, force_mag;
        if (i == j) {
          g_cur_forces[i][j].val[0] = 0.0;
          g_cur_forces[i][j].val[1] = 0.0;
          g_cur_forces[i][j].val[2] = 0.0;
          continue;
        }
        SET_POS_DIFF(dist, i, j);
        dist_squared = VECTOR_MAG_SQUARED(dist);
        dist_tot = sqrt(dist_squared);
        if (dist_tot <= 0.0) {
          fprintf(stderr,
            " tech_euler_1: distance of %le -- quitting \n", dist_tot);
          return (1);
        }
        top = g_G * g_body_array[i].mass * g_body_array[j].mass;
        bottom = (dist_tot + g_softening) * (dist_tot + g_softening);
        force_mag = top / bottom;
        for (c = 0; c < 3; c++) {
          dist_frac.val[c] = dist.val[c] / dist_tot;
          g_cur_forces[i][j].val[c] = 0.0 - (force_mag * dist_frac.val[c]);
          g_cur_forces[j][i].val[c] = 0.0 - g_cur_forces[i][j].val[c];
        }
      }
    }
    /* do the work here */
    for (i = 0; i < g_body_number; i++) {
      VECTOR tot_force;
      VECTOR tot_accel;
      /* 
       * we will compute a new position and velocity
       *   for each object in turn, placing the new
       *   values into the 'new_pos[]' and 'new_vel[]' 
       *   arrays.  Afterwards,
       *   we'll copy the new values back into the
       *   original g_body[] array.
       */
      /* use the current velocity to compute new positions */
      for (c = 0; c < 3; c++) {
        new_pos[i].val[c] = g_body_array[i].pos[c] + g_body_array[i].vel[c] * timestep;
      }
      /* use the current forces to compute new velocities */
      for (c = 0; c < 3; c++) {
        /* 
         * compute the total force on this body
         *    in each direction 
         */
        tot_force.val[c] = 0.0;
        for (j = 0; j < g_body_number; j++) {
          tot_force.val[c] += g_cur_forces[i][j].val[c];
        }
        /* from total force, compute total acceleration */
        tot_accel.val[c] = tot_force.val[c] / g_body_array[i].mass;
        /* use the acceleration to compute new velocity */
        new_vel[i].val[c] = g_body_array[i].vel[c] + tot_accel.val[c] * timestep;
      }
    }
    /* 
     * having computed the new positions and velocities,
     *    we now copy those values into the g_body_array[]
     */
    for (i = 0; i < g_body_number; i++) {
      for (c = 0; c < 3; c++) {
        g_body_array[i].pos[c] = new_pos[i].val[c];
        g_body_array[i].vel[c] = new_vel[i].val[c];
      }
    } * actual_timestep = timestep;
    return (0);
  }

/********************************************************************
 * PROCEDURE: tech_euler_2
 *
 * DESCRIPTION: Advance particles one timestep 
 *              using Euler's method to second order,
 *              which is also known as Heun's method.
 *
 *              The basic idea:
 *
 *                a) compute current accel
 *                b) use current accel to predict poor future vel
 *                c) use poor future vel to predict poor future pos
 *                d) use poor future pos to predict poor future accel
 *                e) calc average of current and future accel
 *                f) use average accel to compute better future vel
 *                g) calc average of current and future vel
 *                h) use average vel to compute better future pos
 *
 *              Thi method does not (yet) attempt to change
 *              the timestep.
 *
 * RETURNS:
 *              0       if all goes well
 *              1       if an error occurs
 */
static int tech_euler_2(double suggested_timestep, double *actual_timestep) {
    int i, j, c;
    double timestep;
    VECTOR poor_new_pos[g_body_number];
    VECTOR better_new_pos[g_body_number];
    VECTOR poor_new_vel[g_body_number];
    VECTOR better_new_vel[g_body_number];
    VECTOR average_vel[g_body_number];
    VECTOR cur_tot_force[g_body_number];
    VECTOR poor_new_force[g_body_number];
    VECTOR average_force[g_body_number];

    timestep = suggested_timestep;

    /* 
     * compute the forces between all objects
     *   using the current positions 
     */
    for (i = 0; i < g_body_number; i++) {
      for (j = i; j < g_body_number; j++) {
        VECTOR dist, dist_frac;
        double dist_tot;
        double dist_squared;
        double top, bottom, force_mag;
        if (i == j) {
          g_cur_forces[i][j].val[0] = 0.0;
          g_cur_forces[i][j].val[1] = 0.0;
          g_cur_forces[i][j].val[2] = 0.0;
          continue;
        }
        SET_POS_DIFF(dist, i, j);
        dist_squared = VECTOR_MAG_SQUARED(dist);
        dist_tot = sqrt(dist_squared);
        if (dist_tot <= 0.0) {
          fprintf(stderr,
            " tech_euler_2: distance of %le -- quitting \n", dist_tot);
          return (1);
        }
        top = g_G * g_body_array[i].mass * g_body_array[j].mass;
        bottom = (dist_tot + g_softening) * (dist_tot + g_softening);
        force_mag = top / bottom;
        for (c = 0; c < 3; c++) {
          dist_frac.val[c] = dist.val[c] / dist_tot;
          g_cur_forces[i][j].val[c] = 0.0 - (force_mag * dist_frac.val[c]);
          g_cur_forces[j][i].val[c] = 0.0 - g_cur_forces[i][j].val[c];
        }
      }
    }
    /* 
     * first, we go through several steps to predict poor future
     *    position for all object
     */
    for (i = 0; i < g_body_number; i++) {
      /* 
       * we will compute a new position and velocity
       *   for each object in turn, placing the new
       *   values into the 'new_pos[]' and 'new_vel[]' 
       *   arrays.  Afterwards,
       *   we'll copy the new values back into the
       *   original g_body[] array.
       */
      /* use the current accel to predict poor future vel */
      /* 
       * compute the total force on this body
       *    in each direction 
       */
      for (c = 0; c < 3; c++) {
        cur_tot_force[i].val[c] = 0.0;
      }
      for (j = 0; j < g_body_number; j++) {
        for (c = 0; c < 3; c++) {
          cur_tot_force[i].val[c] += g_cur_forces[i][j].val[c];
        }
      }
      for (c = 0; c < 3; c++) {
        poor_new_vel[i].val[c] = g_body_array[i].vel[c] + (cur_tot_force[i].val[c] / g_body_array[i].mass) * timestep;
      }
      /* use poor future vel to predict poor future pos */
      for (c = 0; c < 3; c++) {
        poor_new_pos[i].val[c] = g_body_array[i].pos[c] + poor_new_vel[i].val[c] * timestep;
      }
    }
    /* 
     * Now we use poor future pos of all objects to predict 
     *    poor future accel for all objects 
     */
    for (i = 0; i < g_body_number; i++) {
      for (c = 0; c < 3; c++) {
        poor_new_force[i].val[c] = 0.0;
      }
      for (j = 0; j < g_body_number; j++) {
        VECTOR dist, dist_frac;
        double dist_tot;
        double dist_squared;
        double top, bottom, force_mag;
        if (i == j) {
          continue;
        }
        for (c = 0; c < 3; c++) {
          dist.val[c] = poor_new_pos[i].val[c] - poor_new_pos[j].val[c];
        }
        dist_squared = VECTOR_MAG_SQUARED(dist);
        dist_tot = sqrt(dist_squared);
        if (dist_tot <= 0.0) {
          fprintf(stderr,
            " tech_euler_2: distance of %le -- quitting \n", dist_tot);
          return (1);
        }
        top = g_G * g_body_array[i].mass * g_body_array[j].mass;
        bottom = (dist_tot + g_softening) * (dist_tot + g_softening);
        force_mag = top / bottom;
        for (c = 0; c < 3; c++) {
          dist_frac.val[c] = dist.val[c] / dist_tot;
          poor_new_force[i].val[c] += 0.0 - (force_mag * dist_frac.val[c]);
        }
      }
    }
    /* 
     * and now we can walk through the list of all bodies,
     *    one at a time, and calculate improved future quantities
     */
    for (i = 0; i < g_body_number; i++) {
      /* calc average of current and future accel */
      for (c = 0; c < 3; c++) {
        average_force[i].val[c] =
          0.5 * (cur_tot_force[i].val[c] + poor_new_force[i].val[c]);
      }
      /* use average accel to compute better future vel */
      for (c = 0; c < 3; c++) {
        better_new_vel[i].val[c] = g_body_array[i].vel[c] +
          (average_force[i].val[c] / g_body_array[i].mass) * timestep;
      }
      /* calc average of current and future vel */
      for (c = 0; c < 3; c++) {
        average_vel[i].val[c] =
          0.5 * (g_body_array[i].vel[c] + better_new_vel[i].val[c]);
      }
      /* use average vel to compute better future pos */
      for (c = 0; c < 3; c++) {
        better_new_pos[i].val[c] = g_body_array[i].pos[c] +
          average_vel[i].val[c] * timestep;
      }
    }

    /* 
     * having computed the new positions and velocities,
     *    we now copy those values into the g_body_array[]
     */
    for (i = 0; i < g_body_number; i++) {
      for (c = 0; c < 3; c++) {
        g_body_array[i].pos[c] = better_new_pos[i].val[c];
        g_body_array[i].vel[c] = better_new_vel[i].val[c];
      }
    }

    * actual_timestep = timestep;
    return (0);
  }

/********************************************************************
 * PROCEDURE: tech_euler_1a
 *
 * DESCRIPTION: Advance particles one timestep 
 *              using Euler's method to first order.
 *              However, we use the velocity for time N+1
 *              to advance the position for time N;
 *              in other words, use the new velocity
 *              to advance the old position.
 *
 *              This method does not (yet) attempt to change
 *              the timestep.
 *
 * RETURNS:
 *              0       if all goes well
 *              1       if an error occurs
 */
static int tech_euler_1a(double suggested_timestep, double *actual_timestep) {
    int i, j, c;
    double timestep;
    VECTOR new_pos[g_body_number];
    VECTOR new_vel[g_body_number];

    timestep = suggested_timestep;

    /* 
     * compute the forces between all objects
     *   using the current positions 
     */
    for (i = 0; i < g_body_number; i++) {
      for (j = i; j < g_body_number; j++) {
        VECTOR dist, dist_frac;
        double dist_tot;
        double dist_squared;
        double top, bottom, force_mag;

        if (i == j) {
          g_cur_forces[i][j].val[0] = 0.0;
          g_cur_forces[i][j].val[1] = 0.0;
          g_cur_forces[i][j].val[2] = 0.0;
          continue;
        }

        SET_POS_DIFF(dist, i, j);
        dist_squared = VECTOR_MAG_SQUARED(dist);
        dist_tot = sqrt(dist_squared);
        if (dist_tot <= 0.0) {
          fprintf(stderr,
            " tech_euler_1: distance of %le -- quitting \n", dist_tot);
          return (1);
        }

        /* this is where we could add a softening parameter */
        top = g_G * g_body_array[i].mass * g_body_array[j].mass;
        bottom = (dist_tot + g_softening) * (dist_tot + g_softening);
        force_mag = top / bottom;

        for (c = 0; c < 3; c++) {
          dist_frac.val[c] = dist.val[c] / dist_tot;
          g_cur_forces[i][j].val[c] = 0.0 - (force_mag * dist_frac.val[c]);
          g_cur_forces[j][i].val[c] = 0.0 - g_cur_forces[i][j].val[c];
        }
      }
    }
    /* do the work here */
    for (i = 0; i < g_body_number; i++) {
      VECTOR tot_force;
      VECTOR tot_accel;
      /* 
       * we will compute a new position and velocity
       *   for each object in turn, placing the new
       *   values into the 'new_pos[]' and 'new_vel[]' 
       *   arrays.  Afterwards,
       *   we'll copy the new values back into the
       *   original g_body[] array.
       */
      /* use the current forces to compute new velocities */
      for (c = 0; c < 3; c++) {
        /* 
         * compute the total force on this body
         *    in each direction 
         */
        tot_force.val[c] = 0.0;
        for (j = 0; j < g_body_number; j++) {
          tot_force.val[c] += g_cur_forces[i][j].val[c];
        }

        /* from total force, compute total acceleration */
        tot_accel.val[c] = tot_force.val[c] / g_body_array[i].mass;

        /* use the acceleration to compute new velocity */
        new_vel[i].val[c] = g_body_array[i].vel[c] + tot_accel.val[c] * timestep;
      }

      /* use the NEW velocity to compute new positions */
      for (c = 0; c < 3; c++) {
        new_pos[i].val[c] = g_body_array[i].pos[c] + new_vel[i].val[c] * timestep;
      }
    }

    /* 
     * having computed the new positions and velocities,
     *    we now copy those values into the g_body_array[]
     */
    for (i = 0; i < g_body_number; i++) {
      for (c = 0; c < 3; c++) {
        g_body_array[i].pos[c] = new_pos[i].val[c];
        g_body_array[i].vel[c] = new_vel[i].val[c];
      }
    } * actual_timestep = timestep;
    return (0);
  }
/*-----------------------------------------------------------*/