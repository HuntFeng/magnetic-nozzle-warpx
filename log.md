## This log records the observations of each diagnostics 
The detail parameters can be found in each diags folder.
### diags20231121045
INFO:
1. produced a simulation with total time of 1 ion crossing time 
FIXME:
1. It seems the ions and electrons are crossing the magnetic field lines, i.e. the injection spreads in radial direction. 
2. The B_max=30T is too large.
TODO:
1. set the time step so that it's less than 1 cyclotron period. 
2. reduce magnetic field 100 times

### diags202311292146
INFO:
1. [x] set the time step to 0.5 cyclotron period. 
2. [x] reduce magnetic field 100 times
FIXME: 
1. The potential near the exit is too large, some oscillations developed, why is that? due to bc?
2. The spread is too large, what causes the spread?
3. The potential solver, is it doing things correctly? 
TODO: 
- [x] ensure that we resolve debye length every time step
- [x] case 1: set ion temperature lower, 60eV
- [x] case 2: set 0 electric field and check if the particles follow the magnetic field correctly, there should be no spread. What causes the spread?
- [x] case 3: don't change temperatures at all, but change the bc to all dirichlet

### diags202312061235
INFO:
1. set dimensions to 160x1600
2. Ti=60eV
FIXME:
1. simulation is too long to finish (estimated 47 hours for 0.5 ict using 80 cores)

### diags202312061237
INFO:
1. set dimensions to 160x1600
2. Dirichlet BCs
FIXME:
1. the potential is negative in the core region. Because we are injecting same number of electrons and ions every time step?

### diags202312061915
INFO:
1. set dimensions to 160x1600
2. no electric field
FIXME:
1. particles does not follow magnetic field lines