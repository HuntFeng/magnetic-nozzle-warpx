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

### diags202312071328
INFO:
1. dimensions to 160x1600
2. set $dt = 0.1 / \omega_c$
3. no electric field
FIXME:
1. the spread is still wide
TODO:
1. [x] set Bmax = 200G
2. [x] ion temperature 1ev
3. [x] make electron larmor radius 1/50 of the system radius (and make sure ion is small as well)

### diags20201212
INFO:
1. dimensions to 160x1600
2. set $dt = 0.1 / \omega_c$
3. still keep Bmax = 0.3T (if setting it to 200G, larmor radius is still large)
4. T_i = 1eV, T_e = 300eV
5. set mirror ratio R=2, so that B is not too small at the two ends
6. add 1/sqrt(2) in the vx, vy random numbers, so that v_perp = v_thermal
7. larmor radius of both species ~ 1/50 of system radius
8. Electric field is 0