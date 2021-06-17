################################### SETS ######################################

# The set 'state_machine' tells if an appliance works as a state machine while
set state_machine;

# The set 'always_on' tells if an appliance works the whole time (i.e. is
# always active)
set always_on;

# The set 'clipped' tells if a state of an appliance at a specific time is 
# implausible to be on, this information is calculated in a pre-procesing stage
# on the basis of the power states and the y
set clipped dimen 3;

################################# PARAMETERS ##################################

# The parameter 'N' is the number of appliances chosen while 'T' is the time 
# horizon (config file)
param N > 0 integer;
param T > 0 integer;

# Each appliance has at least one non-off state, the number of these states is
# stored in the vector 'S' while the power consumptions related to each power
# state is stored in the parameter 'p' (power_states_count and 
# power_states_power files)
param S {i in 1..N} > 0 integer;
param p {i in 1..N, j in 1..S[i]} > 0;

# Each appliance power state also has a minimum in state time 'a', a maximum 
# in state time 'b' and a maximum power consumption 'm' in the scheduled 
# horizon (min_in_state_time and max_power_consumption files)
param a {i in 1..N, j in 1..S[i]} > 0;
param b {i in 1..N, j in 1..S[i]} > 0;
param m {h in 1..2, i in 1..N} > 0;

# The aggregated power consumption of all the appliances chosen for every time
# step in time horizon can be found in the vector 'y' (total file)
param y {t in 1..T} >= 0;

# The parameters 'w' are appliance-specific weights, if an appliance 
# that change state in a very short time period this variable is set to 0, 
# this variable is also affected by the granularity of the dataset in fact for 
# the grain of minute this variable is set to 0 (penalty_weights file)
param w {i in 1..N};

# The parameters 's' are appliance and time specific weights
param s {t in 1..T, i in 1..N} >= 0, <= 1;
param l {i in 1..N};

# The matrices 'U' and 'D' tells what state changes are possible 
# (U_matrices and D_matrices files)
param U {i in state_machine, j in 1..S[i], k in 1..S[i]};
param D {i in state_machine, j in 1..S[i], k in 1..S[i]};

# The non-negative parameters 'lambda1' and 'lambda2' represents the penalty 
# weight used in the objective function to enforce peice-wise constant 
# behaviour and to penalize a lot of appliances on together (config file)
param lambda1 >= 0;
param lambda2 >= 0;

param Tzone1 >= 0;
param Tzone2 >= 0;
param u_bound {i in 1..N} >= 0;

################################# VARIABLES ###################################

# The main variable 'x' is a binary one that represents the state vector for 
# each appliance during the whole time horizon
var x {t in 1..T, i in 1..N, j in 1..S[i]} binary;

# The other variable are binary support variables added to better fit the 
# problem and create a rapresentative model, those variable are:
# 	'u' that is one if the state j of the appliance i at time t has an upward 
# 		transition
#	'd' that is the dual variable of u, in fact this is one if ther is a 
#		downward transition
#	'c' that is a variable that, with the first two constraints, model the 
#		l-infinity norm of (x[t, i] - x[t-1, i]) used in the objective function
# 		to enforce temporal smoothness
var u {t in 2..T, i in 1..N, j in 1..S[i]} binary;
var d {t in 2..T, i in 1..N, j in 1..S[i]} binary;

############################# OBJECTIVE FUNCTION ##############################

minimize objective_binary:
	0.5 * sum{t in 1..T}(y[t] - sum{i in 1..N, j in 1..S[i]}(p[i, j] * x[t, i, j]))^2 +
	lambda1 * sum{t in 2..T, i in 1..N}(w[i] * sum{j in 1..S[i]}(x[t, i, j] - x[t-1, i, j])^2) +
	lambda2 * sum{t in 2..T, i in 1..N}(l[i] * (1 - s[t, i]) * sum{j in 1..S[i]}(x[t, i, j]^2));

################################ CONSTRAINTS ##################################

# This constraint enforce appliance to have at most one active state
s.t. one_active_state {t in 1..T, i in 1..N}:
	sum{j in 1..S[i]}(x[t, i, j]) <= 1;

# This contraint enforce always-on device to have exactly one active state
s.t. always_on_appliances {t in 1..T, i in always_on}:
	sum{j in 1..S[i]}(x[t, i, j]) = 1;

# Those constraints model the variables u and d to represent the transitions
s.t. link_transition {t in 2..T, i in 1..N, j in 1..S[i]}:
	u[t, i, j] - d[t, i, j] = (x[t, i, j] - x[t-1, i, j]);

s.t. at_most_one_transition {t in 2..T, i in 1..N, j in 1..S[i]}:
	u[t, i, j] + d[t, i, j] <= 1;

# This constraint enforce state machine behaviour
s.t. state_machine_behaviour {t in 2..T, i in state_machine, j in 1..S[i]}:
	sum{k in 1..S[i]}(U[i, j, k] * u[t, i, k]) 
	= sum{k in 1..S[i]}(D[i, j, k] * d[t, i, k]);

# This constraint enforce the applaiances to stay in a state for some time 
# before change to another state
s.t. minimum_in_state_time {i in 1..N, j in 1..S[i], 
							t in 2..(T - a[i, j] + 1)}:
	sum{k in t..(t + a[i, j] - 1)}(x[k, i, j]) >= a[i, j] * u[t, i, j];

# This constraint enforce the applaiances to stay in a state for at most a 
# predefined time
s.t. maximum_in_state_time {i in 1..N, j in 1..S[i], 
							t in 1..(T - b[i, j])}:
	sum{k in t..(t + b[i, j])}(x[k, i, j]) <= b[i, j];

# This constraint enforce each appliance to consume at most a maximum amount of
# energy in the time horizon
s.t. maximum_appliance_consumption1 {i in 1..N}:
	sum{t in (Tzone1+1)..Tzone2, j in 1..S[i]}(p[i, j] * x[t, i, j]) <= m[1, i];

s.t. maximum_appliance_consumption2 {i in 1..N}:
	sum{t in 1..Tzone1, j in 1..S[i]}(p[i, j] * x[t, i, j]) + 
	sum{t in (Tzone2+1)..T, j in 1..S[i]}(p[i, j] * x[t, i, j]) <= m[2, i];

s.t. maximum_activations {i in 1..N}:
	sum{t in 2..T, j in 1..S[i]} u[t, i, j] <= u_bound[i];

# This constraint keep the sum of appliances' consumptions during the time 
# horizon below the sum of the aggregated value in the same period
s.t. total_consumption{t in 1..T}:
	sum{i in 1..N, j in 1..S[i]}(p[i, j] * x[t, i, j]) <= y[t];

# This constraint force some variable to zero to lower complexity
s.t. clip_to_zero{(t, i, j) in clipped}:
	x[t, i, j] = 0;
