param N;
param S;
param p > 0 integer;
param train_size > p integer;
param y {t in 1..train_size} >= 0;
param delta {t in 1..train_size, j in 1..S} binary;

var beta {l in 1..(p+1), j in 1..S};

minimize ar_obj:
	sum{j in 1..S, t in (p+1)..train_size} ( delta[t, j] *
		( y[t] - beta[1, j] - sum{k in 1..p}(beta[k+1, j] * y[t-k]) )^2
	);