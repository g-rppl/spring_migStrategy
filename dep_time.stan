// non-centered parameterisation

data{
  int Nind; // number of individuals
  int  Nsp; // number of species
  vector[Nind] y; // departure relative to night length
  array[Nind] int sp; // species index
  array[Nind] int st; // state index
}

parameters{
  vector<lower=0>[2] sigma_b; // group-level SD
  cholesky_factor_corr[2] L_Rho; // Cholesky correlation factor
  matrix[2,Nsp] z; // z-scores
  real<lower=0> v; // scaling parameter
  real<lower=0> sigma; // population-level SD
}

transformed parameters{
    matrix[Nsp,2] b; // beta
    b = (diag_pre_multiply(sigma_b, L_Rho) * z)';
}

model{
  vector[Nind] mu;

  // Priors and constraints
  sigma_b ~ exponential(1);
  L_Rho   ~ lkj_corr_cholesky(2);
  to_vector(z) ~ normal(0, 1);
  v     ~ exponential(1);
  sigma ~ exponential(1);
  for (i in 1:Nind){
    mu[i] = b[sp[i], st[i]];
  }

  // Likelihood
  target += student_t_lpdf(y | v, mu, sigma);
}

generated quantities{
  vector[Nind] yrep;
  for (i in 1:Nind){
    yrep[i] = student_t_rng(v, b[sp[i], st[i]], sigma);
  }
}
