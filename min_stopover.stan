data{
  int<lower=0> Nind; // number of individuals
  int<lower=0>  Nsp; // number of species
  array[Nind] int<lower=0> y; // min stopover duration
  array[Nind] int<lower=1, upper=Nsp> sp; // species index
  array[Nind] int<lower=1, upper=2>   st; // state index
}

parameters{
  vector[2] mu; // group-level mean
  array[Nsp] vector[2] b; // beta
  corr_matrix[2] Rho; // correlation matrix
  vector<lower=0>[2] sigma; // group-level SD
  real<lower=0>  phi; // scaling parameter
}

model{
  vector[Nind] lambda;
  
  // Priors and constraints
  mu    ~ normal(log(7), 2);
  b     ~ multi_normal(mu, quad_form_diag(Rho, sigma));
  Rho   ~ lkj_corr(2);
  sigma ~ exponential(0.5);
  phi   ~ exponential(1);
  for (i in 1:Nind) {
    lambda[i] = b[sp[i], st[i]];
  }
  
  // Likelihood
  target += neg_binomial_2_log_lpmf(y | lambda, phi);
}

generated quantities{
  vector[Nind] yrep;
  for (i in 1:Nind){
    yrep[i] = neg_binomial_2_log_rng(b[sp[i], st[i]], phi);
  }
}
