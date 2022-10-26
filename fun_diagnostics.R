

########## extract model draws ##########

get_draws <- function(var = NULL, d = drws) {
  class(d) <- "data.frame"
  dd <- as.matrix(d[,startsWith(colnames(d), var)])
  return(dd)
}


########## extract posterior predictions ##########

get_sims <- function(d = NULL, sims = 50) {
  sim <- data.frame(x = rep(d, sims),
                    sim = c(fitmat[,round(runif(sims, 1, ncol(fitmat)))]),
                    draw = rep(1:sims, each = nrow(newdat)))
  return(sim)
}


########## posterior probabilities ##########

p_probs <- function(d = drws) {
  
  class(d) <- "data.frame"
  f <- c()
  
  for (i in 1:(ncol(d)-3)) {
    prob <- ifelse(mean(d[,i]) > 0, sum(d[,i] > 0), sum(d[,i] < 0)) / nrow(d)
    f    <- c(f, prob)
  }
  
  return(f)
  
}


########## summaries posterior ##########

mod_smry <- function(model = NULL, prms = NULL, trnsf = NULL) {
  
  require(cmdstanr)
  require(coda)
  
  smry <- model$summary(prms, "mean")   # mean
  drws <- model$draws(prms, format = "df")   # draws
  
  if (!is.null(trnsf)) {
    # transformations
    smry$mean <- trnsf(smry$mean)
    drws[,-tail(1:ncol(drws), 3)] <- trnsf(drws[,-tail(1:ncol(drws), 3)]) 
  }
  
  HPDI <- HPDinterval(as.mcmc(drws), prob = 0.9)[1:nrow(smry),]   # highest posterior density interval
  if (!is.null(dim(HPDI))) {
    smry <- cbind(smry, HPDI)   # add 90% HPDI
  } else {
    smry$lower <- HPDI[1]
    smry$upper <- HPDI[2]
  }

  smry <- cbind(smry, f = p_probs(drws))   # add posterior probabilities
  smry <- cbind(smry, model$summary(prms, "rhat", "ess_bulk", "ess_tail")[,-c(1)])   # add diagnostics
  smry[,7:8] <- round(smry[,7:8])   # round ess
  rownames(smry) <- smry$variable
  return(smry[,-c(1)])

}


########## MCMC diagnostics ##########

diagMCMC <- function(drws) {
  
  require(gridExtra)
  require(bayesplot)
  
  for(i in 1:(ncol(drws)-3)) {
    
    d <- drws[,c(i, (ncol(drws)-2):ncol(drws))]
    
    g1 <- mcmc_trace(d) + ylab(NULL) + theme(legend.position = "none")
    g2 <- mcmc_acf_bar(d) + ylab(NULL)
    g3 <- mcmc_rank_overlay(d) + theme(legend.position = "none")
    g4 <- mcmc_dens_overlay(d) + theme(legend.position = "none")

    grid.arrange(g1, g2, g3, g4)
    
  }
}
