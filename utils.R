


run_prop_test <- function(tab_levels){
  
  # number of obs in each node
  n_lo <- tab_levels[[1]] + tab_levels[[2]]
  n_hi <- tab_levels[[3]] + tab_levels[[4]]
  n_0 <- n_lo + n_hi
  
  # number of obs choosing the higher level 
  q_lo <- tab_levels[[2]]
  q_hi <- tab_levels[[4]]
  q_0 <- tab_levels[[3]] + tab_levels[[4]]
  
  # Hypothesis: p(lo) >= p(0) >= p(hi)
  # t1 <- prop.test(x=c(q_lo,q_0),n=c(n_lo,n_0),alternative = 'less')
  # t2 <- prop.test(x=c(q_0,q_hi),n=c(n_0,n_hi),alternative = 'less')
  
  matrix_1 <- matrix(c(q_lo,q_0,n_lo,n_0),nrow=2)
  matrix_2 <- matrix(c(q_0,q_hi,n_0,n_hi),nrow=2)
  
  t1 <- fisher.test(matrix_1, alternative = 'less')
  t2 <- fisher.test(matrix_2, alternative = 'less')
  
  # Hochberg step-up correction
  p_value_up <- min(min(sort(c(t1$p.value,t2$p.value)) * c(2,1)),1)
  
  # Holm step-down method
  p_value_down <- min(max(sort(c(t1$p.value,t2$p.value)) * c(2,1)),1)
  
  new_row <- data.frame(n_0 = n_0,
                        prop.lo = q_lo/n_lo,
                        prop.0 = q_0/n_0,
                        prop.hi = q_hi/n_hi,
                        p.lo_0 = t1$p.value,
                        p.0_hi = t2$p.value,
                        p.up = p_value_up,
                        p.down = p_value_down)
  return(new_row)
}



mloglike <- function(p,data){
  sum_tab <- table(data$choice1)
  n_0 <- sum_tab[3] + sum_tab[4]
  LL1 <- n_0 * log(p[1]) + (sum(sum_tab) - n_0) * log(1-p[1])
  LL2 <- sum_tab[4] * log(p[2]) + sum_tab[3] * log(1-p[2])
  LL3 <- sum_tab[2] * log(p[3]) + sum_tab[1] * log(1-p[3])
  return(LL1 + LL2 + LL3)
}

sigmoid <- function(x) {
  return(1 / (1 + exp(-x)))
}

obj <- function(q, data) {
  # Transform q to p
  p1 <- sigmoid(q[1])
  p2 <- sigmoid(q[2]) * p1
  p3 <- p1 + sigmoid(q[3]) * (1 - p1)
  
  # Ensure the order constraint: p[2] < p[1] < p[3]
  if (p3 <= p1 || p1 <= p2) {
    return(Inf)  # Return a high penalty if constraints are violated
  }
  
  # Evaluate the log-likelihood with transformed p
  return(-mloglike(c(p1, p2, p3), data))
}

# Optimize
init_q <- c(0.5,0.5,0.5)
result_1 <- optim(init_q, obj, data = AR, method = "BFGS")

# Extract the optimal p
optimal_q <- result_1$par
optimal_p1 <- sigmoid(optimal_q[1])
optimal_p2 <- sigmoid(optimal_q[2]) * optimal_p1
optimal_p3 <- optimal_p1 + sigmoid(optimal_q[3]) * (1 - optimal_p1)
optimal_p <- c(optimal_p1, optimal_p2, optimal_p3)

# Without the p[2]<p[1]<p[3] constraint
obj_without_ineq <- function(p, data) {
  loglike_value <- mloglike(p, data)
  
  if (!is.finite(loglike_value)) {
    return(Inf)  # If the log-likelihood is not finite, return a large penalty
  }
  
  return(-loglike_value)
}

# Bounds for p (all dimensions between [0, 1])
lower_bounds <- c(0.001, 0.001, 0.001)
upper_bounds <- c(0.999, 0.999, 0.999)

# Optimize using L-BFGS-B method with box constraints
init_p <- c(0.5, 0.5, 0.5)
result_2 <- optim(init_p, obj_without_ineq, data = AR, 
                  method = "L-BFGS-B", 
                  lower = lower_bounds, upper = upper_bounds)

c(result_1$value, optimal_p)
c(result_2$value, result_2$par)


freq_1 <- (table(AR$choice1)[3]+table(AR$choice1)[4])/sum(table(AR$choice1))
freq_2 <- table(AR$choice1)[4]/sum(table(AR$choice1)[3:4])
freq_3 <- table(AR$choice1)[2]/sum(table(AR$choice1)[1:2])

c(freq_1,freq_2,freq_3)




