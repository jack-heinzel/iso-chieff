We want to create a JAX-based sampling routine that samples from the conditional distribution p(a1, a2, cost1, cost2 | chi_eff, q, iso). 

That is, if p(a1, a2, cost2, cost2) = U(a1; [0, 1]) U(a2 ; [0, 1]) U(cost1 ; [-1, 1]) U(cost2 ; [-1, 1]), and chi_eff = (a1 cost1 + q a2 cost2) / (1 + q), then we want to sample from the conditional distribution of p(a1, a2, cost2, cost2 | chi_eff, iso) = p(a1, a2, cost2, cost2) delta(chi_eff - (a1 cost1 + q a2 cost2) / (1 + q)) / p(chi_eff) (using Bayes' theorem). 

We will accomplish this in two stages. First, we derive the conditional p(chi | iso) where chi = a * cost. For a uniform between 0 and 1 and cost uniform between -1 and 1, p(chi | iso) = - (ln |chi|) / 2 for |chi| < 1. 

Given a particular chi value, then P(cost < tau | chi, iso) = 1 - (ln |tau|) / (ln |chi|) and so we can use inverse CDF sampling to sample cost given a chi value, which then of course implies a particular value for a = chi / tau.

We also need to sample from the implied isotropic in chi distribution given a particular value for chi_eff. 

This works by p(chi1 | chi_eff, q, iso) = \frac{1+q}{q}\frac{p(chi1 | iso)}{p(chi_eff | iso, q)} p(chi_1 | iso) p(chi_2 = \frac{1+q}{q}chi_eff - \frac{chi_1}{q} | iso).

We can use inverse CDF sampling to sample from this marginal, then this implies a particular value for chi2 given chi1. 

With that, we have the full stack to produce the sample of a1, a2, cost1, cost2 given a isotropic underlying distribuiton and a particular chi_Eff and mass ratio (q) value.