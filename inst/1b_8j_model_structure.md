# 1b — Model structure (8j preliminary forecasting framework)

Academic-style specification of the model implemented in
`src/8j_preliminary_forecast.ipynb` and the modules it loads
(`framework.jl`, `degree_agepair.jl`, `renewal.jl`, `ngm.jl`, `joint_model.jl`,
`scoring.jl`). It extends the age-stratified next-generation-matrix (NGM) renewal
model of Munday et al. (2023, *PLoS Comput Biol* 19(9):e1011453) by replacing the
latent contact *matrix* with age-pair contact **degree distributions**, from which
two NGMs are derived. Notation and priors follow the analysis plan
(`analysis_plan_heavy_tail_mean.docx`) and, where it fixes numeric choices, the
reference implementation (`CovidAgeGroupForecast`).

## 1. Notation and data

Indices: age groups $a,b \in \{1,\dots,A\}$ with $A=7$ CIS bins
(2–10, 11–15, 16–24, 25–34, 35–49, 50–69, 70+); epidemiological weeks $t$,
Sunday-aligned to the inc2prev grid. Four weekly forecast **origins** $T$
(2021-01-03 … 2021-01-24; inst/1e) are analysed. For each, the estimation window comprises
$n_{\text{fit}}=8$ fitting weeks ending at $T$ preceded by $s_{\max}=4$ weeks of
renewal history, giving $\mathcal{T}=\{1,\dots,12\}$ with $t=T$ at the last index.

Data:
- $N_a$ — population of age group $a$ (England, inc2prev `age_school`).
- $I^{\text{obs}}_a(t)$, $\sigma^{I}_a(t)$ — weekly infection incidence (count) and its
  standard deviation, obtained by aggregating the **daily** inc2prev per-capita
  incidence to weekly totals, $I^{\text{obs}}_a(t)=N_a\sum_{d\in t}\hat p_a(d)$, with
  standard deviations combined in quadrature.
- $A_a(t)\in[0,1]$ — antibody (seropositive) prevalence, from inc2prev `gen_dab`.
- Contact reports: for participant-day $i$ of age group $a$, the number and
  duration-type of contacts to each contactee age group $b$, from CoMix-UK. Age
  groups of participant and contactee are assigned to CIS bins by a single seeded
  population-weighted draw for ambiguous ages (as in the 7j pipeline).

Age groups are aggregated into two blocks $B(a)\in\{\text{child},\text{adult}\}$
(child $=$ bins 1–2) for partial pooling of dispersion/shape parameters.

## 2. Contact degree distributions

For each ordered age pair $(a,b)$ (contactor age $a$ → contactee age $b$) define
the per-participant-day contact degree $k^{a\to b}_i$. Two families are considered;
in the lean preliminary the per-cell mean is treated as constant across the window
(the temporal random walk / Gaussian process of §8 is the intended extension).

**Unweighted (Negative Binomial).** Integer contact counts, zeros included:
$$
k^{a\to b}_i \sim \mathrm{NB}\!\left(\mu_{ab},\,\phi_{B(a)B(b)}\right),\qquad
\mathbb{E}[k]=\mu_{ab},\quad \mathrm{Var}[k]=\mu_{ab}+\mu_{ab}^2/\phi_{B(a)B(b)} .
$$
Its first two raw moments are
$\langle k\rangle_{ab}=\mu_{ab}$ and
$\langle k^2\rangle_{ab}=\mu_{ab}+\mu_{ab}^2\left(1+1/\phi_{B(a)B(b)}\right)$.

**Duration-weighted (hurdle-Weibull).** The weighted degree
$z^{a\to b}_i=\sum_\ell w_{\text{dur}}(\ell)$ sums per-contact duration weights over
that day's contacts to $b$. Individually-reported contacts use the five duration-bin
weights $\{2.5,10,37.5,150,240\}/240$; **group (mass) contacts**, which carry no
recorded duration, receive the explicit fixed weight $w_{\text{dur,group}}=2.5/240$
(inst/1e; fixed now, estimable later). Zeros are modelled separately (hurdle) with an
**empirical** zero probability $p^0_{ab}$ taken from the survey cell; the positive
part is Weibull with shape $\kappa_{B(a)B(b)}$ and a mean parameterisation
$$
z^{a\to b}_i \mid z>0 \;\sim\; \mathrm{Weibull}\!\left(\kappa_{B(a)B(b)},\,\lambda_{ab}\right),\qquad
\lambda_{ab}=\frac{\mu^{W}_{ab}}{\Gamma(1+1/\kappa_{B(a)B(b)})},
$$
so $\mu^{W}_{ab}=\mathbb{E}[z\mid z>0]$. With
$\mathrm{CV}_W^2=\Gamma(1+2/\kappa)/\Gamma(1+1/\kappa)^2-1$, the zero-included raw
moments are
$\langle k\rangle_{ab}=(1-p^0_{ab})\,\mu^{W}_{ab}$ and
$\langle k^2\rangle_{ab}=(1-p^0_{ab})\,(\mu^{W}_{ab})^2\,(1+\mathrm{CV}_W^2)$.

**Latent means — reciprocity-structural, GP-smoothed (inst/1e).** Reciprocity
$N_a\mu_{ab}=N_b\mu_{ba}$ is imposed *within the estimation* by modelling a single symmetric
log-rate $r_{\{a,b\}}$ per **unordered** age pair ($A(A{+}1)/2=28$ for $A=7$) and recovering the
directional means through the contactee-population offset,
$$
\log\mu_{ab}=r_{\{a,b\}}+\log N_b,\qquad r_{\{a,b\}}=r_{\{b,a\}},
$$
so $N_a\mu_{ab}=N_b\mu_{ba}$ holds exactly (docx "Unweighted…estimation"). The rate field is
smoothed across the age-pair grid by a **separable, shared-length-scale Gaussian process** with a
squared-exponential kernel over age-bin midpoints $x_a$ (each bin takes its interval midpoint; the
open-ended 70+ is fixed to $x_7=74.5$),
$$
r=c+\eta\,Lz,\quad z\sim\mathcal N(0,I),\quad L=\operatorname{chol}(K),\quad
K_{\{a,b\},\{c,d\}}=\exp\!\Big(-\tfrac{(x_a-x_c)^2}{2\rho^2}-\tfrac{(x_b-x_d)^2}{2\rho^2}\Big),
$$
evaluated on the 28 unordered pairs (the upper-triangular submatrix of the $7^2$ separable
kernel), with a global intercept $c$, marginal scale $\eta$ and shared length-scale $\rho$. This
replaces the earlier 49 independent deviations $\varepsilon_{ab}$; the same construction serves both
families ($\mu_{ab}$ for NegBin, $\mu^W_{ab}$ for the hurdle-Weibull). The docx's *temporal* GP —
and it declined age-smoothing on too-few bins — is overridden here per inst/1e; the window mean
stays constant in time (temporal GP is the remaining swap-in seam).

## 3. Effective contact matrix and reciprocity

A single scalar per cell summarises the degree distribution for transmission. The
two NGM builders differ only here:
$$
C^0_{ab}=
\begin{cases}
\langle k\rangle_{ab}, & \text{mean NGM},\\[2pt]
\dfrac{\langle k^2\rangle_{ab}}{\langle k\rangle_{ab}}\,g_{ab}, & \text{neighbourhood-degree NGM,}
\end{cases}
$$
the neighbourhood case being the size-biased (excess / "friendship-paradox") degree
$\langle k^2\rangle/\langle k\rangle$ (a configuration network with no degree–degree
correlation, cf. Saumell-Mendiola et al. 2012) **conditioned on non-zero contacts**
via a per-cell non-zero factor $g_{ab}$ (analysis-plan docx configuration-network
$C_{\text{cf}}=(z^2/z)\odot P_{nz}$; inst/1c, 1d — this **overrides** the earlier
"$(1-p^0)$ cancels" reading):
$$
g_{ab}=
\begin{cases}
1/(1-P^0_{ab}),\quad P^0_{ab}=\bigl(\phi/(\phi+\mu_{ab})\bigr)^{\phi}, & \text{NegBin (left-truncated fit)},\\[2pt]
1-p^0_{ab}, & \text{hurdle-Weibull (empirical }p^0\text{)}.
\end{cases}
$$
Thus the neighbourhood $C^0$ is $(\mu+1+\mu/\phi)/(1-P^0)$ for NegBin and
$(1-p^0)\,\mu^{W}(1+\mathrm{CV}^2_W)$ for the Weibull; $(1-P^0)$ is floored at
$\varepsilon=10^{-3}$ so near-empty cells ($\mu\to0$) stay finite. The mean NGM
($C^0=\langle k\rangle$) is unchanged.

Total-contact **reciprocity** ($N_a\,\mu_{ab}=N_b\,\mu_{ba}$) is imposed by
symmetrising totals,
$$
C^\ast_{ab}=\frac{N_a\,C^0_{ab}+N_b\,C^0_{ba}}{2\,N_a},
\qquad\text{so } N_a\,C^\ast_{ab}=N_b\,C^\ast_{ba}.
$$
Since the *mean* is now reciprocity-structural (§2), for the mean NGM ($C^0=\langle
k\rangle=\mu$) this symmetrisation is the identity; it remains active for the
neighbourhood NGM, whose size-biased $C^0$ is not reciprocal even when $\mu$ is.

## 4. Next-generation matrix

Age-specific susceptibility and infectivity are time-constant hierarchical random
effects (non-centred log-normal),
$$
s_a=\exp(\mu_s+\sigma_s z^s_a),\qquad
\iota_a=\exp(\mu_\iota+\sigma_\iota z^\iota_a),\qquad z^s_a,z^\iota_a\sim\mathcal N(0,1).
$$
Immunity enters through a *leaky* antibody term with effectiveness $F$,
$$
\tilde s_a(t)=s_a\bigl(1+(F-1)\,A_a(t)\bigr).
$$
The next-generation matrix for week $t$ is
$$
N_{ab}(t)=\tilde s_a(t)\,C^\ast_{ab}\,\iota_b ,
$$
i.e. $\mathbf N(t)=\operatorname{diag}\!\big(\tilde{\mathbf s}(t)\big)\,\mathbf C^\ast\,\operatorname{diag}(\boldsymbol\iota)$.
The overall transmissibility scale (the household secondary-attack-rate role) is
carried by the informative priors on $s_a$ and $\iota_a$; no separate scalar is fit.

## 5. Generation interval and renewal equation

The weekly generation-interval weights are a discretised, truncated log-normal,
$$
w(s)=\frac{F_{\mathrm{LN}}(s)-F_{\mathrm{LN}}(s-1)}{\sum_{u=1}^{s_{\max}}\bigl[F_{\mathrm{LN}}(u)-F_{\mathrm{LN}}(u-1)\bigr]},
\qquad s=1,\dots,s_{\max}=4,
$$
with $F_{\mathrm{LN}}$ the CDF of a log-normal (in weeks) matched to a 5-day mean
and standard deviation (reference value; fixed in the preliminary). Incident
infections follow the age-structured renewal equation
$$
I_a(t)=\sum_{s=1}^{s_{\max}} w(s)\sum_{b} N_{ab}(t)\,I_b(t-s),
$$
a single NGM per target week applied to the $s_{\max}$ lagged infection vectors. In
the preliminary the lagged terms use the observed incidence $I^{\text{obs}}_b(t-s)$
(as in the reference production model); the fully latent-state recursion is the
intended alternative.

## 6. Likelihood

The joint model combines a **contact** term and an **infection** term.

Contact likelihood (sum over cells; per family):
$$
\mathcal L_{\text{contact}}=
\sum_{a,b}\begin{cases}
\sum_i \log \mathrm{NB}\!\left(k^{a\to b}_i\mid\mu_{ab},\phi_{B(a)B(b)}\right), & \text{NegBin},\\[4pt]
\sum_{i:\,z^{a\to b}_i>0}\log \mathrm{Weibull}\!\left(z^{a\to b}_i\mid\kappa_{B(a)B(b)},\lambda_{ab}\right), & \text{hurdle-Weibull},
\end{cases}
$$
the empirical hurdle probability $p^0_{ab}$ being constant in the parameters and
omitted from the positive-part contribution.

Infection likelihood over the fitting weeks $t\in\{s_{\max}+1,\dots,12\}$, with an
incidence-scaled observation error,
$$
I^{\text{obs}}_a(t)\sim \mathcal N\!\left(\hat I_a(t),\,\sigma_a(t)\right),\quad
\hat I_a(t)=\sum_{s=1}^{s_{\max}} w(s)\sum_b N_{ab}(t)\,I^{\text{obs}}_b(t-s),
$$
$$
\sigma_a(t)=\sqrt{\bigl(\sigma_{\text{inf}}\,I^{\text{obs}}_a(t)\bigr)^2+\bigl(\sigma^{I}_a(t)\bigr)^2}.
$$

## 7. Priors

$$
\begin{aligned}
&z_{\{a,b\}}\sim\mathcal N(0,1),\quad c\sim\mathcal N(c_0,3),\quad
\log\rho\sim\mathcal N(\log 15,0.5),\quad \log\eta\sim\mathcal N(0,0.5),\\
&\log\phi_{BB'}\sim\mathcal N(0,1),\qquad
\log\kappa_{BB'}\sim\mathcal N(0,0.5),\\
&\mu_s\sim\mathrm{Beta}(24,24),\quad \sigma_s\sim\mathcal N_{+}(0.1,0.02),\quad z^s_a\sim\mathcal N(0,1),\\
&\mu_\iota\sim\mathrm{Beta}(4,12),\quad \sigma_\iota\sim\mathcal N_{+}(0.1,0.02),\quad z^\iota_a\sim\mathcal N(0,1),\\
&F\sim\mathrm{Beta}(5,1),\qquad \sigma_{\text{inf}}\sim\mathcal N_{+}(0.05,0.025),
\end{aligned}
$$
where $\mathcal N_{+}$ denotes a normal truncated to the positive half-line and
$B,B'$ index the child/adult blocks. The GP intercept is centred at the empirical
grand-mean log-rate $c_0=\overline{\log\hat\mu_{ab}-\log N_b}$, and the length-scale
$\rho$ (in age-years) is shared across both kernel dimensions. (Log-scale parameters
— including $\log\rho,\log\eta$ — are clamped to safe ranges inside the model to
prevent optimiser overflow; the posterior mode lies well inside these bounds.)

## 8. Inference

Each of the four model configurations (degree family × NGM builder) is a single
joint fit. Pathfinder (Zhang et al. 2022) provides the parsimonious variational fit
and an initialisation, from which Turing.jl runs the No-U-Turn sampler for the
formal fit (`Random.seed!(1236)`). The spatial GP of §2 (age-pair smoothing) is now
part of the estimation; the remaining temporal extension replaces the
constant-in-time mean with a per-cell first-order random walk (or a shared Matérn GP)
over the window weeks, entering only through $\log\mu_{ab}(t)$ and requiring no change
to §§3–6.

## 9. Forecasting

Forecasts for horizons $h=1,\dots,4$ use a **contact-updated iterate** scheme
(inst/1c, 1d): only the contact matrix is assumed available at $T+h-1$, so the degree
distribution is **re-estimated each horizon** from a contact window ending at
$T+h-1$, while the infection likelihood and antibody prevalence stay **frozen at the
origin** $T$. The refreshed NGM $\mathbf N_h$ advances the renewal one week;
intervening weeks enter as mean-plugged lags. Posterior-predictive draws add
observation noise $\sigma=\sigma_{\text{inf}}\,\hat I$, and one MCMC chain is saved
per (origin, horizon) (`dt_intermediate/8j_chn_*.jld2`). The scheme is run for **four
forecast origins** ($T\in\{$2021-01-03, 01-10, 01-17, 01-24$\}$; inst/1e), and scores
are aggregated **by horizon** across origins. (The earlier frozen-NGM roll-forward —
one fit at $T$, iterated with $\mathbf N(T)$ held fixed — remains available as
`posterior_forecast`.)

## 10. Evaluation

For each configuration and age group the posterior-predictive forecast is
summarised at quantile levels $\{0.05,\dots,0.95\}$ and scored against realized
weekly incidence with the **weighted interval score (WIS)** and its
over-/under-prediction and dispersion components, bias, and 50%/90% central-interval
coverage, computed with the R `scoringutils` package (v2). WIS is computed on a **log
scale** (`transform_forecasts(fun=log_shift, offset=1)`; inst/1e) — appropriate for
incidence spanning orders of magnitude — and reported **aggregated by horizon** across
the four origins (both scales are written out; log is the headline). The four
configurations are compared by mean log-scale WIS (lower is better).

---

**References.** Munday et al. (2023) *PLoS Comput Biol* 19(9):e1011453 —
`inst/pcbi.1011453.pdf`. Saumell-Mendiola, Serrano & Boguñá (2012) *Phys Rev E* 86,
026106 — epidemic thresholds on interconnected networks. Zhang et al. (2022)
Pathfinder, *JMLR* 23. Bosse et al. — `scoringutils`. Analysis plan:
`inst/analysis_plan_heavy_tail_mean.docx`. Reference code:
`CovidAgeGroupForecast` (`stan/multi-option-contact-model.stan`).
