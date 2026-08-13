# Preliminary Model Structure: A Joint Age-Pair Contact-Degree / Renewal Forecasting Model

*Reverse-engineered specification of the model implemented in `src/8j_preliminary_forecast.ipynb`
and the framework modules it loads (`forecast_utils.jl` → `framework.jl`, `infection_data.jl`,
`degree_agepair.jl`, `renewal.jl`, `ngm.jl`, `joint_model.jl`, `scoring.jl`). This document
records the model exactly as coded; it is written to stand on its own as a methods description.*

*It describes the model **as it is**, not how it got here. Superseded variants, reverted
experiments and the measurements behind each choice are deliberately not kept here — they are in
git history, in `tasks/lessons.md`, and in far more detail in the field docstrings of
`src/framework.jl` (`ar1_phi_prior`, `RHO_BOUNDS`, `stage1_z_init_scale`, `stage1_phi_init_scale`,
`stage1_phi_pf_max`, `contacts_label`) and the model bodies in `src/joint_model.jl`.*

---

## 1. Overview

We forecast weekly, age-stratified SARS-CoV-2 infection incidence in England by coupling two
components:

1. a **contact-degree model** that describes, for each ordered pair of age groups, the
   distribution of the number (or duration-weighted amount) of daily social contacts reported
   in the CoMix-UK survey; and
2. an **age-structured renewal / next-generation-matrix (NGM) transmission model** whose
   effective-contact matrix is constructed directly from the raw moments of those contact-degree
   distributions.

The design is *composable*, organised around two orthogonal modelling axes that are each fixed
once per fit and dispatched deterministically:

| Axis | Symbol in code | Options | Meaning |
|------|----------------|---------|---------|
| **1. Contact-degree model** | `ContactDegreeModel` | `NegBinAgePair`, `HurdleWeibullAgePair`, `NoContactDegree` | How the age-pair degree distribution is modelled (unweighted counts vs. duration-weighted hurdle; or not at all) |
| **2. NGM builder** | `NGMBuilder` | `MeanNGM`, `NeighbourhoodDegreeNGM`, `DiagonalMeanNGM`, `NullNGM` | How the per-capita effective contact $C^0$ is formed from the degree distribution's moments |

The Cartesian product of the first two options on each axis gives the **"four ways"** — a
$2\times2$ grid of model variants — that the analysis fits and scores side by side. **Two
baselines** (`inst/6_null_interaction_model.md`) sit alongside that grid as two further *pairings*
of the same axes, not as a new axis:

| Model | Pairing | What it removes |
|-------|---------|-----------------|
| **No-interaction** `unweighted-negbin\|mean-diagonal` | `NegBinAgePair` × `DiagonalMeanNGM` | Off-diagonal transmission: only $\mathrm{diag}(C^\ast)$ enters the NGM, so each age group's epidemic is self-contained. Age-dependent **infectivity is pinned to $\mathrm{inf}\equiv1$** (§6) because a diagonal NGM identifies only the product $\mathrm{susc}_a\!\cdot\!\mathrm{inf}_a$. Reuses the NegBin **Stage-1 chains verbatim** (they carry no NGM token), so it costs no extra Stage-1 fits. This is `REF_MODEL`, the relative-skill reference in `9j_viz_utils.jl`. |
| **Null** `no-contact\|null` | `NoContactDegree` × `NullNGM` | **All** contact data: $C^\ast$ is a fixed uniform constant $\bar c$ (§5.1), so only the transmission parameters (age-dependent susceptibility/infectivity, generation interval) drive the forecast. Stage 1 is skipped entirely. |

Inference is a **two-stage cut** (§6.0, `inst/4_cut_Bayes.md`): the contact-degree likelihood
(`model_degree`, Stage 1) and the infection likelihood (`model_transmission`, Stage 2) are
**separate** probabilistic programs, with Stage 2 conditioning on Stage-1 draws and no feedback the
other way. Stage 1 is NGM-independent (Axis 2 enters only downstream), so one Stage-1 fit per degree
model serves both builders; the two axes remain fixed arguments so each stage's parameter space is
well defined.

Notationally we use $A$ age groups indexed $a,b,i,j \in \{1,\dots,A\}$, with $i$ (or $a$) the
**participant / contactor / susceptible** group and $j$ (or $b$) the **contactee / infectious**
group. Time is discretised into calendar weeks indexed $t$.

---

## 2. Data and discretisation

### 2.1 Age grid

The age partition is the set of $A = 7$ England **Coronavirus Infection Survey (CIS)**
`age_school` bins, read from `inc2prev/data-processed/populations.csv`
(`cis_age_grid`). Restricting to lower age limit $\ge 2$ and sorting gives the bins

$$
[2,10],\ [11,15],\ [16,24],\ [25,34],\ [35,49],\ [50,69],\ [70+],
$$

each carrying an England population $N_a$ used as the demographic offset in the contact-mean
reciprocity structure (§5). For the spatial smoother (§5) each bin is assigned a numeric age
coordinate equal to its interval midpoint, except the open-ended $70+$ bin, which is fixed to
$74.5$ years (the observed mean age of $70+$ CoMix participants):

$$
\text{mid} = (6.0,\ 13.0,\ 20.0,\ 29.5,\ 42.0,\ 59.5,\ 74.5).
$$

The first `child_bins = 2` groups ($[2,10]$, $[11,15]$) are labelled **child** and the remaining
five **adult**; this two-level block structure indexes the degree-dispersion parameters (§4.3).

### 2.2 Weekly grid

Weeks are Sunday-anchored to `WEEK_ANCHOR = 2021-03-21`. For a date $d$ the week index is
$\lfloor (d - \text{anchor})/7 \rfloor$; the week is labelled by its Wednesday mid-date. A forecast
is organised around a **`WeeklyWindow`** with origin $t_0$ (snapped to its Sunday week), comprising

- `fit_weeks`: the `n_fit = 8` weeks ending at $t_0$,
- `lag_weeks`: the `smax = 4` weeks immediately preceding the fit weeks (renewal history),
- `all_weeks = lag_weeks ++ fit_weeks`: the $T = 12$ contiguous weeks,
- `forecast_weeks`: the horizon target weeks $t_0 + h$ for $h \in$ `horizons` $= \{1,2,3,4\}$.

⚠ **The two data series are assembled over DIFFERENT spans.** Infections and antibody
(`load_window_data`, §2.3) span all $T = 12$ `all_weeks`, because the renewal needs $s_{\max}$
weeks of history $I(t-s)$ that are never themselves predicted. The **contact-degree data**
(`prepare_degree_data`, §2.4) span `win.fit_weeks ++ win.forecast_weeks` of a window built by
`degree_window(origin, h, cfg)`:

$$[\,t_0 - n_{\text{fit}} + 1,\ \dots,\ t_0 + h\,], \qquad T_n = n_{\text{fit}} + h = 9,10,11,12
\ \text{at}\ h = 1..4,$$

**anchored at the origin and extended to the horizon.** What the contact window omits is the
$s_{\max}$ renewal LAG weeks: the Stage-2 likelihood runs $t = s_{\max}+1,\dots,T$ (§6), so their
$C^*$ would never reach an NGM. Stage-1 latents are therefore $5+32T_n = \mathbf{293/325/357/389}$
(NegBin) and $5+81T_n = \mathbf{734/815/896/977}$ (hurdle-Weibull) at $h = 1..4$.

⚠ **Build it with `degree_window(origin, h, cfg)`, never `WeeklyWindow(origin + Day(7h))`.** The
shifted form spans $[t_0+h-n_{\text{fit}}+1,\dots,t_0+h+4]$, which has the right *length* at $h=4$
and the wrong *dates* at every horizon.

The renewal consumes only the **last** $n_{\text{fit}}$ columns of the contact window (contacts $h$
weeks ahead of their fit week), so the leading $h$ columns are fitted but unused — a deliberate,
accepted cost. Consequently the two per-week indices differ: `Cstar_weeks[t − smax + h]` pairs with
infection week $t$. Both `model_transmission` and `fit_window_infection_draws` (10j) recover $h$ as
`length(Cstar_weeks) − n_fit` rather than being passed it, and assert its range. A length check
cannot catch a wrong-*dated* window of the right length, so **`stage2_inputs` asserts
`apd_h.weeks[1:n_fit] == win0.fit_weeks`** — the one place both windows are in scope. Without it a
missed call site would produce a complete, plausible, wrongly-dated forecast.

Because every horizon's contact window is anchored at $t_0-n_{\text{fit}}+1$, the origin sits at
column `n_fit` $= 8$ in *every* Stage-1 chain (10j asserts `t_o_est == cfg.n_fit`).

### 2.3 Infection and antibody series

Age-stratified estimates are taken from the `inc2prev` output `estimates_age_ab.csv`
(`infection_data.jl`):

- **Weekly infections** $I_{a,t}$ and their SDs $s^{I}_{a,t}$. The `name == "infections"` rows give
  a *daily per-capita infection proportion* with mean and SD; the weekly count is the 7-day sum of
  (proportion $\times$ age-group population), and weekly SDs are combined in quadrature across the
  days of the week (a working independence approximation).
- **Weekly antibody prevalence** $A_{a,t} \in [0,1]$, the within-week mean of the `name == "gen_dab"`
  rows. Those rows carry no date column, so their dates are reconstructed from the `t_index` field
  via the `t_index → date` map built from the infection rows.

  Antibody is assembled **twice**: over the $T$ window weeks (`antibody`, used in the Stage-2 fit
  loop) and separately over the $H$ **forecast target weeks** $t_0+h$ (`antibody_fc`, used by the
  forecast NGM — §3.2). The two are kept in distinct fields rather than one widened matrix so that
  `antibody[:, t]` keeps its exact meaning inside the likelihood. `weekly_antibody` fills any week it
  cannot match with $0$, which is indistinguishable from *genuinely zero antibody prevalence* — and on
  a forecast column that reads as full susceptibility and would inflate the forecast without erroring.
  It therefore **warns** on unmatched weeks, loudly for forecast targets. The zero fill is retained as
  the value (a `NaN`/`missing` would propagate into the NGM); the warning is a tripwire for when the
  origin cap is raised past the `gen_dab` series' end, and is silent over the current 63 origins.

Populations $N_a$ and proportions $N_a / \sum_b N_b$ complete the `WindowData` container.

### 2.4 Age-pair contact-degree data

For each window, `prepare_degree_data` assembles per-cell $(t,i,j)$ contact data from the CoMix-UK
participant roster (`part_uk`) and contact table (`contacts_uk.arrow`), reusing the age-pair
binning of `7j_weekly_age_pair`:

- **Participant / contactee binning.** Each participant's reported age interval and each contact's
  estimated age interval $[\text{lo},\text{hi}]$ are mapped to a CIS bin. A single overlapping bin
  is assigned directly; an interval overlapping several bins is resolved by one **seeded,
  population-weighted random draw** (`MersenneTwister(cfg.seed)`), and an interval below the grid's
  minimum age is dropped. This yields a participant bin $i$ and, per contact, a contactee bin $j$.
- **Zeros from the roster.** The denominator of cell $(t,i,j)$ is $n_{t,i}$, the number of sampled
  participant-days with participant bin $i$ in week $t$ (i.e. from the participant *roster*, not the
  contact table). Participant-days with no contact in cell $(t,i,j)$ contribute observed zeros;
  hence the degree distribution is correctly zero-inflated at the roster level.
- **Duration weighting.** Each individually-reported contact receives a weight
  $w = t_{\text{mid}}/d_{\max}$, where $t_{\text{mid}}$ is the midpoint (minutes) of its duration
  bin — $(2.5, 10, 37.5, 150, \text{cap})$ for the five CoMix levels, the open-ended top level
  capped at $d_{\max} = 240$ — and missing durations are treated as the shortest bin. **Group
  ("mass") contacts** carry no recorded duration and instead receive a fixed group weight
  $w_{\text{group}} = 2.5/240$ (a constant now, estimable later); their integer count is unchanged.
- **Setting.** The pipeline supports `:all`, `:home`, and `:nonhome` (filtering on `cnt_home`); the
  8j notebook uses `:all`.

Each cell $(t,i,j)$ therefore yields four objects used downstream:

| Object | Symbol | Used by |
|--------|--------|---------|
| Integer contact-count distribution incl. zeros | $D^{\text{cnt}}_{t,i,j}$ | NegBin path |
| Vector of positive duration-weighted degrees | $\{W\}_{t,i,j}$ | Hurdle-Weibull path |
| Empirical zero probability $n_{\text{zero}}/n_{t,i}$ | $p^0_{t,i,j}$ | initialisation / diagnostics |
| Roster count | $n_{t,i,j}$ | denominators |

---

## 3. Transmission core: renewal equation and NGM

### 3.1 Generation interval (estimated)

The weekly generation-interval PMF $w = (w_1,\dots,w_{s_{\max}})$, $s_{\max} = 4$, is a discretised
log-normal whose **two log-parameters are estimated** as part of the Stage-2 transmission block.
The construction follows Munday et al. 2023 Eq 2 (`inst/pcbi.1011453.pdf`, p. 6):

$$
w_s \;=\; \frac{F(s;\,w_\mu, w_\sigma) - F(s-1;\,w_\mu, w_\sigma)}{F(s_{\max};\,w_\mu, w_\sigma)},
\qquad s = 1,\dots,s_{\max},
$$

where $F$ is the CDF of $\mathrm{LogNormal}(\text{meanlog}=w_\mu,\ \text{sdlog}=\sqrt{w_\sigma})$ on a
**weekly** time axis. Because $F(0)=0$, the denominator equals the numerator sum exactly, so $w$ is a
proper right-truncated-and-renormalised PMF on $\{1,\dots,4\}$ with no further normalisation step.

**$w_\sigma$ is the log-VARIANCE, not the log-SD.** Munday's Table 1 names the pair "log-mean and
log-variance" and builds its prior centre as $w_{\sigma,0} = \log\!\big((\sigma_w/\mu_w)^2+1\big)$,
which *is* $\sigma^2_{\log}$; the paper's p. 8 prose calling it a "log-standard-deviation" is
inconsistent with its own construction. We take the Table-1 reading, i.e. $\text{sdlog}=\sqrt{w_\sigma}$.
The practical consequence is that the **prior mean reproduces the previous fixed GI exactly** —
$\mu_w = \sigma_w = 5$ days — so the estimated-GI model **nests** the fixed-GI model.

**Priors** (Munday p. 8: normal, with an SD of $20\%$ of the prior mean), centred on the moment-matched
log-parameters of `cfg.gen_mean_days` / `cfg.gen_sd_days` $= 5/5$ days $\Rightarrow$
$(w_{\mu,0}, w_{\sigma,0}) = (-0.6830,\ \log 2 = 0.6931)$:

$$
w_\mu \sim \mathcal N\!\big(w_{\mu,0},\ (|w_{\mu,0}|\cdot r)^2\big),
\qquad
w_\sigma \sim \mathcal N^{+}\!\big(w_{\sigma,0},\ (w_{\sigma,0}\cdot r)^2\big),
\qquad r = \texttt{gen\_prior\_rel\_sd} = 0.2 .
$$

Only $w_\sigma$ is truncated at $0$. The paper prints $T[0,]$ on **both** and a *negative* prior SD
for $w_\mu$ — which is not a valid statement, and truncating $w_\mu$ at $0$ would be wrong anyway: a
5-day generation interval is shorter than a week, so $\text{meanlog} = -0.683 < 0$ is *required*. This
is a deliberate, documented departure from the printed table.

Both latents are soft-clamped inside the model — $w_\mu \in [\log\tfrac1{28}, \log 3]$ and
$w_\sigma \in [0.002, 4]$, with a **transition width $s = 0.05$** (`W_MU_BOUNDS`,
`W_SIGMA_BOUNDS`, `W_GI_SOFT` in `framework.jl`, the single source of truth) — per the codebase idiom
that a clamp is an outer safety bound with the weakly-informative prior living inside it. With this
box the effective prior reproduces $5.000$ d / $5.000$ d and tracks the raw latents to $<0.06\%$
across $\pm3$ prior SDs.

⚠ **The upper $w_\mu$ bound is load-bearing and $\log 3$ is its maximum.** It is the $F(s_{\max})$
guard: $\min F(4)$ over the box is attained exactly at $(\log 3, 4)$ and equals $0.5572$, so the
division above cannot blow up. Raising it collapses that — $\log 4\Rightarrow0.500$,
$\log 6\Rightarrow0.0021$, and $0$ at small $w_\sigma$, i.e. $w/F(4)$ divides by $\approx0$.
$w_\sigma$'s ceiling is held for the same reason. Its floor cannot be usefully widened either, since
a variance is pinned at $0$; the tight $s$ is the only lever that reaches it.

At the prior mean the PMF is $w \approx (0.799,\ 0.158,\ 0.033,\ 0.010)$.

$w$ is a **per-draw** quantity: each Stage-2 posterior draw carries its own $(w_\mu,w_\sigma)$
and hence its own $w$, which the forecast (§8) and the fit-window diagnostic must both use.

### 3.2 Next-generation matrix

For a target week the $A\times A$ NGM is

$$
N_{ab}(t) \;=\; \gamma_{\mathrm{SAR}}\;\cdot\;\underbrace{\text{susc}_a\big(1 + (F-1)\,A_a(t)\big)}_{\text{full\_susceptibility}_a(t)}
\;\cdot\; C^\ast_{ab} \;\cdot\; \text{inf}_b ,
$$

with $\gamma_{\mathrm{SAR}}$ the **per-contact secondary attack rate** — $C^\ast$ is **not**
normalised, so $\gamma_{\mathrm{SAR}}$ carries the NGM level and reproduces the reference cell
$N_{rr}=\text{susc}_r\cdot\text{inf}_r=\gamma_{\mathrm{SAR}}$, $r=\texttt{cfg.ref\_bin}$, and is
comparable across origins — $\text{susc}_a$ the **relative** inherent susceptibility of group $a$
and $\text{inf}_b$ the **relative** infectivity of group $b$, both normalised so the reference bin
$r = 4$ ("25-34") is $1$ ($\text{susc}_r=\text{inf}_r=1$; the other $A-1$ bins estimated), and
$F$ a **leaky** antibody-protection factor scaling susceptibility by the group's antibody
prevalence $A_a(t)$ (at $F=1$ antibodies confer no protection; smaller $F$ gives stronger
protection). $C^\ast_{ab}$ is the per-capita effective contact matrix produced by the NGM builder
(§5.1).

**$F$ is pinned to $1$**, so $\text{full\_susceptibility}_a(t)=\text{susc}_a$ exactly, the $(F-1)$
multiplier is $0$ and $A_a(t)$ does not enter the NGM. It is not a Stage-2 latent — dropping it from
the parameter space rather than leaving it unused keeps it out of the Pathfinder approximation — but
it is still *returned* as the constant $1.0$, so every downstream consumer takes the same shape.
`ngm.jl` is deliberately left general: `build_ngm`/`full_susceptibility` take $F$ and are simply
called with $1$. That is NaN-safe only because `weekly_antibody` zero-fills unmatched weeks rather
than NaN-filling them (§2.3).

**Antibody at the target week.** Inside the Stage-2 *fit* loop $A_a(t)$ is the week-$t$ antibody of
the $t_0$-anchored infection window. In the **forecast** (§8) the frozen NGM instead uses
$A_a(t_0+h)$ — the antibody prevalence at the horizon target week — mirroring the availability
assumption already made for the contact data (whose Stage-1 window ends at $t_0+h$). The fit loop is
deliberately **not** shifted: the infection outcomes it is evaluated against exist only up to $t_0$,
so an $h$-shifted antibody there would pair future antibody with present infections for no gain. One
consequence is that the likelihood pairs $t{+}h$ contacts with $t$ antibody, and only the forecast
step has both at $t_0+h$. Note also that antibody (`gen_dab`) comes from the **same** inc2prev/CIS
pipeline as the infection targets, whereas CoMix is an independent survey — so assuming $A(t_0+h)$ is
known is a stronger assumption than assuming contacts at $t_0+h$ are.

### 3.3 Renewal recursion

New infections propagate by the weekly renewal equation

$$
I_a(t) \;=\; \sum_{b} N_{ab}(t)\, \sum_{s=1}^{s_{\max}} w_s\, I_b(t-s)
\;=\; \Big[N(t)\, \textstyle\sum_{s} w_s\, I(t-s)\Big]_a ,
$$

i.e. a single NGM applied to the $w$-weighted sum of the $s_{\max}$ lagged infection vectors
(`renewal_next`). Multi-step forecasting is done by `two_stage_forecast` (§8), which iterates this
per pooled draw. `renewal.jl` also carries `forecast_forward`, a deterministic frozen-NGM iterate —
**currently unused**, superseded by `two_stage_forecast`; if revived it must be handed the draw's own
$w$, not the prior centre.

---

## 4. Contact-degree models (Axis 1)

Both families parameterise the **directional mean** $\mu_{i\to j}$ of cell $(i,j)$ through the shared
reciprocity/GP construction of §5; they differ in the observation likelihood and in how the raw
moments $\langle k\rangle$, $\langle k^2\rangle$ and a zero-conditioning factor $g$ are computed for
the NGM.

### 4.1 Unweighted negative binomial (`NegBinAgePair`)

The integer contact counts (including zeros) of cell $(i,j)$ are modelled as
$\mathrm{NegBin}(\mu_{ij}, \phi_{ij})$, parameterised by **mean** $\mu_{ij}$ and a **dispersion**
$\phi_{ij}$ shared within the child/adult block pair $(\beta(i),\beta(j))$ (§4.3), with
$\mathrm{Var} = \mu + \mu^2/\phi$. The log-likelihood sums over the empirical count distribution:

$$
\ell_{ij} = \sum_{k} y_k\,\log \mathrm{NegBin}(k;\mu_{ij},\phi_{ij}),
$$

where $(k,y_k)$ are the distinct degrees and their observed frequencies. Zeros are modelled
directly (this path is not a hurdle). The raw moments and zero factor supplied to the NGM are

$$
\langle k\rangle = \mu,\qquad
\langle k^2\rangle = \mu + \mu^2\!\left(1 + \tfrac1\phi\right),\qquad
g = \frac{1}{1 - P_0},\quad P_0 = \left(\tfrac{\phi}{\phi+\mu}\right)^{\!\phi},
$$

with $1 - P_0$ floored at `_P0_FLOOR` $= 10^{-3}$ for numerical safety. $g$ conditions the
neighbourhood degree on having at least one contact (left-truncating the fitted NegBin).

### 4.2 Duration-weighted hurdle-Weibull (`HurdleWeibullAgePair`)

Here $\mu_{ij}$ denotes the mean of the **positive** duration-weighted degrees. The positive weights
$\{W\}_{ij}$ are modelled as $\mathrm{Weibull}(\kappa_{ij}, \lambda_{ij})$ with a shape
$\kappa_{ij}$ shared within the block pair (§4.3) and scale chosen so the Weibull mean equals
$\mu_{ij}$:

$$
\lambda_{ij} = \frac{\mu_{ij}}{\Gamma(1 + 1/\kappa)},\qquad
\ell_{ij} = \sum_{W \in \{W\}_{ij}} \log \mathrm{Weibull}(W;\kappa,\lambda_{ij}).
$$

The zero part is a genuine **hurdle**, and its probability is **fitted** rather than plugged in
empirically. Each cell carries its own zero probability with a flat prior, and the roster supplies a
Binomial likelihood:

$$
p^0_{ij,t} \sim \mathrm{Beta}(1,1),
\qquad
n^{0}_{t,i,j} \sim \mathrm{Binomial}\big(n_{t,i,j},\ p^0_{ij,t}\big),
$$

where $n_{t,i,j}$ is the roster count of §2.4 and $n^{0}_{t,i,j}$ the number of those
participant-days with no contact in the cell. So the zero part contributes to the Stage-1
likelihood, not only to the moments. (In code the $\log\binom{n}{n^0}$ normaliser is dropped — it
depends only on data, so the posterior is unchanged, and it keeps `lgamma` out of a $49\times T_n$
inner loop that runs on every gradient evaluation.) **No pooling**: the $A^2 \times T_n$ zero
probabilities are independent. That is safe here precisely because each is directly identified by its
own Binomial with a large $n$ — there is no funnel, so a non-centred re-parameterisation is
unnecessary. Cells with $n_{t,i,j} = 0$ contribute no Binomial term (the whole roster row is absent —
there is no trial to observe), and their $p^0$ is prior-only.

With $\mathrm{CV}_W^2 = \Gamma(1+2/\kappa)/\Gamma(1+1/\kappa)^2 - 1$ the zero-included raw moments,
evaluated at the *fitted* $p^0$, are

$$
\langle k\rangle = (1-p^0)\,\mu,\qquad
\langle k^2\rangle = (1-p^0)\,\mu^2\big(1 + \mathrm{CV}_W^2\big),\qquad
g = 1 - p^0 .
$$

Two consequences. The zero factor $g$ is a **latent**, so the neighbourhood builder's
$C^\ast$ inherits its uncertainty. And a cell where the roster exists but no contacts landed
($n>0$, $n^0 = n$) gets $p^0$ posterior just *below* $1$ rather than exactly $1$, so
$\langle k\rangle > 0$ and the $k_1 > 0$ guard in `base_contact` (§5.1) does not fire in that common
case — the guard is still required for $n = 0$ rows, where $p^0$ can reach $1$.

### 4.3 Dispersion / shape parameterisation (block-linear × week)

The dispersion (NegBin $\log\phi$) or shape (Weibull $\log\kappa$) is a **per-child/adult-block**
quantity, estimated separately each window week, with **no per-cell term**. Writing $d$ for either
family's log-parameter, for every ordered cell $(i,j)$ and week $t$:

$$
\log d_{ij,t} \;=\; m_{\ell(i,j),\,t},
\qquad
m_{\ell,t} \sim \mathcal N(0,\sigma_m^2),
\qquad
\ell = 2(\beta(i)-1) + \beta(j) \in \{1,2,3,4\},
$$

with the block-linear code $\ell$ running over contactor block $\times$ contactee block, and
$\sigma_m = 1.0$ for the NegBin dispersion, $0.5$ for the Weibull shape. In code this is
`log_k ~ filldist(Normal(0, 1.0), 4, Tn)` / `log_kappa ~ filldist(Normal(0, 0.5), 4, Tn)` — a
$4\times T_n$ array, two-dimensional so `generated_quantities` can reconstruct it (a 3-D `filldist`
cannot be). The block means are **re-drawn iid each week** — the dispersion is *not* temporally
smoothed, unlike the mean field (§5).

**All 49 ordered cells inside a block share one value each week.** The index $\ell$ is *directional*
(child$\to$adult $\ne$ adult$\to$child) and self-pairs $(i,i)$ are included, so the four blocks
partition the $A^2 = 49$ ordered pairs, not the 28 unordered ones. (Contrast the contact **mean**,
§5, whose reciprocity construction *is* defined on the 28 unordered pairs.)

The block mean is the **whole** term: there is no per-cell random effect and no shrinkage prior over
cells. With 49 ordered cells per week, many of them empty, per-cell dispersion is not identified by
the data. The standing verification is that the **within-block SD of $\log d$ is identically $0$**
(11j `plot_within_block_sd`).

#### Soft-clamping the composed value

The log-parameter is soft-clamped (`_softclamp`, not `clamp` — ReverseDiff-safe), keeping the mode
interior and avoiding Weibull/exponential underflow: $\log\kappa \in [-4.3,5]$, i.e.
$\kappa \in [0.0136,148]$; $\log\phi \in [-4,5]$, i.e. $\phi \in [0.018,148]$.

**The transition width matters.** `_softclamp(x, lo, hi, s = 0.25)` is
$lo + s\,\mathrm{softplus}\big((hi - s\,\mathrm{softplus}((hi-x)/s) - lo)/s\big)$. The unscaled form
(equivalently $s=1$) inherits `_softplus`'s **O(1)** transition width, so the claim that a clamp
"equals $x$ in the interior" holds only when $hi-lo$ is several nats. Measure the **derivative**, not
the width: a $2.7$-nat window has a maximum $d(\text{softclamp})/dx$ of $0.600$ *anywhere* — no
interior at all, i.e. a hard modelling constraint disguised as a numerical guard. The wide clamps
here are unaffected ($\log\kappa$ $0.980\to1.000$, $\mu$ $0.997\to1.000$). Keep $s$ well below
$hi-lo$; $s\to0$ recovers a hard `clamp` with a flat, hard-to-escape exterior, so $0.25$ is
deliberately moderate. Inf-safety is preserved — every intermediate stays finite, $f(-\infty)=lo$ and
$f(+\infty)=hi+s\log(1+e^{-(hi-lo)/s})$.

> **⚠ The $\kappa$ floor is $\approx-4.446$, and it is set by $\Gamma(1+2/\kappa)$ — not by
> $\lambda$.** $\Gamma$ overflows above argument $\approx 171.6$. There are **two** $\Gamma$ calls on
> this path and the *second* is the binding one:
>
> | quantity | $\Gamma$ argument | overflows at | $\log\kappa$ floor |
> |---|---|---|---|
> | scale $\lambda_W = \mu/\Gamma(1+1/\kappa)$ | $1+1/\kappa$ | $\kappa \lesssim 0.00586$ | $-5.14$ |
> | $\mathrm{CV}^2 = \Gamma(1+2/\kappa)/\Gamma(1+1/\kappa)^2$ | $1+2/\kappa$ | $\kappa \lesssim 0.01172$ | $\mathbf{-4.446}$ |
>
> `_weibull_moments` computes **both**, so the tighter floor governs. At $\log\kappa=-5$ the scale is
> still finite ($1.5\times10^{-263}$) but $\mathrm{CV}^2 = \infty/\infty = $ **NaN**, which propagates
> into $\langle k^2\rangle$ and aborts the fit. $-4.3$ leaves $\approx0.15$ in $\log\kappa$ of margin.
> The NegBin $\phi$ clamp has no such constraint (its moments are polynomial in $1/\phi$).

---

## 5. The contact mean: structural reciprocity and spatio-temporal-GP smoothing

The directional mean $\mu_{i\to j}$ that both degree families share is built to satisfy
**total-contact reciprocity exactly** and to be **smooth over the age-pair grid and across weeks**.

**Reciprocity by construction.** Each of the $P = A(A+1)/2 = 28$ *unordered* age pairs $(a\le b)$
carries a single symmetric log-rate $r_{a,b}$; ordered pairs $(i,j)$ and $(j,i)$ share it. The
directional mean is offset by the contactee-group population:

$$
\log \mu_{i\to j} = r_{\min(i,j),\,\max(i,j)} + \log N_j
\;\;\Longrightarrow\;\;
N_i\,\mu_{i\to j} = N_j\,\mu_{j\to i},
$$

so the total number of $i\!\to\! j$ contacts equals that of $j\!\to\! i$ contacts. In code the offset
is taken **relative to bin 1** (`logpop = log.(pop ./ pop[1])`), which leaves reciprocity exact — the
constant shift $\log N_1$ cancels — and keeps the latent level $c$ at $O(1)$.

**Spatial kernel over the age-pair grid.** The 28 log-rates are smoothed by a non-centred GP over
the age-pair coordinates, and the kernel smooths **both** directions of the age-pair plane. The age
pair $(\text{mid}_{p_1},\text{mid}_{p_2})$ is rotated $45°$ into a total-age (along-diagonal)
coordinate and an age-gap (across-diagonal) coordinate,

$$
u_p = \frac{\text{mid}_{p_1}+\text{mid}_{p_2}}{\sqrt 2}, \qquad
v_p = \frac{\text{mid}_{p_1}-\text{mid}_{p_2}}{\sqrt 2},
$$

each carrying its own length-scale — $\rho_{\text{diag}}$ on $u$ and $\rho_{\text{gap}}$ on $v$. Then

$$
m_{3/2}(x) = \left(1 + \sqrt 3\,x\right)\exp\!\left(-\sqrt 3\,x\right), \qquad
K^{\text{age}}_{pq} = m_{3/2}\!\left(\frac{|u_p-u_q|}{\rho_{\text{diag}}}\right)\cdot
                      m_{3/2}\!\left(\frac{|v_p-v_q|}{\rho_{\text{gap}}}\right).
$$

So the kernel is **separable and anisotropic**: $\rho_{\text{diag}}$ smooths along total age and
$\rho_{\text{gap}}$ across the age gap (assortativity), each a 1-D Matérn 3/2. It is unit-diagonal by
construction ($m_{3/2}(0) = 1$ in both factors) and PSD as a product of PSD kernels. Being a product
of two 1-D Matérns it is a *separable process*, not a 2-D Matérn, so
$\rho_{\text{diag}} = \rho_{\text{gap}}$ does **not** recover an isotropic kernel. The $\sqrt 2$
normalisation keeps both $\rho$, `RHO_BOUNDS` and `gp_len_prior` on the age-year scale — so
$\rho = 20$ is an effective age-difference length-scale of $20/\sqrt2 = 14.1$ yr.

**Separable spatio-temporal GP over age-pairs × weeks.** Over the $T_n$ contact weeks the field is
*not* drawn independently each week. Each age-pair carries its own temporally-correlated log-rate,
with the temporal correlation **shared** across all age-pairs — a separable (Kronecker) GP whose
covariance factorises into the spatial kernel above and a temporal **AR(1)** correlation over the
week indices $t=1,\dots,T_n$ (**time direction only** — the spatial kernel is untouched):

$$
K^{\text{time}}_{st} = \phi^{\,|s-t|}, \qquad \phi \in (0,1),
\qquad L_{\text{time}} = \mathrm{chol}(K^{\text{time}} + 10^{-4} I).
$$

An AR(1) correlation matrix *is* the exponential (Matérn 1/2) kernel; being Markov, it stays well
conditioned in the near-pooled limit $\phi\to1$, at the cost of non-differentiable sample paths and
a longer memory at long lag. The matrix-normal below gives every age pair its own temporal
trajectory under one shared amplitude $\eta$, with the pairs correlated across age through $L_A$.
The larger $10^{-4}$ jitter (against $10^{-6}$ spatially) keeps $L_{\text{time}}$ positive-definite
as $\phi\to1$; ⚠ do not reduce it — the Pathfinder call is not `try`/`catch`ed, so a
`PosDefException` aborts the whole fit.

**The $P\times T_n$ structure field is drawn matrix-normal, non-centred, and constrained to sum to
zero over the $P$ age pairs within each week.** Writing $Q\in\mathbb R^{P\times(P-1)}$ for the
constant orthonormal basis of $\mathbf 1^{\perp}$ (Helmert contrasts, `_sum_zero_basis`), so that
$QQ^{\!\top} = M = I - \tfrac{1}{P}\mathbf 1\mathbf 1^{\!\top}$,

$$
A_p = Q^{\!\top} K^{\text{age}} Q,
\qquad L_A = \mathrm{chol}(A_p + 10^{-6} I),
$$
$$
R = \eta\,\big(Q\, L_A\, Z\, L_{\text{time}}^{\!\top}\big),
\qquad Z \sim \mathcal N(0,1)^{(P-1)\times T_n},
\qquad \operatorname{Cov}(\operatorname{vec} R) = \eta^2\,\big(K^{\text{time}}\!\otimes M K^{\text{age}} M\big),
$$

so fixing a week gives the spatial kernel conditioned on $\sum_p R_{p,t}=0$, and fixing an age-pair
gives a temporal GP with shared AR(1) coefficient $\phi$. This is the same GP **conditioned**, not
approximated. The constraint is what separates $\eta$ from $\sigma_c$: without it the field's
per-week mean over the pairs is a second copy of the level $c_t$ below, and the two amplitudes are
confounded. $Z$ loses a row.

$K^{\text{age}}$ has unit diagonal but $M K^{\text{age}} M$ does not, so **$\eta$ is not exactly the
marginal SD** — the field's SD is $\eta\sqrt{\operatorname{diag}(M K^{\text{age}} M)}$, a factor of
$\times0.71$–$\times1.08$ at the `gp_len_prior` mode. That sits well inside `gp_scale_prior`, which
is therefore left as it is. ⚠ Do **not** renormalise $A_p$ by $\operatorname{tr}(A_p)/P$ to
"restore" the unit diagonal: as $\rho\to\infty$ that ratio is dominated by the jitter and the field
degenerates to *white noise* of scale $\eta$, inverting the correct limit (field $\to 0$, $c_t$
carrying everything).

**The overall weekly level** carries its own amplitude $\sigma_c$ **decoupled** from $\eta$, is
**conditioned to sum to zero over the $T_n$ window weeks**, and — since `-lcar1` (2026-08-13) —
**shares the field's AR(1), through a SCALED projection of the same kernel**:

$$
c_t = c + \sigma_c\,(Q_t\, L_c\, z_c)_t,
\qquad Q_t = \texttt{\_sum\_zero\_basis}(T_n),
\qquad z_c \sim \mathcal N(0,1)^{T_n-1},
$$
$$
P_r = Q_t^{\!\top} K_t\, Q_t,
\qquad
L_c = \operatorname{chol}\!\Big(\tfrac{P_r}{\operatorname{tr}(P_r)/(T_n-1)} + 10^{-4} I\Big),
\qquad
\operatorname{Cov}(\sigma_c\,\text{dev}) = \sigma_c^2\, Q_t L_c L_c^{\!\top} Q_t^{\!\top}.
$$

i.e. an AR(1) weekly deviation *conditioned* on summing to zero — exact, not a soft penalty — with
the projected kernel **rescaled to unit mean eigenvalue before the jitter**. The week-$t$ log-rate
field is $r_{p,t} = c_t + R_{p,t}$. The per-week marginal SD is $0.921$–$0.958\,\sigma_c$ over
$T_n = 9..12$ **at every $\phi$** (against $\sigma_c\sqrt{1-1/T_n} = 0.943$–$0.958\,\sigma_c$ for the
`-lc0` iid level it replaces), so `gp_level_scale_prior` needs no rescaling.

⚠ **The rescaling is the design, not a detail.** Without it — the pre-`-lc0` form — the level's
*amplitude* collapses as $\phi\to1$, because $Q_t^{\!\top} J Q_t = 0$ **exactly**, so $P_r\to0$
however well conditioned $K_t$ is and $L_c\to\operatorname{chol}(10^{-4}I)$. Measured at $T_n=12$,
mean per-week SD per unit $\sigma_c$:

| $\phi$ | 0 | 0.75 | 0.95 | 0.99 | 0.9999 |
|---|---|---|---|---|---|
| raw $L_c$ | 0.958 | 0.757 | 0.412 | 0.193 | **0.022** |
| scaled $L_c$ | 0.958 | 0.953 | 0.941 | 0.936 | 0.935 |

That is a difference **at the posterior mode**, not only in the tail: the one NUTS read under
$\mathrm{Beta}(3,3)$ (§11) puts hurdle-Weibull at $\phi\approx0.990$–$0.994$. Scaled, the projected
kernel keeps min eigenvalue $\ge 0.117$ and $\operatorname{cond}(L_c)\le 7.6$ over
$T_n = 9..12 \times \phi\in[0,1)$, and the $\phi\to1$ limit is a proper **Brownian bridge**
($M_t K_t M_t \propto -M_t D M_t$ with $D_{st} = |s-t|$; lag-1 correlation $\to 0.69$).

⚠ **Scale BEFORE the jitter — the order is load-bearing**, and it is exactly what separates this
from the $A_p$ renormalisation forbidden above. That one degenerates because the jitter is added
first: once $\operatorname{tr}(A_p)$ falls below it the normalised matrix is essentially
$\text{jitter}\cdot I$, so the field becomes white noise and the correct limit is inverted. Scaling
first leaves the mean eigenvalue at exactly 1, so $10^{-4}$ is always negligible and no such
inversion is possible.

⚠ It does **not** rescue the field-side arm ($L_{\text{time}}$'s first column still absorbs the
field as $\phi\to1$), which `ar1_phi_prior` restrains alone.

The sum-to-zero projection identifies $c$ against the time-mean of the deviation, which would
otherwise be two parameterisations of one quantity. ⚠ **It applies to the LEVEL only** — the
structure field's per-pair mean over weeks duplicates nothing, so constraining that too would force
every pair's structure to average to zero across the window: a model restriction rather than a
reparameterisation. The field keeps the full $L_{\text{time}}$.

The level carries no temporal kernel, so **$\phi$ reaches the likelihood only through
$L_{\text{time}}$**, i.e. only as the per-age-pair temporal correlation. ⚠ As $\phi\to1$,
$L_{\text{time}}$'s first column grows while its last shrinks until most of $Z$ stops reaching the
likelihood; that flat subspace is restrained by $\phi$'s prior alone (below).

The intercept is anchored at the grand mean
$c_0 = \overline{\log(\text{emp mean})_{ij} - \log N_j}$ (so $c \sim \mathcal N(c_0,3^2)$).
$\phi\to 0$ recovers independent weeks; $\phi\to 1$ collapses the field to one pooled matrix.
Numerically, $\rho_{\text{diag}}$ and $\rho_{\text{gap}}$ are clamped to $[0.5,500]$ (`RHO_BOUNDS`,
the single source of truth — **inert** under the current prior, which puts its floor $10.5\sigma$
below and its ceiling $9.2\sigma$ above; the prior restrains the length-scale, the clamp only stops a
stray optimiser step from overflowing `exp`), $\eta$ and $\sigma_c$ to $[e^{-3}, e^{2}]$, and the
per-cell exponent $r_{p,t} + \log N_j$ to $[-8,6]$ (so $\mu \in [3\times10^{-4}, 403]$); the modes
stay interior so reciprocity is not distorted.

⚠ **$\phi$ carries no clamp at all** — it is $\mathrm{Beta}$-distributed on $(0,1)$ by construction
and $\phi^k$ cannot overflow, so `RHO_TIME_BOUNDS` and the temporal soft-clamp are dead on this path
(the constant is retained only so the read-only mirrors can replay archived Matérn-temporal chains).
**On this path the prior is the entire restraint on $\phi$** (§6, §11).

The kernels $L_A, L_{\text{time}}$, the sum-to-zero bases $Q, Q_t$, the spatial length-scales
$\rho_{\text{diag}}, \rho_{\text{gap}}$, the AR(1) coefficient $\phi$ and the scales $\eta, \sigma_c$
are all **shared across weeks**; the per-week variation of the field is **temporally correlated**
(through $L_{\text{time}}$) rather than an independent draw per week (§6).

### 5.1 NGM builder (Axis 2)

The per-capita effective contact $C^\ast_{ab}$ is the single line that differs between builders
(`base_contact`), and is used **directly** in the NGM — there is no post-hoc reciprocity
symmetrisation:

$$
\textbf{MeanNGM:}\quad C^\ast_{ab} = \langle k\rangle_{ab},
\qquad
\textbf{NeighbourhoodDegreeNGM:}\quad C^\ast_{ab} = \frac{\langle k^2\rangle_{ab}}{\langle k\rangle_{ab}}\, g_{ab}
$$

(the size-biased / *excess* degree among non-zero contacts; returned as $0$ when
$\langle k\rangle = 0$ — keep that guard, since per-week Weibull cells can be fully empty and the raw
$k_2/k_1$ is then $0/0$). Note $\langle k^2\rangle/\langle k\rangle = m(1+\mathrm{CV}^2)$, so the
neighbourhood builder up-weights high-variance cells. Reciprocity is carried entirely by the
contact-mean estimation (§5): the **Mean NGM** therefore inherits exact total-contact reciprocity
from $\mu$ ($N_a\,C^\ast_{ab} = N_b\,C^\ast_{ba}$), whereas the **Neighbourhood NGM**'s size-biased
$C^\ast$ is generally *not* reciprocal (it depends on the block-dependent dispersion and zero
factor) and is left un-symmetrised by design — this overrides the docx's total-contact balance.
$C^\ast$ is independent of the transmission parameters, so it is computed once per week and reused
across the renewal recursion.

The two **baseline** builders (`inst/6_null_interaction_model.md`) are:

$$
\textbf{DiagonalMeanNGM:}\quad C^\ast = \mathrm{diag}\big(\langle k\rangle_{11},\dots,\langle k\rangle_{AA}\big),
\qquad
\textbf{NullNGM:}\quad C^\ast_{ab} = \bar c \ \ \forall a,b .
$$

`DiagonalMeanNGM` is a **matrix-level** functional (it needs the cell index), so unlike the other
builders it overrides `contact_star` rather than `base_contact`; it returns a *dense* matrix
because `fit_stage2_pooled` stores $C^\ast$ into a `Vector{Matrix{Float64}}`.

$\bar c$ (`null_contact_level`) is the roster-weighted mean number of **unweighted** contacts a
participant-day reports over the origin window's 8 focal fit weeks, divided by $A$:

$$
\bar c = \frac{1}{A}\cdot
\frac{\sum_{t\in\text{fit}}\sum_i n_{t,i}\sum_j \overline{k}^{\,\text{emp}}_{t,i,j}}
     {\sum_{t\in\text{fit}}\sum_i n_{t,i}} ,
$$

so each row of the uniform $C^\ast$ sums to the average *total* daily contacts and
$\gamma_{\mathrm{SAR}}$ stays on the same per-contact scale as the mean-NGM models. **The null
model's forecasts are invariant to this convention**: $N_{ab}=\gamma_{\mathrm{SAR}}\cdot
\text{fs}_a\cdot\bar c\cdot\mathrm{inf}_b$, so $\gamma_{\mathrm{SAR}}$ and $\bar c$ enter only as a
product and $\gamma_{\mathrm{SAR}}$ is freely estimated — the choice only fixes what
$\gamma_{\mathrm{SAR}}$ *means*. (Verified: scaling $\bar c$ by $10$ moves the posterior median
$\gamma_{\mathrm{SAR}}$ by $\times0.105$ — the residual $5\%$ is the log-normal prior's pull — and
leaves the implied $R$ unchanged to $0.3\%$.) $\bar c$ is computed from the origin's 8 focal weeks
and reused for every horizon, which is what "the used average number should be fixed while
forecasting" requires: every horizon window contains all 8 of them as its leading columns
(`stage2_inputs` asserts exactly that), so $\bar c$ is identical across horizons — verified, spread
$0.0$.

---

## 6. The model (two-stage cut)

### 6.0 Two-stage cut inference (`inst/4_cut_Bayes.md`)

**Stage 1** `model_degree(dm, ds, pop, cfg)` fits the contact-degree GP **alone** — it carries only
the contact block below and returns the per-week raw moments
$(\langle k\rangle_t, \langle k^2\rangle_t, g_t)$, so it is **NGM-independent** (one fit serves both
builders; the builder is applied downstream via `contact_star`). **Stage 2**
`model_transmission(Cstar_weeks, wd, cfg, nb = MeanNGM())` fits the transmission block **alone**,
conditioning on a *fixed* $\{C^\ast_t\}$ built from one Stage-1 posterior draw. The trailing `nb` is
consulted **only** for `fix_infectivity`; the $C^\ast$ functional was already applied upstream. The
generation interval is estimated inside Stage 2 (§3.1) and is *not* an argument.

Stage-1 uncertainty is propagated by a cut Monte Carlo: draw $M=100$ Stage-1 posterior samples; for
each, form $\{C^\ast_t\}$ and run Stage 2 keeping $D=100$ draws; **pool** the $M\times D = 10{,}000$
infection draws as the predictive distribution scored by WIS. There is **no feedback** from the
infection likelihood to the contact GP (the "cut"): $\mu$ is estimated purely from the contact data,
so it does not depend on the NGM builder.

**The model estimates contacts per week, temporally smoothed.** A single **separable
spatio-temporal GP** (§5) governs all $T_n = n_{\text{fit}} + h$ contact weeks
($[t_0-n_{\text{fit}}+1 \dots t_0+h]$; the infection window is still $T = 12$, §2.2) — sharing the
spatial kernel $L_{A}$, the temporal kernel $L_{\text{time}}$, the spatial length-scales
$\rho_{\text{diag}}, \rho_{\text{gap}}$, the AR(1) coefficient $\phi$ and the scales
$\eta, \sigma_c$ — so the weekly log-rate fields are **temporally correlated** rather than
independent draws. Each week yields its own $C^\ast_t$ (through the per-week level $c_t$,
structure-field column $R_{\cdot,t}$, and the per-week block-linear dispersion $m_{\ell,t}$, §4.3),
and the transmission NGM $N(t)$ therefore varies in time through **both** antibody prevalence and
contacts. `constant_contacts = true` recovers the pooled one-$C^\ast$-per-window preliminary;
`model_degree` branches on the flag but returns per-week moments of length $T_n$ either way.

**Sampling statements.**

*Stage 1 — contact block (one separable spatio-temporal GP over the contact weeks $t = 1,\dots,T_n$):*

$$
\begin{aligned}
\log\rho_{\text{diag}},\ \log\rho_{\text{gap}} &\sim \mathcal N(\log 20,\ 0.35^2), &
\phi &\sim \mathrm{Beta}(3,3), &
\log\eta &\sim \mathcal N(0,\ 0.5^2), \\
c &\sim \mathcal N(c_0,\ 3^2), &
\log\sigma_c &\sim \mathcal N(0,\ 0.5^2), &
z_c &\sim \mathcal N(0,1)^{T_n-1}, \\
z &\sim \mathcal N(0,1)^{(P-1)\times T_n}, &
m_{t} &\sim \mathcal N(0,\sigma_d^2)^{4}, &
p^{0}_{t} &\sim \mathrm{Beta}(1,1)^{A^2}
\end{aligned}
$$

$$(\sigma_d = 0.5\ \text{Weibull},\ 1.0\ \text{NegBin};\ A^2 = 49\ \text{cells};\ P = 28\ \text{age pairs}).$$

$z$ loses a row to the spatial sum-to-zero projection and $z_c$ an entry to the temporal one, giving
$5 + 32T_n$ (NegBin) / $5 + 81T_n$ (hurdle-Weibull) unconstrained latents — with
$T_n = n_{\text{fit}} + h$, that is **293/325/357/389** and **734/815/896/977** at $h=1..4$. The
dispersion contributes only the $4\times T_n$ block array $m$ — there is no per-cell dispersion
latent (§4.3). The temporal parameter is the AR(1) coefficient $\phi$ (`phi_time`), *not* a Matérn 3/2 length-scale
$\rho_{\text{time}}$ (`log_rho_time`) — that name is the one in-chain signal distinguishing this
generation from an archived one; everything else needs the cache token (§10).

$p^{0}_{t}$ is declared on the **weighted/Weibull path only** (§4.2); the NegBin path's parameter
space does not contain it, so the two degree models' Stage-1 chains differ in shape.

⚠ **$\phi$'s prior (`ar1_phi_prior` $= \mathrm{Beta}(3,3)$) is not in the cache token** —
`contacts_label` encodes no prior — so changing it does not fork the grid and stale artefacts must be
moved aside by hand. It is symmetric with mode $0.5$ and density $\to 0$ at both boundaries, so it
asserts no temporal pooling; it only makes the boundary progressively expensive to reach
($P(\phi>0.99) = 9.85\times10^{-6}$, against $2.98\times10^{-4}$ under $\mathrm{Beta}(2,2)$ and
$0.0100$ under Uniform). ⚠ **$\mathrm{Beta}(3,3)$ has never been fitted at scale** — re-run the
divergence census (§11) before trusting the weighted path.

The derived quantities are the level $c_t = c + \sigma_c (Q_t L_c z_c)_t$, the structure field
$R = \eta\,(Q\, L_A\, z\, L_{\text{time}}^{\!\top})$ with temporal factor
$K^{\text{time}}_{st} = \phi^{|s-t|}$, the week-$t$ log-rate $r_{p,t} = c_t + R_{p,t}$ (§5), the
contact-degree log-likelihood of §4 injected via `Turing.@addlogprob!`, and
$C^\ast_t = $ `contact_star`$(nb, \langle k\rangle_t, \langle k^2\rangle_t, g_t)$ downstream.

*(Note the name collision: $\phi$ here is the AR(1) temporal coefficient. The NegBin dispersion is
also conventionally written $\phi$ in §4.1; in code they are `phi_time` and `log_k` respectively and
never meet.)*

*Stage 2 — transmission block* (per-contact $\gamma_{\mathrm{SAR}}$ + reference-normalised susc/inf,
non-centred; conditions on the fixed $\{C^\ast_t\}$ of one Stage-1 draw). Susceptibility and
infectivity are **relative** to the reference bin $r=\texttt{cfg.ref\_bin} = 4$ ("25-34"), fixed to
$1$; only the $A-1$ non-reference offsets are estimated (the fixed $1$ is **spliced in at position
$r$**, not prepended), and $\gamma_{\mathrm{SAR}}$ carries the level.

$$
\begin{aligned}
\log\gamma_{\mathrm{SAR}} &\sim \mathcal N(\log 0.1,\, 1.8^2), &&
\gamma_{\mathrm{SAR}} = \exp(\operatorname{softclamp}(\log\gamma_{\mathrm{SAR}},\log 0.001,\log 10)),\\
\sigma_s &\sim \mathcal N^+(0, 0.25^2), & z_s &\sim \mathcal N(0,1)^{A-1}, &
\text{susc} &= \operatorname{splice}_r\big(1,\ \exp(\operatorname{softclamp}(\sigma_s\,z_s, \log 0.05, \log 20))\big),\\
\sigma_i &\sim \mathcal N^+(0, 0.25^2), & z_i &\sim \mathcal N(0,1)^{A-1}, &
\text{inf} &= \operatorname{splice}_r\big(1,\ \exp(\operatorname{softclamp}(\sigma_i\,z_i, \log 0.05, \log 20))\big),\\
F &\equiv 1 \ \text{(pinned, §3.2)}, & \sigma_{\text{inf}} &\sim \mathcal N^+(0.05, 0.025^2), & & \\
w_\mu &\sim \mathcal N(-0.6830,\ 0.1366^2), & w_\sigma &\sim \mathcal N^{+}(0.6931,\ 0.1386^2), &
w &= \text{Eq. §3.1}\big(w_\mu, w_\sigma\big).
\end{aligned}
$$

**The reference bin $r$ is a gauge**: the NGM likelihood is invariant to it (rescaling all
$\text{susc}$ by $c$ and $\gamma_{\mathrm{SAR}}$ by $1/c$ leaves $N$ unchanged), so it acts *only*
through the priors — which bin is pinned to 1 vs. carries the log-offset, and what
$\gamma_{\mathrm{SAR}}=N_{rr}$ anchors to. 25-34 is a large, well-mixed, well-sampled adult group
(Munday/Davies convention); anchoring on children would use an extreme, poorly-identified,
antibody-sparse bin.

The prior controls the typical age spread and the **soft-clamp** is a looser safety bound: the offset
scale $\sigma_{s,i}\sim\mathcal N^+(0,0.25^2)$ is a mode-at-0 half-normal — the conventional
weakly-informative scale, which lets the age profile **shrink to no-variation**
($\text{susc},\text{inf}\to1$) when the data are silent rather than *asserting* age spread — and each
non-reference log-offset $\sigma\,z$ is soft-clamped to $[\log 0.05,\log 20]\approx[-3,3]$,
**hard-bounding** $\text{susc},\text{inf}\in[0.05,20]$: wide enough that realistic profiles never
touch it (the prior's $\pm2$ SD sits at $\pm1$, the clamp at $\sim\pm6$ SD), but still capping a stray
Pathfinder draw's supercritical NGM at the source. The $A-1$ non-reference offsets are **independent
per age bin** ($z\sim\mathcal N(0,1)^{A-1}$ iid) — **no cross-bin smoothing**. The
**generation-interval** latents $(w_\mu, w_\sigma)$ live here too (§3.1); they enter only the
renewal, so the cut keeps them clear of the contact GP. Stage 2 returns the generated quantities
$(\text{susc}, \text{inf}, F, \gamma_{\mathrm{SAR}}, \sigma_{\text{inf}}, w_\mu, w_\sigma)$ —
$\gamma_{\mathrm{SAR}}$, susc/inf and the GI parameters **post-clamp**, so downstream reconstructions
reproduce the fit exactly.

**Fixed infectivity for the no-interaction model.** When `fix_infectivity(nb)` is `true` — only for
`DiagonalMeanNGM` — the $\sigma_i$/$z_i$/$\text{inf}$ block is **not sampled at all** and
$\text{inf}\equiv \mathbf 1$. The reason is exact non-identifiability, not preference: with $C^\ast$
diagonal the NGM is diagonal, so

$$N_{aa} = \gamma_{\mathrm{SAR}}\cdot\text{susc}_a\big(1+(F-1)A_a(t)\big)\cdot C^\ast_{aa}\cdot\text{inf}_a ,$$

and $\text{susc}_a$ and $\text{inf}_a$ enter **only** through their product. Pinning
$\text{inf}\equiv1$ puts the whole age profile in $\text{susc}$. (Leaving $\sigma_i,z_i$ in the
program as unused latents would leave them prior-driven and pollute the Pathfinder approximation,
so they are dropped from the parameter space rather than merely ignored — the same treatment, and
the same reason, as $F$.) This is *distinct* from, and additional to, the reference-bin
normalisation $\text{susc}_r=\text{inf}_r=1$ that all variants share.

*Infection likelihood*, over the fit weeks $t = s_{\max}+1,\dots,T$ (the first $s_{\max}$ weeks serve
only as renewal history). For each age $a$,

$$
\hat I_a(t) = \big[N(t)\, \textstyle\sum_{s} w_s\, I(t-s)\big]_a,\qquad
I_{a,t} \sim \mathcal N\!\Big(\hat I_a(t),\ \sigma_{a,t}^2\Big),\quad
\sigma_{a,t}^2 = (\sigma_{\text{inf}}\, I_{a,t})^2 + (s^{I}_{a,t})^2,
$$

i.e. the observation SD combines a multiplicative process-noise term $\sigma_{\text{inf}} I_{a,t}$
in quadrature with the inc2prev estimate SD $s^{I}_{a,t}$. The $\{C^\ast_t\}$ here are a **fixed**
input (one Stage-1 draw run through `contact_star`), not sampled, and week $t$ reads
`Cstar_weeks[t − smax + h]` (§2.2).

**The two stages' log-likelihoods.** Under the cut the contact and infection terms live in
**separate** models:

$$
\log L_{\text{Stage 1}} = \underbrace{\sum_{t=1}^{T_n}\ \sum_{i,j=1}^{A} \ell^{(t)}_{ij}}_{\text{contact degree (§4)}},
\qquad
\log L_{\text{Stage 2}} = \underbrace{\sum_{t=s_{\max}+1}^{T}\ \sum_{a=1}^{A}
\log \mathcal N\!\big(I_{a,t}\,;\ \hat I_a(t),\ \sigma_{a,t}^2\big)}_{\text{infection renewal}},
$$

where $\ell^{(t)}_{ij}$ is the §4.1 NegBin (over the count histogram) or §4.2 Weibull-hurdle cell
log-likelihood, evaluated at the week-$t$ contact mean $\mu_{ij,t}$ and its block dispersion
$d_{ij,t}$ (§4.3). On the hurdle path $\ell^{(t)}_{ij}$ has **two** terms — the Weibull over the
positive duration-weighted degrees *plus* the Binomial zero term, omitted where $n_{t,i,j}=0$.

Every log-scale latent that feeds an exponential ($\log\rho, \log\eta, \log\sigma_c, \log\kappa,
\log\phi$, $\log\gamma_{\mathrm{SAR}}$, and the per-cell rate) is soft-clamped inside the model body
so that aggressive optimiser/Pathfinder steps cannot over- or underflow; the clamps are wide enough
that the posterior mode is interior and gradients are unaffected.

**Per-horizon window offset.** Although both blocks index weeks by $t$, they do not span the same
calendar weeks. In forecasting (§8) both stages are re-fit once **per horizon** $h$: Stage 1's
contact window runs $[t_0-n_{\text{fit}}+1 \dots t_0+h]$ (cached as
`8j_s1_<degree>_<contacts>_<origin>_h<h>.jld2`, NGM-independent), and Stage 2 conditions on that
stage's $\{C^\ast_t\}$ against the infection/renewal window anchored at $t_0$ (`wd` fixed; pooled
draws cached as `8j_s2_<degree>_<ngm>_<contacts>_<origin>_h<h>.jld2`). So for every horizon the
contact term is fit over weeks offset $h$ **ahead** of the infection term — the age-pair degree
distribution is observed **at** the target week $t_0+h$ — whereas infections and the fit-loop
antibody are anchored at $t_0$. The forecast NGM additionally takes its antibody at $t_0+h$
(§3.2, §8).

---

## 7. Inference (two-stage cut)

For one $(dm, nb, \text{origin}, h)$:

1. **Stage 1** — `fit_stage1(dm, ds, pop, cfg)` fits the contact GP by **NUTS**
   (`cfg.stage1_use_nuts`, the default), **initialised from the Pathfinder mean**: Pathfinder runs
   first regardless, and NUTS starts from its `fit_distribution` mean in the *unconstrained* space,
   so the cost is **additive**, not a replacement. Set `stage1_use_nuts = false` for the
   Pathfinder-only generation. NUTS is configured explicitly — `cfg.stage1_nuts_adapts = 1000`,
   `_draws = 2000`, `_target_accept = 0.95`, `_max_depth = 10` — because the convenience constructor
   `NUTS()` derives `n_adapts = min(1000, n_sample ÷ 2)`, i.e. far too few warmup iterations to adapt
   a step size and diagonal metric in $293$–$389$ / $734$–$977$ dimensions. **One chain per fit**, so
   there is no $\hat R$; health is reported by `_nuts_diagnostics` (divergence count, fraction of
   transitions saturating `max_depth`, minimum ESS) and, split at the half-way point, by 12j's
   split-$\hat R$. Neither the draw count nor `target_accept` nor either AD backend nor any
   initialisation setting is in the cache token, so each is recorded *inside* every `8j_s1_*`
   artefact (`nuts_draws`, `nuts_adapts`, `target_accept`, `ad_backend`, `phi_init_scale`,
   `phi_pf_max`, `phi_pf_override`) and audited by `tmp/check_grid.jl`.
   `stage1_moment_draws` then takes $M = $ `cfg.n_stage1_post` $= 100$ posterior draws' raw moments
   (deterministic even-grid subsample).

   **Initialisation.** `_stage1_init` shrinks the non-centred blocks: `z`/`z_c` start at
   $\mathcal N(0, \texttt{stage1\_z\_init\_scale}^2)$ with `stage1_z_init_scale = 0.1`, and
   `phi_time` at $\mathrm{logistic}(\mathcal N(0, \texttt{stage1\_phi\_init\_scale}^2))\approx 0.5$
   on the **logit** scale. ⚠ The units differ: `z` is identity-linked, so $0.1$ is a shrunken start
   near its prior mean; $\phi$ is logit-linked, so $0.1$ unconstrained is the prior *median*, not a
   small $\phi$. On the NUTS path `_pf_mean_init` then hands NUTS the Pathfinder mean, **except for
   $\phi$ when it comes back at or above `cfg.stage1_phi_pf_max = 0.9999`**, where it is replaced by
   the same prior-median draw and nothing else in the mean is touched. $\phi$ is logit-linked and
   $\mathrm{logit}(1.0) = \infty$ makes the logjoint non-finite, which fails the cell outright; the
   guard converts that hard failure, and a family of extreme starts, into a defined reproducible one.
   It fired on 3 of 504 cells of the Pathfinder grid, all weighted-hurdle-Weibull at $h=1$. The
   replacement draw is taken **inside** the branch, so a non-firing fit consumes no extra RNG and is
   bit-identical to the ungated code.

2. **Stage 2** — `fit_stage2_pooled(nb, moment_draws, wd, cfg)` forms $\{C^\ast_t\}$ for each Stage-1
   draw (via `contact_star`) and **Pathfinder**-fits `model_transmission` conditioning on it, keeping
   $D = $ `cfg.n_stage2_draws` $= 100$ draws. The $M\times D = 10{,}000$ pooled draws
   $(\gamma_{\mathrm{SAR}}, \text{susc}, \text{inf}, F, \sigma_{\text{inf}}, w_\mu, w_\sigma,
   \text{post\_index}, \{C^\ast_{\text{end}}\})$ are the infection predictive. **Stage 2 has no NUTS
   path at all** — 100 cheap fits per Stage-1 draw is the point of the cut, not an omission.

**Null-model bypass.** `stage2_inputs(dm, …)` is the single fork between the two paths. When
`needs_stage1(dm)` is `false` (i.e. `NoContactDegree`) it skips `build_degree_stats`,
`fit_or_load_stage1` and `stage1_moment_draws` altogether and returns **one** constant-$C^\ast$
"draw" from `null_moment_draws(\bar c, A, T_n)`, taking the full $M\times D = 10{,}000$ samples from
that single Stage-2 fit instead of $D$ from each of $M$. $M=1$ is deliberate: the null model has no
contact-degree uncertainty to propagate, so repeating $100$ identical Pathfinder fits would inject
only fit-to-fit approximation noise, at $100\times$ the cost. The pooled draw count — and hence
comparability of WIS — is preserved. `prefit_stage1!` drops such degree models up front,
so **no `8j_s1_no-contact_*` file is ever written**.

**Seeding and concurrency.** `cfg.seed = 1236`. Each Stage-1 fit is handed a fresh
`Xoshiro(cfg.seed)`; Stage-2 draw $m$ at horizon $h$ uses `Xoshiro(cfg.seed + 1000h + m)`. Fits are
mutually independent: `prefit_stage1!` fans the Stage-1 chains out over Julia threads under a
semaphore (BLAS pinned to one thread, and **one warm fit per *degree-model type*** run serially
before the fan-out — each `typeof(dm)` is a distinct `model_degree` signature and therefore a
distinct AD-rule derivation), then `prefit_stage2!` runs each Stage-2 cell's 100 per-draw fits under
the same concurrency cap. Origins are processed **sequentially** with a full `GC.gc()` between them:
the per-origin working set is garbage once the fan-out returns, but Julia's heuristic will not
collect while the heap looks healthy, and the accumulation OOM-killed a 63-origin run at origin 16
with nothing in its log. Artefacts cache to `dt_intermediate/8j_s1_*.jld2` and `8j_s2_*.jld2`;
cached files are skipped ⟹ resumable, and every write goes through `_atomic_jldsave` (temp file then
rename) so a killed process cannot leave a truncated artefact that the `isfile` skip counts as
complete.

`src/8j_run_grid.jl` is the headless driver: it fits a bounded batch of origins and exits `0` (more
to do) / `10` (slice complete) so a supervisor can restart it in a fresh process, which is the only
thing that resets the per-origin memory climb. It also partitions the grid by origin
(`ORIGIN_STRIDE`/`ORIGIN_OFFSET`, round-robin) so a Slurm array can run disjoint slices, and
`DRY_RUN=1` resolves and prints the token, slice and batch then exits `10` without fitting. The
cluster layer — image, array scripts, in-container supervisor — is `hpc/`; see `hpc/README.md`.

### 7.1 Automatic differentiation

Both stages' gradients go through `ADTypes`/`DifferentiationInterface`, resolved by
`_resolve_adtype` in `framework.jl`, and each stage takes `:mooncake` | `:reversediff` |
`:forwarddiff`. **The two stages use DIFFERENT backends**, and this is a measurement rather than an
oversight:

| | field | default | resolver |
|---|---|---|---|
| Stage 1 (`model_degree`, Pathfinder **and** NUTS) | `cfg.ad_backend` | `:mooncake` | `ad_type(cfg)` |
| Stage 2 (`model_transmission`, Pathfinder) | `cfg.stage2_ad_backend` | `:reversediff` | `stage2_ad_type(cfg)` |

Mooncake is far faster **per gradient** — measured on `model_transmission` (18 dims), 30 551 vs
1 712 gradients/s, a $17.9\times$ advantage that reproduces exactly, and $9$–$11\times$ on the
Stage-1 models — and its rule *is* cached across constructions. But the two stages are in opposite
regimes:

- **Stage 1** is *one* fit of a 293–389 / 734–977-dimension model per (degree × origin × horizon),
  needing $O(10^5)$ gradients. The gradient advantage dominates a one-off `build_rrule` (66 s NegBin
  / 14 s hurdle-Weibull), which `prefit_stage1!` warms once per model *type* before its fan-out.
  A sysimage does **not** remove that warm-up: `build_rrule` keys on the concrete `DynamicPPL.Model`
  type, which does not exist until `forecast_utils.jl` is included at runtime.
- **Stage 2** is *100 independent* fits of that tiny model per cell, so per-fit **setup** dominates
  and gradient throughput is irrelevant: one `pathfinder()` fit costs **10.64 s under Mooncake
  against 0.30 s under ReverseDiff**, and the 100-fit fan-out recovers none of it ($1.09\times$
  speed-up at `max_concurrent = 9` vs ReverseDiff's $2.31\times$ — Mooncake's per-fit setup
  serialises). End to end that is 16.5 min vs 0.2 min per pooled cell ⇒ **17 days vs 5.0 h** over the
  1512-cell grid.

Gradients agree to $\le 4\times10^{-14}$ relative across backends, so this is a pure numerical means
and **not** a cache-token component; each artefact records its own backend under `ad_backend`
instead. Two consequences worth stating:

- ReverseDiff and ForwardDiff are **tracked-number** backends, which is why `model_degree` promotes
  `ETp = promote_type(...)` before allocating its `K1`/`K2`/`G` buffers — miss a term and that
  latent's tape is silently cut. **Mooncake is source-to-source and substitutes no element type**, so
  under it `ETp` collapses to `Float64` and the promotion is an inert compile-time constant. It is
  still load-bearing for the ReverseDiff path and must not be removed.
- `NegBin`'s struct fields must stay **parametric**. With abstract `::Real` fields Mooncake ran the
  Stage-1 NegBin model at 2.7 grad/s against ReverseDiff's 41.8 — a $15\times$ regression —
  while the concrete-typed hurdle-Weibull path was already $9\times$ faster.

---

## 8. Forecasting: the contact-updated pooled iterate

The notebook forecasts with `two_stage_forecast`, the *contact-updated* iterate over the **pooled**
draws. For a baseline origin $t_0$, the infection series is **frozen at $t_0$**, while the
contact/degree window extends: for horizon $h$ the Stage-1 degree window ends at $t_0 + h$ and the
Stage-2 pooled draws for $(dm, nb, t_0, h)$ are reloaded (or fit). Per pooled draw $d$ (from Stage-1
draw $m = $ `post_index[d]`) a fresh NGM is formed from that draw's origin-week $C^\ast$
(`Cstar_end[m]`) with **antibody at the target week $A(t_0+h)$** (§3.2) and its Stage-2 infection
parameters, and a single renewal step is taken with **that draw's own** generation interval
$w^{(d)} = $ §3.1$(w_\mu^{(d)}, w_\sigma^{(d)})$,

$$
\hat I_a(t_0+h) = \Big[N\, \textstyle\sum_{s=1}^{s_{\max}} w^{(d)}_s\, I(t_0+h-s)\Big]_a,
$$

so the renewal-weighted lag sum is computed **inside** the draw loop, not once per horizon as it was
under a fixed $w$. Observation noise $\sigma = \max(\sigma_{\text{inf}}\hat I_a, 10^{-6})$ is added
per draw; non-finite predictions are passed through as $\pm\infty$ rather than being turned into
`NaN`, so a degenerate draw stays visible to the scorer (§9).

⚠ **The renewal lags use observed infections up to $t_0$ plus the MEDIAN over finite draws of the
intervening weeks' predictions.** The plug is deliberately the median, not the mean: the pooled
predictive is heavy-tailed — a few pathological Pathfinder draws give an astronomically supercritical
$N$ — and a mean plug lets one outlier poison the shared iteration for *all* draws into a mass
Inf/NaN blow-up. Individual pathological draws still blow up in their own fan column, which the
finite-robust quantiles downstream handle. Per-draw coherence across horizons is undefined either
way.

This produces an $A \times H \times N$ array of posterior-predictive draws, where
$N = $ `n_stage1_post` $\times$ `n_stage2_draws` $= 10{,}000$ is the pooled predictive — fed
directly to WIS (the draw axis is pooling-agnostic to `scoring.jl`).

---

## 9. Scoring

The primary metric is the **weighted interval score (WIS)** computed by the R package
`scoringutils` (v2) via `RCall`, on quantile-format forecasts (`scoring.jl`):

- Posterior-predictive draws are summarised at the 19 quantile levels
  $\{0.05, 0.10, \dots, 0.95\}$, **rounded to two decimals** in `to_quantile_long` — `scoringutils`
  matches interval endpoints by exact `Float64` equality, and one drifted endpoint makes the whole
  set asymmetric, which silently drops `wis` and most other metrics from every returned frame while
  emitting only an R *warning*. `score_wis` therefore asserts on the **columns**, never on row count,
  and errors with the received levels if any required metric is missing.
- Forecasts are scored on **both the natural and the log scale**, and the **headline metric is the
  log-scale WIS aggregated by horizon** across all origins. WIS is reported with its
  over-prediction, under-prediction and dispersion components, bias, and 50%/90% interval coverage,
  aggregated by model, by model $\times$ horizon, by model $\times$ origin, by model $\times$ origin
  $\times$ horizon, and by model $\times$ horizon $\times$ age (`res/8j_scores_by_model*.csv`).
- ⚠ **The two scales are scored from different objects.** The Gaussian observation fan of
  `two_stage_forecast` can push low quantiles below zero, which the natural scale must see and
  penalise but `log_shift` cannot represent (scoringutils v2 *errors* on it). Natural scale = the raw
  quantiles; log scale = a copy **truncated at 0**, with the truncation counted, returned and
  `@warn`ed. Verified identical to `transform_forecasts(fun = log_shift, offset = 1)` when no
  quantile is negative.
- ⚠ **Non-finite predictions drop the whole forecast UNIT**, never individual quantiles — removing
  part of a fan would leave an asymmetric interval set and break WIS itself. The count and the
  percentage of units dropped per model are returned and warned, because "this model produced an
  unusable forecast for N% of its units" is a more important result than the WIS of the remainder.
- A native sample **CRPS** (energy form) provides a cheap cross-check.

The relative-skill reference (`REF_MODEL`) for relative WIS is the **no-interaction** model
`unweighted-negbin|mean-diagonal`.

---

## 10. The 8j experiment

The notebook (`8j_preliminary_forecast.ipynb`) runs the full grid.

**Configuration** (`FrameworkConfig`, with the notebook setting `constant_contacts = false`):
$d_{\max}=240$, $w_{\text{group}}=2.5/240$, $s_{\max}=4$, `n_fit` $=8$, `horizons` $=1{:}4$,
seed $=1236$, GI prior centre `gen_mean_days`/`gen_sd_days` $=5/5$ days with
`gen_prior_rel_sd` $=0.2$ (§3.1 — these set the *prior* on the estimated $w_\mu,w_\sigma$, not a
fixed $w$), `child_bins` $=2$, quantiles $0.05{:}0.05{:}0.95$, cut sizes `n_stage1_post` $=100$ /
`n_stage2_draws` $=100$ (⟹ 10 000 pooled), `stage1_use_nuts` $=$ **`true`**, `ad_backend`
$=$ `:mooncake` / `stage2_ad_backend` $=$ `:reversediff` (§7.1), and the priors

| latent | prior | field |
|---|---|---|
| $\log\rho_{\text{diag}},\log\rho_{\text{gap}}$ | $\mathcal N(\log20,0.35^2)$ | `gp_len_prior` (shared) |
| $\log\eta$ | $\mathcal N(0,0.5^2)$ | `gp_scale_prior` |
| $\phi$ | $\mathrm{Beta}(3,3)$ | `ar1_phi_prior` (dimensionless) |
| $\log\sigma_c$ | $\mathcal N(0,0.5^2)$ | `gp_level_scale_prior` |
| $\log\gamma_{\mathrm{SAR}}$ | $\mathcal N(\log0.1,1.8^2)$, softclamp $[\log0.001,\log10]$ | `gamma_sar_prior` |
| $\sigma_s,\sigma_i$ | $\mathcal N^+(0,0.25^2)$ | `susc_inf_sd_prior` |

plus **block-linear per-week dispersion** (a $4\times T_n$ array of block means, no per-cell term,
§4.3), **fitted hurdle zero probability** $p^0 \sim \mathrm{Beta}(1,1)$ per cell × week on the
weighted path (§4.2), and **per-week temporally-smoothed contact estimation** (one separable
spatio-temporal age-pair GP across `[t₀−n_fit+1 … t₀+h]`).

**Cache token.** `contacts_label(cfg)` is the short **`"temporal-w8h-lcar1"`**, plus `-nuts` when
`stage1_use_nuts` is set. It names only the *current* generation's distinguishing changes — `-w8h`
the origin-anchored $n_{\text{fit}}+h$ contact window (§2.2) and `-lcar1` the AR(1)-smoothed,
scaled weekly level (§5). ⚠ `-lcar1` changes **no parameter name and no dimension** — it reverses
`-lc0`, whose tokens are themselves current-style — so `reconstruct_mu_draws` forks on the token
**three ways**: `-lc0` ⇒ iid, else legacy ⇒ raw $L_c$, else ⇒ scaled $L_c$.
Every **retained** generation on disk is named by a literal in `framework.jl`, because no `cfg`
reproduces it:

| constant | generation | where |
|---|---|---|
| `CONTACTS_TOKEN_AR1` = `"temporal-gsar-cut-sc-p0-gi-s0-m32-t0-ar1"` | the **complete 504 s1 / 1512 s2 Pathfinder grid** — currently the only complete grid on disk | needs `CONTACTS_SAVE_DIR_AR1` = `dt_intermediate_ar1/` |
| `CONTACTS_TOKEN_PF` = `"temporal-gsar-cut-sc-p0-gi"` | the pre-`-s0` Pathfinder generation | `dt_intermediate/` |
| `CONTACTS_TOKEN_HD` = `"temporal-gsar-cut-sc-hd-p0-gi"` | an older Stage-1 generation, retained as the non-zero comparison line in `plot_within_block_sd` | needs `CONTACTS_SAVE_DIR_HD` = `dt_intermediate_hierarchical/` |

⚠ A token needs its **directory** as well: passing `CONTACTS_TOKEN_AR1` against the default
`save_dir` silently finds nothing. And ⚠ **never sniff a model property out of the token with a bare
`occursin`** — current-style tokens carry no historical markers, so a positive test would reject
every current chain. Write such a guard as `is_legacy_token(c) ? occursin("-marker", c) : true`; the
partition is exact, since every token minted before the prefix was dropped carries `-gsar-cut` and
none minted after does.

**Rolling origins.** The forecast origin is rolled weekly over the whole *available period* the
current data support (`available_forecast_origins`): bounded below by the first inc2prev week
($2020$-$08$-$02$) plus the 12-week fit/lag lookback, and above by the last CoMix contact week minus
$\max h$ weeks (the iterate needs contacts out to $t_0 + 4$). `ENV["FIT_END"]` (default
`2021-12-31` ⇒ 63 origins, 2020-10-18 … 2021-12-26) and `ENV["ORIGIN_MIN"]` cap the roll; 9j reads
the same two variables and **must** agree with 8j, or its lookups miss and it silently refits.

**Four ways ++ two baselines.** At each origin the four combos
$\{$`NegBinAgePair`, `HurdleWeibullAgePair`$\} \times \{$`MeanNGM`, `NeighbourhoodDegreeNGM`$\}$,
plus `NegBinAgePair`×`DiagonalMeanNGM` (no-interaction) and `NoContactDegree`×`NullNGM` (null), are
fit (two-stage) and forecast $1$–$4$ weeks ahead. The run is memory-bounded and resumable: per origin
only that origin's four degree windows are built (reusing a single raw read of the CoMix tables),
Stage-1 GP chains (8 = 2 degree × 4 horizons — the no-interaction model reuses the NegBin chains and
the null needs none) and Stage-2 pooled files (24 = 6 combos × 4 horizons) are pre-fit, forecasts are
assembled, and the degree data discarded. The two baselines therefore cost one extra model's worth of
Stage-2 fits plus 4 cheap fits per origin.

**Outputs.** Quantile scores (`res/8j_scores_by_model*.csv`)
and diagnostic figures: WIS by horizon, four-ways WIS bars, WIS over the forecast period,
forecast-vs-observed fans by origin, and fitted transmission structure — susceptibility/infectivity
ratios to the reference group, and the two spatial GP length-scales
$\rho_{\text{diag}}, \rho_{\text{gap}}$ over time plus the AR(1) coefficient $\phi$ on the same axis
(⚠ the units differ, age-years against a dimensionless correlation, so `plot_lengthscales`' third
series is *not* a length-scale and must not be read as weeks). Further panels:

- a **generation-interval** panel (9j): the posterior of the GI mean and SD **in days**
  (back-transformed from $w_\mu,w_\sigma$) and of $w=(w_1..w_4)$, overlaid on the prior, by model and
  over the rolling origins — the identifiability check for the $w$–$\gamma_{\mathrm{SAR}}$
  confounding of §11;
- a **$p^0$** panel (10j, weighted path): the fitted zero probability against the empirical $n^0/n$
  per cell — the check that the Binomial denominator is wired to the roster correctly;
- an **antibody-protection factor** panel (9j, `res/9j_protection_factor_F.png`): the posterior of
  $F$ per model over the rolling origins. With $F$ pinned at $1$ (§3.2) this is degenerate by
  construction — six flat lines at exactly $1.0$ — and is deliberately kept as the visible
  confirmation that the antibody term is off (`plot_F`'s `ylims` is $(0,1.05)$ so the pinned line
  does not vanish into the border);
- **dispersion panels** (10j/11j): a $7\times7$ map of per-cell dispersion against its block mean,
  four flat quadrants by construction; and `plot_within_block_sd`, whose current-model line **must be
  identically 0** (§4.3);
- **Stage-1 GP hyperparameters vs prior** (10j §7): per-draw $\rho_{\text{diag}}$, $\rho_{\text{gap}}$,
  $\phi$, $\eta$, $\sigma_c$ against their priors, by horizon and marginally;
- **Stage-1 NUTS convergence** (12j): per-block worst ESS and split-$\hat R$, E-BFMI, and the
  fraction of transitions at the depth cap.

**Reproduction number** (two separate figures). *(1)* `res/9j_reproduction_number.png` — the
"contact & transmission" $R$: the dominant (Perron) eigenvalue $\rho(N)$ of the frozen origin-week
NGM, per origin, over the inc2prev national $R$ and the $R=1$ line. *(2)*
`res/9j_contact_reproduction_number.png` — the **contacts-only relative** $R$:
$\rho(C^\ast_t)/\rho(C^\ast_{t_{\text{first}}})$, the dominant eigenvalue of the bare contact matrix
$C^\ast$ alone (no $\gamma_{\text{SAR}}$, susceptibility, infectivity or antibody) normalised to the
**first forecast origin** (=1.0), isolating how contact structure alone drove transmissibility
relative to the baseline week. Overlaid with one **model-free** line —
$\rho(\hat{E}_t)/\rho(\hat{E}_{t_{\text{first}}})$, the same ratio for the RAW empirical weekly
mean-contact matrix $\hat{E}_t$ (`AgePairData.emp_mean`, no GP / no reciprocity / no fit) — the
data-only baseline the four fitted $C^\ast$ curves smooth.

---

## 11. Deliberate simplifications and open issues

The model is intentionally lean. The following are the documented simplifications, the live hazards,
and the seams at which each would be relaxed. **These should be revisited before any scientific
interpretation of the fitted transmission parameters.**

- **The hurdle-Weibull path wants the pooled temporal limit, and that is its likelihood, not its
  prior.** With $p^0 \approx 0.95$ most cells carry almost no positive observations, so the weighted
  likelihood is nearly flat in time. The preference has now surfaced under **four** temporal
  parameterisations — $\rho_{\text{time}}$ 20–27 wk under a log-normal length-scale prior, 47–66 wk
  under an inverse-gamma one, $\phi \to 0.9985$–$0.9998$ under AR(1), and $\rho_{\text{time}}$ at or
  past the window under a Matérn 3/2 temporal kernel *whose prior put $8.7\times10^{-6}$ of its mass
  there*. A prior that tight being overridden that far is the clearest evidence that the likelihood,
  not the prior, is what wants the limit. **The indicated action for that path is
  `constant_contacts = true`, not a fifth temporal prior.** The reason AR(1) is nonetheless the
  kernel (§5) is not that it prevents the limit but that it stays conditioned *in* it.
- **$\phi$ is not identified by the data under Pathfinder, on either degree model.** Holding
  everything fixed but the starting value, the final $\phi$ is largely a function of where the
  optimiser started — hurdle-Weibull spans $0.011$–$1.000$ and is not even monotone in $\phi_0$;
  NegBin spans $0.150$–$0.726$, monotone, with only $\approx30\%$ shrinkage toward the middle. Two
  ordinary starting points therefore produce opposite conclusions about the temporal structure of the
  same data. `stage1_phi_init_scale` and `stage1_phi_pf_max` (§7) make the start *defined and
  reproducible*; they do not identify $\phi$ and are not claimed to. **Read $\phi$ from NUTS, never
  from a Pathfinder fit** — including any $\phi$ median quoted from a Pathfinder survey, which partly
  measures the optimiser's starting distribution.
- **There is no gate on a diverged Stage-1 fit.** The standing criterion is raw
  $\max|z| > 10$ or $|\log\eta| > 5$; a chain failing it is a diverged optimiser path, not a
  posterior, and nothing in the pipeline reports it — Pathfinder returns normally, the artefact
  caches, and 9j/10j/11j read it as a fit. `fit_stage1` should reject such a chain instead of caching
  it silently. Re-run that census after any refit before trusting the weighted path.
- **Contact temporal structure.** *Implemented* — the age-pair structure field is smoothed across
  weeks by the temporal factor of a separable spatio-temporal GP (§5), with one shared **AR(1)
  coefficient $\phi$**; since `-lcar1` the **weekly level shares it too** (through a scaled
  projection of the same kernel — §5), so $\phi$ describes both the age-pair trajectories and the
  overall level. It did **not** share it from `-lc0` to `-lcar1` ($c_t$ was iid across weeks,
  sum-to-zero, so $\phi$ then described the individual age-pair trajectories and nothing else).
  *Remaining seams*: a non-separable space–time kernel; and the field-side $\phi\to1$ limit, where
  $L_{\text{time}}$'s first column absorbs the field and most of $z$ stops reaching the likelihood.
  $\phi$'s prior is the only restraint on that (§5, §6).
- **Contact-degree dispersion** is a **block mean per week and nothing more** (§4.3). *Remaining
  seams*: **(i)** it is **not temporally smoothed** — the block means are redrawn iid each week,
  unlike the mean field; **(ii)** with no per-cell term there is no within-block spread at all, so
  genuine cell-level heterogeneity (if any) is absorbed into the block mean; **(iii)** many per-week
  cells are **empty**, so their dispersion is prior-only yet still feeds $\langle k^2\rangle$ into the
  NGM — noise that the **neighbourhood** builder, which divides by $\langle k\rangle$, amplifies.
- **Hurdle zero probability.** *Implemented* — $p^0$ is fitted per cell × week with a Binomial roster
  likelihood rather than plugged in empirically (§4.2), so its uncertainty propagates into
  $\langle k\rangle$, $\langle k^2\rangle$ and $g$. *Remaining seams*: **no pooling** across cells or
  weeks (each $p^0$ stands alone under a flat $\mathrm{Beta}(1,1)$ — defensible while roster counts
  are large, but sparse cells lean entirely on the prior), and cells with $n_{t,i,j} = 0$ have no
  likelihood at all.
- **Group-contact weight** $w_{\text{group}} = 2.5/240$ is fixed, not estimated.
- **Generation interval.** *Implemented* — the two log-normal parameters are estimated (§3.1) with
  Munday's informative prior. *Remaining seams*: it is **not variant-specific** (one $w$ per fitting
  window, re-estimated per origin), and $s_{\max}=4$ stays fixed. Note that $w$ and
  $\gamma_{\mathrm{SAR}}$ are **confounded** — both scale the renewal predictor, so raising $w_1$ and
  lowering $\gamma_{\mathrm{SAR}}$ nearly compensate over an 8-week window. The $20\%$ prior SD is
  what keeps the pair identified and should not be loosened; if the posterior equals the prior, the
  GI is adding nothing, and if it parks on a soft-clamp with a tight CI that is clamp compression,
  not certainty.
- **Transmission block** — *partially addressed*: the level is an explicit, data-identified
  **per-contact secondary attack rate** $\gamma_{\mathrm{SAR}}$ (un-normalised $C^\ast$) with
  susceptibility/infectivity **relative** to the reference bin, fit as **Stage 2** of the cut (§6.0).
  The relative offsets are **independent per age bin** with separately estimated marginal scales
  $\sigma_s,\sigma_i$ and **no cross-bin smoothing**, each log-offset soft-clamped to
  $[\log 0.05,\log 20]$. *Remaining seams*: $F$ is **pinned at 1** (§3.2), so the open question is
  whether to re-enable the antibody term at all rather than which prior to give it (the reference
  prior is $\mathrm{Beta}(5,1)$; the paper uses $\Gamma(2,2)T[0,1]$); and infection observation error
  uses an independence approximation across the week and across ages.
- **Antibody availability.** The forecast NGM assumes $A(t_0+h)$ is known at forecast time (§3.2),
  parallel to the contact data. This is a *stronger* assumption than the contact one, because
  `gen_dab` shares the inc2prev/CIS pipeline with the infection targets while CoMix is an independent
  survey. The Stage-2 fit loop is left at $t_0$-anchored antibody, so the likelihood pairs $t{+}h$
  contacts with $t$ antibody.
- **NGM lag index.** Munday Eq 1/6 places the NGM *inside* the lag sum,
  $\sum_s w(s)\,N(t-s)\,I(t-s)$ — the NGM at the infector's primary event. This implementation applies
  a single $N(t)$ outside the sum (§3.3). The deviation is unresolved.
