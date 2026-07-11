# Preliminary Model Structure: A Joint Age-Pair Contact-Degree / Renewal Forecasting Model

*Reverse-engineered specification of the model implemented in `src/8j_preliminary_forecast.ipynb`
and the framework modules it loads (`forecast_utils.jl` → `framework.jl`, `infection_data.jl`,
`degree_agepair.jl`, `renewal.jl`, `ngm.jl`, `joint_model.jl`, `scoring.jl`). This document
records the model exactly as coded; it is written to stand on its own as a methods description.*

---

## 1. Overview

We forecast weekly, age-stratified SARS-CoV-2 infection incidence in England by coupling two
components inside a single Bayesian model:

1. a **contact-degree model** that describes, for each ordered pair of age groups, the
   distribution of the number (or duration-weighted amount) of daily social contacts reported
   in the CoMix-UK survey; and
2. an **age-structured renewal / next-generation-matrix (NGM) transmission model** whose
   effective-contact matrix is constructed directly from the raw moments of those contact-degree
   distributions.

The design is deliberately *composable*, organised around two orthogonal modelling axes that are
each fixed once per fit and dispatched deterministically:

| Axis | Symbol in code | Options | Meaning |
|------|----------------|---------|---------|
| **1. Contact-degree model** | `ContactDegreeModel` | `NegBinAgePair`, `HurdleWeibullAgePair` | How the age-pair degree distribution is modelled (unweighted counts vs. duration-weighted hurdle) |
| **2. NGM builder** | `NGMBuilder` | `MeanNGM`, `NeighbourhoodDegreeNGM` | How the per-capita effective contact $C^0$ is formed from the degree distribution's moments |

The Cartesian product of the two axes gives the **"four ways"** — a $2\times2$ grid of model
variants — that the preliminary analysis fits and scores side by side. A single joint Turing model
(`model_joint`) serves all four combinations: the contact-degree likelihood and the infection
likelihood live in the same probabilistic program, and the two axes enter only as fixed model
arguments, so the parameter space is well defined per fit.

Notationally we use $A$ age groups indexed $a,b,i,j \in \{1,\dots,A\}$, with $i$ (or $a$) the
**participant / contactor / susceptible** group and $j$ (or $b$) the **contactee / infectious**
group. Time is discretised into ISO-like calendar weeks indexed $t$.

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
reciprocity structure (§5). For the spatial smoother (§5) each bin is assigned a numeric age coordinate equal to
its interval midpoint, except the open-ended $70+$ bin, which is fixed to $74.5$ years (the
observed mean age of $70+$ CoMix participants):

$$
\text{mid} = (6.0,\ 13.0,\ 20.0,\ 29.5,\ 42.0,\ 59.5,\ 74.5).
$$

The first `child_bins = 2` groups ($[2,10]$, $[11,15]$) are labelled **child** and the remaining
five **adult**; this two-level block structure indexes the degree-dispersion parameters (§4).

### 2.2 Weekly grid

Weeks are Sunday-anchored to `WEEK_ANCHOR = 2021-03-21`. For a date $d$ the week index is
$\lfloor (d - \text{anchor})/7 \rfloor$; the week is labelled by its Wednesday mid-date. A forecast
is organised around a **`WeeklyWindow`** with origin $t_0$ (snapped to its Sunday week), comprising

- `fit_weeks`: the `n_fit = 8` weeks ending at $t_0$,
- `lag_weeks`: the `smax = 4` weeks immediately preceding the fit weeks (renewal history),
- `all_weeks = lag_weeks ++ fit_weeks`: the $T = 12$ contiguous weeks over which data are assembled,
- `forecast_weeks`: the horizon target weeks $t_0 + h$ for $h \in$ `horizons` $= \{1,2,3,4\}$.

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

Populations $N_a$ and proportions $N_a / \sum_b N_b$ complete the `WindowData` container.

### 2.4 Age-pair contact-degree data

For each window, `prepare_degree_data` assembles per-cell $(t,i,j)$ contact data from the CoMix-UK
participant roster (`part_uk`) and contact table (`contacts_uk.arrow`), reusing the age-pair
binning of `7j_weekly_age_pair`:

- **Participant / contactee binning.** Each participant's reported age interval and each contact's
  estimated age interval $[\text{lo},\text{hi}]$ are mapped to a CIS bin. A single overlapping bin
  is assigned directly; an interval overlapping several bins is resolved by one **seeded,
  population-weighted random draw** (`MersenneTwister(seed)`), and an interval below the grid's
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
| Empirical zero probability $n_{\text{zero}}/n_{t,i}$ | $p^0_{t,i,j}$ | Hurdle path + moments |
| Roster count | $n_{t,i,j}$ | denominators |

---

## 3. Transmission core: renewal equation and NGM

### 3.1 Generation interval

The weekly generation-interval PMF $w = (w_1,\dots,w_{s_{\max}})$, $s_{\max} = 4$, is a discretised
log-normal (`gen_interval_pmf`). With mean and SD given in days (both $5$ days by default,
i.e. $\mathrm{CV}=1$), converted to weeks $\mu_w, \sigma_w$, the log-normal has
$\text{sdlog}^2 = \log\!\big((\sigma_w/\mu_w)^2 + 1\big)$ and
$\text{meanlog} = \log\mu_w - \text{sdlog}^2/2$, and

$$
w_s \propto F(s) - F(s-1),\qquad s = 1,\dots,s_{\max},
$$

renormalised to sum to one, where $F$ is the log-normal CDF.

### 3.2 Next-generation matrix

For a target week the $A\times A$ NGM is

$$
N_{ab}(t) \;=\; \gamma_{\mathrm{SAR}}\;\cdot\;\underbrace{\text{susc}_a\big(1 + (F-1)\,A_a(t)\big)}_{\text{full\_susceptibility}_a(t)}
\;\cdot\; C^\ast_{ab} \;\cdot\; \text{inf}_b ,
$$

with $\gamma_{\mathrm{SAR}}$ the single **absolute transmissibility** scalar (analysis-plan reparam;
it alone carries the NGM level), $\text{susc}_a$ the **relative** inherent susceptibility of group $a$
and $\text{inf}_b$ the **relative** infectivity of group $b$ — both normalised so the reference bin $1$
("2-10") is $1$ ($\text{susc}_1=\text{inf}_1=1$; bins $2..A$ estimated) — and $F \in (0,1)$ a **leaky**
antibody-protection factor scaling
susceptibility by the group's antibody prevalence $A_a(t)$ (at $F=1$ antibodies confer no
protection; smaller $F$ gives stronger protection). $C^\ast_{ab}$ is the per-capita effective
contact matrix produced by the NGM builder (§5.1).

### 3.3 Renewal recursion and forecast

New infections propagate by the weekly renewal equation

$$
I_a(t) \;=\; \sum_{b} N_{ab}(t)\, \sum_{s=1}^{s_{\max}} w_s\, I_b(t-s)
\;=\; \Big[N(t)\, \textstyle\sum_{s} w_s\, I(t-s)\Big]_a ,
$$

i.e. a single NGM applied to the $w$-weighted sum of the $s_{\max}$ lagged infection vectors
(`renewal_next`). A deterministic multi-step forecast (`forecast_forward`) freezes $N$ at the origin
week and iterates $H$ weeks, appending each prediction to the history to serve as the next lag.

---

## 4. Contact-degree models (Axis 1)

Both families parameterise the **directional mean** $\mu_{i\to j}$ of cell $(i,j)$ through the shared
reciprocity/GP construction of §5; they differ in the observation likelihood and in how the raw
moments $\langle k\rangle$, $\langle k^2\rangle$ and a zero-conditioning factor $g$ are computed for
the NGM.

### 4.1 Unweighted negative binomial (`NegBinAgePair`)

The integer contact counts (including zeros) of cell $(i,j)$ are modelled as
$\mathrm{NegBin}(\mu_{ij}, \phi_{\beta(i)\beta(j)})$, parameterised by **mean** $\mu_{ij}$ and a
**dispersion** $\phi$ that depends only on the child/adult block pair $(\beta(i),\beta(j))$, with
$\mathrm{Var} = \mu + \mu^2/\phi$. The log-likelihood sums over the empirical count distribution:

$$
\ell_{ij} = \sum_{k} y_k\,\log \mathrm{NegBin}(k;\mu_{ij},\phi_{\beta(i)\beta(j)}),
$$

where $(k,y_k)$ are the distinct degrees and their observed frequencies. Zeros are modelled
directly (this path is not a hurdle). The raw moments and zero factor supplied to the NGM are

$$
\langle k\rangle = \mu,\qquad
\langle k^2\rangle = \mu + \mu^2\!\left(1 + \tfrac1\phi\right),\qquad
g = \frac{1}{1 - P_0},\quad P_0 = \left(\tfrac{\phi}{\phi+\mu}\right)^{\!\phi},
$$

with $1 - P_0$ floored at $10^{-3}$ for numerical safety. $g$ conditions the neighbourhood degree
on having at least one contact (left-truncating the fitted NegBin).

### 4.2 Duration-weighted hurdle-Weibull (`HurdleWeibullAgePair`)

Here $\mu_{ij}$ denotes the mean of the **positive** duration-weighted degrees. The positive weights
$\{W\}_{ij}$ are modelled as $\mathrm{Weibull}(\kappa_{\beta(i)\beta(j)}, \lambda_{ij})$ with a
block-indexed shape $\kappa$ and scale chosen so the Weibull mean equals $\mu_{ij}$:

$$
\lambda_{ij} = \frac{\mu_{ij}}{\Gamma(1 + 1/\kappa)},\qquad
\ell_{ij} = \sum_{W \in \{W\}_{ij}} \log \mathrm{Weibull}(W;\kappa,\lambda_{ij}).
$$

The zero part is a genuine **hurdle**: the zero probability is *not* a fitted parameter but the
empirical $p^0_{ij}$, which enters only the moments (not the likelihood). With
$\mathrm{CV}_W^2 = \Gamma(1+2/\kappa)/\Gamma(1+1/\kappa)^2 - 1$ the zero-included raw moments are

$$
\langle k\rangle = (1-p^0)\,\mu,\qquad
\langle k^2\rangle = (1-p^0)\,\mu^2\big(1 + \mathrm{CV}_W^2\big),\qquad
g = 1 - p^0 .
$$

### 4.3 Dispersion/shape parameterisation

The block-indexed dispersion (NegBin $\log\phi$) or shape (Weibull $\log\kappa$) carries one value
per child/adult block pair — four values indexed by the block-linear code
$\ell = 2(\beta(i)-1) + \beta(j) \in \{1,2,3,4\}$ (contactor block $\times$ contactee block), stored
as a $4 \times T$ array (one block-vector per week). Inside the model these log-parameters are
clamped to keep the mode interior and avoid Weibull/exponential underflow
($\log\kappa \in [-3,3]$, i.e. $\kappa \in [0.05,20]$; $\log\phi \in [-4,5]$, i.e.
$\phi \in [0.018,148]$).

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

so the total number of $i\!\to\! j$ contacts equals that of $j\!\to\! i$ contacts.

**Spatial kernel over the age-pair grid.** The 28 log-rates are smoothed by a non-centred GP over
the age-pair coordinates. The spatial kernel is an **anisotropic** separable RBF in **diagonal
coordinates**: the age pair $(\text{mid}_{p_1},\text{mid}_{p_2})$ is rotated $45°$ into a total-age
(along-diagonal) coordinate and an age-gap (across-diagonal) coordinate,

$$
u_p = \frac{\text{mid}_{p_1}+\text{mid}_{p_2}}{\sqrt 2}, \qquad
v_p = \frac{\text{mid}_{p_1}-\text{mid}_{p_2}}{\sqrt 2},
$$

each with its **own** length-scale — $\rho_{\text{diag}}$ on total age, $\rho_{\text{gap}}$ on the
age gap (assortativity):

$$
K^{\text{age}}_{pq} = \exp\!\left(-\frac{(u_p-u_q)^2}{2\rho_{\text{diag}}^2} - \frac{(v_p-v_q)^2}{2\rho_{\text{gap}}^2}\right),
\qquad L_{\text{age}} = \mathrm{chol}(K^{\text{age}} + 10^{-6} I).
$$

The rotation is orthonormal, so $(u_p-u_q)^2+(v_p-v_q)^2 = (\text{mid}_{p_1}-\text{mid}_{q_1})^2 +
(\text{mid}_{p_2}-\text{mid}_{q_2})^2$ and $\rho_{\text{diag}}=\rho_{\text{gap}}$ recovers the
isotropic RBF exactly.

**Separable spatio-temporal GP over age-pairs × weeks.** Over the $T$ window weeks the field is *not*
drawn independently each week. Each age-pair carries its own temporally-correlated log-rate, with the
temporal length-scale **shared** across all age-pairs — a separable (Kronecker) GP whose covariance
factorises into the spatial kernel above and a temporal RBF over the week indices $t=1,\dots,T$,

$$
K^{\text{time}}_{st} = \exp\!\left(-\frac{(s-t)^2}{2\rho_{\text{time}}^2}\right),
\qquad L_{\text{time}} = \mathrm{chol}(K^{\text{time}} + 10^{-4} I),
$$

($\rho_{\text{time}}$ in weeks; the larger $10^{-4}$ jitter keeps $L_{\text{time}}$ positive-definite
in the near-pooled limit). The $P\times T$ structure field is drawn matrix-normal, non-centred,

$$
R = \eta\,\big(L_{\text{age}}\, Z\, L_{\text{time}}^{\!\top}\big),
\qquad Z \sim \mathcal N(0,1)^{P\times T},
\qquad \operatorname{Cov}(\operatorname{vec} R) = \eta^2\,\big(K^{\text{time}}\!\otimes K^{\text{age}}\big),
$$

so fixing a week gives the spatial RBF and fixing an age-pair gives a temporal GP with shared
$\rho_{\text{time}}$. The **overall weekly level** is likewise temporally smoothed, but with its own
amplitude $\sigma_c$ **decoupled** from $\eta$: a scalar intercept $c$ plus a 1-D temporal GP sharing
$L_{\text{time}}$,

$$
c_t = c + \sigma_c\,(L_{\text{time}}\, z_c)_t,
\qquad z_c \sim \mathcal N(0,1)^{T},
$$

and the week-$t$ log-rate field is $r_{p,t} = c_t + R_{p,t}$. The intercept is anchored at the grand
mean $c_0 = \overline{\log(\text{emp mean})_{ij} - \log N_j}$ (so $c \sim \mathcal N(c_0,3^2)$).
$\rho_{\text{time}}\to 0$ recovers independent weeks; $\rho_{\text{time}}\to\infty$ collapses to one
pooled field. Numerically, each of $\rho_{\text{diag}}, \rho_{\text{gap}}$ is clamped to $[3,45]$,
$\rho_{\text{time}}$ to $[0.5,26]$ weeks, $\eta$ and $\sigma_c$ to $[e^{-3}, e^{2}]$, and the per-cell
exponent $r_{p,t} + \log N_j$ to $[-8,6]$ (so $\mu \in [3\times10^{-4}, 400]$); the modes stay
interior so reciprocity is not distorted.

The kernels $L_{\text{age}}, L_{\text{time}}$, the length-scales
$\rho_{\text{diag}}, \rho_{\text{gap}}, \rho_{\text{time}}$ and the scales $\eta, \sigma_c$ are all
**shared across weeks**; the per-week variation is now **temporally correlated** (through
$L_{\text{time}}$) rather than an independent draw per week (§6).

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
$\langle k\rangle = 0$). Note $\langle k^2\rangle/\langle k\rangle = m(1+\mathrm{CV}^2)$, so the
neighbourhood builder up-weights high-variance cells. Reciprocity is carried entirely by the
contact-mean estimation (§5): the **Mean NGM** therefore inherits exact total-contact reciprocity
from $\mu$ ($N_a\,C^\ast_{ab} = N_b\,C^\ast_{ba}$), whereas the **Neighbourhood NGM**'s size-biased
$C^\ast$ is generally *not* reciprocal (it depends on the block-dependent dispersion and zero
factor) and is left un-symmetrised by design. $C^\ast$ is independent of the transmission
parameters, so it is computed once per week and reused across the renewal recursion.

---

## 6. The joint model

`model_joint(dm, nb, ds, wd, w, cfg)` is one Turing `@model` combining the contact-degree
likelihood and the infection likelihood.

**The model estimates contacts per week, temporally smoothed.** A single **separable
spatio-temporal GP** (§5) governs all $T$ window weeks — sharing the spatial kernel $L_{\text{age}}$,
the temporal kernel $L_{\text{time}}$, the length-scales
$\rho_{\text{diag}}, \rho_{\text{gap}}, \rho_{\text{time}}$ and the scales $\eta, \sigma_c$ — so the
weekly log-rate fields are **temporally correlated** rather than independent draws. Each week yields
its own $C^\ast_t$ (through the per-week level $c_t$, structure-field column $R_{\cdot,t}$, and
per-week block dispersion $\phi_{\beta(i)\beta(j),t}$/$\kappa_{\beta(i)\beta(j),t}$), and the transmission NGM $N(t)$ therefore varies in time through
**both** antibody prevalence and (now temporally-smooth) contacts.

**Sampling statements.**

*Contact block (one separable spatio-temporal GP over all weeks $t = 1,\dots,T$):*

$$
\begin{aligned}
\log\rho_{\text{diag}},\ \log\rho_{\text{gap}} &\sim \mathcal N(\log 15,\ 0.5^2), &
\log\rho_{\text{time}} &\sim \mathcal N(\log 4,\ 0.5^2), &
\log\eta &\sim \mathcal N(0,\ 0.5^2), \\
c &\sim \mathcal N(c_0,\ 3^2), &
\log\sigma_c &\sim \mathcal N(0,\ 0.5^2), &
z_c &\sim \mathcal N(0,1)^{T}, \\
z &\sim \mathcal N(0,1)^{P\times T}, &
\log\kappa_{t}\ \text{or}\ \log\phi_{t} &\sim \mathcal N(0,\sigma_d^2)^{4}
& & (\sigma_d = 0.5\ \text{Weibull},\ 1.0\ \text{NegBin};\ \text{4 block pairs}).
\end{aligned}
$$

with the derived level $c_t = c + \sigma_c (L_{\text{time}} z_c)_t$ and structure field
$R = \eta\,(L_{\text{age}}\, z\, L_{\text{time}}^{\!\top})$ giving the week-$t$ log-rate
$r_{p,t} = c_t + R_{p,t}$ (§5), the contact-degree log-likelihood of §4 injected via
`Turing.@addlogprob!`, and
$C^\ast_t = $ `contact_star`$(nb, \langle k\rangle_t, \langle k^2\rangle_t, g_t)$. The dispersion is
**block-linear per week** (§4.3): one $\log\kappa$/$\log\phi$ value per child/adult block pair (a
$4\times T$ array), re-drawn each week with no age-pair random effect; it is **not** temporally
smoothed (unlike the mean field).

*Transmission block (absolute $\gamma_{\mathrm{SAR}}$ + reference-normalised susc/inf, non-centred):*
susceptibility and infectivity are **relative** to the reference bin $1$ ("2-10"), fixed to $1$; only the
$A-1$ non-reference offsets are estimated, and the single absolute-transmissibility scalar
$\gamma_{\mathrm{SAR}}$ carries the level (it replaces the old confounded $\mu_s,\mu_i$ pair).

$$
\begin{aligned}
\log\gamma_{\mathrm{SAR}} &\sim \mathcal N(\log 0.33,\, 0.56^2), &&
\gamma_{\mathrm{SAR}} = \exp(\log\gamma_{\mathrm{SAR}}),\\
\sigma_s &\sim \mathcal N^+(0.1, 0.02^2), & z_s &\sim \mathcal N(0,1)^{A-1}, &
\text{susc} &= \big(1,\ \exp(\sigma_s z_s)\big),\\
\sigma_i &\sim \mathcal N^+(0.1, 0.02^2), & z_i &\sim \mathcal N(0,1)^{A-1}, &
\text{inf} &= \big(1,\ \exp(\sigma_i z_i)\big),\\
F &\sim \mathrm{Beta}(5,1), & \sigma_{\text{inf}} &\sim \mathcal N^+(0.05, 0.025^2). & & &
\end{aligned}
$$

The $\gamma_{\mathrm{SAR}}$ prior is **calibrated** to the fitted reference-cell level
$\text{susc}_1\!\cdot\!\text{inf}_1$ (the quantity $\gamma_{\mathrm{SAR}}$ absorbs under bin-1
normalisation) read from the pre-reparam chains: median $0.33$, log-SD $0.56$. The model returns the
generated quantities $(\text{susc}, \text{inf}, F, \gamma_{\mathrm{SAR}}, \sigma_{\text{inf}}, \{C^\ast_t\})$.

*Infection likelihood*, over the fit weeks $t = s_{\max}+1,\dots,T$ (the first $s_{\max}$ weeks serve
only as renewal history). For each age $a$,

$$
\hat I_a(t) = \big[N(t)\, \textstyle\sum_{s} w_s\, I(t-s)\big]_a,\qquad
I_{a,t} \sim \mathcal N\!\Big(\hat I_a(t),\ \sigma_{a,t}^2\Big),\quad
\sigma_{a,t}^2 = (\sigma_{\text{inf}}\, I_{a,t})^2 + (s^{I}_{a,t})^2,
$$

i.e. the observation SD combines a multiplicative process-noise term $\sigma_{\text{inf}} I_{a,t}$
in quadrature with the inc2prev estimate SD $s^{I}_{a,t}$. The model returns the generated
quantities $(\text{susc}, \text{inf}, F, \gamma_{\mathrm{SAR}}, \sigma_{\text{inf}}, \{C^\ast_t\})$ for forecasting.

**Joint log-likelihood.** Stacking the two `Turing.@addlogprob!` contributions, the joint model
accumulates

$$
\log L \;=\;
\underbrace{\sum_{t=1}^{T}\ \sum_{i,j=1}^{A} \ell^{(t)}_{ij}}_{\text{contact degree (§4)}}
\;+\;
\underbrace{\sum_{t=s_{\max}+1}^{T}\ \sum_{a=1}^{A}
\log \mathcal N\!\big(I_{a,t}\,;\ \hat I_a(t),\ \sigma_{a,t}^2\big)}_{\text{infection renewal}},
$$

where $\ell^{(t)}_{ij}$ is the §4.1 NegBin (over the count histogram) or §4.2 Weibull-hurdle (over
the positive duration-weighted degrees) cell log-likelihood, evaluated at the week-$t$ contact mean
$\mu_{ij,t}$ and its block-pair dispersion $\phi_{\beta(i)\beta(j)}$ / shape
$\kappa_{\beta(i)\beta(j)}$; the contact term runs over **all** $T$ window weeks while the infection
term uses only the $t>s_{\max}$ fit weeks. Adding the priors of the two sampling blocks gives the
log-posterior that Pathfinder/NUTS target (§7).

Every log-scale latent that feeds an exponential ($\log\rho, \log\eta, \log\kappa, \log\phi$, and
the per-cell rate) is clamped inside the model body so that aggressive optimiser/Pathfinder steps
cannot underflow (e.g. Weibull scale $\to 0$); the clamps are wide enough that the posterior mode is
interior and gradients are unaffected.

**Per-horizon window offset (forecasting use).** Although the contact and infection blocks share the
index $t = 1,\dots,T$, they need not span the same calendar weeks. In forecasting (§8) the joint
model is re-fit once **per horizon** $h$: the contact-degree block's window is slid forward to end at
$t_0 + (h-1)$, while the infection/renewal block stays anchored at the origin $t_0$ (`wd` is fixed;
only the degree data `ds` changes with $h$, and each $(dm, nb, t_0, h)$ chain is cached separately as
`..._h<h>.jld2`). Thus for $h>1$ the contact term is fit over weeks offset $h-1$ **ahead** of the
infection term — the age-pair degree distribution is observed up to one week before the target
$t_0+h$, whereas infections and antibody are frozen at $t_0$. For $h=1$ the two windows coincide.

---

## 7. Inference

`fit_joint` fits one $(dm, nb)$ combination for one window:

1. **Pathfinder** (`Pathfinder.pathfinder`) produces a parsimonious variational approximation to the
   posterior (default 200 draws), used both as the standalone fit when `USE_NUTS = false` and as an
   initialiser otherwise.
2. **NUTS** (optional; `USE_NUTS = true`) is then run, initialised from the Pathfinder posterior
   mean. If NUTS fails it falls back to the Pathfinder draws (which carry the same parameter names).

The random seed is `cfg.seed = 1236`; a per-fit RNG stream can be supplied to make the parallel
pre-fit thread-safe. Fits are mutually independent, so `prefit_chains!` fans the
$(\text{origin} \times \text{combo} \times \text{horizon})$ chains out over Julia threads with a
CPU- and memory-balanced concurrency cap, pins BLAS to one thread to avoid oversubscription, warms
compilation on one spec first, and caches each chain to
`dt_intermediate/8j_chn_<degree>_<ngm>_<contacts>_<origin>_h<h>.jld2`. Cached chains are skipped, so
runs are resumable.

**The 8j notebook uses `USE_NUTS = false` — Pathfinder is the operative fitting method.**

---

## 8. Forecasting: the contact-updated iterate

The notebook forecasts with `iterated_forecast`, the *contact-updated* iterate. For a baseline
origin $t_0$, the infection and antibody series are **frozen at $t_0$**, while the contact/degree
window is allowed to slide: for horizon $h$ the degree window ends at $t_0 + (h-1)$ weeks, the joint
model is re-fit (or its cached chain reloaded), and a fresh NGM is formed from that window's
origin-week $C^\ast$ (with antibody held at $t_0$). A single renewal step is then taken,

$$
\hat I_a(t_0+h) = \Big[N\, \textstyle\sum_{s=1}^{s_{\max}} w_s\, I(t_0+h-s)\Big]_a,
$$

and observation noise $\sigma = \max(\sigma_{\text{inf}}\hat I_a, 10^{-6})$ is added per draw. The
renewal lags use observed infections up to $t_0$ plus the **mean** predictions of the intervening
weeks (a deterministic mean-plugged lag; per-draw coherence across the independent re-fits is
undefined). This produces an $A \times H \times K$ array of posterior-predictive draws
($K = $ `n_forecast_draws` $= 200$).

A simpler `posterior_forecast` also exists — it freezes the NGM at the origin and iterates $H$ weeks
with `forecast_forward` — but the 8j run uses the contact-updated iterate.

---

## 9. Scoring

The primary metric is the **weighted interval score (WIS)** computed by the R package
`scoringutils` (v2) via `RCall`, on quantile-format forecasts (`scoring.jl`):

- Posterior-predictive draws are summarised at the 19 quantile levels
  $\{0.05, 0.10, \dots, 0.95\}$ (rounded to two decimals so `scoringutils` matches the interval
  endpoints exactly), one row per (age $\times$ horizon $\times$ quantile level).
- Forecasts are scored on **both the natural and the log scale**
  (`transform_forecasts(fun = log_shift, offset = 1)`); the **headline metric is the log-scale WIS
  aggregated by horizon** across all origins. WIS is reported with its over-prediction,
  under-prediction and dispersion components, bias, and 50%/90% interval coverage, aggregated by
  model, by model $\times$ horizon, by model $\times$ origin, and by model $\times$ origin $\times$
  horizon (`res/8j_scores_by_model*.csv`).
- A native sample **CRPS** (energy form) provides a cheap cross-check.

---

## 10. The 8j experiment

The notebook (`8j_preliminary_forecast.ipynb`) runs the full grid:

- **Configuration** (`FrameworkConfig`): $d_{\max}=240$, $w_{\text{group}}=2.5/240$,
  $s_{\max}=4$, `n_fit` $=8$, `horizons` $=1{:}4$, seed $=1236$, generation interval mean/SD
  $=5/5$ days, `child_bins` $=2$, quantiles $0.05{:}0.05{:}0.95$, `n_forecast_draws` $=200$,
  GP priors $\log\rho_{\text{diag}},\log\rho_{\text{gap}}\sim\mathcal N(\log15,0.5^2)$ (shared
  prior for both diagonal length-scales), $\log\eta\sim\mathcal N(0,0.5^2)$,
  $\log\rho_{\text{time}}\sim\mathcal N(\log4,0.5^2)$ (`gp_time_len_prior`, weeks) and
  $\log\sigma_c\sim\mathcal N(0,0.5^2)$ (`gp_level_scale_prior`), **block-linear per-week dispersion**
  (one $\log\kappa$/$\log\phi$ per child/adult block pair, §4.3), and **per-week temporally-smoothed
  contact estimation** (one separable spatio-temporal age-pair GP across the window weeks). The
  chain-cache `contacts_label` is `"temporal-gsar"` — the `-gsar` tag marks block-linear dispersion
  plus the absolute-transmissibility $\gamma_{\mathrm{SAR}}$ reparam. (The earlier `-hdisp`/`-hn`
  dispersion random-effect tags were reverted 2026-07-11, so the current `-gsar` caches are disjoint
  from those and from the pre-$\gamma_{\mathrm{SAR}}$ `"temporal"` chains.)
- **Rolling origins.** The forecast origin is rolled weekly over the whole *available period* the
  current data support (`available_forecast_origins`): bounded below by the first inc2prev week
  ($2020$-$08$-$02$) plus the 12-week fit/lag lookback, and above by the last CoMix contact week
  minus $(\max h - 1)$ weeks (the iterate needs contacts out to $t_0 + 3$).
- **Four ways.** At each origin the four combos
  $\{$`NegBinAgePair`, `HurdleWeibullAgePair`$\} \times \{$`MeanNGM`, `NeighbourhoodDegreeNGM`$\}$
  are fit and forecast $1$–$4$ weeks ahead. The run is memory-bounded and resumable: per origin only
  that origin's four degree windows are built (reusing a single raw read of the CoMix tables), its
  16 chains are pre-fit in parallel, forecasts are assembled, and the degree data discarded.
- **Outputs.** Quantile scores (`res/8j_scores_by_model*.csv`) and diagnostic figures: WIS by
  horizon, four-ways WIS bars, WIS over the forecast period, forecast-vs-observed fans by origin,
  and fitted transmission structure (susceptibility/infectivity ratios to a reference group and the
  three GP length-scales $\rho_{\text{diag}}, \rho_{\text{gap}}, \rho_{\text{time}}$ over time).

---

## 11. Deliberate simplifications

The preliminary model is intentionally lean; the following are the documented simplifications and
the seams at which they would be relaxed:

- **Contact temporal structure.** *Implemented* — contacts are smoothed across weeks by a separable
  spatio-temporal GP (§5), both the structure field and the overall level, sharing one temporal
  length-scale $\rho_{\text{time}}$. The **dispersion** remains **block-linear** — one $\log\kappa$/
  $\log\phi$ value per child/adult block pair (§4.3), re-drawn **per week** (not temporally smoothed)
  with no age-pair random effect. Remaining seams: temporally smoothing the block dispersion, an
  age-pair dispersion random effect (partial pooling within a block), and a longer-memory or
  non-separable space–time kernel (a stationary RBF is used now).
- **Group-contact weight** $w_{\text{group}} = 2.5/240$ is fixed, not estimated.
- **Fixed generation interval** (5-day mean, log-normal) rather than an estimated or
  variant-specific one.
- **Transmission block** — *partially addressed*: the level is now an explicit, data-identified
  **absolute transmissibility** $\gamma_{\mathrm{SAR}}$ with susceptibility/infectivity **relative** to the
  reference bin $1$ (the plan's $\gamma_{\mathrm{SAR}}$ + baseline-"2-10" form), replacing the old
  confounded $\mu_s,\mu_i$ level pair. *Remaining seams*: the relative-offset scales
  $\sigma_s,\sigma_i\sim\mathcal N^+(0.1,0.02^2)$ are still tight (fitted profiles want more spread — carried
  by large $z$ rather than $\sigma$), $F$ keeps the $\mathrm{Beta}(5,1)$ reference prior (paper uses
  $\Gamma(2,2)T[0,1]$), and infection observation error uses an independence approximation across the week
  and across ages.
- **Hurdle zero probability** in the weighted path is taken empirically ($p^0$), not fitted.

These should be revisited before any scientific interpretation of the fitted transmission
parameters.
