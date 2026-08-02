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
| **1. Contact-degree model** | `ContactDegreeModel` | `NegBinAgePair`, `HurdleWeibullAgePair`, `NoContactDegree` | How the age-pair degree distribution is modelled (unweighted counts vs. duration-weighted hurdle; or not at all) |
| **2. NGM builder** | `NGMBuilder` | `MeanNGM`, `NeighbourhoodDegreeNGM`, `DiagonalMeanNGM`, `NullNGM` | How the per-capita effective contact $C^0$ is formed from the degree distribution's moments |

The Cartesian product of the first two options on each axis gives the **"four ways"** — a
$2\times2$ grid of model variants — that the preliminary analysis fits and scores side by side.

**Two baselines** (added 2026-07-30, `inst/6_null_interaction_model.md`) sit alongside that grid as
two further *pairings* of the same axes, not as a new axis:

| Model | Pairing | What it removes |
|-------|---------|-----------------|
| **No-interaction** `unweighted-negbin\|mean-diagonal` | `NegBinAgePair` × `DiagonalMeanNGM` | Off-diagonal transmission: only $\mathrm{diag}(C^\ast)$ enters the NGM, so each age group's epidemic is self-contained. Age-dependent **infectivity is pinned to $\mathrm{inf}\equiv1$** (§6.1) because a diagonal NGM identifies only the product $\mathrm{susc}_a\!\cdot\!\mathrm{inf}_a$. Reuses the NegBin **Stage-1 chains verbatim** (they carry no NGM token), so it costs no extra Stage-1 fits. |
| **Null** `no-contact\|null` | `NoContactDegree` × `NullNGM` | **All** contact data: $C^\ast$ is a fixed uniform constant $\bar c$ (§5.1), so only the transmission parameters (age-dependent susceptibility/infectivity, generation interval) drive the forecast. Stage 1 is skipped entirely. |

As of 2026-07-12
(inst/4_cut_Bayes.md) the fit is a **two-stage cut inference** (§6.0): the contact-degree likelihood
(`model_degree`, Stage 1) and the infection likelihood (`model_transmission`, Stage 2) are **separate**
probabilistic programs, with Stage 2 conditioning on Stage-1 draws and no feedback the other way.
Stage 1 is NGM-independent (Axis 2 enters only downstream), so one Stage-1 fit per degree model serves
both builders; the two axes remain fixed arguments so each stage's parameter space is well defined.

> **Formal-model update, 2026-07-30 (`inst/5_formal_pathfinder_impl.md`).** Four extensions move the
> model from *preliminary* to *formal*, all inside the same two-stage cut and still fitted with
> Pathfinder (Stage 1 switchable to NUTS later):
> **(1)** the contact-degree **dispersion becomes a two-level hierarchy** — a random term on every
> age-pair cell, drawn within its child/adult block, independent per week, non-centred, with a block
> **mean** per block × week and a **scale $\tau_t$ shared across the four blocks**, and no hyperprior
> above the blocks (§4.3, §6);
> **(2)** the hurdle **zero probability $p^0$ is now fitted** rather than taken empirically, via a
> Binomial likelihood on the roster (§4.2, §6) — weighted/Weibull path only;
> **(3)** the **generation-interval parameters are estimated** rather than fixed, following Munday 2023
> Eq 2 and Table 1 (§3.1);
> **(4)** the forecast NGM's **antibody moves to the target week $A(t_0+h)$** (§3.2, §8).
> Three deliberate **overrides of the analysis-plan docx** are recorded with these: the level-2 priors
> (§4.3), the fitted $p^0$ (§4.2) and the serial-interval discretisation (§3.1). Both stages'
> parameter spaces change, so all cached chains under the previous `…-sc` token are stale (§10).

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

  Antibody is assembled **twice**: over the $T$ window weeks (`antibody`, used in the Stage-2 fit
  loop) and separately over the $H$ **forecast target weeks** $t_0+h$ (`antibody_fc`, used by the
  forecast NGM — §3.2). The two are kept in distinct fields rather than one widened matrix so that
  `antibody[:, t]` keeps its exact meaning inside the likelihood. `weekly_antibody` fills any week it
  cannot match with $0$, which is indistinguishable from *genuinely zero antibody prevalence* — and on
  a forecast column that reads as full susceptibility and would inflate the forecast without erroring.
  It therefore **warns** on unmatched weeks (fixed 2026-07-30), loudly for forecast targets. The zero
  fill is retained as the value (a `NaN`/`missing` would propagate into the NGM); the warning is a
  tripwire for when the origin cap is raised past the `gen_dab` series' end, and is silent over the
  current 63 origins.

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

### 3.1 Generation interval (**estimated**, 2026-07-30)

The weekly generation-interval PMF $w = (w_1,\dots,w_{s_{\max}})$, $s_{\max} = 4$, is a discretised
log-normal whose **two log-parameters are estimated** as part of the Stage-2 transmission block
(`inst/5_formal_pathfinder_impl.md`; previously they were fixed). The construction follows
Munday et al. 2023 Eq 2 (`inst/pcbi.1011453.pdf`, p. 6):

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
$\mu_w = \sigma_w = 5$ days — so the estimated-GI model **nests** the fixed-GI model. (Reading
$w_\sigma$ as the sdlog instead would imply a $4.5$-day mean and $3.5$-day SD at the prior centre,
i.e. not the $5/5$ the paper states it wants.)

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

Both latents are soft-clamped inside the model — $w_\mu \in [\log\tfrac17, \log 3]$ (GI mean between
about one day and three weeks) and $w_\sigma \in [0.02, 4]$ — far outside the prior's $\pm 2$ SD, per
the codebase idiom that a clamp is an outer safety bound with the weakly-informative prior living
inside it. At either clamp $F(s_{\max}) \ge 0.55$, so the division above cannot blow up.

At the prior mean the PMF is $w \approx (0.799,\ 0.158,\ 0.033,\ 0.010)$.

> **Deliberate override of the docx.** The analysis plan discretises the serial interval as
> $w(s) = \big(F(s+1) - F(s-1)\big)\big/\big(F(S_{\max}+1) + F(S_{\max})\big)$ (citing Park 2024). We
> use Munday's Eq 2 instead: it is what the implementation was asked to follow, it is what
> `gen_interval_pmf` already computes, and its denominator is *exactly* the numerator sum — whereas the
> docx's is not, since $\sum_{s=1}^{S}\!\big(F(s{+}1)-F(s{-}1)\big) = F(S{+}1)+F(S)-F(1)$. Taken
> literally the docx weights therefore sum to $\approx 0.60$, not $1$, at the 5 d / 5 d centre (a
> constant factor that would simply be absorbed into $\gamma_{\mathrm{SAR}}$, but it is not a PMF).
> Once renormalised the two forms are in fact numerically close here —
> $(0.795, 0.159, 0.036, 0.011)$ for the docx against $(0.799, 0.158, 0.033, 0.010)$ for Eq 2 — because
> the docx numerator is just a two-lag moving sum of Eq 2's and the PMF decays sharply. So this
> override of the usual "docx wins" rule is about correctness of form, not about materially different
> weights.

$w$ is therefore a **per-draw** quantity: each Stage-2 posterior draw carries its own $(w_\mu,w_\sigma)$
and hence its own $w$, which the forecast (§8) and the fit-window diagnostic must both use.

### 3.2 Next-generation matrix

For a target week the $A\times A$ NGM is

$$
N_{ab}(t) \;=\; \gamma_{\mathrm{SAR}}\;\cdot\;\underbrace{\text{susc}_a\big(1 + (F-1)\,A_a(t)\big)}_{\text{full\_susceptibility}_a(t)}
\;\cdot\; C^\ast_{ab} \;\cdot\; \text{inf}_b ,
$$

with $\gamma_{\mathrm{SAR}}$ the **per-contact secondary attack rate** (analysis-plan reparam; it
carries the NGM level, and because $C^\ast$ is **not** normalised it reproduces the reference cell
$N_{rr}=\text{susc}_r\cdot\text{inf}_r=\gamma_{\mathrm{SAR}}$, $r=\texttt{cfg.ref\_bin}$), $\text{susc}_a$ the **relative** inherent susceptibility of group $a$
and $\text{inf}_b$ the **relative** infectivity of group $b$ — both normalised so the reference bin $r$
(**2026-07-31** set to $r=4$, "25-34"; formerly $r=1$, "2-10") is $1$ ($\text{susc}_r=\text{inf}_r=1$; the other $A-1$ bins estimated) — and $F \in (0,1)$ a **leaky**
antibody-protection factor scaling
susceptibility by the group's antibody prevalence $A_a(t)$ (at $F=1$ antibodies confer no
protection; smaller $F$ gives stronger protection). $C^\ast_{ab}$ is the per-capita effective
contact matrix produced by the NGM builder (§5.1).

> **Antibody at the target week (2026-07-30, `inst/5_formal_pathfinder_impl.md`).** Inside the
> Stage-2 *fit* loop $A_a(t)$ is the week-$t$ antibody of the $t_0$-anchored infection window, as
> before. In the **forecast** (§8) the frozen NGM instead uses $A_a(t_0+h)$ — the antibody prevalence
> at the horizon target week — mirroring the availability assumption already made for the contact
> data (whose Stage-1 window ends at $t_0+h$). The fit loop is deliberately **not** shifted: the
> infection outcomes it is evaluated against exist only up to $t_0$, so an $h$-shifted antibody there
> would pair future antibody with present infections for no gain. One consequence is that the
> likelihood pairs $t{+}h$ contacts with $t$ antibody, and only the forecast step has both at
> $t_0+h$. Note also that antibody (`gen_dab`) comes from the **same** inc2prev/CIS pipeline as the
> infection targets, whereas CoMix is an independent survey — so assuming $A(t_0+h)$ is known is a
> stronger assumption than assuming contacts at $t_0+h$ are.

> **Update 2026-07-12 (two-stage cut + γ_SAR revert, inst/4_cut_Bayes.md).** The `-gnorm`
> $C^\ast\!\to\!C^\ast/\bar S$ normalisation (added 2026-07-11 to decouple the transmissibility scalar
> from the contact scale) was **reverted**: $C^\ast$ is **not** normalised, so it feeds the NGM at its
> raw level and $\gamma_{\mathrm{SAR}}$ is again the **per-contact secondary attack rate** — it
> reproduces $N_{11}=\text{susc}_1\text{inf}_1$ directly and IS comparable across origins
> (`gamma_sar` / `log_gamma_sar`, `build_ngm(…; gamma_sar=…)`, prior $\mathrm{Normal}(\log 0.1, 1.8^2)$
> — **loosened 2026-07-13** to span $\gamma_{\mathrm{SAR}}\!\in\![0.001,10]$. The earlier $\mathrm{Normal}(\log0.27,1.05^2)$
> [90% $\gamma_{\mathrm{SAR}}\!\in\![0.048,1.52]$] together with the softclamp lower bound $\log0.02$ was **pinning
> the low-$\gamma$ configs**: the negbin$|$neighbourhood posterior median (~0.021) sat on the $\log0.02$ clamp with an
> implausibly tight CI (clamp compression). New centre $\log 0.1$ = geometric mean of $[0.001,10]$, log-SD $1.8$
> ⇒ 90% $\gamma_{\mathrm{SAR}}\!\in\![0.0052,1.93]$; the softclamp is widened to $[\log0.001,\log10]$, now at ≈±2.56σ
> (outside the band, tails ≈0.5% each), so it contains the prior and no longer biases the low tail. This invalidates
> cached `8j_s2_*` chains — regenerate them; `8j_s1_*` are $\gamma_{\mathrm{SAR}}$-independent and unaffected.)
> Concurrently the single joint fit was split into a **two-stage cut inference** (see §6.0 below): the
> $C^\ast$–$\gamma$ posterior correlation that motivated normalisation is now moot because $\gamma_{\mathrm{SAR}}$
> and the contact structure are estimated in *separate* stages. See `tasks/lessons.md`.

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
$\mathrm{NegBin}(\mu_{ij}, \phi_{ij})$, parameterised by **mean** $\mu_{ij}$ and a **per-cell
dispersion** $\phi_{ij}$ drawn hierarchically within the child/adult block pair
$(\beta(i),\beta(j))$ (§4.3), with $\mathrm{Var} = \mu + \mu^2/\phi$. The log-likelihood sums over
the empirical count distribution:

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

with $1 - P_0$ floored at $10^{-3}$ for numerical safety. $g$ conditions the neighbourhood degree
on having at least one contact (left-truncating the fitted NegBin).

### 4.2 Duration-weighted hurdle-Weibull (`HurdleWeibullAgePair`)

Here $\mu_{ij}$ denotes the mean of the **positive** duration-weighted degrees. The positive weights
$\{W\}_{ij}$ are modelled as $\mathrm{Weibull}(\kappa_{ij}, \lambda_{ij})$ with a **per-cell** shape
$\kappa_{ij}$ drawn hierarchically within the block pair (§4.3) and scale chosen so the Weibull mean
equals $\mu_{ij}$:

$$
\lambda_{ij} = \frac{\mu_{ij}}{\Gamma(1 + 1/\kappa)},\qquad
\ell_{ij} = \sum_{W \in \{W\}_{ij}} \log \mathrm{Weibull}(W;\kappa,\lambda_{ij}).
$$

The zero part is a genuine **hurdle**, and as of 2026-07-30 its probability is **fitted** rather than
plugged in empirically. Each cell carries its own zero probability with a flat prior, and the roster
supplies a Binomial likelihood:

$$
p^0_{ij,t} \sim \mathrm{Beta}(1,1),
\qquad
n^{0}_{t,i,j} \sim \mathrm{Binomial}\big(n_{t,i,j},\ p^0_{ij,t}\big),
$$

where $n_{t,i,j}$ is the roster count of §2.4 and $n^{0}_{t,i,j}$ the number of those
participant-days with no contact in the cell. So the zero part now contributes to the Stage-1
likelihood, not only to the moments. **No pooling**: the $A^2 \times T$ zero probabilities are
independent. That is safe here precisely because each is directly identified by its own Binomial with
a large $n$ — there is no funnel, unlike the dispersion scale $\tau_t$ (§4.3), so a non-centred
re-parameterisation is unnecessary. Cells with $n_{t,i,j} = 0$ contribute no Binomial term (the whole
roster row is absent — there is no trial to observe), and their $p^0$ is prior-only.

With $\mathrm{CV}_W^2 = \Gamma(1+2/\kappa)/\Gamma(1+1/\kappa)^2 - 1$ the zero-included raw moments are
unchanged in form, now evaluated at the *fitted* $p^0$:

$$
\langle k\rangle = (1-p^0)\,\mu,\qquad
\langle k^2\rangle = (1-p^0)\,\mu^2\big(1 + \mathrm{CV}_W^2\big),\qquad
g = 1 - p^0 .
$$

Two consequences. The zero factor $g$ is now a **latent**, so the neighbourhood builder's
$C^\ast$ inherits its uncertainty. And a cell where the roster exists but no contacts landed
($n>0$, $n^0 = n$) gets $p^0$ posterior just *below* $1$ rather than exactly $1$, so
$\langle k\rangle > 0$ and the $k_1 > 0$ guard in `base_contact` (§5.1) stops firing in that common
case — the guard is still required for $n = 0$ rows, where $p^0$ can reach $1$.

> **Deliberate override of the docx.** The analysis plan states: *"We do not estimate the parameter
> $p_{0,xy}^{t}$ and use the empirical value from each survey for this parameter."* Fitting it
> propagates the zero-probability uncertainty into $\langle k\rangle$, $\langle k^2\rangle$ and $g$,
> which the empirical plug-in discards. This overrides the usual "docx wins" rule and is recorded as
> such. The unweighted NegBin path (§4.1) is untouched — it is not a hurdle, and models its zeros
> directly.

### 4.3 Dispersion/shape parameterisation (**regularised horseshoe**, 2026-08-02)

The dispersion (NegBin $\log\phi$) or shape (Weibull $\log\kappa$) is a **per-age-pair-cell** quantity
drawn from a **child/adult block** distribution — the analysis plan's
$\log k^{t}_{xy} \sim \mathcal N\big(\mu^{t}_{k,XY},\ (\sigma^{t}_{k,XY})^2\big)$
(`inst/analysis_plan_heavy_tail_mean.md`, "Unweighted contact degree distribution estimation"; the
hand-drawn `inst/media/image4.png`, *"variance is hierarchical"*). Writing $d$ for either family's
log-parameter, for every ordered cell $(i,j)$ and week $t$:

$$
\log d_{ij,t} \;=\; \underbrace{m_{\ell(i,j),\,t}}_{\text{block mean}}
\;+\; \underbrace{\tau}_{\text{global}} \cdot
\underbrace{\tilde\lambda_{ij,t}}_{\text{local, regularised}} \cdot
\underbrace{z_{ij,t}}_{\text{per-cell}},
\qquad
\tilde\lambda^2_{ij,t} \;=\; \frac{c^2\,\lambda^2_{ij,t}}{c^2 + \tau^2\lambda^2_{ij,t}},
\qquad z_{ij,t} \sim \mathcal N(0,1),
$$

with the block-linear code $\ell = 2(\beta(i)-1) + \beta(j) \in \{1,2,3,4\}$ (contactor block
$\times$ contactee block), and

$$
\lambda_{ij,t} \sim \text{half-}t_{\nu_\lambda}(0,1),\qquad
c^2 \sim \text{Inv-Gamma}\!\left(\tfrac{\nu_c}{2},\ \tfrac{\nu_c s^2}{2}\right),\qquad
\tau \sim \mathcal N^{+}(0, \tau_0^2),
$$

with $(\nu_\lambda,\ \nu_c,\ s) = (3,\ 4,\ 1)$ and $\tau_0$ set **per degree family** —
`cfg.disp_rhs_local_df`, `cfg.disp_rhs_slab_df`, `cfg.disp_rhs_slab_scale`, and
`disp_tau0_prior(cfg, dm)` selecting `cfg.disp_re_scale_prior_unweighted` (NegBin $\phi$) or
`cfg.disp_re_scale_prior_weighted` (Weibull $\kappa$). Both are **0.01** as of 2026-08-02; the
remaining fields are shared. This is the **regularised
horseshoe** of Piironen & Vehtari (2017, [arXiv:1707.01694](https://arxiv.org/pdf/1707.01694),
eq. 11): $\tau$ shrinks every cell onto its block mean by default, $\lambda_{ij,t}$ lets an
individually well-evidenced cell escape, and the slab $c$ caps how far it can go.

- **Level 1 — one random term per age-pair cell.** The index runs over the $A^2 = 49$ **ordered**
  pairs, $p = (i-1)A + j$, not the 28 unordered ones: the four blocks are *directional*
  (child$\to$adult $\ne$ adult$\to$child), and self-pairs $(i,i)$ are included. (Contrast the contact
  **mean**, §5, whose reciprocity construction is defined on the 28 *unordered* pairs.) Each cell
  carries its **own** local scale $\lambda_{ij,t}$, one per cell **per week** — 49 $\times$ $T$.
- **Level 2 — block mean per block, global scale shared.** The mean $m_{\ell,t}$ is indexed by the
  block pair (still $4\times T$, unchanged priors), so the four blocks have separately estimated
  centres. The global scale $\tau$ is **one scalar for the whole fitting window** (8 fit + 4 lag =
  12 weeks), shared by all four blocks and all weeks.
- **No level 3.** The block means, $\tau$ and $c^2$ are top-level latents with fixed priors; there is
  **no** overall hyperprior pooling them.
- **Non-centred.** The composition above is written out in the model body from a standard-normal
  $z$; the centred form $\log d \sim \mathcal N(m, (\tau\tilde\lambda)^2)$ is *not* used. The centred
  version puts a sharp funnel between the scales and their 49 cells, which the Pathfinder/LBFGS path
  negotiates badly (and NUTS, the intended Stage-1 refinement, worse).

Storage shapes are kept $\le$ 2-D — $m$ is $4\times T$, $z$ is $A^2\times T$, $\lambda$ is
$A^2\times T$, $\tau$ and $c^2$ are scalars — because DynamicPPL's `generated_quantities` cannot
reconstruct a 3-D `filldist` (see `tasks/lessons.md`). $\tau$ is drawn from a **half-Normal**, so it
is already non-negative and needs no exponential/soft-clamp transform.

Stage-1 unconstrained dimension: **1580** (NegBin) and **2168** (hurdle-Weibull), from
1002/1590 before ($+588$ for $\lambda$, $+1$ for $c^2$, $-11$ as $\tau$ collapses $T\to1$).

> **Why $\tau$ moved from per-week to per-window (2026-08-02, user).** Under the previous
> `-hd` hierarchy $\tau_t$ was estimated afresh each of the 12 weeks from that week's 49 cells.
> `tasks/lessons.md` 2026-07-30 records the result: $\tau_t$ medians $\approx 2$ against a prior
> median of $0.34$, every fitted $\kappa$ pinned on its clamp, and — decisively — a four-way sweep of
> the $\tau$ prior that came out **non-monotone**, which is optimiser-path luck rather than a
> posterior. `framework.jl` already named *"a single scalar $\tau$ for the window"* as the documented
> fallback. The per-cell, per-week adaptivity that $\tau_t$ was carrying now lives in
> $\lambda_{ij,t}$, where it is a *local* scale that must be earned cell by cell rather than a global
> width applied to all 49 at once.
>
> **The assumption is testable and the evidence is already on disk.** The `-hd` chains are retained,
> and 11j's final cell plots their $\tau_t$ across the window. If $\tau_t$ varies materially week to
> week, a scalar $\tau$ is forcing individual cells to absorb a *week*-level effect, and a
> $\tau_t\cdot\tilde\lambda_{ij,t}$ variant (per-week global scale) should be reconsidered.

> **Why half-$t_3$ and not the canonical half-Cauchy (2026-08-02, user).** The textbook horseshoe
> uses $\lambda \sim C^{+}(0,1) = $ half-$t_1$. Degrees of freedom 3 keeps the same spike-and-slab
> shape with a lighter tail, for Pathfinder/NUTS geometry. The tail exponent *is* the restoring
> force: in the unconstrained coordinate $v = \log\lambda$ the half-$t_\nu$ log-density behaves as
> $-\nu\cdot v$, so **measured** $d\log p/dv \to -3.000$ for $\nu=3$ versus $-1.000$ for
> $\nu=1$ — three times the pull-back against a runaway — and the extreme tail is 227$\times$
> lighter ($q_{99.99} = 28$ vs $6366$). This is what keeps the large-$\lambda$ region
> *prior-dominated* rather than the flat, gradient-free trap the $\kappa$ soft-clamp created.

> **The slab is what un-binds the $\kappa$ clamp.** Because $\tau\tilde\lambda = c\,u/\sqrt{c^2+u^2}
> \le c$ (with $u = \tau\lambda$), the deviation obeys $|\delta| \le c\,|z|$ **exactly**. With
> $s = 1$ a fully escaped cell at $|z| = 2$ sits $\approx\pm2.2$ in log from its block mean — a
> factor $\approx 9$ — comfortably inside $[-4.3, 5]$. Prior-implied multiplier $\tau\tilde\lambda$ at
> the $\lambda$ quantiles $q_{50}/q_{90}/q_{99}/q_{99.99}$ (with $c$, $\tau$ at their prior medians
> $1.09$, $0.067$): $\approx 0.052 / 0.157 / 0.371 / 0.945$. So the default is *hard* shrinkage
> ($0.05$ in log $\Rightarrow$ within $\pm10\%$ of the block mean) and an escapee reaches $\approx
> 0.37$, comparable to the **old** $\tau$ prior's *median*.

> **$\tau_0$ is PER FAMILY, and it cannot be set analytically (2026-08-02).**
> The two families' dispersion REs live at very different scales — measured at origin 2021-05-09 h1
> under a shared $\tau_0 = 0.1$, the per-cell multiplier $\tau\tilde\lambda$ was $1.61$ (NegBin
> $\phi$) against $0.73$ (Weibull $\kappa$), and the within-block SD of log-dispersion $0.5$–$2.4$
> against $0.05$–$0.42$ — so one shared value cannot suit both. `disp_tau0_prior(cfg, dm)` is the
> single access point; reading the two fields directly is a bug, since they are tuned independently.
>
> Piironen & Vehtari's $\tau_0 = \frac{p_0}{D-p_0}\cdot\frac{\sigma}{\sqrt n}$ **does not apply
> here**: it is derived for a linear model with a residual scale $\sigma$ and sample size $n$, and a
> NegBin / hurdle-Weibull likelihood on counts and durations has neither. A prior-predictive Monte
> Carlo of $m_{\text{eff}}$ (`prior_shrinkage_reference`) is well-defined but describes only the
> *prior*, and the pilot showed the likelihood overwhelming that prior by 7–15 SDs. **The realised
> escape rate must be measured from fitted chains** — `src/tune_tau0.jl` fits one origin per
> $\tau_0$ step and reports $\tau$, $c$, $\lambda$, the multiplier, escape, $m_{\text{eff}}$,
> within-block SD and the count of cells whose $\delta$ CI excludes 0.
>
> Because `contacts_label` now encodes both $\tau_0$ values (`_tau0_tag`), successive tuning steps
> write distinct filenames and stay side by side on disk — without that, `fit_or_load_stage1`'s
> `isfile` short-circuit would silently reload the previous step's chain.

#### Numerical form (AD-load-bearing)

The multiplier is computed as $\tau\tilde\lambda = c\,u/\sqrt{c^2+u^2}$ with
$u = \tau\cdot\texttt{\_softcap}(\lambda, 10^6)$. This is not a stylistic choice. Measured with
`ReverseDiff.gradient` at the corner cases ($\checkmark$ = value **and** all partials finite):

| candidate | $u = 0$ exactly | $\lambda = 10^{200}$ | typical |
|---|---|---|---|
| $c^2\lambda^2/(c^2+\tau^2\lambda^2)$ — literal eq. 11 | $\checkmark$ | **NaN** ($\infty/\infty$) | $\checkmark$ |
| $\exp(\texttt{softclamp}(\log u))$ guard | **NaN grad** ($1/u\to\infty$); value floors at $4.7\times10^{-14}$, not 0 | $\checkmark$ | $\checkmark$ |
| $c\sqrt{w}$, $w = u^2/(c^2+u^2)$ | **NaN grad** ($d\sqrt{w}/dw\to\infty$) | $\checkmark$ | $\checkmark$ |
| **$c\,u/\sqrt{c^2+u^2}$ + `_softcap`** | $\checkmark$ | $\checkmark$ | $\checkmark$ |

The $u = 0$ corner is the one that matters most: under a working horseshoe *most* cells sit there, so
a NaN at full shrinkage would poison the entire gradient every iteration — and the two rejected
guards return a perfectly finite **value** there, failing only in the reverse pass. `_softcap(x, hi)
= hi - \texttt{softplus}(hi - x)` is `_softclamp`'s upper half; the two-sided
`_softclamp(λ, 0, hi)` must **not** be used, because near $lo = 0$ it distorts
($\texttt{softclamp}(0.7,0,10^6) = \texttt{softplus}(0.7) = 1.10$), putting a floor of $\approx\log 2$
under $\lambda$ and destroying the shrinkage.

$c = \sqrt{c^2}$ is taken without an epsilon: Inv-Gamma's $e^{-\nu s^2/2c^2}$ factor gives an
unbounded restoring gradient as $c^2\to0$, so it cannot reach the `sqrt`'s infinite-derivative point.
This is asserted in the pilot verification, not assumed.

Note `logpdf(TDist, x)` returns $-\infty$ once $x^2$ overflows ($x > 1.34\times10^{154}$) — but so
does `logpdf(Normal, x)`, the prior on every existing $z$ latent, at exactly the same point. That is
upstream of the composition, shared by the whole model, and not something `_softcap` can address.

**Pathfinder initialisation.** `_stage1_init` pins $\lambda = 1$ and $c^2 = s^2$ rather than drawing
them from their priors. These are the **exact prior modes in the unconstrained coordinates**
Pathfinder optimises, for any $\nu$ and $s$: for $v = \log\lambda$, $\ d\log p/dv = 0
\Leftrightarrow \nu + e^{2v} = (\nu+1)e^{2v} \Leftrightarrow \lambda = 1$; for $w = \log c^2$,
$\ d\log p/dw = 0 \Leftrightarrow e^{-w} = 1/s^2 \Leftrightarrow c^2 = s^2$. A *prior draw* of
$\lambda$ would be far worse than merely diffuse: half-$t_3$ over 588 cells routinely yields values
in the tens, putting several cells straight into the slab with a full-strength random effect before
the likelihood has constrained anything.

The **composed** value is soft-clamped (not the block mean alone), keeping the mode interior and
avoiding Weibull/exponential underflow: $\log\kappa \in [-4.3,5]$, i.e. $\kappa \in [0.0136,148]$;
$\log\phi \in [-4,5]$, i.e. $\phi \in [0.018,148]$.

> **$\kappa$ clamp WIDENED $[-3,3]\to[-4.3,5]$, 2026-07-30.** With the per-cell random effect added,
> the old bound bound *hard*: every fitted $\kappa$ sat exactly on $0.0498$ — the clamp-compression
> signature — and the flat region it creates let the Stage-1 LBFGS path run away, producing block
> means near $-441$ (≈900 prior SDs from $\mathcal N(0,0.5)$, i.e. a diverged optimiser rather than a
> posterior). Widening moves the flat region far enough out that the likelihood keeps steering.
> The horseshoe's $|\delta| \le c|z|$ bound now keeps the composed value away from it structurally.
>
> **The floor is $\approx-4.446$, set by $\Gamma(1+2/\kappa)$ — not by $\lambda$.** $\Gamma$ overflows
> above argument $\approx 171.6$. There are **two** $\Gamma$ calls on this path and the *second* is
> the binding one:
>
> | quantity | $\Gamma$ argument | overflows at | $\log\kappa$ floor |
> |---|---|---|---|
> | scale $\lambda_W = \mu/\Gamma(1+1/\kappa)$ | $1+1/\kappa$ | $\kappa \lesssim 0.00586$ | $-5.14$ |
> | $\mathrm{CV}^2 = \Gamma(1+2/\kappa)/\Gamma(1+1/\kappa)^2$ | $1+2/\kappa$ | $\kappa \lesssim 0.01172$ | $\mathbf{-4.446}$ |
>
> `_weibull_moments` computes **both**, so the tighter floor governs. At $\log\kappa=-5$ the scale is
> still finite ($1.5\times10^{-263}$) but $\mathrm{CV}^2 = \infty/\infty = $ **NaN**, which propagates
> into $\langle k^2\rangle$ and aborts the fit. $-4.3$ leaves ≈0.15 in $\log\kappa$ of margin.
> The NegBin $\phi$ clamp has no such constraint (its moments are polynomial in $1/\phi$) and is
> unchanged at $[-4,5]$.

#### Reading the shrinkage (11j)

Per cell and week, the **shrinkage factor** is

$$\text{shrink}_{ij,t} = \frac{c^2}{c^2+\tau^2\lambda^2_{ij,t}} = 1 - \Big(\frac{\tau\tilde\lambda}{c}\Big)^2 \in [0,1],
\qquad m_{\text{eff},t} = \sum_{ij}\big(1-\text{shrink}_{ij,t}\big),$$

1 meaning fully shrunk onto the block mean and 0 meaning escaped into the slab.

> ⚠ This is the *model-internal* analogue of the paper's $\kappa_j$, **not** the same number. Piironen
> & Vehtari's $\kappa_j = 1/(1+n\sigma^{-2}\tau^2\tilde\lambda_j^2)$ measures shrinkage against the
> **data** information $n\sigma^{-2}$, which requires a Gaussian-likelihood approximation this model
> does not have; ours measures it against the **slab** scale $c^2$, which is exact. Both run 0→1 in
> the same direction; do not quote one as the other.

> ⚠ **`shrink` has a non-zero prior baseline** ($\approx 0.998$ at the prior medians), so
> $m_{\text{eff}}$ has a prior floor of order 1 per 49-cell week, **not** 0. `prior_shrinkage_reference`
> (11j) Monte-Carlos the prior-predictive band through the identical formulas, and every
> $m_{\text{eff}}$ panel draws it. Quoting $m_{\text{eff}}$ in absolute terms would badly overstate how
> many cells have genuinely escaped.

11j's panels: the window-level scales vs their priors (`plot_rhs_globals`, 10j); $m_{\text{eff}}$ per
week vs the prior band (`plot_meff_over_weeks`); the 7$\times$7 escape map
(`plot_shrinkage_cells`); the ranked RE deviation with the $\pm c$ ceiling (`plot_re_ranked`); escape
vs per-cell sample size (`plot_shrinkage_vs_n` — the headline check, since escape concentrated at
*low* $n_{\text{pos}}$ is prior noise leaking in where there is no data); and the old-vs-new
within-block SD (`plot_within_block_sd`), which reads both cache generations off disk.

> **Deliberate override of the docx.** The plan specifies $\mu_{k,XY} \sim \mathrm{Gamma}(2,1/4)$ and
> $\sigma_{k,XY} \sim \mathrm{Gamma}(2,1/2)$ — a per-block mean *and* a per-block SD, both with
> positive support. Three departures: **(i)** the Gamma on the mean would force the block **mean of
> $\log k$** above zero, i.e. $k > 1$, against the fitted values ($\phi \approx 0.28$,
> $\kappa \approx 0.9$–$1.0$; `tasks/lessons.md`), so the block means keep their existing Normal
> priors; **(ii)** the scale is shared across blocks rather than per-block — a per-block scale is
> estimated from that block's cells alone, and child$\to$child holds only $2\times2 = 4$ ordered
> cells; **(iii)** the shared scale is a *regularised horseshoe* (global $\times$ local $\times$ slab)
> rather than a single half-Normal SD. This overrides the usual "docx wins" rule and is recorded as
> such.

> **Relation to the earlier generations.** A hierarchical dispersion was added on 2026-07-11 and
> reverted the same day, then restored on 2026-07-30 as the `-hd` model: a per-week half-Normal
> $\tau_t$ shared across blocks, prior $\mathcal N^{+}(0,0.5^2)$. That generation is the *previous*
> one, not the current model; its chains remain on disk under `CONTACTS_TOKEN_HD` for the 11j
> comparison. The slab scale $s = 1$ deliberately reproduces roughly that generation's RE width — but
> now only for the escaped minority, with everyone else shrunk hard.


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
leaves the implied $R$ unchanged to $0.3\%$.) $\bar c$ is computed **once per origin** from that
origin's focal weeks and reused for every horizon, which is what "the used average number should be
fixed while forecasting" requires.

---

## 6. The model (two-stage cut)

### 6.0 Two-stage cut inference (inst/4_cut_Bayes.md)

The former single joint `@model` was split into a **two-stage cut inference**. **Stage 1**
`model_degree(dm, ds, pop, cfg)` fits the contact-degree GP **alone** — it carries only the contact
block below and returns the per-week raw moments $(\langle k\rangle_t, \langle k^2\rangle_t, g_t)$, so
it is **NGM-independent** (one fit serves both builders; the builder is applied downstream via
`contact_star`). **Stage 2** `model_transmission(\{C^\ast_t\}, wd, w, cfg)` fits the transmission
block **alone**, conditioning on a *fixed* $\{C^\ast_t\}$ built from one Stage-1 posterior draw.

Stage-1 uncertainty is propagated by a cut Monte Carlo: draw $M=100$ Stage-1 posterior samples; for
each, form $\{C^\ast_t\}$ and run Stage 2 keeping $D=100$ draws; **pool** the $M\times D = 10{,}000$
infection draws as the predictive distribution scored by WIS. There is **no feedback** from the
infection likelihood to the contact GP (the "cut"): $\mu$ is estimated purely from the contact data,
so it no longer depends on the NGM builder. The two blocks' sampling statements are unchanged from
the joint model and are given below as the two stages.

**The model estimates contacts per week, temporally smoothed.** A single **separable
spatio-temporal GP** (§5) governs all $T$ window weeks — sharing the spatial kernel $L_{\text{age}}$,
the temporal kernel $L_{\text{time}}$, the length-scales
$\rho_{\text{diag}}, \rho_{\text{gap}}, \rho_{\text{time}}$ and the scales $\eta, \sigma_c$ — so the
weekly log-rate fields are **temporally correlated** rather than independent draws. Each week yields
its own $C^\ast_t$ (through the per-week level $c_t$, structure-field column $R_{\cdot,t}$, and
per-week hierarchical dispersion $\phi_{ij,t}$/$\kappa_{ij,t}$, §4.3), and the transmission NGM $N(t)$ therefore varies in time through
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
m_{t} &\sim \mathcal N(0,\sigma_d^2)^{4}, &
\tau_{t} &\sim \mathcal N^{+}(0,\ 0.5^2), \\
z^{d}_{t} &\sim \mathcal N(0,1)^{A^2}, &
p^{0}_{t} &\sim \mathrm{Beta}(1,1)^{A^2}
& & (\sigma_d = 0.5\ \text{Weibull},\ 1.0\ \text{NegBin};\ A^2 = 49\ \text{cells}).
\end{aligned}
$$

$p^{0}_{t}$ is declared on the **weighted/Weibull path only** (§4.2); the NegBin path's parameter
space does not contain it, so the two degree models' Stage-1 chains now differ in shape.

with the derived level $c_t = c + \sigma_c (L_{\text{time}} z_c)_t$ and structure field
$R = \eta\,(L_{\text{age}}\, z\, L_{\text{time}}^{\!\top})$ giving the week-$t$ log-rate
$r_{p,t} = c_t + R_{p,t}$ (§5), the contact-degree log-likelihood of §4 injected via
`Turing.@addlogprob!`, and
$C^\ast_t = $ `contact_star`$(nb, \langle k\rangle_t, \langle k^2\rangle_t, g_t)$. The dispersion is
**hierarchical per week** (§4.3): the block mean $m_{\ell,t}$ (a $4\times T$ array) and the shared
scale $\tau_t$ (length $T$) combine with the per-cell standard-normal $z^{d}_{p,t}$ (an $A^2\times T$
array) as $\log d_{ij,t} = m_{\ell,t} + \tau_t\,z^{d}_{p,t}$, $\ell = 2(\beta(i)-1)+\beta(j)$,
$p = (i-1)A+j$ — non-centred, and re-drawn each week with **no** temporal smoothing (unlike the mean
field). The block-mean priors $\mathcal N(0,\sigma_d^2)$ are unchanged from the block-only model, so
$\tau_t\to 0$ recovers it exactly; the scale takes the half-Normal $\mathcal N^{+}(0,0.5^2)$
(`disp_re_scale_prior`), whose marginal RE SD of $\approx 0.5$ in log puts a typical cell within
$\approx[0.37,\,2.7]\times$ its block mean at $\pm2$ SD.

*Stage 2 — transmission block (per-contact $\gamma_{\mathrm{SAR}}$ + reference-normalised susc/inf,
non-centred; conditions on the fixed $\{C^\ast_t\}$ of one Stage-1 draw):* susceptibility and
infectivity are **relative** to the reference bin $r=\texttt{cfg.ref\_bin}$ (**2026-07-31** set to
$r=4$, "25-34"; formerly $r=1$, "2-10"), fixed to $1$; only the $A-1$
non-reference offsets are estimated (the fixed $1$ is **spliced in at position $r$**, not prepended),
and the per-contact secondary attack rate $\gamma_{\mathrm{SAR}}$
carries the level (it replaces the old confounded $\mu_s,\mu_i$ pair).

$$
\begin{aligned}
\log\gamma_{\mathrm{SAR}} &\sim \mathcal N(\log 0.1,\, 1.8^2), &&
\gamma_{\mathrm{SAR}} = \exp(\operatorname{softclamp}(\log\gamma_{\mathrm{SAR}},\log 0.001,\log 10)),\\
\sigma_s &\sim \mathcal N^+(0, 0.25^2), & z_s &\sim \mathcal N(0,1)^{A-1}, &
\text{susc} &= \operatorname{splice}_r\big(1,\ \exp(\operatorname{softclamp}(\sigma_s\,z_s, \log 0.05, \log 20))\big),\\
\sigma_i &\sim \mathcal N^+(0, 0.25^2), & z_i &\sim \mathcal N(0,1)^{A-1}, &
\text{inf} &= \operatorname{splice}_r\big(1,\ \exp(\operatorname{softclamp}(\sigma_i\,z_i, \log 0.05, \log 20))\big),\\
F &\sim \mathrm{Beta}(5,1), & \sigma_{\text{inf}} &\sim \mathcal N^+(0.05, 0.025^2), & & \\
w_\mu &\sim \mathcal N(-0.6830,\ 0.1366^2), & w_\sigma &\sim \mathcal N^{+}(0.6931,\ 0.1386^2), &
w &= \text{Eq. §3.1}\big(w_\mu, w_\sigma\big).
\end{aligned}
$$

With $C^\ast$ **un-normalised** (§3.2 update, 2026-07-12), $\gamma_{\mathrm{SAR}}$ is the per-contact
secondary attack rate: it reproduces the reference cell $N_{rr}=\text{susc}_r\cdot\text{inf}_r$
directly and is comparable across origins. **The reference bin $r$ is a gauge**: the NGM likelihood is
invariant to it (rescaling all $\text{susc}$ by $c$ and $\gamma_{\mathrm{SAR}}$ by $1/c$ leaves $N$
unchanged), so the **2026-07-31** switch $r=1\to4$ ("2-10"$\to$"25-34", user request) acts *only*
through the priors — which bin is pinned to 1 vs. carries the log-offset, and what
$\gamma_{\mathrm{SAR}}=N_{rr}$ anchors to. 25-34 is a large, well-mixed, well-sampled adult group
(Munday/Davies convention); anchoring on children (2-10) — an extreme, poorly-identified,
antibody-sparse bin — was the weaker choice. Note the switch re-anchors $\gamma_{\mathrm{SAR}}$ to the
25-34 cell, so its posterior is no longer directly comparable to pre-switch runs (the wide
$\mathcal N(\log0.1,1.8^2)$ prior absorbs the level shift). Its prior was originally calibrated by reading 18
pre-`-gnorm` `temporal` chains (both degree models × 9 origins): $\text{susc}_1\!\cdot\!\text{inf}_1$
had median $0.33$, log-SD $0.56$ ⟹ $\mathcal N(\log 0.33, 0.56^2)$. It was **loosened twice since**
and now stands at $\mathcal N(\log 0.1, 1.8^2)$ with softclamp $[\log 0.001,\log 10]$ — see the
§3.2 update box (2026-07-13); the calibration above now informs only the centre. Age variation in inherent susceptibility/infectivity
admits genuine age variation, and the **soft-clamp** is a looser safety bound: the offset scale
$\sigma_{s,i}\sim\mathcal N^+(0,0.25^2)$ (**2026-07-31 set to a mode-at-0 half-normal**, user request —
the conventional weakly-informative scale that lets the age profile **shrink to no-variation**
($\text{susc},\text{inf}\to1$) when the data are silent, rather than the previous
$\mathcal N^+(0.5,0.25^2)$ which sat the mode away from 0 and *asserted* age spread; upper reach
$\approx$ unchanged, marginal SD still $\le\!\sim0.5$ ⟹ realistic
$\text{susc},\text{inf}$ well inside the clamp), and each non-reference
log-offset $\sigma\,z$ is soft-clamped to $[\log 0.05,\log 20]\approx[-3,3]$ (**also loosened
2026-07-13 from $[\log 0.2,\log 5]$**), **hard-bounding** $\text{susc},\text{inf}\in[0.05,20]$ — wide
enough that realistic profiles never touch it (the prior's $\pm2$ SD sits at $\pm1$, clamp at
$\sim\pm6$ SD), but still capping a stray Pathfinder draw's supercritical NGM at the source. Prior
mass $\subset$ clamp ⟹ profiles interior and undistorted. The $A-1$ non-reference offsets are **independent per age bin**
(offset $\sigma\,z$, $z\sim\mathcal N(0,1)^{A-1}$ iid) — **no cross-bin smoothing**. (A
shared-length-scale squared-exponential GP over the age-bin index, $\sigma\,L_{si}z$, previously
correlated neighbouring bins; it was **removed 2026-07-13** (user request), along with the length-scale
latent $\rho_{si}$. Because that kernel had unit diagonal, dropping it leaves each bin's marginal SD
$=\sigma_{s,i}$ unchanged, so the $\sigma_{s,i}$-driven spread (now the shrink-to-1
$\mathcal N^+(0,0.25^2)$ band) and the $[0.05,20]$ clamp are preserved — the age
profile is simply rougher. See the GP→RW1→RW2→GP→none history in `tasks/lessons.md`.)
The **generation-interval** latents $(w_\mu, w_\sigma)$ live here too (§3.1) — they enter only the
renewal, so the cut keeps them clear of the contact GP. Stage 2 returns the generated quantities
$(\text{susc}, \text{inf}, F, \gamma_{\mathrm{SAR}}, \sigma_{\text{inf}}, w_\mu, w_\sigma)$; Stage 1
returns the raw moments $(\{\langle k\rangle_t\}, \{\langle k^2\rangle_t\}, \{g_t\})$.

**Fixed infectivity for the no-interaction model** (2026-07-30, `inst/6_null_interaction_model.md`).
`model_transmission` takes the NGM builder as a trailing argument purely to consult the trait
`fix_infectivity(nb)`. When it is `true` — only for `DiagonalMeanNGM` — the block

$$\sigma_i \sim \mathcal N^+(0,0.25^2),\quad z_i\sim\mathcal N(0,1)^{A-1},\quad
\text{inf} = \operatorname{splice}_r(1, \exp(\operatorname{softclamp}(\sigma_i z_i,\cdot)))$$

is **not sampled at all** and $\text{inf}\equiv \mathbf 1$. The reason is exact non-identifiability,
not a modelling preference: with $C^\ast$ diagonal the NGM is diagonal, so

$$N_{aa} = \gamma_{\mathrm{SAR}}\cdot\text{susc}_a\big(1+(F-1)A_a(t)\big)\cdot C^\ast_{aa}\cdot\text{inf}_a ,$$

and $\text{susc}_a$ and $\text{inf}_a$ enter **only** through their product. Pinning
$\text{inf}\equiv1$ puts the whole age profile in $\text{susc}$. (Leaving $\sigma_i,z_i$ in the
program as unused latents would leave them prior-driven and pollute the Pathfinder approximation,
so they are dropped from the parameter space rather than merely ignored.) This is *distinct* from —
and additional to — the reference-bin normalisation $\text{susc}_r=\text{inf}_r=1$ that all
variants share.

*Infection likelihood*, over the fit weeks $t = s_{\max}+1,\dots,T$ (the first $s_{\max}$ weeks serve
only as renewal history). For each age $a$,

$$
\hat I_a(t) = \big[N(t)\, \textstyle\sum_{s} w_s\, I(t-s)\big]_a,\qquad
I_{a,t} \sim \mathcal N\!\Big(\hat I_a(t),\ \sigma_{a,t}^2\Big),\quad
\sigma_{a,t}^2 = (\sigma_{\text{inf}}\, I_{a,t})^2 + (s^{I}_{a,t})^2,
$$

i.e. the observation SD combines a multiplicative process-noise term $\sigma_{\text{inf}} I_{a,t}$
in quadrature with the inc2prev estimate SD $s^{I}_{a,t}$. In the cut, the $\{C^\ast_t\}$ here are a
**fixed** input (one Stage-1 draw run through `contact_star`), not sampled.

**The two stages' log-likelihoods (no longer one target).** Under the cut the contact and infection
terms live in **separate** models:

$$
\log L_{\text{Stage 1}} = \underbrace{\sum_{t=1}^{T}\ \sum_{i,j=1}^{A} \ell^{(t)}_{ij}}_{\text{contact degree (§4)}},
\qquad
\log L_{\text{Stage 2}} = \underbrace{\sum_{t=s_{\max}+1}^{T}\ \sum_{a=1}^{A}
\log \mathcal N\!\big(I_{a,t}\,;\ \hat I_a(t),\ \sigma_{a,t}^2\big)}_{\text{infection renewal}},
$$

where $\ell^{(t)}_{ij}$ is the §4.1 NegBin (over the count histogram) or §4.2 Weibull-hurdle cell
log-likelihood, evaluated at the week-$t$ contact mean $\mu_{ij,t}$ and its per-cell dispersion
$d_{ij,t}$ (§4.3). On the hurdle path $\ell^{(t)}_{ij}$ now has **two** terms — the Weibull over the
positive duration-weighted degrees *plus* the Binomial zero term
$\log\mathrm{Binomial}\big(n^0_{t,i,j};\,n_{t,i,j},\,p^0_{ij,t}\big)$ of §4.2, omitted where
$n_{t,i,j}=0$. Stage 1's contact term runs over **all** $T$ window weeks;
Stage 2's infection term (conditioning on that stage-1 draw's fixed $\{C^\ast_t\}$) uses only the
$t>s_{\max}$ fit weeks. Because the two are fit separately, the infection likelihood does **not**
feed back into the contact GP — this is the "cut" (§6.0).

Every log-scale latent that feeds an exponential ($\log\rho, \log\eta, \log\kappa, \log\phi$, and
the per-cell rate) is clamped inside the model body so that aggressive optimiser/Pathfinder steps
cannot underflow (e.g. Weibull scale $\to 0$); the clamps are wide enough that the posterior mode is
interior and gradients are unaffected.

**Per-horizon window offset (forecasting use).** Although the contact and infection blocks share the
index $t = 1,\dots,T$, they need not span the same calendar weeks. In forecasting (§8) both stages are
re-fit once **per horizon** $h$: Stage 1's contact-degree window is slid forward to end at $t_0 + h$
(cached as `8j_s1_<degree>_<contacts>_<origin>_h<h>.jld2`, NGM-independent), and Stage 2 conditions on
that stage's $\{C^\ast_t\}$ against the infection/renewal window anchored at the origin $t_0$ (`wd`
fixed; the pooled draws cached as `8j_s2_<degree>_<ngm>_<contacts>_<origin>_h<h>.jld2`). Thus for every
horizon the contact term is fit over weeks offset $h$ **ahead** of the infection term — the age-pair
degree distribution is observed **at** the target week $t_0+h$ (contemporaneous with it), whereas
infections and the fit-loop antibody are anchored at $t_0$. The two windows never coincide (even at
$h=1$ the contacts lead the infection block by one week). The forecast NGM additionally takes its
antibody at $t_0+h$ (§3.2, §8).

---

## 7. Inference (two-stage cut)

For one $(dm, nb, \text{origin}, h)$:

1. **Stage 1** — `fit_stage1(dm, ds, pop, cfg)` fits the contact GP by **Pathfinder** (default; or
   NUTS when `cfg.stage1_use_nuts = true`, initialised from the Pathfinder mean). `stage1_moment_draws`
   then takes $M = $ `cfg.n_stage1_post` $= 100$ posterior draws' raw moments (deterministic even-grid
   subsample).
2. **Stage 2** — `fit_stage2_pooled(nb, moment_draws, wd, cfg)` forms $\{C^\ast_t\}$ for each Stage-1
   draw (via `contact_star`) and Pathfinder-fits `model_transmission` conditioning on it, keeping
   $D = $ `cfg.n_stage2_draws` $= 100$ draws. The $M\times D = 10{,}000$ pooled draws
   $(\gamma_{\mathrm{SAR}}, \text{susc}, \text{inf}, F, \sigma_{\text{inf}}, w_\mu, w_\sigma,
   \text{post\_index}, \{C^\ast_{\text{end}}\})$ are the infection predictive.

**Null-model bypass.** `stage2_inputs(dm, …)` is the single fork between the two paths. When
`needs_stage1(dm)` is `false` (i.e. `NoContactDegree`) it skips `build_degree_stats`,
`fit_or_load_stage1` and `stage1_moment_draws` altogether and returns **one** constant-$C^\ast$
"draw" from `null_moment_draws(\bar c, A, T_n)`, taking the full $M\times D = 10{,}000$ samples from
that single Stage-2 fit instead of $D$ from each of $M$. $M=1$ is deliberate: the null model has no
contact-degree uncertainty to propagate, so repeating $100$ identical Pathfinder fits would inject
only fit-to-fit approximation noise, at $100\times$ the cost. The pooled draw count — and hence
comparability of WIS/log score — is preserved. `prefit_stage1!` drops such degree models up front,
so **no `8j_s1_no-contact_*` file is ever written**.

The random seed is `cfg.seed = 1236` (Stage-2 draw $m$ uses `Xoshiro(seed + m)`). Fits are mutually
independent: `prefit_stage1!` fans the Stage-1 chains out over Julia threads (BLAS pinned, warm-compile
first), then `prefit_stage2!` runs each Stage-2 cell's 100 per-draw fits under the same concurrency cap;
origins are processed sequentially (bounded memory). Artefacts cache to
`dt_intermediate/8j_s1_<degree>_<contacts>_<origin>_h<h>.jld2` (Stage 1) and
`8j_s2_<degree>_<ngm>_<contacts>_<origin>_h<h>.jld2` (Stage 2); cached files are skipped ⟹ resumable.

**The 8j notebook uses Pathfinder for both stages (`STAGE1_USE_NUTS = false`); switching Stage 1 to
NUTS is the intended later refinement.**

> **Stage-1 dimension after the hierarchical dispersion and fitted $p^0$ (2026-07-30).** The contact
> block gains $T = 12$ shared scales $\tau_t$ and $A^2 T = 588$ per-cell standard normals, and — on
> the **weighted path only** — a further $A^2 T = 588$ zero probabilities. So the Stage-1 latent
> count goes from $\approx 400$ to $\approx 1000$ for `NegBinAgePair` ($\approx 2.5\times$) and
> $\approx 1590$ for `HurdleWeibullAgePair` ($\approx 4\times$). **The two degree models' parameter
> spaces now differ materially in size**, so per-fit cost and memory diverge between them — the
> Weibull fits are the binding constraint. Pathfinder's LBFGS path is correspondingly slower and more
> prone to wandering, and the planned NUTS switch will be substantially more expensive.
>
> `fit_concurrency`'s `mem_per_fit_gib = 1.0` is an unmeasured guess and should be set from an actual
> fit (separately per degree model, given the above). Note also that `_mem_available_gib` reads
> `/proc/meminfo`, which does not exist on macOS — the darwin path silently falls through to
> `Sys.free_memory()`, which reports only *truly free* pages (macOS keeps these near zero), so the
> memory cap can collapse and force fully serial fitting without any warning. Measure the returned
> concurrency before committing to a full refit.

---

## 8. Forecasting: the contact-updated pooled iterate

The notebook forecasts with `two_stage_forecast`, the *contact-updated* iterate over the **pooled**
draws. For a baseline origin $t_0$, the infection series is **frozen at $t_0$**, while the
contact/degree window slides: for horizon $h$ the Stage-1 degree window ends at $t_0 + h$ and the
Stage-2 pooled draws for $(dm, nb, t_0, h)$ are reloaded (or fit). Per pooled draw $d$ (from Stage-1
draw $m = $ `post_index[d]`) a fresh NGM is formed from that draw's origin-week $C^\ast$
(`Cstar_end[m]`) with **antibody at the target week $A(t_0+h)$** (§3.2, changed 2026-07-30 from
$A(t_0)$) and its Stage-2 infection parameters, and a single renewal step is taken with **that draw's
own** generation interval $w^{(d)} = $ §3.1$(w_\mu^{(d)}, w_\sigma^{(d)})$,

$$
\hat I_a(t_0+h) = \Big[N\, \textstyle\sum_{s=1}^{s_{\max}} w^{(d)}_s\, I(t_0+h-s)\Big]_a,
$$

(the renewal-weighted lag sum is therefore computed **inside** the draw loop, not once per horizon
as it was under a fixed $w$)

with observation noise $\sigma = \max(\sigma_{\text{inf}}\hat I_a, 10^{-6})$ added per draw. The
renewal lags use observed infections up to $t_0$ plus the **mean** predictions of the intervening
weeks (a deterministic mean-plugged lag; per-draw coherence across horizons is undefined). This
produces an $A \times H \times N$ array of posterior-predictive draws, where
$N = $ `n_stage1_post` $\times$ `n_stage2_draws` $= 10{,}000$ is the pooled predictive — fed
directly to WIS (the draw axis is pooling-agnostic to `scoring.jl`).

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

### 9.1 Log score (2026-07-30, inst/6_null_interaction_model.md)

Reported **alongside** WIS. Note the naming trap: the "log-scale WIS" above is WIS computed after a
log *transform* of forecasts and observations; the **log score** is the logarithmic *scoring rule*
$-\log f(y)$ of the predictive density. `scoringutils` defines it only for the **sample** forecast
class (`as_forecast_sample` ⟹ `scoringRules::logs_sample`, a Gaussian-KDE estimate) — it is *not*
available for the quantile class the WIS path uses — so `scoring.jl` carries a second R path:

- `to_sample_long` emits one row per (age × horizon × retained draw), and `score_logs` scores
  **one origin at a time**. `score()` returns one row per forecast unit, so the accumulated per-unit
  table stays small while the per-origin sample table handed to R is ~$10^5$ rows; a single global
  table would be ~$10^8$.
- Two sanitisations, both **counted and reported** rather than silently applied: (i) non-finite
  draws are dropped — `two_stage_forecast` deliberately keeps $\pm\infty$ draws, which the KDE
  cannot consume, and dropping them narrows the retained fan; (ii) draws are thinned to
  `n_sample` (default $1000$) per cell on a deterministic even grid.
- Scored on both scales, mirroring `score_wis`'s five output frames (`res/8j_logscore_*.csv`). The
  log-scale copy is built **explicitly** (`log(pmax(\cdot,0)+1)`) rather than with
  `transform_forecasts(log_shift)`, because individual sample draws can be negative
  ($\text{draw}=\hat I + \sigma\varepsilon$) and `log_shift` would return `NaN`; $\mathrm{pmax}(\cdot,0)$
  censors at the model's support. The **headline scale is natural**, since the log-scale variant
  additionally depends on that censoring.
- Relative log score is reported as a **difference** vs the reference model, never a ratio: a log
  score is not sign-stable (it goes negative wherever the predictive density exceeds 1).

The relative-skill reference (`REF_MODEL`, for both relative WIS and relative log score) is the
**no-interaction** model `unweighted-negbin|mean-diagonal`. Before it existed,
`unweighted-negbin|mean` stood in for Munday's no-interaction reference.

---

## 10. The 8j experiment

The notebook (`8j_preliminary_forecast.ipynb`) runs the full grid:

- **Configuration** (`FrameworkConfig`): $d_{\max}=240$, $w_{\text{group}}=2.5/240$,
  $s_{\max}=4$, `n_fit` $=8$, `horizons` $=1{:}4$, seed $=1236$, generation-interval prior centre
  `gen_mean_days`/`gen_sd_days` $=5/5$ days with `gen_prior_rel_sd` $=0.2$ (§3.1 — these now set the
  *prior* on the estimated $w_\mu,w_\sigma$, not a fixed $w$), `child_bins` $=2$, quantiles $0.05{:}0.05{:}0.95$, cut sizes
  `n_stage1_post` $=100$ / `n_stage2_draws` $=100$ (⟹ 10 000 pooled), `stage1_use_nuts` $=$ `false`,
  GP priors $\log\rho_{\text{diag}},\log\rho_{\text{gap}}\sim\mathcal N(\log15,0.5^2)$ (shared
  prior for both diagonal length-scales), $\log\eta\sim\mathcal N(0,0.5^2)$,
  $\log\rho_{\text{time}}\sim\mathcal N(\log4,0.5^2)$ (`gp_time_len_prior`, weeks),
  $\log\sigma_c\sim\mathcal N(0,0.5^2)$ (`gp_level_scale_prior`),
  $\log\gamma_{\mathrm{SAR}}\sim\mathcal N(\log0.1,1.8^2)$ (`gamma_sar_prior`; loosened 2026-07-13 to span $\gamma_{\mathrm{SAR}}\!\in\![0.001,10]$, 90% $\in[0.0052,1.93]$, softclamp $[\log0.001,\log10]$),
  **hierarchical per-week dispersion** (block mean + shared per-week scale
  `disp_re_scale_prior` $=\mathcal N^{+}(0,0.5^2)$ + per-cell random term, §4.3),
  **fitted hurdle zero probability** $p^0 \sim \mathrm{Beta}(1,1)$ per cell × week on the weighted
  path (§4.2),
  and **per-week temporally-smoothed contact estimation** (one separable spatio-temporal age-pair GP
  across the window weeks). The chain-cache `contacts_label` is `"temporal-gsar-cut-sc-hd-p0-gi"` — the
  `-gsar-cut` tag marks the **two-stage cut** split with per-contact $\gamma_{\mathrm{SAR}}$ and
  **un-normalised** $C^\ast$ (the S̄-normalising `-gnorm` chains, differently-scaled `log_gamma`, are
  disjoint and left on disk); `-sc` the Stage-2 susc/inf soft-clamp; and, added 2026-07-30, `-hd` the
  **h**ierarchical **d**ispersion, `-p0` the fitted hurdle zero probability (both Stage 1) and `-gi`
  the estimated **g**eneration **i**nterval (Stage 2). Both stages' parameter spaces changed, and the
  token is shared by `stage1_path` and `stage2_path`, so the 504 `8j_s1_*` and 1008 `8j_s2_*` files
  under `…-sc` are stale — they are left on disk and simply never reloaded, and a full refit is
  required. The derived 9j caches (`9j_rt_*`, `9j_relrt_*`, `9j_obsrt_*`) and any stored forecast
  assembly must also be regenerated, because the forecast changed (per-draw $w$, antibody at
  $t_0+h$) even where the pooled draws did not.
- **Rolling origins.** The forecast origin is rolled weekly over the whole *available period* the
  current data support (`available_forecast_origins`): bounded below by the first inc2prev week
  ($2020$-$08$-$02$) plus the 12-week fit/lag lookback, and above by the last CoMix contact week
  minus $\max h$ weeks (the iterate needs contacts out to $t_0 + 4$).
- **Four ways ++ two baselines.** At each origin the four combos
  $\{$`NegBinAgePair`, `HurdleWeibullAgePair`$\} \times \{$`MeanNGM`, `NeighbourhoodDegreeNGM`$\}$,
  plus `NegBinAgePair`×`DiagonalMeanNGM` (no-interaction) and
  `NoContactDegree`×`NullNGM` (null), are fit (two-stage) and forecast $1$–$4$ weeks ahead. The run
  is memory-bounded and resumable: per origin only that origin's four degree windows are built
  (reusing a single raw read of the CoMix tables), Stage-1 GP chains (still 8 = 2 degree × 4
  horizons — the no-interaction model reuses the NegBin chains and the null needs none) and
  Stage-2 pooled files (24 = 6 combos × 4 horizons) are pre-fit, forecasts are assembled, and the
  degree data discarded. Adding the baselines therefore costs one extra model's worth of Stage-2
  fits (no-interaction) plus 4 cheap fits per origin (null), and **invalidates no cached file** —
  their tokens are new. The `9j_assembly_*` cache does self-invalidate on the model-label change.
- **Outputs.** Quantile scores (`res/8j_scores_by_model*.csv`) and diagnostic figures: WIS by
  horizon, four-ways WIS bars, WIS over the forecast period, forecast-vs-observed fans by origin,
  and fitted transmission structure (susceptibility/infectivity ratios to a reference group and the
  three GP length-scales $\rho_{\text{diag}}, \rho_{\text{gap}}, \rho_{\text{time}}$ over time).
  Added 2026-07-30: *(i)* a **generation-interval** panel (9j) showing the posterior of the GI mean
  and SD **in days** (back-transformed from $w_\mu,w_\sigma$) and of $w=(w_1..w_4)$, overlaid on the
  prior, by model and over the rolling origins — the identifiability check for the
  $w$–$\gamma_{\mathrm{SAR}}$ confounding noted in §11; *(ii)* two **dispersion-hierarchy** panels
  (10j) — the shared scale $\tau_t$ across the window weeks (one series $\times$ $T$, against its
  prior band; if it hugs the prior the random term is not earning its place) and a $7\times7$ map of
  the per-cell dispersion against its block mean; and *(iii)* a **$p^0$** panel (10j, weighted path)
  plotting the fitted zero probability against the empirical $n^0/n$ per cell — the check that the
  Binomial denominator is wired to the roster correctly.
  Added 2026-07-31: an **antibody-protection factor** panel (9j, `res/9j_protection_factor_F.png`) —
  the posterior of the leaky factor $F$ (median + 90% band) per model over the rolling origins, the
  direct analog of the $\gamma_{\mathrm{SAR}}$-over-time figure. $F$ enters
  $\text{full\_susceptibility}_a(t)=\text{susc}_a(1+(F-1)A_a(t))$, so $F\to0$ ⟹ antibodies fully
  protect, $F\to1$ ⟹ no protection (prior $\mathrm{Beta}(5,1)$ ⟹ $F\in(0,1)$); it is a scalar per
  Stage-2 draw, surfaced from the pooled artefact via `collect_transmission_structure` and drawn by
  `plot_F`.
- **Reproduction number** (two separate figures). *(1)* `res/9j_reproduction_number.png` — the
  "contact & transmission" $R$: the dominant (Perron) eigenvalue $\rho(N)$ of the frozen origin-week
  NGM, per origin, over the inc2prev national $R$ and the $R=1$ line. *(2)*
  `res/9j_contact_reproduction_number.png` — the **contacts-only relative** $R$:
  $\rho(C^\ast_t)/\rho(C^\ast_{t_{\text{first}}})$, the dominant eigenvalue of the bare contact matrix
  $C^\ast$ alone (no $\gamma_{\text{SAR}}$, susceptibility, infectivity or antibody) normalised to the
  **first forecast origin** (=1.0), isolating how contact structure alone drove transmissibility
  relative to the baseline week. Overlaid with one **model-free** line —
  $\rho(\hat{E}_t)/\rho(\hat{E}_{t_{\text{first}}})$, the same ratio for the RAW empirical weekly
  mean-contact matrix $\hat{E}_t$ (`AgePairData.emp_mean`, no GP / no reciprocity / no fit) — the data-only
  baseline the four fitted $C^\ast$ curves smooth.

---

## 11. Deliberate simplifications

The preliminary model is intentionally lean; the following are the documented simplifications and
the seams at which they would be relaxed:

- **Contact temporal structure.** *Implemented* — contacts are smoothed across weeks by a separable
  spatio-temporal GP (§5), both the structure field and the overall level, sharing one temporal
  length-scale $\rho_{\text{time}}$. Remaining seams: a longer-memory or non-separable space–time
  kernel (a stationary RBF is used now).
- **Contact-degree dispersion.** *Implemented 2026-07-30* — the age-pair dispersion random effect
  (partial pooling within a child/adult block) is now in the model, with a per-block mean and a
  **shared per-week scale** $\tau_t$ (§4.3). The earlier child$\to$child identifiability worry is
  **resolved by that sharing**: a per-block scale would have been 12 SDs estimated from 4 ordered
  cells each, whereas $\tau_t$ draws on all 49 cells of its week. *Remaining seams*: **(i)** the
  dispersion is still **not temporally smoothed** — $m$, $\tau$ and $z$ are redrawn iid each week,
  unlike the mean field; **(ii)** $\tau_t$'s prior scale $\mathcal N^{+}(0,0.5^2)$ is *not*
  data-derived (the previously measured $0.109$ described between-block, not within-block, spread).
  Two failure directions, both diagnosed by the 10j $\tau_t$ panel: if the posterior *hugs* the prior
  the random term is not earning its place (fall back to a single scalar $\tau$, or to $0.109$); if
  it runs far *above* the prior, the composed $m + \tau z$ saturates the $\log\kappa$ soft-clamp,
  which is clamp compression rather than a real fit.
  *What the smoke-scale fits actually showed, and the correction:* an early reading of
  $\tau_t \approx 2$ against a prior median of $0.34$ looked like the second direction, but a
  four-way sweep of the $\tau$ prior ($10^{-6}$, $0.109$, $0.25$, $0.5$) was **not monotone** —
  $0.109$ and $0.5$ diverged while $10^{-6}$ and $0.25$ were healthy. That is optimiser-path luck,
  not a prior-scale effect: the diverged runs had block means near $-441$, so the large $\tau$ was a
  *symptom* of a runaway LBFGS path in the clamp's flat region, not evidence of real between-cell
  spread. The response was to **widen the $\kappa$ clamp** (§4.3), not to retune $\tau$. Re-read
  $\tau_t$ from a converged fit before drawing any conclusion about between-cell dispersion;
  **(iii)** many per-week
  cells are **empty** (no sampled participant-days), so their random term is prior-only yet still
  feeds $\langle k^2\rangle$ into the NGM — noise that the **neighbourhood** builder, which divides by
  $\langle k\rangle$, amplifies. Whether to zero the random term on empty cells is an open
  implementation choice; on the weighted path the fitted $p^0$ (§4.2) partly mitigates it.
- **Hurdle zero probability.** *Implemented 2026-07-30* — $p^0$ is fitted per cell × week with a
  Binomial roster likelihood rather than plugged in empirically (§4.2), so its uncertainty now
  propagates into $\langle k\rangle$, $\langle k^2\rangle$ and $g$. *Remaining seams*: **no pooling**
  across cells or weeks (each $p^0$ stands alone under a flat $\mathrm{Beta}(1,1)$ — defensible while
  roster counts are large, but sparse cells lean entirely on the prior), and cells with
  $n_{t,i,j} = 0$ have no likelihood at all.
- **Group-contact weight** $w_{\text{group}} = 2.5/240$ is fixed, not estimated.
- **Generation interval.** *Implemented 2026-07-30* — the two log-normal parameters are estimated
  (§3.1) with Munday's informative prior. *Remaining seams*: it is **not variant-specific** (one
  $w$ per fitting window, re-estimated per origin), and $s_{\max}=4$ stays fixed. Note that $w$ and
  $\gamma_{\mathrm{SAR}}$ are **confounded** — both scale the renewal predictor, so raising $w_1$ and
  lowering $\gamma_{\mathrm{SAR}}$ nearly compensate over an 8-week window. The $20\%$ prior SD is what
  keeps the pair identified and should not be loosened; if the posterior equals the prior, the GI is
  adding nothing, and if it parks on a soft-clamp with a tight CI that is clamp compression, not
  certainty (cf. the $\gamma_{\mathrm{SAR}}\approx0.021$ episode, `tasks/lessons.md` 2026-07-13).
- **Transmission block** — *partially addressed*: the level is now an explicit, data-identified
  **per-contact secondary attack rate** $\gamma_{\mathrm{SAR}}$ (un-normalised $C^\ast$) with
  susceptibility/infectivity **relative** to the reference bin $r=\texttt{cfg.ref\_bin}$ (the plan's
  $\gamma_{\mathrm{SAR}}$ + baseline form; **2026-07-31** $r=4$ "25-34", formerly $r=1$ "2-10"),
  replacing the old confounded $\mu_s,\mu_i$ level pair; the block is fit as
  **Stage 2** of the cut (§6.0), conditioning on Stage-1 contact draws. The relative offsets are
  **independent per age bin** — offset $\sigma\,z$, $z\sim\mathcal N(0,1)^{A-1}$ iid, with **separately
  estimated** marginal scales $\sigma_s,\sigma_i\sim\mathcal N^+(0,0.25^2)$ (**2026-07-31** mode-at-0
  half-normal, shrink-to-reference) — with **no cross-bin
  smoothing** (the shared-length-scale squared-exponential GP $\sigma\,L_{si}z$ was removed 2026-07-13,
  user request; ending the GP→RW1→RW2→GP→none sequence), and each log-offset is soft-clamped to
  $[\log 0.05,\log 20]$ **hard-bounding** susc/inf to $[0.05,20]$. *Remaining seams*: $F$ keeps the $\mathrm{Beta}(5,1)$ reference prior (paper uses
  $\Gamma(2,2)T[0,1]$), and infection observation error uses an independence approximation across the week
  and across ages.
- **Antibody availability.** The forecast NGM assumes $A(t_0+h)$ is known at forecast time (§3.2),
  parallel to the contact data. This is a *stronger* assumption than the contact one, because
  `gen_dab` shares the inc2prev/CIS pipeline with the infection targets while CoMix is an independent
  survey. The Stage-2 fit loop is left at $t_0$-anchored antibody, so the likelihood pairs $t{+}h$
  contacts with $t$ antibody.
- **NGM lag index.** Munday Eq 1/6 places the NGM *inside* the lag sum,
  $\sum_s w(s)\,N(t-s)\,I(t-s)$ — the NGM at the infector's primary event. This implementation applies
  a single $N(t)$ outside the sum (§3.3). The deviation predates the formal model and is unresolved.

These should be revisited before any scientific interpretation of the fitted transmission
parameters.
