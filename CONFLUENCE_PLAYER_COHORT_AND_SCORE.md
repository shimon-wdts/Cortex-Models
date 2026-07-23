# Player Cohort and Score

| Field | Value |
|---|---|
| Owner | David Bouskila |
| Status | Beta / back-of-house review |
| Target Release | TBD |
| Last Updated | July 22, 2026 |
| Reviewers | TBD |
| Work Stream | Player Cohort, Player Score, and Player Tier Lift |

## 1. Executive Summary

Player Cohort and Score is a behavioral decision-support model suite for casino operations, marketing, product, and analytics teams. It is designed to answer three practical questions about each player:

| Question | Plain-English Meaning | Model Area |
|---|---|---|
| Who is this player today? | What behavioral group does this player currently resemble? | Player Cohort |
| How should this player be prioritized? | How strong is the player's current value, play quality, frequency, and volatility profile? | Player Score |
| What is the most reasonable development direction? | Is there a credible path to a stronger future state, a better fit, a risk-control path, or a stronger current-cohort position? | Player Tier Lift |

The model suite does not exist to tell operators exactly what to do with a player. It exists to organize the evidence behind a player, explain the player's current behavioral state, and recommend an uplift path for human review. The output should help a casino team understand whether the player is a table-fit opportunity, a frequency opportunity, a side-bet or feature-engagement opportunity, a current-cohort strengthening opportunity, or a risk-control case that should be stabilized before any growth-oriented action.

This page replaces prior working notes and should be treated as the definitive reference for the current beta version of the Player Cohort and Score work stream.

## 2. Audience and How to Read This Page

This page is written for casino operations and marketing first, with enough detail for product, engineering, and analytics teams to implement and validate the work stream.

| Audience | What This Page Should Help Them Understand |
|---|---|
| Casino operations | What the model is saying about a player, how to interpret the recommendation path, and where human judgment is required. |
| Marketing and loyalty teams | Which players may be ready for a better state, which behaviors matter, and how outputs can inform campaign or host-review strategy. |
| Product managers and designers | What fields the model produces, how to explain them in Cortex Floor, and which caveats must be visible in the user experience. |
| Engineers | What the model pipeline does, what output layers are expected, and where integrations should consume model fields. |
| Data science and analytics | What assumptions, model targets, thresholds, validation results, and limitations are currently accepted for beta use. |

## 3. Definitions

These terms are used throughout the page. They are defined here the first time so the rest of the page can use them consistently.

| Term | Definition |
|---|---|
| Player | A casino customer who plays table games. |
| Cohort | A group of players who behave similarly. In this work stream, cohorts are intended to be behavior-led, not simply value buckets. |
| Player Score | A 1 to 100 prioritization score that blends worth, deal hold, frequency, and volatility. It is not a customer-facing rating. |
| Tier | A shorthand for a stronger behavioral or value-development state. It does not necessarily mean a loyalty-program tier. |
| Lift | Improvement from one observed period to a later observed period. Lift may mean engagement lift, expected value lift, cohort movement, or healthier risk behavior depending on the recommendation path. |
| Tier Lift | Movement toward a stronger state, strengthening within the current state, or stabilizing a player before a growth path is appropriate. |
| Theo | Theoretical win, meaning the expected casino value implied by the player's wager pattern and game math. In this page, theo is used as a business outcome and validation signal, not as the sole basis for cohorts. |
| EV | Expected value. In this model suite, EV is used as a directional estimate of expected theo-dollar lift, not guaranteed return on investment. |
| ADT | Average daily theoretical value. It is a common casino value metric, but this model suite intentionally does not rely only on ADT-style ranking. |
| Recommendation Path | A business-readable direction such as "Improve table fit and access" or "Protect momentum after losses." |
| Path Fit Score | A 0 to 100 measure of how strongly the player matches a recommendation path. It is not the probability that the action will succeed. |
| Confidence | A plain-language label such as low, medium, high, or critical that summarizes how strong the model evidence is for presentation. |
| Beta | Controlled back-of-house review. In beta, outputs support human review and feedback capture; they are not production automation. |

## 4. Problem Statement

Casino teams have many players and limited operational attention. Traditional player ranking often answers the question "who is already valuable?" by sorting players by theo, turnover, average bet, or recent play volume. That is useful, but incomplete.

The harder question is: "Which players have a realistic path to a better future state, and what kind of path should the business consider?"

A player may be:

| Player Situation | Why a Simple Value Rank Can Miss It |
|---|---|
| Valuable but unstable | Current value may be high, but volatility or loss-response behavior may make a growth action inappropriate. |
| Lower value but improving | Current theo may look modest even when behavior suggests a stronger future state is possible. |
| Near a stronger cohort | The player may be close to a better behavioral group even if current financial metrics do not stand out. |
| Best served by current-cohort strengthening | The player may not need a new tier or new cohort; they may need a better fit within their current pattern. |
| Showing risk-control signals | The right operational posture may be stabilization, not escalation. |

Player Cohort and Score exists to close this gap. It turns raw table-game behavior into a structured view of who the player is, how the player scores today, and what uplift path is most reasonable for operator review.

## 5. What Has Been Built

The current beta model suite has three connected model outputs.

| Model Output | What It Does | Why It Matters |
|---|---|---|
| Player Cohort | Assigns each player to a behavior-based cohort and identifies a nearby stronger target cohort when available. | Gives operators a plain-language picture of the player's current behavioral state and possible development direction. |
| Player Score | Produces a 1 to 100 score using worth, deal hold, frequency, and volatility. | Helps prioritize players without relying on only one value metric. |
| Player Tier Lift | Selects one recommendation path per candidate player and attaches fit, confidence, expected EV, engagement, risk, and cohort movement signals. | Converts model evidence into an operator-reviewable development path. |

These outputs are designed to work together. The cohort explains the player's behavioral identity. The player score summarizes prioritization. The tier lift recommendation suggests the most reasonable development direction.

## 6. Core Product Principle

The product principle is:

**Behavior first, value second, human decision always.**

That means:

| Principle | Meaning |
|---|---|
| Behavior first | The model should understand how a player plays: rhythm, session behavior, game fit, range, loss response, and feature engagement. |
| Value second | Financial fields such as theo, turnover, average bet, and worth are important, but they should not be the only reason a player surfaces. |
| Human decision always | The output is decision support. It should recommend an uplift path and explain the evidence, not automatically prescribe or execute an action. |

### 6.1 Success Criteria

The model suite should be judged by whether it helps the business make better review decisions, not by whether it produces a high score alone.

| Success Criterion | Definition of Success |
|---|---|
| Plain-English explainability | Casino operations and marketing stakeholders can understand what a cohort is, what Player Score means, and what a tier-lift path is recommending. |
| Behavior-led grouping | Cohorts describe recognizable player behavior rather than simply recreating value buckets. |
| Useful prioritization | Player Score helps focus review attention while remaining clearly separate from the recommendation path. |
| Reviewable recommendation paths | Each tier-lift path explains what kind of opportunity or caution the player represents. |
| Threshold control | Product and analytics teams can tune thresholds and gates to control candidate volume, path mix, and review quality. |
| Production-shaped outputs | Outputs are structured enough for downstream product and engineering use, while still preserving the model context needed for review. |

## 7. Assumptions and Design Decisions

This section is intentionally explicit. These assumptions are not hidden model facts; they are decisions the business, product, and analytics teams should be able to review and challenge.

### 7.1 Business Assumptions

| Area | Current Assumption | Why It Is Reasonable | What It Affects |
|---|---|---|---|
| Behavioral cohorts | Players can be meaningfully grouped by how they behave. | Players often show repeatable patterns in frequency, session duration, game preference, side-bet usage, bet range, and loss response. | Cohort assignment, cohort labels, target cohort direction. |
| Financial agnosticism for cohorts | Cohorts should be primarily behavior-led, not directly driven by money fields. | If cohorts are built only from theo or worth, the system mostly rediscovers current value instead of explaining behavior. | Cohort explainability and future opportunity detection. |
| Money as outcome | Theo, turnover, worth, and average bet remain important as context and validation outcomes. | The business still needs to know whether model-selected players have better future value signals. | Validation, EV lift estimation, business review. |
| Tier meaning | A tier is a stronger behavioral or value-development state, not necessarily a loyalty tier. | The model is intended to support player development and Cortex Floor decisions, not redefine the loyalty program by itself. | UI language, documentation, operator interpretation. |
| Tier movement | Tier lift can mean moving toward a better cohort, strengthening the current cohort position, improving fit, improving engagement, or reducing risk. | Not every player should be pushed to bet more or move to a different cohort. | Recommendation-path selection and success metrics. |
| Operator control | The system recommends a direction, not exact operator action. | Hosts, marketing, and floor teams understand relationship context, property context, and policy constraints. | Product copy, workflow design, beta review. |
| Responsible-gaming caution | Responsible-gaming policy must be visible before external or automated use. | The model touches player behavior and should never encourage harmful escalation. | Beta status, caveats, product governance. |
| Beta readiness | The current version is suitable for controlled back-of-house review. | Test validation is useful, but live feedback, policy review, and additional holdout testing are still needed. | Release scope and operator workflow. |
| Minimum player history | The model is more reliable when a player has enough observed play history to describe behavior. | A very sparse player record can create unstable behavior labels, cohort assignment, and score components. | Confidence, candidate gating, and operator interpretation. |
| Cohort stability | Cohorts describe the player's observed behavior in the current review context and may change as the player's behavior changes. | Player behavior is not fixed; the model should reflect recent evidence rather than permanently label a player. | Cohort assignment, target cohort, monitoring, and refresh cadence. |
| Target cohort realism | Target better cohort means a plausible nearby stronger state, not a guaranteed destination. | The target cohort is a directional reference point for review, not an operational command. | Tier-lift interpretation and UI language. |
| Property transferability | Score distributions, thresholds, cohort labels, and path mix may need calibration by property or operating context. | Different properties can have different game mix, player mix, table limits, and operating practices. | Threshold tuning, rollout, and validation. |
| Intervention attribution | Future lift can be associated with model-selected signals, but should not be claimed as caused by an action without live testing. | Historical model validation is not the same as a controlled intervention test. | Claims, dashboards, and executive interpretation. |
| Action boundary by path | Some paths are growth-oriented, some are stabilization-oriented, and some are monitor-only. | A risk-control path should not be interpreted like a revenue-growth path. | Recommendation wording, success metrics, and operator review. |

### 7.2 What the Model Is Not

| Not This | Explanation |
|---|---|
| Not a customer-facing rating | The output is for internal review. It should not be shown to players. |
| Not an automatic-offer engine | It does not decide a comp, incentive, contact strategy, or host action by itself. |
| Not a responsible-gaming decision engine | It can surface caution signals, but policy decisions must be defined outside the model. |
| Not a guarantee of lift | The model estimates directional opportunity. It does not prove that an action caused future improvement. |
| Not a replacement for casino judgment | Local context, host knowledge, guest history, compliance, and property policy still matter. |

### 7.3 EV and Uplift Assumptions

EV, or expected value, is used directionally. The current EV signal estimates next observed period theo-dollar lift compared with the current period. This is useful for prioritization, but it should not be interpreted as guaranteed dollars or causal return on investment.

Important assumptions:

| EV Assumption | Interpretation |
|---|---|
| EV is directional | It helps compare opportunity groups but should not be presented as guaranteed revenue. |
| EV is noisy | Theo-dollar outcomes are heavy-tailed. A small number of very large players or sessions can move averages. |
| EV is not causal proof | Historical association does not prove that the recommendation action causes the outcome. |
| EV should be paired with path meaning | A risk-control path can be successful even when revenue lift is not the primary goal. |

### 7.4 Data Sufficiency and Sparse-History Assumptions

The model can score a player only from the behavior it has observed. Players with limited history should be interpreted with more caution because their cohort, score, and recommendation path may be based on a smaller behavioral sample.

| Situation | Recommended Interpretation |
|---|---|
| New or sparse-history player | Treat the output as early directional context, not a stable behavioral profile. |
| Short or unusual recent activity | Review whether the player's observed behavior is representative before taking action. |
| Stable repeated behavior | Higher confidence that the cohort and behavior traits describe a real pattern. |
| Missing or incomplete features | Do not overinterpret the recommendation; inspect data completeness and fallback behavior. |

### 7.5 Cohort and Target-Cohort Assumptions

Cohorts should be treated as behavioral snapshots, not permanent identities. A player can move cohorts as their rhythm, game preference, loss response, session behavior, or bet range changes.

The target better cohort is a useful direction for product and operator review. It should not be presented as a guaranteed next state or as a required action target. The safest language is: "the player is near this stronger behavioral state" rather than "the player will become this."

### 7.6 Score Threshold and Calibration Assumptions

Player Score thresholds are review controls. They help decide which players should be reviewed first or included in a shortlist, but they are not fixed business policy by themselves.

| Threshold Principle | Interpretation |
|---|---|
| Thresholds are configurable | Product and analytics can tune them as beta feedback comes in. |
| Thresholds are not action rules | Crossing a threshold does not automatically mean an offer, outreach, or intervention should happen. |
| Thresholds should be property-aware | A useful cutoff at one property may be too broad or too narrow at another. |
| Thresholds should be validated operationally | The business should review whether thresholds create a useful number of candidates and a useful path mix. |

### 7.7 Intervention Attribution and Action-Boundary Assumptions

The model can say that a player resembles historical cases with certain future outcomes. It cannot, by itself, prove that an operator action caused the outcome. Causal claims require live tests, holdouts, or structured feedback analysis.

Each recommendation path also has an action boundary:

| Path Type | Boundary |
|---|---|
| Growth-oriented path | Review possible development opportunity, but do not automate escalation. |
| Stabilization-oriented path | Focus on safer session quality or risk-control interpretation before growth. |
| Fit-oriented path | Review environment, game, table, or feature fit rather than simply increasing value pressure. |
| Monitor path | The right decision may be no immediate action. |

## 8. Conclusions From Exploration and Testing

The current beta conclusions are based on our test data. The data source should be treated generically in product documentation.

### 8.1 High-Level Conclusions

| Conclusion | What It Means |
|---|---|
| Behavior-led grouping is useful | Players can be described in recognizable behavioral categories such as chasing after losses, flexible game pattern, regular side-bet behavior, or late-session fade. |
| Current value alone is not enough | A player can have a stronger future path even if the current value rank is not the highest. |
| Path type matters | A table-fit recommendation should not be judged the same way as a risk-control recommendation. |
| Path fit is not outcome probability | A high path fit score means the player strongly matches the path criteria; it does not guarantee lift. |
| EV adds business context but must be caveated | Predicted theo-dollar lift is helpful for prioritization, but it is noisy and directional. |
| Beta review is the right release posture | The current version is strong enough for back-of-house review and feedback, not automated production action. |

### 8.2 Current Test Snapshot

| Metric | Current Test Result |
|---|---:|
| Players scored | 10,887 |
| Recommendation candidates | 6,595 |
| Candidate rate | 60.6% |
| Average predicted EV per candidate | $242.77 |
| Median predicted EV per candidate | $38.51 |
| Total predicted candidate EV | $1,601,056.97 |
| Candidates with positive predicted EV | 4,657 |
| Candidates with negative predicted EV | 1,938 |

These numbers should be read as beta-test evidence, not final production rates. They show that the model is producing a broad candidate pool and that many candidates have positive directional EV, but they also show that some paths need filtering, operator review, or more data before being treated as production-ready actions.

### 8.3 Current Recommendation Mix

| Recommendation Path | Candidate Count | Candidate Share | Avg Predicted EV | Positive EV Rate | Avg Engagement Lift Score | Avg Path Fit Score |
|---|---:|---:|---:|---:|---:|---:|
| Expose to preferred game features | 2,734 | 41.5% | $285.11 | 78.0% | 15.2 | 99.8 |
| Build confidence and engagement | 1,564 | 23.7% | $3.61 | 31.8% | 1.2 | 52.0 |
| Strengthen current cohort position | 912 | 13.8% | $54.48 | 93.2% | 6.5 | 51.4 |
| Protect momentum after losses | 547 | 8.3% | $767.54 | 69.7% | 29.1 | 99.9 |
| Improve table fit and access | 441 | 6.7% | $593.88 | 96.4% | 32.9 | 96.3 |
| Develop from lowest cohort | 236 | 3.6% | $62.47 | 98.3% | 7.5 | 52.8 |
| Invite to higher-limit path | 69 | 1.0% | $777.43 | 78.3% | 35.6 | 96.9 |
| Move toward adjacent better cohort | 44 | 0.7% | $315.87 | 86.4% | 17.8 | 78.6 |
| Maintain current trajectory | 43 | 0.7% | $14.21 | 100.0% | 1.5 | 63.2 |
| Increase return rhythm | 4 | 0.1% | $1,435.04 | 100.0% | 57.1 | 100.0 |
| Develop hidden opportunity | 1 | 0.0% | -$4,142.13 | 0.0% | 11.4 | 99.1 |

Interpretation:

| Observation | Product Meaning |
|---|---|
| Feature and game-fit paths are common | Many candidates are being surfaced because their play pattern suggests a preferred game, side-bet, or feature path. |
| Confidence-building is broad but low EV | This path may need tighter filters or stronger explanation before it becomes a primary beta action. |
| Table-fit and momentum-protection paths show strong average EV | These may be useful review paths, but momentum protection must still be framed as stabilization first. |
| Higher-limit paths are small but high EV | These should remain carefully reviewed because higher-limit recommendations carry stronger risk and policy considerations. |
| A few paths have tiny sample sizes | Paths with very low counts should be treated as directional until more examples exist. |

### 8.4 Model Validation Snapshot

The recommendation models use CatBoost classifiers for behavioral predictions and a CatBoost regressor for expected EV lift. CatBoost is a machine-learning method that works well with tabular data and mixed feature types.

| Model Signal | What It Predicts | Validation AUC | Baseline Positive Rate | Top-Decile Positive Rate | Top-Decile Lift |
|---|---|---:|---:|---:|---:|
| Engagement lift | Whether visits, sessions, or time are likely to improve next period | 0.773 | 19.4% | 44.2% | 2.28x |
| Tilt risk | Whether chase, volatility, or loss-exit behavior is likely to rise | 0.767 | 14.8% | 41.1% | 2.77x |
| Baccarat engagement | Whether Baccarat engagement is likely to remain or become strong | 0.983 | 36.1% | 98.2% | 2.72x |
| Side-bet engagement | Whether side-bet-heavy behavior is likely to continue or grow | 0.982 | 67.9% | 99.7% | 1.47x |
| Limit path readiness | Whether upper-range or stretch behavior is likely to improve | 0.932 | 35.7% | 100.0% | 2.80x |

EV regressor validation:

| EV Metric | Current Test Result |
|---|---:|
| Validation rows | 3,863 |
| Baseline actual EV lift | $358.85 |
| Top-decile actual EV lift | $438.93 |
| Top-decile EV delta | $80.08 |
| Validation MAE | $2,668.92 |
| Validation RMSE | $9,315.90 |

Interpretation: the EV model has directional signal, but dollar outcomes are noisy. It should support prioritization and product context, not exact ROI promises.

## 9. Player Cohort Model

### 9.1 Purpose

The Player Cohort model answers: **Who is this player behaviorally?**

It groups players by observable behavior rather than only by financial value. This is important because two players with similar theo can behave very differently, and two players with different current value can share a similar development path.

### 9.2 What the Cohort Model Uses

The cohort model uses behavior areas such as:

| Input Area | Example Signals | Business Meaning |
|---|---|---|
| Engagement | Active days, sessions, hours played, continuation rate | How consistently the player participates. |
| Session behavior | Average session duration, total hours, session fade, return rhythm | Whether play is sustained, short, fading, or repeatable. |
| Volatility and range | Bet spread, range width, stretch capacity, ceiling pressure | Whether the player keeps bet size stable or explores a wider range. |
| Loss response | Chase rate, loss exit rate, post-loss stop rate | How the player behaves after losses. |
| Game preference | Baccarat engagement, Blackjack engagement, game concentration | Which game environment the player already responds to. |
| Side-bet behavior | Side-bet rate, side-handle share | Whether feature or side-bet paths are relevant. |
| Trend behavior | Later-period versus earlier-period changes | Whether behavior is improving, weakening, or changing. |

### 9.3 What the Cohort Model Produces

| Output | Meaning |
|---|---|
| Current cohort | The behavioral group the player currently resembles. |
| Cohort confidence | How strongly the player fits the assigned cohort. |
| Cohort distance | How far the player is from the center of the assigned cohort. |
| Cohort development score | A relative score used to reason about stronger or weaker states. |
| Cohort development rank | A relative ordering of cohorts by development strength. |
| Target better cohort | A nearby stronger cohort the player could plausibly move toward. |
| Target margin | How close the player is to the target cohort boundary. |
| Edge-to-better flag | Whether the player is close enough to a stronger cohort to be treated as an edge candidate. |

### 9.4 Current Cohort Labels in Test Output

| Cohort Label | Player Count |
|---|---:|
| Low Engagement / Side-Bet Heavy | 5,530 |
| Low Engagement / Balanced | 2,892 |
| Low Engagement / Chasing | 1,185 |
| High Engagement / Chasing | 861 |
| Medium Engagement / Chasing | 419 |

These labels are product-facing summaries of behavioral groupings. They should be reviewed for clarity before broad release because cohort labels heavily influence how operators understand the model.

## 10. Behavior Characteristics

Behavior characteristics are named traits that explain what the player is showing. They are meant to be plain-English descriptors, not abstract feature names.

### 10.1 Behavior Trait Library

| Behavior Category | Characteristic Wording | Plain-English Meaning |
|---|---|---|
| Loss response | Chases after losses | The player tends to increase wager or continue risk after losing outcomes. |
| Loss response | Stops quickly after losses | The player often stops, exits, or pauses after losing outcomes. |
| Loss response | Stabilizes after losses | The player keeps betting steadier after losses. |
| Side-bet engagement | Regular side-bet player | The player participates in side bets or feature bets at an elevated rate. |
| Side-bet engagement | Avoids side bets | The player rarely participates in side bets. |
| Bet range | Expands bet range | The player uses a wider wager range or plays closer to upper range. |
| Bet range | Keeps bet size stable | The player's wager spread is comparatively stable. |
| Return rhythm | Returns frequently | The player has stronger active-day or session-return rhythm. |
| Return rhythm | Infrequent return pattern | The player returns less frequently or less consistently. |
| Session fade | Fades late in session | The player shows late-session weakening or post-loss fade signals. |
| Session endurance | Sustains longer sessions | The player sustains longer sessions or more total hours. |
| Game preference | Strong game preference | The player's play is concentrated in one preferred game. |
| Game preference | Game-flexible player | The player's play is spread across games rather than concentrated. |

### 10.2 Primary Behavior Logic

The model chooses a primary behavior using priority groups. The goal is to show the most operationally useful behavior, not simply the most generic or highest raw score.

| Priority Group | Threshold | Traits Considered | Why This Comes First |
|---|---:|---|---|
| Group 1 | 70+ | Chases after losses, Expands bet range, Fades late in session | These are strong, actionable behavior signals that can materially affect what an operator should consider. |
| Group 2 | 65+ | Stops quickly after losses, Stabilizes after losses, Returns frequently, Infrequent return pattern, Sustains longer sessions, Game-flexible player, Avoids side bets, Regular side-bet player | These describe meaningful play style, rhythm, and engagement patterns. |
| Group 3 | 60+ | Strong game preference, Keeps bet size stable | These are useful but can be more general, so they are used after stronger signals are checked. |

If no trait clears the priority thresholds, the model falls back to the strongest available characteristic. This prevents a player from being labeled only by a generic trait when a more actionable behavior is present.

### 10.3 Current Primary Behavior Mix in Test Output

| Primary Behavior | Player Count |
|---|---:|
| Game-flexible player | 3,637 |
| Chases after losses | 3,007 |
| Fades late in session | 1,630 |
| Regular side-bet player | 1,117 |
| Expands bet range | 685 |
| Returns frequently | 292 |
| Avoids side bets | 257 |
| Stops quickly after losses | 131 |
| Sustains longer sessions | 113 |
| Strong game preference | 14 |
| Stabilizes after losses | 4 |

Interpretation: the current test output shows many game-flexible and chase/loss-response players. This may be a true property pattern, a feature-engineering artifact, or a naming/calibration area to keep reviewing during beta.

## 11. Player Score Model

### 11.1 Purpose

The Player Score model answers: **How should this player be prioritized today?**

The score is a 1 to 100 composite measure. It is not a loyalty tier, not a player-facing score, and not a guarantee of future value. It is a compact internal score that summarizes current player strength across value, quality, frequency, and volatility.

### 11.2 Player Score Components and Weights

| Component | Weight | Why It Is Included |
|---|---:|---|
| Worth | 35% | Captures the player's value potential. Casino teams need prioritization to remain connected to business value, but worth should not be the only driver. |
| Deal hold | 35% | Captures how the player's play translates into retained casino value or quality of revenue outcome. It balances worth by considering realized or modeled value quality. |
| Frequency | 20% | Captures how often and how consistently the player returns. A valuable player who rarely returns should be treated differently from a frequent player with growing rhythm. |
| Volatility | 10% | Captures instability or spread in play behavior. It is weighted lower because volatility is important context, but the score should not be dominated by risk style alone. |

The weights intentionally balance value and behavior:

| Design Choice | Rationale |
|---|---|
| Worth and deal hold together receive the largest weight | The business must still prioritize players who matter economically. |
| Frequency receives a meaningful but smaller weight | Repeatability and rhythm matter for development, but frequency alone does not define value. |
| Volatility receives the smallest weight | Volatility helps explain stability and risk, but high or low volatility should not automatically make a player good or bad. |

### 11.3 How to Interpret Score Thresholds

Player Score thresholds should be treated as configurable review controls, not permanent production policy. The score helps prioritize attention, but the right threshold depends on the property, the operating team, the review capacity, and the beta goal.

| Threshold Concept | Plain-English Interpretation |
|---|---|
| Lower review threshold | Used when the team wants broader coverage and more examples for learning. |
| Priority review threshold | Used when the team wants a tighter list of players with stronger combined score evidence. |
| High-priority threshold | Used when the team wants to focus on the strongest internal-priority players. |
| Outlier review threshold | Used to inspect unusually high or unusual scores before they are trusted operationally. |

The threshold should answer an operational question: "How many players can the team responsibly review, and how much evidence should be required before a player enters that workflow?"

### 11.4 Player Score Caveats

| Caveat | Why It Matters |
|---|---|
| The score is not the recommendation | Tier Lift can still recommend different paths for players with similar scores. |
| The score is not responsible-gaming clearance | Policy review still applies before action. |
| The score should not be treated as a customer label | It is an internal prioritization signal. |
| The score should be reviewed with the cohort and recommendation path | A high score without context can lead to generic decisions. |

## 12. Player Tier Lift Model

### 12.1 Purpose

The Player Tier Lift model answers: **What uplift path should the business review for this player?**

In this context, tier lift means movement toward a stronger state. That may be:

| Lift Type | Meaning |
|---|---|
| Move to a stronger cohort | The player is near a stronger behavioral group and may be able to move toward it. |
| Strengthen current cohort position | The player may not need a different cohort; they may need better fit within the current one. |
| Improve table or game fit | The player may perform better in the environment they already respond to. |
| Improve engagement | The player may need stronger rhythm, session continuation, or feature engagement. |
| Stabilize risk behavior | The player may need momentum protection or risk-control review before any growth action. |

### 12.2 Candidate Gates

Players become recommendation candidates through one or more gates.

| Candidate Gate | Meaning | Why It Exists |
|---|---|---|
| Edge to better cohort | The player is close to a stronger cohort boundary. | Captures plausible movement to a better behavioral state. |
| Lowest-development cohort | The player is in one of the weakest behavioral states. | Captures development opportunities that may not be boundary cases. |
| In-cohort optimization | The player is not necessarily moving cohorts, but has opportunity signals and sits away from the current cohort center. | Prevents forcing every player into a different cohort when current-cohort strengthening is more realistic. |

In-cohort optimization currently uses:

| Gate Parameter | Current Value | Meaning |
|---|---:|---|
| Minimum distance percentile inside cohort | 0.55 | Player is meaningfully away from the center of their current cohort. |
| Minimum opportunity score | 60.0 | Player has enough hidden opportunity, rhythm, table fit, worth, frequency, or engagement signal. |
| Minimum engagement lift probability | 0.30 | The model sees some chance of engagement improvement. |
| Rank offset from lowest | 2.0 | Avoids using this path for the very lowest-development cohorts when a clearer development path exists. |

### 12.3 Recommendation Models

The tier lift layer uses multiple predictive signals:

| Model Signal | Target |
|---|---|
| Engagement lift | Likelihood that active days, sessions, or hours improve next period. |
| Tilt risk | Likelihood that chase, volatility, or loss-exit behavior rises. |
| Baccarat engagement | Likelihood of stronger or sustained Baccarat engagement. |
| Side-bet engagement | Likelihood of continued or growing side-bet-heavy behavior. |
| Limit path readiness | Likelihood of stronger upper-range, range-width, or stretch behavior. |
| Expected EV lift | Directional next-period theo-dollar lift versus current period. |

The classifier outputs are probabilities. The expected EV model is a regressor, meaning it predicts a numeric dollar value rather than a yes/no probability.

### 12.4 Recommendation Paths

| Recommendation Path | Intended Business Meaning | Primary Signals | Example Operator-Review Direction | Success Metric |
|---|---|---|---|---|
| Protect momentum after losses | The player shows chase, post-loss sensitivity, or volatility. Stabilization should come before growth. | Tilt risk, post-loss response, volatility, chase rate. | Review whether the player needs a lower-pressure experience or monitoring rather than an escalation. | Lower chase rate, healthier session continuation, reduced risk pattern. |
| Strengthen current cohort position | The player has opportunity signals but may be best served by improving within the current cohort. | In-cohort opportunity, cohort distance, rhythm, table fit, hidden opportunity. | Improve fit, rhythm, or current pattern rather than forcing a new cohort. | Lower distance to cohort center, stronger rhythm, improved worth/theo. |
| Invite to higher-limit path | The player shows range or stretch capacity and enough rhythm for careful review. | Limit readiness, stretch capacity, range width, ceiling pressure, rhythm. | Consider whether a higher-limit experience is appropriate after policy and host review. | Higher upper-range play share, stronger next-trip max wager, non-negative risk review. |
| Improve table fit and access | The player shows stronger engagement in a specific game or table environment. | Baccarat probability, table fit, game affinity, primary game. | Route toward the environment where the player already engages best. | Higher preferred-game share, stronger session continuation. |
| Expose to preferred game features | The player responds to side bets, features, or game-specific mechanics. | Side-bet probability, side-bet rate, side-handle share, game affinity. | Consider feature-led engagement or table/game options. | Higher feature-engagement share, repeat-session behavior. |
| Develop hidden opportunity | The player may have more growth potential than current top-line metrics suggest. | Hidden opportunity, rhythm, range width, stretch capacity. | Treat as a behavioral growth candidate, not only as a low-value player. | More active days, stronger range, improved worth trajectory. |
| Build confidence and engagement | The player has some capacity but shows fade, uneven rhythm, or loss sensitivity. | Confidence need, session fade, stretch capacity. | Stabilize experience before pushing a stronger growth path. | Longer controlled sessions, steadier return rhythm. |
| Increase return rhythm | The model sees near-term engagement lift through another trip or session. | Engagement lift probability, rhythm, hours trend. | Nudge toward return frequency or session continuation. | More active days, sessions, or hours. |
| Develop from lowest cohort | The player is in a lower-development behavioral state and needs a clearer development direction. | Lowest-development flag, hidden opportunity, rhythm, cohort edge. | Give the player a path toward stronger rhythm, fit, or controlled range expansion. | Movement toward stronger cohort or improved development score. |
| Move toward adjacent better cohort | The player is near a stronger cohort boundary. | Cohort edge score, target better cohort, target margin. | Focus on the smallest behavioral gap to the target cohort. | Lower target distance and eventual cohort movement. |
| Maintain current trajectory | No stronger path is currently more compelling than the player's existing pattern. | Cohort distance, rhythm, volatility. | Monitor and avoid unnecessary intervention. | Stable rhythm and no deterioration in volatility or engagement. |

### 12.5 User Configuration Sections

The model suite is configurable. The configuration should be understood as product and analytics control, not as fixed business truth. During beta, these controls help the team manage candidate volume, path quality, review burden, and responsible-gaming posture.

| Configuration Area | What It Controls | Why It Matters |
|---|---|---|
| Cohort input features | Which behavioral fields are used to describe players. | Determines whether cohorts feel behavior-led and recognizable to operators. |
| Behavior trait thresholds | Which traits are eligible to become primary or secondary behaviors. | Prevents generic labels from crowding out more actionable behavior characteristics. |
| Player Score weights | How worth, deal hold, frequency, and volatility contribute to the composite score. | Lets the business adjust prioritization philosophy without changing the whole model. |
| Player Score thresholds | Which players enter broader review, priority review, or high-priority review. | Controls review volume and should be calibrated by property and team capacity. |
| Candidate gates | Edge-to-better, lowest-development, and in-cohort opportunity criteria. | Determines who becomes eligible for tier-lift recommendation. |
| Recommendation path gates | Path-specific requirements such as risk, table fit, feature engagement, limit readiness, and rhythm. | Keeps each recommendation path tied to the right player evidence. |
| Path caps and shortlist controls | Maximum number of players per path or per review batch. | Prevents one path from dominating beta review. |
| Responsible-gaming mode | Whether caution is visible, enforced, or used for ranking. | Keeps product behavior aligned with policy maturity. |
| Feedback fields | What operators record after reviewing a recommendation. | Creates the learning signal needed for future model improvement. |

### 12.6 If an Assumption Is Wrong, What Should Be Tuned

This table gives product and analytics teams a practical way to respond when beta review shows that an assumption is too broad, too narrow, or unclear.

| If This Happens | Tune This |
|---|---|
| Too many players enter the recommendation list | Candidate gates, Player Score thresholds, path caps, or shortlist size. |
| Too few players enter the recommendation list | Candidate gates, score thresholds, or included recommendation paths. |
| One recommendation path dominates | Path-specific gates, path caps, or path-ranking weights. |
| Cohort labels do not feel recognizable | Cohort feature set, cohort count, label wording, or cohort development scoring. |
| Target cohorts feel unrealistic | Target-cohort margin logic, cohort development rank, or edge-to-better threshold. |
| Primary behaviors feel too generic | Behavior trait priority groups and trait thresholds. |
| Player Score feels too value-heavy | Player Score component weights. |
| Player Score feels too behavior-heavy | Player Score component weights and score interpretation thresholds. |
| EV values are overinterpreted | UI copy, tooltip language, and EV display rules. |
| Risk-control paths feel like growth prompts | Recommendation copy, path category labels, and responsible-gaming review language. |
| Operators cannot review the output volume daily | Review thresholds, shortlist size, path caps, and daily ingestion rules. |

## 13. Operational Uses

The model suite is intended for back-of-house review by casino operations, marketing, loyalty, analytics, and product teams.

### 13.1 Core Uses

| Use Case | How the Model Helps |
|---|---|
| Identify players likely to improve | Tier Lift surfaces players with a plausible development path, not only the highest current value. |
| Prioritize host or operator review | Player Score and path fit help teams focus attention on players with stronger evidence. |
| Improve table or game fit | Cohort and recommendation signals can show when a player is better matched to a specific game or table environment. |
| Support marketing segmentation | Cohort and behavior traits can inform campaign audiences in beta review, subject to policy and approval. |
| Optimize loyalty or development strategy | The model can show whether a player is near a stronger behavioral state or needs current-state strengthening. |
| Create a feedback loop | Operator review decisions and observed outcomes can become future training and validation signals. |

### 13.2 Example Actions for Review

These are examples of review directions, not instructions.

| Recommendation Path | Example Review Actions |
|---|---|
| Improve table fit and access | Review preferred game, table environment, pit area, table type, or availability constraints. |
| Expose to preferred game features | Review whether game features, side-bet interest, or table mix are relevant. |
| Strengthen current cohort position | Review current play pattern and decide whether consistency, rhythm, or fit can be improved. |
| Protect momentum after losses | Review whether the player should be stabilized or monitored before any growth-oriented outreach. |
| Build confidence and engagement | Review lower-pressure engagement options or ways to reduce late-session fade. |
| Invite to higher-limit path | Review only with host, policy, and responsible-gaming context because the path can imply escalation. |
| Develop from lowest cohort | Review whether a basic development path could improve engagement or table fit. |
| Maintain current trajectory | No immediate action may be needed beyond monitoring. |

### 13.3 Interpretation Rules for Operators

| Rule | Explanation |
|---|---|
| Read the recommendation path first | The path explains the kind of opportunity or caution. |
| Read the evidence second | Supporting signals explain why the path was selected. |
| Do not treat path fit as outcome probability | A high fit means a strong match to the path, not guaranteed lift. |
| Do not treat EV as guaranteed dollars | EV is directional and should be paired with confidence and caveats. |
| Apply policy before action | Responsible-gaming, host, marketing, and property policy still control action. |

## 14. Example Player Walkthrough

This example is fictional. It is included to show how a reviewer should read the model outputs together instead of treating any single score as the full answer.

### 14.1 Player Snapshot

| Field | Example Value |
|---|---|
| Player | Player 676767 |
| Current cohort | Medium Engagement / Chasing |
| Target better cohort | High Engagement / Controlled Range |
| Primary behavior | Chases after losses |
| Secondary behaviors | Expands bet range, Returns frequently, Regular side-bet player |
| Player Score | 72 / 100 |
| Recommended tier-lift path | Protect momentum after losses |
| Path Fit Score | 91 / 100 |
| Predicted engagement lift score | 29 / 100 |
| Directional expected EV lift | $760 |
| Confidence | High |

### 14.2 What the Cohort Says

The player's current cohort is **Medium Engagement / Chasing**. In plain English, this means the player is active enough to be meaningful, but their play pattern shows elevated chase or post-loss sensitivity. The target better cohort is **High Engagement / Controlled Range**, which means the model sees a stronger nearby state where the player could still be engaged but with healthier range and rhythm.

The important interpretation is not "push this player harder." The important interpretation is "this player has value and activity, but the next step should be controlled because the current behavior includes risk-sensitive signals."

### 14.3 What the Behavior Profile Says

| Behavior | Interpretation |
|---|---|
| Chases after losses | The player tends to increase wager or continue risk after losing outcomes. |
| Expands bet range | The player uses a wider wager range and may move toward upper-range play. |
| Returns frequently | The player has enough rhythm to be reviewed for development. |
| Regular side-bet player | The player responds to side bets or game features. |

This combination matters because it contains both opportunity and caution. The player has rhythm and feature engagement, but the loss-response behavior changes how the operator should think about the path.

### 14.4 What the Player Score Says

The Player Score is **72 / 100**, which puts the player in a priority-review range. That does not mean the player should automatically receive an offer or escalation. It means the player has enough combined worth, deal hold, frequency, and volatility signal to deserve review.

The score should be read as prioritization context:

| Component | Example Interpretation |
|---|---|
| Worth | The player has meaningful value potential. |
| Deal hold | The player's value quality is strong enough to matter. |
| Frequency | The player returns often enough for an intervention path to be observable. |
| Volatility | The player's instability is part of the review, not a reason to blindly pursue lift. |

### 14.5 What the Tier-Lift Recommendation Says

The selected recommendation path is **Protect momentum after losses**. This means the model believes the best review direction is stabilization before growth. Even though the player has positive directional EV, the path should not be presented as "go increase this player's play." It should be presented as "this player may have upside, but the operator should first protect the player's session quality and avoid encouraging unhealthy escalation."

| Recommendation Field | Example Value | How to Read It |
|---|---|---|
| Recommended path | Protect momentum after losses | Stabilization is the main direction. |
| Path Fit Score | 91 / 100 | The player strongly matches this path. |
| Expected EV lift | $760 | There is directional value context, but it is not guaranteed revenue. |
| Engagement lift score | 29 / 100 | Engagement lift is not the main reason this player surfaced. |
| Risk-control score | High | Risk behavior is central to the recommendation. |
| Success metric | Lower chase rate and healthier session continuation | Success is healthier behavior, not just revenue lift. |

### 14.6 How an Operator Should Interpret It

The operator should understand this as a review prompt:

| Operator Question | How the Model Helps |
|---|---|
| Is this player worth reviewing? | Yes. The Player Score and directional EV suggest the player matters. |
| Is the right action a direct growth push? | Not necessarily. The recommendation path is risk control, not escalation. |
| What behavior should be watched? | Post-loss chase, range expansion, and late-session momentum. |
| What would success look like? | The player maintains engagement with less chase behavior and healthier session continuation. |
| What should still happen before action? | Host, property, and responsible-gaming policy review. |

### 14.7 Product Takeaway From the Example

This example shows why the three outputs must be read together.

| Output | What It Adds |
|---|---|
| Cohort | Explains the player's current behavioral state and possible stronger state. |
| Behavior profile | Explains what the player is actually showing. |
| Player Score | Explains why the player is worth review. |
| Tier Lift | Explains the safest and most relevant development direction. |

The same player can have value, opportunity, and caution at the same time. The model suite is useful because it keeps those ideas separate instead of collapsing them into one generic "good player" or "bad player" score.

## 15. Outputs

The model suite produces outputs for analytics, product, and operator review. The output names below are generic and should not be treated as environment-specific paths.

### 15.1 Output Layers

| Layer | Example Outputs | Primary Audience |
|---|---|---|
| Cohort output | `player_cohorts`, `cohort_inference_output` | Analytics, product, engineering |
| Recommendation output | `player_recommendations`, `player_recommendation_candidates` | Product, engineering, analytics |
| Insight output | Per-player cohort insights, tier-lift insights, player-score insights | Product and Cortex Floor builders |
| Validation output | Recommendation path summaries, lift summaries, model validation metrics | Analytics, executives, product |
| Feedback output | Review templates and observed outcome fields | Beta operators and future model training |

### 15.2 Player Cohort Fields

| Field | Meaning |
|---|---|
| `player_id` | Player identifier. |
| `cohort_id` | Current behavioral cohort identifier. |
| `cohort_model_label` | Plain-English label for the current cohort. |
| `cohort_confidence_score` | Strength of fit to the assigned cohort. |
| `cohort_distance` | Distance from the current cohort center. |
| `cohort_development_rank` | Relative rank of the cohort state. |
| `cohort_development_score` | Relative score for the cohort state. |
| `target_better_cohort_id` | Nearby stronger cohort identifier, when available. |
| `target_better_cohort_label` | Plain-English target cohort label. |
| `target_better_cohort_margin` | Gap between the player and target cohort boundary. |
| `cohort_edge_to_better_flag` | Whether the player is close enough to a stronger cohort to be considered an edge candidate. |
| `cohort_edge_score` | Strength of the edge-to-better signal. |

### 15.3 Player Score Fields

| Field | Meaning |
|---|---|
| `player_score` | Composite 1 to 100 score, derived from score components. |
| `worth_score` | Value potential component. |
| `deal_hold_score` | Revenue quality or hold component. |
| `frequency_score` | Return rhythm and play frequency component. |
| `volatility_score` | Play spread or instability component. |
| `score_components` | The component breakdown used to explain the score. |

### 15.4 Behavior Profile Fields

| Field | Meaning |
|---|---|
| `primary_behavior` | The most important behavior characteristic to show for the player. |
| `secondary_behaviors` | Additional behavior characteristics, typically top-ranked traits after the primary behavior. |
| `characteristics` | Full scored set of behavior traits and prevalence-style context. |
| `betting_style` | Plain-English summary of game preference, side-bet intensity, volatility, and limit readiness. |

### 15.5 Tier Lift Recommendation Fields

| Field | Meaning |
|---|---|
| `recommended_path` | Business-readable uplift or stabilization path. |
| `desired_behavior_change` | Plain-language behavior change the path is trying to support. |
| `recommendation_action_type` | Machine-readable category for the recommendation. |
| `player_recommendation` | Operator-facing recommendation text. |
| `recommendation_reason` | Why the path was selected. |
| `supporting_signals` | Model fields or behavior signals behind the path. |
| `success_metric` | What should improve if the path is working. |
| `path_fit_score` | Strength of fit to the selected recommendation path. |
| `engagement_lift_score` | Predicted engagement-lift probability. |
| `expected_ev_lift` | Directional predicted next-period theo-dollar lift. |
| `ev_lift_score` | Normalized expected EV lift score. |
| `tier_movement_likelihood` | Proxy for movement potential using cohort edge, in-cohort fit, limit readiness, and hidden opportunity. |
| `risk_control_score` | Risk signal from tilt, post-loss response, volatility, and chase behavior. |
| `path_selection_reason` | Human-readable explanation of scoring and gating. |

### 15.6 Output Frequency and Use

The intended beta cadence is daily. The model can produce a daily recommendation view, but downstream systems should ingest or surface a recommendation only when there is a new recommendation or a materially changed recommendation for the player.

This distinction matters. Daily scoring keeps the model current, but daily scoring should not create repeated duplicate work for operators. The product workflow should avoid resurfacing the same unchanged recommendation as if it were new.

The key product decision is not just how often the output is produced. It is how often operators can review new or changed recommendations responsibly, give feedback, and act within property policy.

## 16. Pipeline

The end-to-end pipeline turns raw play records into behavior features, cohorts, scores, recommendations, and reviewable outputs.

### 16.1 End-to-End Flow

| Step | What Happens | Output |
|---|---|---|
| 1. Data ingestion | Raw table-game, bet, session, game, and player records are collected from source systems. | Raw data tables or extracts. |
| 2. Feature engineering | Player-period behavior features are calculated. | Player-period feature rows. |
| 3. Behavior trait scoring | Named behavior traits are scored from feature signals. | Primary behavior, secondary behaviors, and characteristics. |
| 4. Cohort assignment | Players are assigned to behavior-based cohorts. | Current cohort and cohort confidence. |
| 5. Target-cohort context | The system finds nearby stronger cohorts and calculates edge-to-better scores. | Target better cohort, target margin, edge flag. |
| 6. Player score calculation | Worth, deal hold, frequency, and volatility are blended into a 1 to 100 score. | Player Score and score components. |
| 7. Recommendation model scoring | CatBoost models estimate engagement lift, risk, game engagement, side-bet engagement, limit readiness, and EV lift. | Model probabilities and predicted EV. |
| 8. Candidate gating | Edge, low-development, and in-cohort optimization gates identify review candidates. | Recommendation candidate flags. |
| 9. Path selection | The system compares recommendation paths and selects one path per candidate. | Recommended path and path fit score. |
| 10. Output generation | Model outputs are packaged for analytics, product, and operator review. | Cohort, score, recommendation, and insight outputs. |
| 11. Beta review | Operators and product reviewers inspect outputs and record feedback. | Feedback data for future retraining and validation. |

### 16.2 Training Pipeline

| Training Area | Description |
|---|---|
| Cohort training | Builds behavioral cohorts using feature rows. Financial value should be used carefully so the cohort remains behavior-led. |
| Recommendation training | Trains CatBoost classifiers on next-period behavior labels such as engagement lift, tilt risk, game engagement, side-bet engagement, and limit readiness. |
| EV training | Trains a regression model for next-period theo-dollar lift compared with current period. |
| Validation | Uses held-out or later-period rows where available to measure classification separation, EV signal, and path-level outcomes. |
| Packaging | Saves model artifacts and metadata for inference. |

### 16.3 Inference Pipeline

| Inference Area | Description |
|---|---|
| Load current features | Read the latest player behavior features. |
| Assign cohort | Determine current cohort, confidence, distance, and target better cohort. |
| Compute score | Generate Player Score and score components. |
| Score recommendation models | Add behavioral probabilities and expected EV lift. |
| Apply candidate logic | Identify whether the player should be considered for recommendation. |
| Select path | Choose the highest-fitting path after gate logic and soft caps. |
| Generate outputs | Write model outputs for review and downstream use. |

### 16.4 Monitoring During Beta

Monitoring should focus on whether the pipeline is stable and whether the outputs remain useful.

| Monitoring Area | What to Watch |
|---|---|
| Data freshness | Are input records recent enough for the review cadence? |
| Row volume | Did the number of players, sessions, or candidates change unexpectedly? |
| Feature completeness | Are required fields missing or drifting? |
| Cohort distribution | Did one cohort suddenly dominate? |
| Primary behavior distribution | Are generic traits crowding out more useful traits? |
| Recommendation mix | Did one path dominate the candidate set? |
| Score distribution | Are scores collapsing into a narrow band? |
| EV distribution | Are predicted EV values extreme or unstable? |
| Operator feedback | Are recommendations being accepted, rejected, deferred, or ignored? |
| Responsible-gaming review | Are caution fields visible and being handled according to policy? |

## 17. Beta Workflow

This work stream is beta and should be used for controlled back-of-house review.

### 17.1 Recommended Beta Process

| Step | Description |
|---|---|
| 1. Generate model outputs | Run the model suite and produce cohort, score, and tier-lift outputs. |
| 2. Review candidate volume | Confirm candidate counts and path mix are reasonable. |
| 3. Review high-priority examples | Inspect players across paths, not only highest score. |
| 4. Validate language | Confirm cohort labels, behavior labels, and recommendation text are understandable to operators. |
| 5. Apply policy review | Check responsible-gaming, host, marketing, and property policy before action. |
| 6. Record feedback | Capture whether the recommendation was shown, accepted, rejected, deferred, or acted on. |
| 7. Track observed outcomes | Record follow-up outcomes for future validation and retraining. |

### 17.2 Beta Success Criteria

| Success Area | What Good Looks Like |
|---|---|
| Comprehension | Operators can explain the cohort, score, and recommendation path without needing a data scientist. |
| Usefulness | Reviewers agree that surfaced candidates are worth discussing, even when they do not act. |
| Focus | The model reduces noise rather than creating a large undifferentiated list. |
| Business signal | Recommended groups show better directional outcomes than low-signal baselines. |
| Risk handling | Risk-control paths are not positioned as revenue escalation. |
| Feedback capture | Operator decisions and outcomes are collected consistently enough to improve future models. |
| Product clarity | UI language separates path fit, confidence, EV, and outcome caveats. |

## 18. Known Limitations and Caveats

| Limitation | Impact | Current Handling |
|---|---|---|
| Historical validation is not causal proof | We can say the model found associated patterns, not that an action caused lift. | Use decision-support language and collect beta feedback. |
| EV is noisy | Dollar estimates can be unstable, especially for high-value or outlier players. | Treat EV as directional and show caveats. |
| Some paths have small sample sizes | Certain recommendation paths may not have enough examples for strong claims. | Keep low-sample paths as review-only or directional. |
| Responsible-gaming policy is not encoded as a blocker | The model may surface a player whose action requires policy review. | Keep caution visible and require human review. |
| Cohort labels require product review | Labels can shape operator interpretation. | Continue reviewing labels with casino stakeholders. |
| Primary behavior mix can be too generic | Some traits may dominate if thresholds are not tuned. | Monitor behavior distribution and adjust priority logic. |
| Score weights are assumptions | Worth, deal hold, frequency, and volatility weights reflect current judgment. | Revisit weights after feedback and validation. |
| Output use cases are beta | The model should not automate offers or interventions yet. | Use back-of-house review only. |

### 18.1 Risks and Controls

| Risk | Why It Matters | Control |
|---|---|---|
| Sparse player history | A player with limited observed play can receive unstable cohort, behavior, or score outputs. | Treat sparse-history outputs as directional and require operator review. |
| Cohort drift | Player behavior and property mix can change over time, making cohort labels less representative. | Monitor cohort distribution and review labels during beta. |
| Target cohort overstatement | A target better cohort can be misunderstood as a guaranteed move. | Use directional language: "near this stronger state" rather than "will move to this state." |
| Score overuse | Teams may treat Player Score as the only prioritization signal. | Pair score with cohort, behavior profile, recommendation path, and responsible-gaming review. |
| Threshold mismatch | A threshold that works in one setting may produce too many or too few candidates elsewhere. | Tune thresholds by property, operating cadence, and review capacity. |
| EV overclaiming | Predicted EV may be interpreted as guaranteed revenue. | Label EV as directional and avoid causal ROI language. |
| Risk-control misinterpretation | A stabilization path may be treated as a growth path. | Separate path categories and success metrics in UI and training materials. |
| Recommendation duplication | Daily scoring can resurface unchanged recommendations. | Ingest or display only new or materially changed recommendations. |
| Responsible-gaming policy gap | The model can surface sensitive behavior before final policy is encoded. | Keep responsible-gaming review visible and do not automate action. |
| Feedback incompleteness | Without feedback, the model cannot learn from beta use. | Make feedback fields lightweight and part of the beta workflow. |

## 19. Roadmap and Open Decisions

These items should be resolved as the beta matures.

| Area | Open Decision |
|---|---|
| Operator workflow | Which roles review cohort, score, and tier-lift outputs? |
| Action governance | Which actions are allowed, blocked, or require approval by recommendation path? |
| Responsible-gaming policy | Which caution signals should block, rerank, or require review? |
| Score calibration | Which Player Score thresholds should control review volume, priority review, and high-priority review? |
| Path calibration | Which paths need tighter gates before wider beta exposure? |
| Feedback design | What fields are easiest for operators to complete consistently? |
| Production readiness | What validation, holdout, and policy checks are required before automation? |

## 20. Acceptance Criteria Mapping

| Jira Scope Area | How This Page Covers It |
|---|---|
| Assumptions | Sections 7 and 18 document business assumptions, EV caveats, responsible-gaming posture, beta status, known limitations, risks, and controls. |
| Conclusions | Section 8 summarizes test results, recommendation mix, model validation, and interpretation. |
| Models | Sections 9 through 12 document Player Cohort, Behavior Characteristics, Player Score, and Player Tier Lift. |
| Uses | Section 13 explains operational uses and example review directions. |
| Outputs | Section 15 documents output layers and fields. |
| Pipelines | Section 16 documents training, inference, and monitoring pipelines. |
| Success criteria | Sections 6 and 17 define product success and beta success criteria. |
| Configuration | Section 12 documents configurable controls and what to tune when assumptions need adjustment. |

## 21. Final Takeaway

Player Cohort and Score is a beta decision-support system for understanding player behavior and reviewing development opportunities. It does not replace operator judgment, responsible-gaming policy, or marketing governance.

The model suite should be understood as a connected story:

| Step | Meaning |
|---|---|
| Player Cohort | Who the player is behaviorally. |
| Behavior Characteristics | What the player is showing. |
| Player Score | How the player ranks as an internal priority. |
| Player Tier Lift | What uplift or stabilization path is most reasonable to review. |
| Validation and Feedback | How the model becomes more trustworthy over time. |

The beta goal is not simply to produce a score. The beta goal is to produce a reviewable, explainable, and policy-aware player opportunity map that casino teams can understand, challenge, and improve.
