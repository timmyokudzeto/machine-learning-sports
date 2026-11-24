# Machine-learning-sports

Two Python forecasting projects test how much historical results tell us before a match kicks off or a race starts. Both notebooks replay events in chronological order and compare their predictions with a simple benchmark.

| Notebook | Forecast | Held-out result |
|---|---|---|
| [Premier League](./Premier_League_Forecast.ipynb) | Scorelines, match outcomes and title probabilities | 57 of 110 outcomes correct, up from 48 for the league-average baseline |
| [Formula 1](./F1_Race_Analysis.ipynb) | Finishing positions and podium probabilities | Average finishing-position error fell from 3.64 to 3.49 places across 462 driver entries |

## Premier League

A regularised Poisson regression learns each club’s attacking and defensive strength and the advantage of playing at home. The preseason model trained on 2,280 matches from 2020/21 to 2025/26, giving recent results more weight. Its estimated goal rates produce a score prediction and probabilities for a home win, draw and away win in each fixture.

**How did it fare against actual scores?** Across the first 110 matches of 2026/27, the reconstructed preseason forecast picked nine more outcomes correctly than the baseline, but gained only one exact score. The notebook also backtests three complete seasons, training only on matches played before each season began.

| First 110 matches, through 23 November 2026 | Club model | League-average baseline |
|---|---:|---:|
| Outcome accuracy | 51.8% | 43.6% |
| Exact scores correct | 13 | 12 |
| Outcome log loss (lower is better) | 1.023 | 1.078 |

**Who wins the league?** After incorporating those 110 results, the model kept the points already earned and simulated the remaining 270 matches 10,000 times. In the 24 November 2026 snapshot, Arsenal won 76.6% of simulations, Manchester City 15.9% and Liverpool 4.7%. The simulation assumes independent goal counts and fixed team strengths for the remaining fixtures.

Match dates, teams and full-time goals come from [Football-Data.co.uk](https://www.football-data.co.uk/englandm.php). The fixture list comes from [FixtureDownload](https://fixturedownload.com/feed/json/epl-2026).

## Formula 1

Once the starting grid is known, does recent form help predict the finish? Before each 2026 Grand Prix, the notebook fits a linear regression using earlier races. Its inputs are starting position, the driver’s average finish over their previous five races and the constructor’s recent form across both cars. A separate logistic regression estimates the probability of a podium.

| Forecast across 21 Grands Prix through Las Vegas | Mean absolute error in finishing places |
|---|---:|
| Driver finishes where they started | 3.64 |
| Linear regression using the grid alone | 3.76 |
| Linear regression using the grid and recent form | 3.49 |

At Las Vegas, the model ranked Verstappen and Norris first and second. Both finished as predicted. Piastri exposed a harder case when he started 17th and finished third despite an estimated podium chance of roughly 1%. Across the full test, adding form improved podium log loss but worsened the Brier score. [Official Las Vegas race result](https://www.formula1.com/en/results/2026/races/1300/las-vegas/race-result)

The notebook embeds the races, drivers and results tables used in the analysis. Driver history begins in 2010 and model training begins in 2014. The original CSV publisher is undocumented.

## Run locally

From the repository root, install the shared dependencies and start JupyterLab.

```bash
python -m pip install jupyterlab pandas numpy scipy scikit-learn matplotlib
python -m jupyterlab
```

Open either notebook and select **Run All Cells**.

- **Premier League** downloads its data on the first run, caches it in `pl_data/` and exports the full fixture ledger to `premier_league_forecasts_2026_27.csv`. To evaluate a later round, save the original predictions, update `CUTOFF` and set `REFRESH = True`.
- **Formula 1** runs offline once the dependencies are installed. To forecast a pending race from the snapshot, place a `future_grid.csv` beside the notebook with one row per driver and the columns `driverId`, `constructorId` and `grid`.
