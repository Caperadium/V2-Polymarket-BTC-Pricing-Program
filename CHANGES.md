# Changes

<!-- Append one entry per logical task. Cleared after each push. -->

- Update `deploy/README.md` for the markout-reporting feature ahead of VPS
  deployment: document the `markout_report.json` artifact (horizons,
  regions, `n_attempted` coverage semantics, 7-day lookback, automatic
  `mid_log` pruning and the snapshot-before-prune note), note that the
  `fold(fills) == inventory` acceptance check now compares `avg_cost` as
  well as `q`, and add a 72h acceptance-test step verifying the markout
  report regenerates with high coverage.
