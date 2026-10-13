# Working rules

## Searching the data share (Z:)

Every file lookup on `Z:\Adam-Lab-Shared` is a network round trip, so recursive searches take minutes.

1. Build paths, do not search. Experiment folders come from `experiments.source_path` in the app database; the layout under them is fixed (`pipeline_results/...`, `sta_movies_tif/cell_N/...`). Open or list that one path.
2. Never run `find`, `**` globs or recursive greps over `Z:\` or a whole experiment tree. Search one experiment folder with a name pattern, or use the file index (below).
3. Keep a local file index (JSON or table, refreshed only on request) for questions like "which cells have X". Answer from it without touching the share.
4. Query a local read-only copy of `runtime/astro_database.sqlite3` (copy it to the scratchpad), not the one on the share. Writes go to the real database only.
5. Use short timeouts (30-60 s) on commands that touch Z:. If one times out, stop and answer from the code, say what is unconfirmed, and check on the data only if asked.
6. Reading movie values or scanning many movies is cluster work: write a job that saves a small summary, do not read the movies from here.

## Where the app gets its data

1. Everything the app shows comes from saved data: files in the experiment folder or app results saved in `runtime/`. A page never computes results from raw movies while it loads, and never reads a big movie to build a view.
2. The code that needs a file just opens it. If it is missing, it raises a clear error that says what is missing and which existing analysis step produces it (rerun that step). No helper whose job is to "ensure" or "prepare" the file, and no silent generation on first view.
3. A step is complete when all of its outputs exist. Do not add a second step to finish what an earlier step should have produced. An experiment is COMPLETED only when every step of the pipeline (`STEPS` in `app/services/runs.py`) has completed successfully; the status is computed from `STEPS`, never from a hard-coded list or count, so adding a step changes what COMPLETED means. No step is excluded from that check.
4. A view may only read and reformat saved data. If it needs a new file written, that is an analysis step.

## Code rules

1. Database: only files that exist solely for data extraction (`app/db/`) query or write the database. Services, blueprints, pipelines and views call those functions; no `connect`/SQL anywhere else.
2. No invented fallback values for measured data. If the frame rate (fps) cannot be read from `Experiment.xml`, raise a clear error to the user, everywhere in the app. The same for any other value that must come from the data.
3. Imports only at the top of the file. No imports inside functions or after code, and no `sys.path` edits between imports.
4. A file or folder name that is used in more than one place is defined once in `utils/globals_utils.py` and imported. Never type such a name as a string literal elsewhere.
5. Path-building functions live only in `utils/paths_utils.py`, built from the names in `utils/globals_utils.py` through one generic registry (kind + anchor + fields), not one function per file. No path helpers in other files.
6. No lint-silencing comments (`# noqa`). Fix the code instead.
7. Reuse before writing. Before adding a function, search `utils/` (and the services that already do similar work) for one that does the same or almost the same thing, and reuse or extend it. Never keep two copies of one piece of logic.
8. No new exception classes to signal an expected situation (for example "no results"). Return a value or a text warning instead. Raise the built-in errors only for real problems.
9. Functions are short and do one thing: split a long function into small named helpers instead of one long body with nested loops and branches.
10. Per-experiment data that is read often (frame rate, Suite2p stats and fluorescence) is read once and reused, never again per request. Web-app caches are rolling (keep only the last few experiments), live in `app/services/caches.py`, and hold small data only (never movies). A cache is refreshed when the user confirms the step that produced its data (for Suite2p: `forget_suite2p` in `confirm_suite2p`). A failed read is never cached. The frame rate is cached for the life of the process in `utils/movie_utils.py` because cluster jobs use it too.
