# Working rules

## Searching the data share (Z:)

Every file lookup on `Z:\Adam-Lab-Shared` is a network round trip, so recursive searches take minutes.

1. Build paths, do not search. Experiment folders come from `experiments.source_path` in the app database; the layout under them is fixed (`pipeline_results/...`, `cropped_sta_movies/cell_N/...`). Open or list that one path.
2. Never run `find`, `**` globs or recursive greps over `Z:\` or a whole experiment tree. Search one experiment folder with a name pattern, or use the file index (below).
3. Keep a local file index (JSON or table, refreshed only on request) for questions like "which cells have X". Answer from it without touching the share.
4. Query a local read-only copy of `runtime/astro_database.sqlite3` (copy it to the scratchpad), not the one on the share. Writes go to the real database only.
5. Use short timeouts (30-60 s) on commands that touch Z:. If one times out, stop and answer from the code, say what is unconfirmed, and check on the data only if asked.
6. Reading movie values or scanning many movies is cluster work: write a job that saves a small summary, do not read the movies from here.
