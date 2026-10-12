PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS experiments(
  id INTEGER PRIMARY KEY, canonical_id TEXT NOT NULL UNIQUE, source_path TEXT NOT NULL UNIQUE,
  cage TEXT NOT NULL, mouse_name TEXT NOT NULL, recording_date TEXT NOT NULL, behavior TEXT NOT NULL,
  fov TEXT NOT NULL, z_plane TEXT NOT NULL, zoom TEXT NOT NULL,
  validation_state TEXT NOT NULL DEFAULT 'valid', validation_report TEXT NOT NULL DEFAULT '{}',
  workflow_stage TEXT NOT NULL DEFAULT 'analysis_ready',
  preprocessing_review_state TEXT NOT NULL DEFAULT 'unreviewed',
  suite2p_review_state TEXT NOT NULL DEFAULT 'unreviewed',
  is_favorite INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS removed_experiments(
  id INTEGER PRIMARY KEY,
  canonical_id TEXT NOT NULL UNIQUE,
  source_path TEXT NOT NULL UNIQUE,
  removed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  note TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS metadata_overrides(
  id INTEGER PRIMARY KEY, experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
  field_name TEXT NOT NULL, original_value TEXT NOT NULL, corrected_value TEXT NOT NULL, correction_note TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, UNIQUE(experiment_id, field_name)
);
CREATE TABLE IF NOT EXISTS experiment_notes(
  id INTEGER PRIMARY KEY,
  experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
  note_text TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS neurons(
  id INTEGER PRIMARY KEY, experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
  pipeline_neuron_id TEXT NOT NULL, roi_source TEXT NOT NULL, roi_summary TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, UNIQUE(experiment_id, pipeline_neuron_id)
);
CREATE TABLE IF NOT EXISTS run_id_sequence(id INTEGER PRIMARY KEY CHECK (id = 1), next_id INTEGER NOT NULL);
INSERT OR IGNORE INTO run_id_sequence(id, next_id) VALUES(1, 1);
CREATE TABLE IF NOT EXISTS analysis_runs(
  id INTEGER PRIMARY KEY, experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
  neuron_id INTEGER REFERENCES neurons(id) ON DELETE CASCADE, step TEXT NOT NULL, backend TEXT NOT NULL,
  status TEXT NOT NULL, is_test INTEGER NOT NULL DEFAULT 0, parameters_json TEXT NOT NULL DEFAULT '{}',
  input_manifest_json TEXT NOT NULL DEFAULT '{}', output_manifest_json TEXT NOT NULL DEFAULT '{}',
  metrics_json TEXT NOT NULL DEFAULT '{}', command_json TEXT NOT NULL DEFAULT '{}',
  code_provenance TEXT NOT NULL DEFAULT '', started_at TEXT, ended_at TEXT, exit_code INTEGER,
  stdout_path TEXT, stderr_path TEXT, log_path TEXT, review_state TEXT NOT NULL DEFAULT 'unreviewed',
  review_note TEXT NOT NULL DEFAULT '', output_dir TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS artifacts(
  id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
  artifact_type TEXT NOT NULL, relative_path TEXT NOT NULL, checksum TEXT, size_bytes INTEGER,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS preprocessing_steps(
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
  step_name TEXT NOT NULL,
  step_order INTEGER NOT NULL,
  slurm_job_id TEXT NOT NULL DEFAULT '',
  slurm_log_path TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'pending',
  submitted_at TEXT,
  started_at TEXT,
  ended_at TEXT,
  exit_code INTEGER,
  message TEXT NOT NULL DEFAULT '',
  UNIQUE(run_id, step_name)
);
CREATE TABLE IF NOT EXISTS confirmed_neurons(
  id INTEGER PRIMARY KEY,
  experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
  neuron_row_id INTEGER NOT NULL REFERENCES neurons(id) ON DELETE CASCADE,
  pipeline_neuron_id TEXT NOT NULL,
  canonical_id TEXT NOT NULL,
  cage TEXT NOT NULL,
  mouse_name TEXT NOT NULL,
  recording_date TEXT NOT NULL,
  behavior TEXT NOT NULL,
  fov TEXT NOT NULL,
  z_plane TEXT NOT NULL,
  zoom TEXT NOT NULL,
  source_path TEXT NOT NULL,
  spike_run_id INTEGER REFERENCES analysis_runs(id) ON DELETE SET NULL,
  behavior_run_id INTEGER REFERENCES analysis_runs(id) ON DELETE SET NULL,
  sta_run_id INTEGER REFERENCES analysis_runs(id) ON DELETE SET NULL,
  status TEXT NOT NULL DEFAULT 'confirmed',
  reviewer_note TEXT NOT NULL DEFAULT '',
  schema_version INTEGER NOT NULL DEFAULT 1,
  confirmed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(experiment_id, pipeline_neuron_id)
);
CREATE TABLE IF NOT EXISTS confirmed_neuron_peaks(
  id INTEGER PRIMARY KEY,
  confirmed_neuron_id INTEGER NOT NULL REFERENCES confirmed_neurons(id) ON DELETE CASCADE,
  all_peak_frames_json TEXT NOT NULL DEFAULT '[]',
  simple_peak_frames_json TEXT NOT NULL DEFAULT '[]',
  complex_peak_frames_json TEXT NOT NULL DEFAULT '[]',
  n_all_peaks INTEGER NOT NULL DEFAULT 0,
  n_simple_peaks INTEGER NOT NULL DEFAULT 0,
  n_complex_peaks INTEGER NOT NULL DEFAULT 0,
  peak_detection_params_json TEXT NOT NULL DEFAULT '{}',
  source_artifact_path TEXT NOT NULL DEFAULT '',
  UNIQUE(confirmed_neuron_id)
);
CREATE TABLE IF NOT EXISTS confirmed_sta_results(
  id INTEGER PRIMARY KEY,
  confirmed_neuron_id INTEGER NOT NULL REFERENCES confirmed_neurons(id) ON DELETE CASCADE,
  result_type TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 0,
  n_peaks INTEGER,
  n_significant_clusters INTEGER NOT NULL DEFAULT 0,
  has_significant_cluster INTEGER NOT NULL DEFAULT 0,
  pre_window INTEGER,
  post_window INTEGER,
  analysis_window_json TEXT NOT NULL DEFAULT '[]',
  baseline_window_json TEXT NOT NULL DEFAULT '[]',
  spatial_bin INTEGER,
  voxel_alpha REAL,
  cluster_alpha REAL,
  n_permutations INTEGER,
  parameters_json TEXT NOT NULL DEFAULT '{}',
  source_npz_path TEXT NOT NULL,
  UNIQUE(confirmed_neuron_id, result_type, version)
);
CREATE TABLE IF NOT EXISTS confirmed_sta_clusters(
  id INTEGER PRIMARY KEY,
  confirmed_sta_result_id INTEGER NOT NULL REFERENCES confirmed_sta_results(id) ON DELETE CASCADE,
  cluster_id INTEGER NOT NULL,
  sign TEXT NOT NULL,
  is_significant INTEGER NOT NULL DEFAULT 0,
  p_value REAL,
  mass REAL,
  size_voxels INTEGER,
  peak_value REAL,
  centroid_y REAL,
  centroid_x REAL,
  area_px INTEGER,
  mean_trace_json TEXT NOT NULL DEFAULT '[]',
  t_rel_json TEXT NOT NULL DEFAULT '[]',
  mask_artifact_path TEXT NOT NULL DEFAULT '',
  UNIQUE(confirmed_sta_result_id, cluster_id)
);
CREATE TABLE IF NOT EXISTS confirmed_cell_metrics(
  id INTEGER PRIMARY KEY,
  confirmed_neuron_id INTEGER NOT NULL REFERENCES confirmed_neurons(id) ON DELETE CASCADE,
  metric_group TEXT NOT NULL,
  metric_name TEXT NOT NULL,
  metric_value_json TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT '',
  UNIQUE(confirmed_neuron_id, metric_group, metric_name)
);
CREATE TABLE IF NOT EXISTS scan_history(id INTEGER PRIMARY KEY, parent_path TEXT NOT NULL, started_at TEXT NOT NULL, completed_at TEXT, summary_json TEXT NOT NULL DEFAULT '{}');
CREATE TABLE IF NOT EXISTS app_settings(key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
INSERT OR IGNORE INTO schema_migrations(version) VALUES(1);
-- Indexes for the WHERE / JOIN columns used by the pages and the run monitor (CS-09). All idempotent.
CREATE INDEX IF NOT EXISTS idx_analysis_runs_exp_step_status ON analysis_runs(experiment_id, step, status);
CREATE INDEX IF NOT EXISTS idx_analysis_runs_exp_test_step ON analysis_runs(experiment_id, is_test, step, id);
CREATE INDEX IF NOT EXISTS idx_analysis_runs_status ON analysis_runs(status);
CREATE INDEX IF NOT EXISTS idx_analysis_runs_neuron ON analysis_runs(neuron_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id);
CREATE INDEX IF NOT EXISTS idx_confirmed_neurons_neuron_row ON confirmed_neurons(neuron_row_id);
CREATE INDEX IF NOT EXISTS idx_confirmed_neurons_exp ON confirmed_neurons(experiment_id);
