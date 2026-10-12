import streamlit as st
import os
import sys
import subprocess
import datetime
import json
import numpy as np
import pandas as pd
import tifffile
from streamlit_autorefresh import st_autorefresh
import glob
import tkinter as tk
from tkinter import filedialog

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import pipeline_constants as consts, pipeline_registry
from utils import files_paths as paths
from utils import pipeline_utils as pipe_utils
from utils import data_utils as data_utils
from utils import pipeline_registry as steps_registry

NUMBER_INPUT = "number_input"
BOOLEAN_INPUT = "boolean_input"
MC_PARAMS_TITLE = "**Motion Correction Parameters**"
PB_PARAMS_TITLE = "**Photobleaching Correction Parameters**"
MAX_TRACE_POINTS = 1200
SPATIAL_DOWNSAMPLE = 8
IS_2CH_USER_KEY = "_is_2ch_user_selected"

########## initialization ###########
class GUI_parameter:
    def __init__(self, name, param_type, default, list_vals=None, display_name=None):
        self.key = name
        if display_name is None:
            self.name = name
        else:
            self.name = display_name
        self.type = param_type
        self.default = default
        self.list_vals = list_vals
        self.help = self.get_help()
        self.st_widget = self.create_widget()
    
    def get_help(self):
        if self.name == consts.GSIG_FILT:
            return "Size of kernel for high pass spatial filtering"
        elif self.name == consts.MAX_SHIFTS_X:
            return "Maximum allowed rigid shift"
        elif self.name == consts.PW_RIGID:
            return "NoRMCorre stand for Non-Rigid Motion Correction. \
                    \nnon-piecewise rigid is faster and sometimes it will be sufficient"
        elif self.name == consts.MAX_DEVIATION_RIGID:
            return "Maximum deviation allowed for patch with respect to rigid shifts"
        elif self.name == consts.OVERLAPS_X:
            return "Overlap between patches (size of patch strides + overlaps.)"
        elif self.name == consts.STRIDES_X:
            return "Start a new patch for pw-rigid motion correction every n pixels"
        else:
            return None
    def create_widget(self):
        if self.type == "list_input":
            st.selectbox(self.name, index=self.default, key=self.key, options=self.list_vals, help=self.help)
        if self.type == "number_input":
            st.number_input(self.name, key=self.key, value=self.default, help=self.help)
        if self.type == "boolean_input":
            st.checkbox(self.name, key=self.key, value=self.default, help=self.help)


def init_pipeline_session():
    """
    Initialize pipeline session only once per browser session.
    """

    if "session_time" not in st.session_state:

        session_time = datetime.datetime.now().strftime("%d-%m-%Y___%H-%M-%S")

        pipe_utils.mkdir(os.path.join(paths.PIPELINE_LOGS_DIR, session_time))

        pipeline_runner_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            paths.PIPELINE_RUNNER_SCRIPT
        )

        print("Launching:", pipeline_runner_script, session_time)

        subprocess.Popen(["python", pipeline_runner_script, session_time])

        st.session_state.session_time = session_time

    return st.session_state.session_time


def init_session_state():
    title_col1, title_col2, title_col3 = st.columns(3)
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False
    with title_col2:
        st.title('**Voltage Imaging Pipeline** :star:')

    defaults = {
        consts.RAW_VIDEO_PATH: "",
        consts.ANALYSIS_VIDEO_PATH: "",
        consts.HOME_DIR: "",
        consts.IS_2CH: False,
        IS_2CH_USER_KEY: False,
        consts.CAGE: "",
        consts.MOUSE_NAME: "",
        consts.FOV: "",
        consts.EXPERIMENT_DATE: "",
        consts.BEHAVIOR: "",
        consts.EXPERIMENT_DETAILS: "",
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if consts.TRIMMED not in st.session_state:
        st.session_state[consts.TRIMMED] = 0
    if consts.PB_TRACE_LOADED not in st.session_state:
        st.session_state[consts.PB_TRACE_LOADED] = False
    if "_pipeline_defaults_initialized" not in st.session_state:
        _refresh_2ch_mode_and_step_defaults(force=True)
        st.session_state["_pipeline_defaults_initialized"] = True

    tab_keys = list(pipeline_registry.TABS_REGISTRY.keys())
    tab_display_names = [
        pipeline_registry.TABS_REGISTRY[k]["display_name"]
        for k in tab_keys
    ]

    whitespace = 28
    tabs = st.tabs([name.center(whitespace, "\u2001") for name in tab_display_names])
    tabs_dict = dict(zip(tab_keys, tabs))

    return tabs_dict

########## 1st Tab: run pipelines ###########
def _path_indicates_2ch(raw_path):
    return "hyp" in str(raw_path).lower()


def _apply_pipeline_step_defaults(is_2ch_mode):
    for step_name in steps_registry.STEPS_REGISTRY.keys():
        if is_2ch_mode:
            st.session_state[step_name] = True
        else:
            st.session_state[step_name] = step_name in (
                consts.MOTION_CORRECTION,
                consts.PHOTOBLEACHING_CORRECTION
            )


def _refresh_2ch_mode_and_step_defaults(force=False):
    user_2ch = bool(st.session_state.get(IS_2CH_USER_KEY, False))
    effective_2ch = user_2ch
    prev_effective_2ch = st.session_state.get("_effective_2ch_mode", None)

    st.session_state[consts.IS_2CH] = effective_2ch
    st.session_state["_effective_2ch_mode"] = effective_2ch

    if force or prev_effective_2ch is None or effective_2ch != prev_effective_2ch:
        _apply_pipeline_step_defaults(effective_2ch)


def _on_raw_video_path_change():
    st.session_state[consts.PB_TRACE_LOADED] = False
    st.session_state.pop(consts.TRIMMED_SLIDER, None)
    raw_path = st.session_state.get(consts.RAW_VIDEO_PATH, "")
    if _path_indicates_2ch(raw_path):
        st.session_state[IS_2CH_USER_KEY] = True
    _refresh_2ch_mode_and_step_defaults(force=False)


def _on_2ch_checkbox_change():
    _refresh_2ch_mode_and_step_defaults(force=False)


def display_mc_params():
    with st.expander(MC_PARAMS_TITLE):
        param_col1, param_col2, param_col3 = st.columns(3)
        with param_col1:
            gSig_filt = GUI_parameter(consts.GSIG_FILT, NUMBER_INPUT, 9)
            max_shifts_x = GUI_parameter(consts.MAX_SHIFTS_X, NUMBER_INPUT, 60)
            max_shifts_y = GUI_parameter(consts.MAX_SHIFTS_Y, NUMBER_INPUT, 60)
        with param_col2:
            place_holder = st.write("")
            place_holder2 = st.write("")
            pw_rigid = GUI_parameter(consts.PW_RIGID, BOOLEAN_INPUT, False)
            place_holder3 = st.write("")
            max_deviation_rigid = GUI_parameter(consts.MAX_DEVIATION_RIGID, NUMBER_INPUT, 3)
        with param_col3:
            overlaps_x = GUI_parameter(consts.OVERLAPS_X, NUMBER_INPUT, 32)
            overlaps_y = GUI_parameter(consts.OVERLAPS_Y, NUMBER_INPUT, 32)
            strides_x = GUI_parameter(consts.STRIDES_X, NUMBER_INPUT, 96)
            strides_y = GUI_parameter(consts.STRIDES_Y, NUMBER_INPUT, 96)
    return


def display_pb_params():
    with st.expander(PB_PARAMS_TITLE):
        raw_path = st.session_state.get(consts.RAW_VIDEO_PATH, "")
        st.number_input(
            consts.TRIMMED,
            key=consts.TRIMMED,
            min_value=0,
            step=1,
            on_change=_sync_slider_from_input
        )

        if not raw_path or not os.path.exists(raw_path):
            st.caption("Select a movie path to preview mean intensity.")
            return

        if st.button("Load/Refresh mean-intensity preview", key="load_pb_trace_btn"):
            st.session_state["pb_trace_loaded"] = True

        if not st.session_state.get("pb_trace_loaded", False):
            st.caption("Preview is on-demand to keep the GUI responsive.")
            return

        try:
            with st.spinner("Computing mean-intensity trace..."):
                n_frames, sampled_frames, sampled_mean = _compute_mean_trace(
                    raw_path,
                    os.path.getmtime(raw_path),
                    os.path.getsize(raw_path)
                )

            max_frame = max(0, n_frames - 1)
            current_trimmed = int(st.session_state.get(consts.TRIMMED, 3000))
            current_trimmed = max(0, min(current_trimmed, max_frame))

            if consts.TRIMMED_SLIDER not in st.session_state:
                st.session_state[consts.TRIMMED_SLIDER] = current_trimmed
            st.session_state[consts.TRIMMED_SLIDER] = max(0, min(int(st.session_state[consts.TRIMMED_SLIDER]), max_frame))

            selected_frame = st.slider(
                "Trim first frames",
                min_value=0,
                max_value=max_frame,
                step=1,
                key=consts.TRIMMED_SLIDER,
                on_change=_sync_input_from_slider
            )
            selected_frame = int(selected_frame)

            trace_df = pd.DataFrame({
                "frame": sampled_frames.astype(int),
                "mean_intensity": sampled_mean.astype(float)
            }).set_index("frame")
            st.line_chart(trace_df)

            nearest_idx = int(np.argmin(np.abs(sampled_frames - selected_frame)))
            selected_mean = float(sampled_mean[nearest_idx])
            removed_pct = (selected_frame / max(1, n_frames - 1)) * 100.0
            st.caption(
                f"Selected trim frame: {selected_frame} / {n_frames - 1} "
                f"(~{removed_pct:.1f}% removed), mean intensity near selection: {selected_mean:.2f}"
            )
        except Exception as e:
            st.warning(f"Could not compute mean-intensity trace: {e}")
    return


@st.cache_data(show_spinner=False)
def _compute_mean_trace(raw_path, _mtime, _size):
    if raw_path.lower().endswith(".raw"):
        n_frames, sampled_frames, sampled_mean = _mean_trace_from_raw(raw_path)
        return n_frames, sampled_frames, sampled_mean
    if raw_path.lower().endswith(".tif") or raw_path.lower().endswith(".tiff"):
        n_frames, sampled_frames, sampled_mean = _mean_trace_from_tif(raw_path)
        return n_frames, sampled_frames, sampled_mean
    raise ValueError("Unsupported file format. Use .raw or .tif/.tiff")


def _sync_input_from_slider():
    if consts.TRIMMED_SLIDER in st.session_state:
        st.session_state[consts.TRIMMED] = int(st.session_state[consts.TRIMMED_SLIDER])


def _sync_slider_from_input():
    trimmed_value = int(st.session_state.get(consts.TRIMMED, 0))
    if consts.TRIMMED_SLIDER in st.session_state:
        st.session_state[consts.TRIMMED_SLIDER] = max(0, trimmed_value)


def _mean_trace_from_raw(raw_path):
    width, height = pipe_utils.get_raw_video_dimensions(raw_path)
    itemsize = np.dtype(np.uint16).itemsize
    total_bytes = os.path.getsize(raw_path)
    frame_size_bytes = width * height * itemsize
    if frame_size_bytes == 0:
        raise ValueError("Raw video dimensions are invalid (zero frame size).")
    n_frames = total_bytes // frame_size_bytes
    if n_frames <= 0:
        raise ValueError("Raw video has no frames.")

    mm = np.memmap(raw_path, dtype=np.uint16, mode="r", shape=(n_frames, height, width))
    temporal_step = max(1, int(np.ceil(n_frames / MAX_TRACE_POINTS)))
    sampled_frames = np.arange(0, n_frames, temporal_step, dtype=np.int64)
    sampled_movie = mm[::temporal_step, ::SPATIAL_DOWNSAMPLE, ::SPATIAL_DOWNSAMPLE]
    sampled_mean = sampled_movie.mean(axis=(1, 2))
    return int(n_frames), sampled_frames, sampled_mean


def _mean_trace_from_tif(tif_path):
    try:
        movie = tifffile.memmap(tif_path)
    except Exception:
        movie = tifffile.imread(tif_path)
    if movie.ndim != 3:
        raise ValueError(f"Expected 3D movie, got shape {movie.shape}")

    n_frames = int(movie.shape[0])
    temporal_step = max(1, int(np.ceil(n_frames / MAX_TRACE_POINTS)))
    sampled_frames = np.arange(0, n_frames, temporal_step, dtype=np.int64)
    sampled_movie = movie[::temporal_step, ::SPATIAL_DOWNSAMPLE, ::SPATIAL_DOWNSAMPLE]
    sampled_mean = sampled_movie.mean(axis=(1, 2))
    return n_frames, sampled_frames, sampled_mean


def choose_file():
    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)
        root.update()
        path = filedialog.askopenfilename(master=root)

        if not path:
            st.session_state["browse_status"] = "No file selected."
            return

        st.session_state[consts.RAW_VIDEO_PATH] = path
        st.session_state["pb_trace_loaded"] = False
        st.session_state.pop(consts.TRIMMED_SLIDER, None)
        if _path_indicates_2ch(path):
            st.session_state[IS_2CH_USER_KEY] = True
        _refresh_2ch_mode_and_step_defaults(force=False)

        cage, mouse_name, fov, date, behavior, exp_details = pipe_utils.get_video_details(path)

        st.session_state.update({
            consts.CAGE: cage or "",
            consts.MOUSE_NAME: mouse_name or "",
            consts.EXPERIMENT_DATE: date or "",
            consts.FOV: fov or "",
            consts.BEHAVIOR: behavior or "",
            consts.EXPERIMENT_DETAILS: exp_details or "",
        })
        st.session_state["browse_status"] = f"Selected: {os.path.basename(path)}"
    except Exception as e:
        st.session_state["browse_status"] = f"Browse failed: {e}"
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass


def display_video_input(text_input="Raw video path",
                        key=consts.RAW_VIDEO_PATH,
                        browse_callback=choose_file,
                        status_key="browse_status",
                        button_key=None,
                        on_change=None
                        ):
    video_input_col_1, video_input_col_2 = st.columns([7,1])

    with video_input_col_1:
        st.text_input(f'**_{text_input}_**', key=key, on_change=on_change)
    with video_input_col_2:
        st.write("")
        st.write("")
        st.button('**_Browse_**', on_click=browse_callback, key=button_key)
    if status_key in st.session_state and st.session_state[status_key]:
        st.caption(st.session_state[status_key])


def display_mouse_details():
    mouse_details_col_1, mouse_details_col_2, mouse_details_col_3, mouse_details_col_4, mouse_details_col_5, mouse_details_col_6 = st.columns(6)
    with mouse_details_col_1:
        cage = st.text_input('**_Cage_**', key=consts.CAGE)
    with mouse_details_col_2:
        mouse_name = st.text_input('**_Mouse name_**',key=consts.MOUSE_NAME)
    with mouse_details_col_3:
        date = st.text_input('**_Date_**',key=consts.EXPERIMENT_DATE)
    with mouse_details_col_4:
        fov = st.text_input('**_Fov_**',key=consts.FOV)
    with mouse_details_col_5:
        behavior = st.text_input('**_Behavior_**', key=consts.BEHAVIOR)
    with mouse_details_col_6:
        exp_details = st.text_input('**_Experiment Details_**', key=consts.EXPERIMENT_DETAILS)
    return


def display_2ch_flag():
    st.checkbox(
        "**2-channel mode**",
        key=IS_2CH_USER_KEY,
        on_change=_on_2ch_checkbox_change,
        help='Enable 2-channel processing. If path contains "Hyp", this is auto-set to True initially.'
    )


def display_pipeline_steps():
    cols = st.columns(len(steps_registry.STEPS_REGISTRY) + 1)

    with cols[0]:
        st.markdown('**_Pipeline steps:_**')

    for i, (step_name, step_properties) in enumerate(steps_registry.STEPS_REGISTRY.items(), start=1):
        with cols[i]:
            GUI_parameter(
                step_name,
                BOOLEAN_INPUT,
                step_properties["default"],
                display_name=step_properties["display_name"]
            )

def run_pipeline_logic(session_time):
    run_col1, run_col2, run_col3, run_col4, run_col5 = st.columns(5)
    with run_col3:
        run_pipeline = st.button('**_run pipeline_**', type="primary")
    if run_pipeline:
        raw_video_path = str(st.session_state.get(consts.RAW_VIDEO_PATH, ""))
        if not raw_video_path.lower().endswith((".raw", ".tif")):
            st.warning('Enter a valid video path (.raw or .tif)', icon="âš ï¸")
        else:
            save_pipeline_params(session_time)


def _get_gui_params_from_session():
        gui_params = {}

        for k, v in st.session_state.items():
            gui_params[k] = data_utils.serialize_value(v)

        return gui_params


def _create_gui_params(gui_params):
    gui_time = datetime.datetime.now().strftime("%d-%m-%Y___%H-%M-%S")
    gui_params[consts.GUI_TIME] = gui_time
    gui_params[consts.RAW_VIDEO_PATH_LINUX] = pipe_utils.windows_to_linux_path(gui_params[consts.RAW_VIDEO_PATH])
    gui_params[consts.HOME_DIR_LINUX] = os.path.split(gui_params[consts.RAW_VIDEO_PATH_LINUX])[0]
    gui_params[consts.HOME_DIR] = os.path.split(gui_params[consts.RAW_VIDEO_PATH])[0]
    return gui_params


def save_pipeline_params(session_time):
    gui_params = _create_gui_params( _get_gui_params_from_session())

    pipe_dir_name = "_".join([gui_params[consts.CAGE], gui_params[consts.MOUSE_NAME], gui_params[consts.GUI_TIME]])
    pipe_dir = os.path.join(paths.PIPELINE_LOGS_DIR, session_time, pipe_dir_name)
    pipe_utils.mkdir(pipe_dir)

    param_file_path = os.path.join(pipe_dir, consts.PARAMS_FILE_SUFFIX_NAME[1:])
    with open(param_file_path , 'w') as fp:
        print("Saved gui_params to {}".format(param_file_path))
        json.dump(gui_params, fp, indent=4)
    return 


########## 2nd Tab: pipelines progress ###########
def display_pipelines_monitor(session_time):
    st.subheader("ðŸ“Š Pipelines Monitor")
    session_dir = os.path.join(paths.PIPELINE_LOGS_DIR, session_time)

    if not os.path.exists(session_dir):
        st.info("No pipelines yet.")
        return

    pipe_dirs = sorted(os.listdir(session_dir))

    if len(pipe_dirs) == 0:
        st.info("No pipelines yet.")
        return

    for pipe_id, pipe_dir_name in enumerate(pipe_dirs):
        pipe_dir = os.path.join(session_dir, pipe_dir_name)
        param_file = os.path.join(pipe_dir, consts.PARAMS_FILE_SUFFIX_NAME[1:])

        if not os.path.exists(param_file):
            continue

        with open(param_file) as f:
            gui_params = json.load(f)

        cage = gui_params.get(consts.CAGE, "")
        mouse = gui_params.get(consts.MOUSE_NAME, "")
        pipe_title = f"Pipeline #{pipe_id+1} â€” {cage} {mouse}"
        st.markdown(f"### {pipe_title}")

        # ---------- progress ----------
        steps_total = get_total_steps_num(gui_params)
        completed_steps = get_completed_steps(pipe_dir)
        progress_ratio = 0 if steps_total == 0 else completed_steps / steps_total
        st.progress(progress_ratio)

        # ---------- logs ----------
        log_files = glob.glob(os.path.join(pipe_dir, "*.txt"))
        log_files = sorted(log_files, key=os.path.getmtime)

        if len(log_files) == 0:
            st.info("Waiting for logs...")
            continue

        for log_file in log_files:
            step_name = os.path.basename(log_file).replace(".txt", "")

            with st.expander(step_name, expanded=False):

                try:
                    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                        logs = f.read()

                    # detect state
                    if consts.JOB_FAILED in logs or consts.JOB_CANCELLD in logs:
                        st.error(logs)
                    elif consts.JOB_FINISHED in logs:
                        st.success(logs)
                    else:
                        st.code(logs, language="bash")

                except Exception as e:
                    st.error(f"Could not read log: {e}")


def get_total_steps_num(gui_params):
    steps_num = 0
    steps_lst = list(steps_registry.STEPS_REGISTRY.keys())
    for step in steps_lst:
        if gui_params[step]:
            steps_num += 1
    return steps_num

def get_completed_steps(pipe_dir):
    completed_steps = 0 
    for file_name in os.listdir(pipe_dir):
        if file_name.endswith('.txt'):
            completed_steps += 1
    return completed_steps


########## 3rd Tab: run analyses ###########
########### Tabs wrapers #############
def display_run_pipeline_tab(run_pipeline_tab, session_time):
    with run_pipeline_tab:
        cols = st.columns([1,2,1])
        with cols[1]:
            display_mc_params()
            display_video_input(on_change=_on_raw_video_path_change)
            display_2ch_flag()
            display_pb_params()
            display_mouse_details()
            display_pipeline_steps()
            run_pipeline_logic(session_time)


def display_monitor_tab(monitor_tab, session_time):
    with monitor_tab:
        display_pipelines_monitor(session_time)


def main():
    st.set_page_config(layout="wide")
    st_autorefresh(interval=5000, key="global_refresh")
    session_time = init_pipeline_session()
    tabs = init_session_state()
    display_run_pipeline_tab(tabs[consts.RUN], session_time)
    display_monitor_tab(tabs[consts.MONITOR], session_time)

if __name__ == "__main__":
    main()
