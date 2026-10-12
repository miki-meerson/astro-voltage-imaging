import os
import sys
import time
import paramiko
import json
import posixpath
import shlex

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import pipeline_constants as consts
from utils import files_paths as paths
from utils import credentials_miki as cred
from utils import pipeline_utils as pipe_utils
from utils import pipeline_registry as steps_registry


class PipelinesManager:
    def __init__(self, session_time):
        self.pipe_counter = -1
        self.pipelines = {}
        self.running_pipelines = []
        self.finished_pipelines = []
        self.failed_pipelines = []
        self.cancelld_pipelines = []
        self.gui_session_time = session_time
        self.param_files = []

    def fetch_new_pipelines(self):
        session_logs_dir = os.path.join(paths.PIPELINE_LOGS_DIR, self.gui_session_time)
        for name in os.listdir(session_logs_dir):
            dir_path = os.path.join(session_logs_dir, name)
            if os.path.isdir(dir_path):
                if name not in self.param_files:
                    param_file = os.path.join(dir_path, consts.PARAMS_FILE_SUFFIX_NAME[1:])
                    # Ignore non-pipeline folders (e.g., analysis jobs directory).
                    if not os.path.exists(param_file):
                        continue
                    self.param_files.append(name)
                    with open(param_file) as json_file:
                        gui_params = json.load(json_file)
                        self.create_new_pipeline(gui_params)
        return

    def create_new_pipeline(self, gui_params):
        self.pipe_counter += 1
        pipe = Pipeline(gui_params, self.pipe_counter)
        self.pipelines[self.pipe_counter] = pipe
        self.running_pipelines.append(self.pipe_counter)

    def manage_pipelines_steps(self):
        for pipe_num in self.running_pipelines:
            pipe = self.pipelines[pipe_num]
            if pipe.state == consts.WAITING:
                pipe.run_queued_step()
            elif pipe.state == consts.EXECUTING:
                pipe.manage_running_step()
            elif pipe.state == consts.FAILED:
                self.failed_pipelines.append(pipe_num)
            elif pipe.state == consts.FINISHED:
                self.finished_pipelines.append(pipe_num)
            elif pipe.state == consts.CANCELLD:
                self.cancelld_pipelines.append(pipe_num)
        return

    def save_pipelines_logs(self):
        for pipe_num in self.running_pipelines:
            pipe = self.pipelines[pipe_num]
            logs = pipe.get_step_info()  # (pipe.current_step.step_name, pipe.current_step.state, pipe.current_step.logs)
            pipe_log_dir = os.path.join(paths.PIPELINE_LOGS_DIR, self.gui_session_time, pipe.log_dir)
            if logs is not None:
                step_log_file = os.path.join(pipe_log_dir, logs[0] + '.txt')
                with open(step_log_file, 'w', encoding="utf-8") as f:
                    f.write(f"Job state: {logs[1]}\n")
                    if logs[2] is not None:
                        for line in logs[2]:
                            f.write(f"{line}\n")
        return

    def fetch_logs(self):
        logs = []
        for pipe_num in self.running_pipelines:
            pipe = self.pipelines[pipe_num]
            l = pipe.current_step.get_logs()
            logs.append(l)
            # send it to gui for some actions
        return logs

    def fetch_step_states(self):
        states = []  # this list just for tests
        for pipe_num in self.running_pipelines:
            pipe = self.pipelines[pipe_num]
            s = pipe.current_step.get_state()
            states.append(s)
            # send it to gui for some actions
        return states

    def pipeline_termination(self):
        """
        handle with crashed pipelines (should stop the pipeline and not continue to send more jobs from it)
        also should handle completion of a pipeline by telling it to the gui
        """
        for pipe_num in self.finished_pipelines + self.failed_pipelines + self.cancelld_pipelines:
            if pipe_num in self.running_pipelines:
                self.running_pipelines.remove(pipe_num)


class Pipeline:
    def __init__(self, gui_params, pipe_counter):
        gui_steps_lst = self._get_pipeline_steps_from_gui(gui_params)
        self.param_path = self._save_params_to_json(gui_params)
        self.pipeline_steps = self.parse_pipeline_steps(gui_steps_lst)
        self.state = consts.WAITING
        self.current_step = None
        self.completed_steps = []
        self.serial_num = pipe_counter
        self.log_dir = self._get_log_dir(gui_params)
        self.gui_params = gui_params

    def _get_log_dir(self, gui_params):
        pipe_dir = "_".join([
            gui_params[consts.CAGE], gui_params[consts.MOUSE_NAME], gui_params[consts.GUI_TIME]])
        return pipe_dir

    def _get_pipeline_steps_from_gui(self, gui_params):
        """
        return a list of tupels containing the different steps
        from GUI and a flag indicate if they need to be run:
        # [(step_name, run_flag) e.g. (MC, True)]  the order is matters!
        """
        steps_names = list(steps_registry.STEPS_REGISTRY.keys())
        gui_steps_lst = []
        for step in steps_names:
            if gui_params[step]:
                gui_steps_lst.append((step, True))
        return gui_steps_lst

    def _save_params_to_json(self, gui_params):
        home_dir = gui_params[consts.HOME_DIR]
        param_dir = os.path.join(home_dir, consts.PARAMS_DIR_NAME)
        pipe_utils.mkdir(param_dir)
        param_file_path = os.path.join(param_dir, gui_params[consts.GUI_TIME] + consts.PARAMS_FILE_SUFFIX_NAME)
        with open(param_file_path, 'w') as fp:
            json.dump(gui_params, fp, indent=4)
        param_file_for_cluster = pipe_utils.windows_to_linux_path(param_file_path)
        return param_file_for_cluster

    def parse_pipeline_steps(self, steps_lst):
        """
        return list of steps to sequentially run for this pipeline
        """
        pipeline_steps = [PipelineStep(step_name, self.param_path)
                          for step_name, run_step in steps_lst
                          if run_step]
        return pipeline_steps

    def run_queued_step(self):
        self.current_step = self.pipeline_steps.pop(0)
        print(f"Running {self.current_step}")
        self.current_step.run()
        self.state = consts.EXECUTING

    def manage_running_step(self):
        current_state = self.current_step.get_state()
        if current_state == consts.JOB_FAILED:
            self.state = consts.FAILED  # the manager will crash the pipeline
        elif current_state == consts.JOB_CANCELLD:
            self.state = consts.CANCELLD  # the manager will crash the pipeline
        elif current_state == consts.JOB_FINISHED:
            self.completed_steps.append(self.current_step)
            self.current_step = None
            if not self.pipeline_steps:  # if current step was the last
                self.state = consts.FINISHED
            else:
                self.state = consts.WAITING

    def get_step_info(self):
        if self.current_step is not None:
            return (self.current_step.step_name, self.current_step.state, self.current_step.logs)
        else:
            if len(self.completed_steps) > 0:
                return (self.completed_steps[-1].step_name, self.completed_steps[-1].state,
                        self.completed_steps[-1].logs)
            else:
                return None


class PipelineStep:
    """
    represent a single step of the pipeline.
    the step logic implemented in different py files - one per step.
    """

    def __init__(self, step_name, param_path):
        self.step_name = step_name
        self.params = param_path
        self.state = consts.NOT_STARTED
        self.cluster_script = self.get_cluster_script()
        self.cluster_job = ClusterJob(self.cluster_script, self.params)
        self.logs = None

    def get_cluster_script(self):
        return steps_registry.STEPS_REGISTRY[self.step_name]["script"]

    def run(self):
        self.cluster_job.run_job()

    def get_state(self):
        self._update_state()
        self.get_logs()
        return self.state

    def _update_state(self):
        self.state = self.cluster_job.update_state_2()

    def cancel_step(self):
        self.cluster_job.cancel_job()

    def get_logs(self):
        logs = self.cluster_job.get_job_logs()
        self.logs = logs
        return logs


class ClusterJob:
    def __init__(self, cluster_script, script_params=""):
        self.script = cluster_script
        self.params = script_params
        self.job_id = None
        self.log_file = None
        self.run_job_command = consts.RUN_JOB_COMMAND
        self.job_state_command = consts.JOB_STATE_COMMAND

        self.job_state_command_2 = consts.JOB_STATE_COMMAND_2

        self.cancel_job_command = consts.CANCEL_JOB_COMMAND
        self.get_log_path_command = consts.GET_LOG_PATH_COMMAND

    def run_job(self):
        ssh = SSH_connection()

        params_dir = posixpath.dirname(self.params)
        pipeline_dir = posixpath.dirname(params_dir)
        script_dir = posixpath.dirname(self.script)
        steps_dir = posixpath.dirname(script_dir)
        step_name = os.path.basename(self.script).replace(".sh", "")
        log_file_linux = posixpath.join(params_dir, f"{step_name}_%j.log")

        ssh.run_command(f"mkdir -p {pipeline_dir}")
        command = " ".join([
            self.run_job_command,
            f"--chdir={shlex.quote(steps_dir)}",
            f"--output={shlex.quote(log_file_linux)}",
            shlex.quote(self.script),
            shlex.quote(self.params),
        ])
        print("Submitting job:", command)
        output = ssh.run_command(command)

        try:
            self.job_id = [s for s in output[0].split() if s.isdigit()][0]
        except Exception:
            raise RuntimeError(f"Could not parse job id from sbatch output: {output}")

        # store real log path
        self.log_file = log_file_linux.replace("%j", self.job_id)
        ssh.close()

    def update_state(self):
        ssh = SSH_connection()
        command = self.job_state_command.format(self.job_id)
        output = ssh.run_command(command)
        ssh.close()
        state = self._parse_state(output)
        return state

    def update_state_2(self):
        ssh = SSH_connection()
        command = self.job_state_command_2.format(self.job_id)
        output = ssh.run_command(command)[0]
        ssh.close()
        state = self._parse_state_2(output)
        return state

    def _parse_state(self, output):
        if output == []:
            state = consts.NOT_STARTED
        elif output[0].strip().split(' ')[0] == consts.SLURM_PENDING:
            state = consts.JOB_PENDING
        elif output[0].strip().split(' ')[0] == consts.SLURM_RUNNING:
            state = consts.JOB_RUNNING
        elif output[0].strip().split(' ')[0] == consts.SLURM_FAILED:
            state = consts.JOB_FAILED
        elif output[0].strip().split(' ')[0] == consts.SLURM_FINISHED:
            state = consts.JOB_FINISHED
        elif output[0].strip().split(' ')[0] == consts.SLURM_CANCELLD:
            state = consts.JOB_CANCELLD
        else:
            print("not handel state")
            print(output)
        return state

    def _parse_state_2_old(self, output):
        if output == []:
            state = consts.NOT_STARTED
        elif consts.SLURM_PENDING in output:
            state = consts.JOB_PENDING
        elif consts.SLURM_RUNNING in output:
            state = consts.JOB_RUNNING
        elif consts.SLURM_FAILED in output:
            state = consts.JOB_FAILED
        elif consts.SLURM_FINISHED in output:
            state = consts.JOB_FINISHED
        elif consts.SLURM_CANCELLD in output:
            state = consts.JOB_CANCELLD
        else:
            print("not handel state")
            print(output)
        return state

    def _parse_state_2(self, output: str):
        out = (output or "").strip()

        # scontrol can't find job (often means it's already finished / purged)
        if "Invalid job id" in out or "slurm_load_jobs error" in out:
            # if log exists â†’ finished
            if self.log_file:
                return consts.JOB_FINISHED
            return consts.NOT_STARTED

        if not out:
            return consts.NOT_STARTED

        if consts.SLURM_PENDING in out:
            return consts.JOB_PENDING
        if consts.SLURM_RUNNING in out:
            return consts.JOB_RUNNING
        if consts.SLURM_FAILED in out:
            return consts.JOB_FAILED
        if consts.SLURM_FINISHED in out:
            return consts.JOB_FINISHED
        if consts.SLURM_CANCELLD in out:
            return consts.JOB_CANCELLD

        # fallback: never crash
        return consts.NOT_STARTED

    def cancel_job(self):
        ssh = SSH_connection()
        command = self.cancel_job_command.format(self.job_id)
        ssh.run_command(command)
        ssh.close()
        self.update_state_2()

    def _set_log_file(self):
        ssh = SSH_connection()
        command = self.get_log_path_command.format(self.job_id)
        output = ssh.run_command(command)
        ssh.close()

        if not output:
            print(f"[WARN] No log path returned for job {self.job_id}")
            return

        line = output[0].strip()

        if "=" not in line:
            print(f"[WARN] Unexpected log path output: {line}")
            return

        self.log_file = line.split("=")[1]


    def get_job_logs(self):
        state = self.update_state_2()

        if state in [consts.JOB_PENDING, consts.NOT_STARTED]:
            return ["Job not started yet"]

        if not self.log_file:
            self._set_log_file()

        if not self.log_file:
            return ["Log file not available yet"]

        ssh = SSH_connection()
        logs = ssh.run_command("cat " + self.log_file)
        ssh.close()

        return [row.strip() for row in logs]


class SSH_connection:

    def __init__(self):
        self.host = cred.CLUSTER_HOST
        self.port = cred.CLUSTER_PORT
        self.username = cred.SSH_USERNAME
        self.password = cred.SSH_PASSWORD
        self.timeout = 600  # [seconds]
        self.ssh = self._connect()

    def _connect(self):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(self.host, self.port, self.username, self.password, timeout=self.timeout)
        return ssh

    def run_command(self, command):
        stdin, stdout, stderr = self.ssh.exec_command(command)
        output = stdout.readlines()
        if output == []:
            output = stderr.readlines()
        return output

    def close(self):
        self.ssh.close()


def main(args):
    session_time = args[1]
    manager = PipelinesManager(session_time)
    while True:
        manager.fetch_new_pipelines()
        manager.manage_pipelines_steps()
        manager.pipeline_termination()
        manager.save_pipelines_logs()
        time.sleep(5)  # how often to check for updates (seconds)


if __name__ == "__main__":
    main(sys.argv)


