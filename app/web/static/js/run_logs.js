(function () {
  const RUN_ID = JSON.parse(document.getElementById("page-data").textContent).runId;
  async function refreshLogs() {
    try {
      const response = await fetch(`/runs/${RUN_ID}/logs`, {cache: "no-store"});
      if (!response.ok) return;
      const data = await response.json();
      const stdout = document.getElementById("stdout-log");
      const stderr = document.getElementById("stderr-log");
      stdout.textContent = data.stdout || "No stdout yet.";
      stderr.textContent = data.stderr || "No stderr yet.";
      stdout.scrollTop = stdout.scrollHeight;
      stderr.scrollTop = stderr.scrollHeight;
      if (!["pending", "queued", "running"].includes(data.status)) {
        window.location.reload();
      }
    } catch (_) {}
  }
  setInterval(refreshLogs, 1500);
  refreshLogs();
})();
