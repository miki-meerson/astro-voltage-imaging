// Light / dark switch: follows the system until the button is used; the choice is remembered in this browser only (storage may be blocked).
(() => {
  const root = document.documentElement;
  const read = () => { try { return localStorage.getItem("theme"); } catch (e) { return null; } };
  const write = value => { try { localStorage.setItem("theme", value); } catch (e) { /* the choice just is not remembered */ } };
  const saved = read();
  if (saved === "light" || saved === "dark") root.dataset.theme = saved;
  const current = () => root.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  document.querySelectorAll(".theme-toggle").forEach(button => {
    const label = () => { button.textContent = current() === "dark" ? "Light theme" : "Dark theme"; };
    label();
    button.addEventListener("click", () => {
      const next = current() === "dark" ? "light" : "dark";
      root.dataset.theme = next;
      write(next);
      label();
    });
  });
})();
