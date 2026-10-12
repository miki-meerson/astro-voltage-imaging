// The one JS definition of the sign convention (Python twin: app/services/polarity.py).
// UP / brighter / red = depolarization, DOWN / darker / blue = hyperpolarization. The raw astro fluorescence goes the other way;
// stored data keep their sign and the server flips what it sends, so views only use the words and colours below.
window.Polarity = (() => {
  const COLORS = {depolarizing: "#ff5d6c", hyperpolarizing: "#4fa3ff"};
  // cluster sign ("positive" / "negative", flipped STA convention) or a polarity word -> "depolarizing" | "hyperpolarizing"
  const of = s => (s === "positive" || s === "depolarizing") ? "depolarizing" : "hyperpolarizing";
  return {
    COLORS,
    DEP: "depolarizing", HYPER: "hyperpolarizing",
    of,
    color: s => COLORS[of(s)],
    AXIS_NOTE: "depolarization up",
  };
})();
