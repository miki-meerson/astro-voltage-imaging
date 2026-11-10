// The cell list of the Cells tab keeps its place when a cell is opened (the page loads again): the scroll position is kept, and the selected cell is brought into view when it is outside.
(function () {
  const list = document.querySelector(".cell-list");
  if (!list) return;
  const KEY = "cell-list-scroll";
  try { const saved = sessionStorage.getItem(KEY); if (saved !== null) list.scrollTop = Number(saved); } catch (_) {}
  const row = list.querySelector("tr.selected");
  if (row) {
    const top = row.getBoundingClientRect().top - list.getBoundingClientRect().top + list.scrollTop, bottom = top + row.offsetHeight;
    if (top < list.scrollTop || bottom > list.scrollTop + list.clientHeight) list.scrollTop = top - (list.clientHeight - row.offsetHeight) / 2;  // centred
  }
  list.addEventListener("scroll", () => { try { sessionStorage.setItem(KEY, String(list.scrollTop)); } catch (_) {} });
})();
