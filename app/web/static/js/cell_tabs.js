// Tabs of the STA panel (maps + movie / motion), one instance per .sta-tabs block.
document.querySelectorAll('.sta-tabs').forEach(function (root) {
  function load(pane) {
    var img = pane.querySelector('img.lazy');
    if (!img) return;
    if (img.getAttribute('src') !== img.dataset.src) img.setAttribute('src', img.dataset.src);
  }
  function show(name) {
    root.querySelectorAll('.tab').forEach(function (t) { t.classList.toggle('active', t.dataset.tab === name); });
    root.querySelectorAll('.tabpane').forEach(function (p) {
      var on = p.dataset.pane === name;
      p.classList.toggle('active', on);
      if (on) load(p);
    });
    try { localStorage.setItem('staTab', name); } catch (e) {}
  }
  root.querySelectorAll('.tab').forEach(function (t) { t.addEventListener('click', function () { show(t.dataset.tab); }); });
  var saved = null;
  try { saved = localStorage.getItem('staTab'); } catch (e) {}
  if (saved && root.querySelector('.tab[data-tab="' + saved + '"]')) show(saved);
});
