// Rules for what the user may choose, checked in the page so the server can trust the form.
// A form with data-require-one="<field name>" cannot be submitted while none of its checkboxes with that name is ticked.
(function () {
  document.querySelectorAll("form[data-require-one]").forEach(form => {
    const boxes = form.querySelectorAll(`input[type=checkbox][name="${form.dataset.requireOne}"]`);
    const button = form.querySelector("button:not([type=button])");
    const message = form.dataset.requireMessage || "Tick at least one option.";
    const update = () => {
      const ok = Array.from(boxes).some(box => box.checked);
      button.disabled = !ok;
      button.title = ok ? "" : message;
    };
    boxes.forEach(box => box.addEventListener("change", update));
    update();
  });
})();
