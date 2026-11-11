// Which cells have the same spikes, shared by the pages that show or count them (the cluster map's "cells with the same spikes" view and the remote clusters panel).
// A cell is {cell, record: {spikes: [frame, ...]}}; two cells have the same spikes when, the smaller of the two ways round, at least `share` of the spikes of the one have a spike
// of the other within `frames`.
window.SameSpikes = (() => {
  // the share of the spikes of the one that have a spike of the other within `frames`, the smaller of the two ways round: 1 when the spikes are the same
  function shared(a, b, frames) {
    const covered = (from, to) => from.filter(s => to.some(u => Math.abs(u - s) <= frames)).length / from.length;
    return Math.min(covered(a, b), covered(b, a));
  }

  // the pair of cells (with at least minSpikes spikes) that comes closest to having the same spikes: {a, b, share}, or null
  function closest(cards, minSpikes, frames) {
    const cells = cards.filter(o => o.record.spikes.length >= minSpikes);
    let best = null;
    cells.forEach((a, i) => cells.slice(i + 1).forEach(b => { const share = shared(a.record.spikes, b.record.spikes, frames); if (!best || share > best.share) best = {a: a.cell, b: b.cell, share}; }));
    return best;
  }

  // groups of two or more cells (with at least minSpikes spikes) that are linked through pairs with the same spikes, the biggest first
  function groups(cards, minSpikes, frames, share) {
    const cells = cards.filter(o => o.record.spikes.length >= minSpikes), parent = cells.map((_, i) => i);
    const top = i => { while (parent[i] !== i) i = parent[i] = parent[parent[i]]; return i; };
    cells.forEach((a, i) => cells.slice(i + 1).forEach((b, k) => { if (shared(a.record.spikes, b.record.spikes, frames) >= share) parent[top(i)] = top(i + 1 + k); }));
    const found = new Map();
    cells.forEach((o, i) => found.set(top(i), [...(found.get(top(i)) || []), o]));
    return [...found.values()].filter(group => group.length > 1).sort((a, b) => b.length - a.length || a[0].cell - b[0].cell);
  }

  return {shared, closest, groups};
})();
