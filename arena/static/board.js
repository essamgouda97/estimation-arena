const POLL_MS = 5000;
const COLUMNS = 7; // rank, bot, author, three ratings, finishing places
const UNRANKED_NOTE = "These bots have a rating from a few games but no place: they are being tested, or can't play on the nightly ladder.";
const $ = (id) => document.getElementById(id);
const metric = () => document.body.dataset.metric;
let state = null, firstElo = {}, lastOk = null, failed = false, seenSets = {}, openKey = null, details = {}, drawn = "";

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}
const signed = (x, digits = 0) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(digits);
const pct = (x) => (100 * x).toFixed(1) + "%";

function duration(seconds) {
  if (seconds < 90) return "under 2 min";
  if (seconds < 5400) return `about ${Math.round(seconds / 60)} min`;
  return `about ${(seconds / 3600).toFixed(1)} h`;
}

function record(strip) {
  const count = (c) => (strip || "").split(c).length - 1;
  return `${count("W")}-${count("D")}-${count("L")}`;
}

function stats(m) {
  const box = el("div", "stats");
  if (!m.done) return box.appendChild(el("span", "", "Waiting for the first set")), box;
  const rows = [["Weighted rank", signed(m.reward.mean, 2)], ["King only", pct(m.king.mean)], ["Score delta", signed(m.delta.mean, 0)], ["Sets won, drawn, lost on weighted rank", record(m.strip)]];
  for (const [label, value] of rows) {
    const s = el("span", "", label + " ");
    s.appendChild(el("b", "", value));
    box.appendChild(s);
  }
  return box;
}

function strip(m) {
  const box = el("div", "strip");
  box.setAttribute("role", "img");
  box.setAttribute("aria-label", `${m.done} of ${m.sets} sets played, record ${record(m.strip)}`);
  const seen = seenSets[m.id] ?? m.done;
  for (let k = 0; k < m.sets; k++) {
    const c = (m.strip || "")[k];
    box.appendChild(el("i", c ? c + (k >= seen ? " new" : "") : ""));
  }
  seenSets[m.id] = m.done;
  return box;
}

function matchRow(m, live) {
  const box = el("article", "match");
  const head = el("div", "match-head");
  head.appendChild(el("span", "match-title", `${m.name} vs ${m.opponent}`));
  if (live) {
    const left = m.done ? ((Date.now() / 1000 - m.started_at) / m.done) * (m.sets - m.done) : null;
    head.appendChild(el("span", "match-time", `${m.done} of ${m.sets} sets` + (left === null ? "" : `, ${duration(left)} left`)));
  } else if (m.state === "stopped") {
    head.appendChild(el("span", "match-time stopped", `Stopped after ${m.done} of ${m.sets} sets`));
  } else {
    head.appendChild(el("span", "match-time", `${m.sets} sets, finished ${new Date(m.updated_at * 1000).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}`));
  }
  box.append(head, strip(m), stats(m));
  return box;
}

function places(shares) {
  if (!shares) return el("span", "places-none", "No games yet");
  const bar = el("div", "places");
  const names = ["King", "Sub-King", "Sub-Kooz", "Kooz"];
  bar.setAttribute("role", "img");
  bar.setAttribute("aria-label", names.map((n, k) => `${n} ${Math.round(shares[k] * 100)}%`).join(", "));
  bar.title = bar.getAttribute("aria-label");
  ["k", "sk", "sko", "ko"].forEach((cls, k) => {
    const part = el("i", cls);
    part.style.flex = `${Math.max(shares[k], 0.001)} 0 0`;
    bar.appendChild(part);
  });
  return bar;
}

function eloCell(row, field) {
  const td = el("td", `num elo m-${field}`);
  const value = row[field];
  td.appendChild(document.createTextNode(value == null ? "–" : Math.round(value)));
  const margin = row[field + "_margin"];
  if (value != null && margin != null) td.appendChild(el("span", "pm", ` ±${Math.round(margin)}`));
  const start = firstElo[row.key]?.[field];
  const moved = value != null && start != null ? Math.round(value - start) : 0;
  td.appendChild(el("span", "elo-move", moved ? `${signed(moved)} since opened` : ""));
  return td;
}

async function loadDetail(key) {
  details[key] = { pending: true };
  try {
    const res = await fetch("/api/board/bot?key=" + encodeURIComponent(key), { cache: "no-store" });
    if (!res.ok) throw new Error(res.status);
    details[key] = await res.json();
  } catch {
    details[key] = { failed: true };
  }
  if (openKey === key) renderRows(true, false);
}

function detailRow(key) {
  const tr = el("tr", "detail");
  const td = el("td");
  td.colSpan = COLUMNS;
  tr.appendChild(td);
  const shell = el("div", "detail-shell");
  td.appendChild(shell);
  const about = details[key];
  if (!about || about.pending) {
    shell.appendChild(el("p", "empty", "Loading"));
  } else if (about.failed) {
    shell.appendChild(el("p", "stopped", "Couldn't load this bot's explanation. Close it and open it again to retry."));
  } else {
    if (about.description) shell.appendChild(el("p", "detail-text", about.description));
    if (about.code_name) shell.appendChild(el("p", "detail-code", `Code name: ${about.code_name}`));
    if (about.method) {
      shell.appendChild(el("h3", "", "How it decides"));
      const steps = el("div", "steps");
      steps.innerHTML = about.method;
      shell.appendChild(steps);
    }
    const trained = about.training;
    shell.appendChild(el("h3", "", "How it was trained"));
    shell.appendChild(el("p", "detail-text", trained.summary || "No training notes for this bot yet."));
    for (const stage of trained.stages) {
      shell.appendChild(el("h4", "", stage.title));
      const steps = el("div", "steps");
      steps.innerHTML = stage.svg;
      shell.appendChild(steps);
    }
    if (trained.curves || trained.lab) {
      const links = el("p", "detail-links");
      if (trained.curves) {
        const a = el("a", "", "See its training charts");
        a.href = "/tb/?runFilter=" + encodeURIComponent(trained.curves) + "#timeseries";
        links.appendChild(a);
      }
      if (trained.lab) {
        const a = el("a", "", "Watch it learn in the lab");
        a.href = "/lab";
        links.appendChild(a);
      }
      shell.appendChild(links);
    }
  }
  return tr;
}

function toggle(key) {
  openKey = openKey === key ? null : key;
  if (openKey) {
    if (!details[key] || details[key].failed) loadDetail(key);
  }
  renderRows(true, false);
}

function unrankedDivider() {
  const tr = el("tr", "unranked"), td = el("td");
  td.colSpan = COLUMNS;
  td.append(el("strong", "", "Not ranked yet"), el("span", "", UNRANKED_NOTE));
  tr.appendChild(td);
  return tr;
}

function renderRows(force = false, animate = true) {
  const body = $("rows"), field = metric();
  const signature = JSON.stringify([field, openKey, state.standings, state.playing.map((m) => m.key)]);
  if (!force && signature === drawn) return;
  drawn = signature;
  const focused = document.activeElement?.closest?.("tr")?.dataset.key;
  const scrolled = [...body.querySelectorAll(".steps")].map((box) => box.scrollLeft);
  const shown = [...body.querySelectorAll("tr[data-key]")];
  const before = new Map(shown.map((tr) => [tr.dataset.key, tr.getBoundingClientRect().top]));
  const previous = new Map(shown.map((tr) => [tr.dataset.key, tr.dataset.value]));
  const playing = new Set(state.playing.map((m) => m.key).concat(state.playing.map((m) => m.opponentKey)));
  const unranked = (row) => (row.provisional ? 1 : 0);
  const rows = [...state.standings].sort((a, b) => unranked(a) - unranked(b) || (b[field] ?? -1e9) - (a[field] ?? -1e9));
  const firstUnranked = rows.find(unranked);
  let place = 0;
  body.replaceChildren(...rows.flatMap((row) => {
    const tr = el("tr");
    tr.dataset.key = row.key;
    tr.dataset.value = row[field];
    tr.appendChild(el("td", "num rank", row[field] == null || row.provisional ? "" : ++place));
    const name = el("td", "bot");
    const open = el("button", "bot-name", row.name);
    open.setAttribute("aria-expanded", openKey === row.key);
    open.addEventListener("click", () => toggle(row.key));
    name.appendChild(open);
    const games = el("a", "tag", "Games");
    games.href = "/games/#bot=" + encodeURIComponent(row.key);
    name.appendChild(games);
    if (row.anchor) name.appendChild(el("span", "tag", "Benchmark"));
    if (row.provisional) name.appendChild(el("span", "tag", `${row.games} games so far`));
    if (playing.has(row.key)) name.appendChild(el("span", "tag on", "Playing"));
    if (row.close_to && field === "elo") name.appendChild(el("span", "tag", `Too close to call with ${row.close_to}`));
    name.appendChild(el("span", "by", row.author));
    tr.appendChild(name);
    tr.appendChild(el("td", "author", row.author));
    for (const f of ["elo", "king_elo", "score_elo"]) tr.appendChild(eloCell(row, f));
    const cell = el("td", "places-cell");
    cell.appendChild(places(row.positions));
    tr.appendChild(cell);
    if (previous.has(row.key) && previous.get(row.key) !== tr.dataset.value) tr.classList.add("flash");
    const own = openKey === row.key ? [tr, detailRow(row.key)] : [tr];
    return row === firstUnranked ? [unrankedDivider(), ...own] : own;
  }));
  if (focused) body.querySelector(`tr[data-key="${CSS.escape(focused)}"] .bot-name`)?.focus({ preventScroll: true });
  body.querySelectorAll(".steps").forEach((box, k) => { box.scrollLeft = scrolled[k] ?? 0; });
  if (!animate || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  for (const tr of body.querySelectorAll("tr[data-key]")) {
    const was = before.get(tr.dataset.key);
    if (was === undefined) continue;
    const shift = was - tr.getBoundingClientRect().top;
    if (shift) tr.animate([{ transform: `translateY(${shift}px)` }, { transform: "none" }], { duration: 500, easing: "ease-out" });
  }
  setTimeout(() => body.querySelectorAll(".flash").forEach((tr) => tr.classList.remove("flash")), 1600);
}

function renderDirect() {
  $("direct-section").hidden = !state.direct.length;
  if (!state.direct.length) return;
  const bots = state.standings.filter((r) => !r.anchor && !r.provisional).slice(0, 8);
  const find = (a, b) => state.direct.find((m) => m.a === a && m.b === b);
  const head = el("tr");
  head.append(el("th", "who", "Bot"), ...bots.map((r, k) => {
    const th = el("th", "num", k + 1);
    th.title = r.name;
    return th;
  }));
  $("cross-head").replaceChildren(head);
  $("cross-body").replaceChildren(...bots.map((row, k) => {
    const tr = el("tr");
    const who = el("td", "who");
    who.append(el("span", "rank", k + 1), ` ${row.name}`);
    tr.appendChild(who);
    bots.forEach((other, j) => {
      if (k === j) return tr.appendChild(el("td", "num self"));
      const match = find(row.key, other.key), flip = find(other.key, row.key);
      if (!match && !flip) return tr.appendChild(el("td", "num close", "-"));
      const mine = match ? match.a_wins : flip.b_wins, theirs = match ? match.b_wins : flip.a_wins;
      const kind = !(match || flip).clear ? "close" : mine > theirs ? "won" : "lost";
      tr.appendChild(el("td", `num ${kind}`, `${mine}-${(match || flip).draws}-${theirs}`));
    });
    return tr;
  }));
  const names = Object.fromEntries(state.standings.map((r) => [r.key, r.name]));
  $("runoff").textContent = state.runoff.length ? `Runoff winner: ${names[state.runoff[0]]}. The runoff drops the bot scoring least against those left, until one remains.` : "";
}

function render() {
  for (const row of state.standings) {
    firstElo[row.key] ??= { elo: row.elo, king_elo: row.king_elo, score_elo: row.score_elo };
  }
  const keyByName = Object.fromEntries(state.standings.map((r) => [r.name, r.key]));
  state.playing.forEach((m) => { m.opponentKey = keyByName[m.opponent]; });
  const list = (id, matches, live, empty) => $(id).replaceChildren(...(matches.length ? matches.map((m) => matchRow(m, live)) : [el("p", "empty", empty)]));
  list("playing", state.playing, true, "No matches are playing right now.");
  list("recent", state.recent, false, "Finished matches appear here.");
  $("standings-title").textContent = state.season ? `Standings, season ${state.season.number}` : "Standings";
  renderRows();
  renderDirect();
}

function showStatus() {
  const box = $("status");
  box.classList.toggle("bad", failed);
  if (!lastOk) { box.textContent = failed ? "Can't reach the server. Retrying." : "Loading"; return; }
  const ago = Math.round((Date.now() - lastOk) / 1000);
  box.textContent = failed ? `Can't reach the server. Last update ${ago} s ago, retrying.` : `Updated ${ago} s ago`;
}

async function poll() {
  if (!document.hidden) {
    try {
      const res = await fetch("/api/board", { cache: "no-store" });
      if (!res.ok) throw new Error(res.status);
      state = await res.json();
      lastOk = Date.now(); failed = false;
      render();
    } catch { failed = true; }
    showStatus();
  }
  setTimeout(poll, POLL_MS);
}

document.querySelectorAll("[data-rank]").forEach((button) => button.addEventListener("click", () => {
  document.body.dataset.metric = button.dataset.rank;
  document.querySelectorAll("[data-rank]").forEach((b) => b.setAttribute("aria-pressed", b === button));
  if (state) renderRows(true);
}));
document.addEventListener("visibilitychange", () => { if (!document.hidden) showStatus(); });
setInterval(showStatus, 1000);
fetch("/api/board/guide").then((r) => r.json()).then((guide) => {
  const style = document.createElement("style");
  style.textContent = guide.css;
  document.head.appendChild(style);
  $("guide-slot").innerHTML = guide.html;
  if (location.hash === "#guide") document.getElementById("guide").scrollIntoView();
}).catch(() => {});
poll();
