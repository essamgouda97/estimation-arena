"use strict";
const $ = (id) => document.getElementById(id);
const SUITS = ["♣", "♦", "♥", "♠"], RANKS = "23456789TJQKA", TRUMPS = [...SUITS, "NT"];
const POS = ["var(--king)", "var(--subking)", "var(--subkooz)", "var(--kooz)"];
const POS_NAMES = ["King", "Sub-King", "Sub-Kooz", "Kooz"];
const STILL = matchMedia("(prefers-reduced-motion: reduce)").matches;
const pct = (x) => `${Math.round(x * 100)}%`;
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function card(c, attrs = "") {
  const red = c >= 13 && c < 39 ? " red" : "";
  const r = RANKS[c % 13] === "T" ? "10" : RANKS[c % 13];
  return `<span class="card${red}" data-c="${c}" ${attrs}>${r}<b>${SUITS[Math.floor(c / 13)]}</b></span>`;
}

const S = { speed: 1, progress: 0, run: null, data: null, at: -1, snap: null, cache: new Map(), round: 0, step: 0, playing: false, auto: false };

// ---- the reel: one column per saved version, stacked by final position
function drawReel() {
  const vs = S.data.versions;
  $("reel").innerHTML = `<div class="head" id="head"></div>` + vs.map((v, i) => `<div class="v${i === S.at ? " on" : ""}${S.playing && i < S.at ? " done" : ""}" data-i="${i}" title="Iteration ${v.iteration.toLocaleString()}: ${v.quick.map((x, k) => `${POS_NAMES[k]} ${pct(x)}`).join(", ")}">${v.quick.map((x, k) => `<i style="height:${x * 100}%;background:${POS[k]}"></i>`).join("")}</div>`).join("");
  const r = $("reel");
  placeHead();
  r.setAttribute("aria-valuemin", 0); r.setAttribute("aria-valuemax", Math.max(vs.length - 1, 0)); r.setAttribute("aria-valuenow", S.at);
}
function placeHead() {
  const head = $("head");
  if (!head) return;
  const n = S.data.versions.length;
  head.style.display = S.playing ? "block" : "none";
  head.style.left = `${((S.at + S.progress) / Math.max(n, 1)) * 100}%`;
}
function pickFromX(x) {
  const r = $("reel").getBoundingClientRect();
  const n = S.data.versions.length;
  go(Math.max(0, Math.min(n - 1, Math.floor(((x - r.left) / r.width) * n))));
}
{
  const reel = $("reel");
  let down = false;
  reel.addEventListener("pointerdown", (e) => { down = true; stopPlay(); reel.setPointerCapture(e.pointerId); pickFromX(e.clientX); });
  reel.addEventListener("pointermove", (e) => { if (down) pickFromX(e.clientX); });
  reel.addEventListener("pointerup", () => { down = false; });
  reel.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") { stopPlay(); go(Math.max(0, S.at - 1)); e.preventDefault(); }
    if (e.key === "ArrowRight") { stopPlay(); go(Math.min(S.data.versions.length - 1, S.at + 1)); e.preventDefault(); }
  });
}

// ---- what it thinks: the fixed probes, updated in place so the colours morph between versions
function buildGrid() {
  let h = `<div></div>`;
  for (let n = 4; n <= 13; n++) h += `<div class="hd">${n}</div>`;
  TRUMPS.forEach((t, k) => {
    h += `<div class="hd${k === 1 || k === 2 ? " red" : ""}">${t}</div>`;
    for (let n = 4; n <= 13; n++) h += `<div class="cell" data-a="${2 + (n - 4) * 5 + k}" title="Bid ${n}${t}"><i></i><span></span></div>`;
  });
  $("grid").innerHTML = h;
}
function drawProbes(p) {
  if (!$("grid").children.length) buildGrid();
  $("probe-hand").innerHTML = [...p.hand].sort((a, b) => b - a).map((c) => card(c)).join("");
  const top = Math.max(0.0001, ...Object.entries(p.opening).filter(([a]) => +a !== 53).map(([, x]) => x));
  $("grid").querySelectorAll(".cell").forEach((cell) => {
    const x = p.opening[cell.dataset.a] ?? 0;
    cell.classList.toggle("off", !(cell.dataset.a in p.opening));
    const strength = Math.pow(x / top, 0.6) * Math.min(1, 0.15 + x * 4);
    const inner = cell.querySelector("i");
    inner.style.transitionDelay = `${(Math.floor((cell.dataset.a - 2) / 5) + (cell.dataset.a - 2) % 5) * 14}ms`;
    inner.style.opacity = Math.min(1, strength * 1.4);
    inner.style.transform = `scale(${Math.sqrt(strength)})`;
    cell.querySelector("span").textContent = x >= 0.1 ? pct(x) : "";
  });
  placeRing(p);
  const pass = p.opening["53"] ?? 0;
  $("dash-fill").style.width = pct(p.dash_call); $("dash-val").textContent = pct(p.dash_call);
  $("pass-fill").style.width = pct(pass); $("pass-val").textContent = pct(pass);
  const lead = Object.entries(p.lead).map(([c, x]) => [+c, x]).sort((a, b) => b[0] - a[0]);
  const ids = lead.map(([c]) => c).join();
  if ($("lead").dataset.ids !== ids) {
    $("lead").dataset.ids = ids;
    $("lead").innerHTML = lead.map(([c]) => `<div class="lead-col" data-c="${c}"><small></small><div class="lead-bar"></div>${card(c)}</div>`).join("");
  }
  lead.forEach(([c, x]) => {
    const col = $("lead").querySelector(`[data-c="${c}"].lead-col`);
    col.querySelector(".lead-bar").style.height = `${Math.round(x * 56)}px`;
    col.querySelector("small").textContent = x >= 0.05 ? pct(x) : "";
    col.classList.toggle("top", x === Math.max(...lead.map(([, y]) => y)));
  });
}
function topBid(p) {
  let best = null;
  for (const [a, x] of Object.entries(p.opening)) if (+a >= 2 && +a <= 51 && (best === null || x > p.opening[best])) best = a;
  return best;
}
function placeRing(p) {
  const grid = $("grid");
  let ring = grid.querySelector(".ring"), trail = grid.querySelector(".trail");
  if (!ring) {
    ring = document.createElement("div"); ring.className = "ring"; grid.append(ring);
    trail = document.createElementNS("http://www.w3.org/2000/svg", "svg"); trail.setAttribute("class", "trail"); grid.append(trail);
  }
  const box = grid.getBoundingClientRect();
  const at = (a) => { const c = grid.querySelector(`.cell[data-a="${a}"]`); return c && c.getBoundingClientRect(); };
  const best = at(topBid(p));
  if (!best || !box.width) return;
  ring.style.display = "block";
  ring.style.left = `${best.left - box.left - 3}px`; ring.style.top = `${best.top - box.top - 3}px`;
  ring.style.width = `${best.width + 6}px`; ring.style.height = `${best.height + 6}px`;
  const seen = S.data.versions.slice(0, S.at + 1).map((v) => at(topBid(v.probes))).filter(Boolean);
  const pts = seen.map((r) => [r.left - box.left + r.width / 2, r.top - box.top + r.height / 2]);
  trail.setAttribute("viewBox", `0 0 ${box.width} ${box.height}`);
  trail.innerHTML = `<path d="${pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`).join("")}"/>` + pts.map(([x, y]) => `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="2.5"/>`).join("");
}

// ---- a version
async function fetchSnap(i) {
  const it = S.data.versions[i].iteration;
  if (!S.cache.has(it)) S.cache.set(it, fetch(`/api/lab/${S.run}/snap/${it}`).then((r) => r.json()));
  return S.cache.get(it);
}
function setEnabled() {
  const have = S.data?.versions.length || 0;
  $("play").disabled = have < 2;
  for (const id of ["prev", "next", "auto"]) $(id).disabled = !S.snap;
}
function clearView(message) {
  stopPlay(); stopAuto();
  S.at = -1; S.snap = null; S.round = 0; S.step = 0;
  $("iter").textContent = "–";
  $("iter-sub").textContent = message;
  $("verified").hidden = true;
  for (const id of ["reel", "rounds", "think", "felt", "race", "probe-hand", "lead", "grid", "move", "board"]) $(id).innerHTML = "";
  delete $("lead").dataset.ids;
  $("dash-fill").style.width = $("pass-fill").style.width = "0";
  $("dash-val").textContent = $("pass-val").textContent = "";
  $("felt").innerHTML = `<p class="felt-empty">${esc(message)}</p>`;
  setEnabled();
}
function go(i) {
  if (!S.data.versions.length) return;
  S.at = i;
  const v = S.data.versions[i];
  const row = nearestRow(v.iteration);
  $("iter").textContent = v.iteration.toLocaleString();
  $("iter-sub").textContent = `Iteration, after ${row ? millions(row.decisions) : "?"} decisions of self-play`;
  drawReel();
  drawProbes(v.probes);
  const ok = $("verified");
  ok.hidden = false;
  ok.className = `verified${v.verified ? "" : " bad"}`;
  ok.textContent = v.verified ? "Both engines agree on this game" : "The two engines disagreed on this game";
  drawMultiples();
  return fetchSnap(i).then((snap) => {
    if (S.at !== i) return;
    S.snap = snap;
    const rounds = snap.game.rounds;
    S.round = Math.min(S.round, rounds.length - 1);
    S.step = Math.min(S.step, rounds[S.round].ev.length);
    if (!S.playing) drawRoundChips();
    drawTable(false);
    drawRace();
    setEnabled();
  });
}
const millions = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(2)} billion` : `${Math.round(n / 1e6)} million`);
function nearestRow(it) {
  const rows = S.data.rows;
  let best = null;
  for (const r of rows) if (!best || Math.abs(r.iteration - it) < Math.abs(best.iteration - it)) best = r;
  return best;
}

// ---- the film: every version plays the same round, full screen
const sleep = (ms) => new Promise((r) => setTimeout(r, STILL ? Math.min(ms, 60) : ms));
let filmId = 0;
function setSpeed(x) {
  S.speed = x;
  $("c-speed").querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.speed === x)));
  try { localStorage.setItem("lab-speed", String(x)); } catch { /* speed just isn't remembered */ }
}
$("c-speed").onclick = (e) => { const b = e.target.closest("button"); if (b) setSpeed(+b.dataset.speed); };
{
  let saved = 1;
  try { saved = +localStorage.getItem("lab-speed") || 1; } catch { /* use the default */ }
  setSpeed([0.5, 1, 2, 4].includes(saved) ? saved : 1);
}
const reel = (ms) => sleep(ms / S.speed); // film time: follows the speed control, even mid-wait
const moved = [];
function moveInto(node, target) { moved.push([node, node.parentNode, node.nextSibling]); target.append(node); }
function openCinema() {
  moveInto($("grid"), $("c-left")); moveInto($("board"), $("c-left")); moveInto($("think"), $("c-left"));
  moveInto($("felt"), $("c-felt")); moveInto($("reel"), $("c-reel"));
  $("cinema").hidden = false; document.body.style.overflow = "hidden";
  $("c-pause").focus();
}
function closeCinema() {
  filmId++; S.playing = false; S.paused = false; S.progress = 0;
  while (moved.length) { const [n, parent, next] = moved.pop(); parent.insertBefore(n, next); }
  $("cinema").hidden = true; document.body.style.overflow = "";
  $("c-pause").textContent = "Pause"; $("play").textContent = "Play timelapse";
  if (S.data?.versions.length) { placeRing(S.data.versions[S.at].probes); drawReel(); drawRoundChips(); }
  $("play").focus();
}
const stopPlay = () => { if (!$("cinema").hidden) closeCinema(); };
function rollCounter(to) {
  const el = $("c-iter"), from = +el.dataset.v || 0;
  el.dataset.v = to;
  if (STILL) { el.textContent = to.toLocaleString(); return; }
  const t0 = performance.now();
  const frame = (t) => {
    const k = Math.min(1, (t - t0) / 450);
    el.textContent = Math.round(from + (to - from) * (1 - (1 - k) ** 3)).toLocaleString();
    if (k < 1) requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}
function drawCinemaTop(v) {
  rollCounter(v.iteration);
  const row = nearestRow(v.iteration);
  $("c-sub").textContent = row ? `iterations, ${millions(row.decisions)} decisions` : "iterations";
  const bar = $("c-bar");
  if (!bar.children.length) bar.innerHTML = POS.map((c) => `<i style="background:${c}"></i>`).join("");
  [...bar.children].forEach((el, k) => { el.style.width = `${v.quick[k] * 100}%`; });
  $("c-names").innerHTML = POS_NAMES.map((n, k) => `<span>${n} ${pct(v.quick[k])}</span>`).join("");
}
async function waitWhilePaused(id) { while (S.paused && id === filmId) await sleep(100); }
async function film() {
  const id = ++filmId;
  S.playing = true; S.paused = false;
  openCinema();
  let i = S.at >= S.data.versions.length - 1 ? 0 : Math.max(S.at, 0);
  $("c-iter").dataset.v = S.data.versions[i].iteration;
  $("c-iter").textContent = S.data.versions[i].iteration.toLocaleString();
  while (id === filmId) {
    S.step = 0; S.progress = 0;
    await go(i);
    if (id !== filmId) return;
    drawCinemaTop(S.data.versions[i]);
    if (S.data.versions[i + 1]) fetchSnap(i + 1);
    const round = S.snap.game.rounds[Math.min(S.round, S.snap.game.rounds.length - 1)];
    const total = Math.max(2800, Math.min(7000, 80000 / S.data.versions.length));
    const per = Math.max(35, Math.min(180, (total - 900) / Math.max(round.ev.length, 1)));
    dealCards(Math.min(650, total * 0.15) / S.speed);
    await reel(Math.min(700, total * 0.17));
    for (let k = 0; k < round.ev.length && id === filmId; k++) {
      await waitWhilePaused(id);
      if (id !== filmId) return;
      S.step = k + 1;
      S.progress = (k + 1) / round.ev.length;
      placeHead();
      drawTable(true, { flight: Math.min(260, per * 1.7) / S.speed, bubble: Math.max(380, per * 4) / S.speed });
      await reel(per * (round.ev[k].w !== undefined ? 2.4 : 1));
    }
    if (id !== filmId) return;
    await reel(Math.max(2400, Math.min(3600, total * 0.6)));
    await waitWhilePaused(id);
    if (id !== filmId) return;
    i += 1;
    if (i >= S.data.versions.length) { i = 0; await reel(STILL ? 300 : 1200); clearTrail(); if (id !== filmId) return; }
  }
}
addEventListener("resize", () => { const v = S.data?.versions[S.at]; if (v) placeRing(v.probes); });
function clearTrail() { const t = $("grid").querySelector(".trail"); if (t) t.innerHTML = ""; }
$("play").onclick = () => { if (S.data?.versions.length > 1) film(); };
$("c-close").onclick = closeCinema;
$("c-pause").onclick = () => { S.paused = !S.paused; $("c-pause").textContent = S.paused ? "Resume" : "Pause"; };
addEventListener("keydown", (e) => {
  if ($("cinema").hidden) return;
  if (e.key === "Escape") closeCinema();
  if (e.key === " " && e.target.tagName !== "BUTTON") { e.preventDefault(); $("c-pause").click(); }
});
function dealCards(ms) {
  if (STILL) return;
  const felt = $("felt").getBoundingClientRect();
  const cx = felt.left + felt.width / 2, cy = felt.top + felt.height / 2;
  const cards = [...$("felt").querySelectorAll(".seat .card")];
  cards.forEach((el, i) => {
    const r = el.getBoundingClientRect();
    el.animate([
      { transform: `translate(${cx - r.left - r.width / 2}px,${cy - r.top - r.height / 2}px) rotate(${(i * 37) % 60 - 30}deg) scale(.6)`, opacity: 0 },
      { transform: "none", opacity: 1 },
    ], { duration: ms * 0.6, delay: (i / cards.length) * ms * 0.5, easing: "cubic-bezier(.2,.8,.2,1)", fill: "backwards" });
  });
}
function seatEl(k) { return $("felt").querySelector(`.seat.s${k}`); }
// ---- the table
function stateAt(round, step) {
  const hands = round.hands.map((h) => new Set(h));
  const st = { hands, trick: [], won: [0, 0, 0, 0], est: [null, null, null, null], calls: ["", "", "", ""], top: null, trump: round.fixed ?? null, caller: null, lastWin: null, turn: round.first, phase: round.fixed == null ? 0 : 2 };
  let clear = false;
  for (let i = 0; i < step; i++) {
    const e = round.ev[i];
    if (clear) { st.trick = []; clear = false; st.lastWin = null; }
    if (e.a >= 68) { hands[e.s].delete(e.a - 68); st.trick.push([e.s, e.a - 68]); } else { st.calls[e.s] = e.l; if (e.a >= 2 && e.a <= 51) st.top = e.l; }
    if (e.ph !== undefined) st.phase = e.ph;
    if (e.est) st.est = e.est;
    if (e.tr !== undefined) st.trump = e.tr;
    if (e.ca !== undefined) st.caller = e.ca;
    if (e.w !== undefined) { st.won[e.w]++; st.lastWin = e.w; clear = true; }
    st.turn = e.w !== undefined ? e.w : (round.ev[i + 1] ? round.ev[i + 1].s : null);
  }
  if (step < round.ev.length) st.turn = round.ev[step].s;
  return st;
}
function drawTable(animate = true, opts = {}) {
  if (!S.snap) return;
  const flight = opts.flight || 260;
  $("felt").querySelectorAll(".flyer").forEach((el) => el.remove());
  const before = new Map();
  if (animate && !STILL) $("felt").querySelectorAll(".card").forEach((el) => before.set(el.dataset.c, el.getBoundingClientRect()));
  const g = S.snap.game;
  S.round = Math.min(S.round, g.rounds.length - 1);
  S.step = Math.min(S.step, g.rounds[S.round].ev.length);
  const round = g.rounds[S.round], st = stateAt(round, S.step);
  const prior = S.step > 0 ? stateAt(round, S.step - 1) : null;
  const seat = (k) => {
    const est = st.est[k];
    const chips = [];
    if (st.caller === k) chips.push("Caller");
    else if (st.calls[k] === "With" && st.phase >= 2) chips.push("With");
    if (st.calls[k] === "Dash Call") chips.push("Dash");
    const said = st.phase < 2 && st.calls[k] && !["No Dash", "Dash Call"].includes(st.calls[k]) ? `<span class="said">${esc(st.calls[k])}</span>` : "";
    const hand = [...st.hands[k]].sort((a, b) => b - a);
    const cards = hand.map((c, i) => card(c, i && Math.floor(hand[i - 1] / 13) !== Math.floor(c / 13) ? 'data-gap="1"' : "")).join("");
    return `<div class="seat s${k}${st.turn === k && S.step < round.ev.length ? " turn" : ""}"><h4><span>${seatName(g, k)}</span>${chips.map((c) => `<em>${c}</em>`).join("")}${said}${pips(est, st.won[k])}</h4><div class="cards">${cards}</div></div>`;
  };
  const pit = st.trick.map(([s, c]) => card(c, `data-p`).replace('class="card', `class="card p${s}${st.lastWin === s ? " win" : ""}`)).join("");
  const done = S.step >= round.ev.length;
  const hud = st.trump == null && st.phase < 2 ? `<div class="hud"><p>Round ${round.no}</p><p>${st.phase === 0 ? "Dash Calls" : st.top ? `Top bid ${esc(st.top)}` : "Bidding"}${round.carry > 1 ? `, stakes ×${round.carry}` : ""}</p></div>`
    : `<div class="hud"><b class="${st.trump === 1 || st.trump === 2 ? "red" : ""}">${st.trump == null ? "" : TRUMPS[st.trump]}</b><p>${callLine(g, round, st)}</p><p>${totalLine(st)}${round.carry > 1 ? ` ×${round.carry}` : ""}</p></div>`;
  $("felt").innerHTML = seat(2) + seat(1) + `<div class="pit">${st.trick.length && !done ? "" : done ? "" : hud}${done ? "" : pit}</div>` + seat(3) + seat(0) + (done ? sheet(g, round, st) : "");
  if (before.size) {
    $("felt").querySelectorAll(".card").forEach((el) => {
      const a = before.get(el.dataset.c);
      if (!a) return;
      const b = el.getBoundingClientRect();
      const dx = a.left - b.left, dy = a.top - b.top;
      if (Math.abs(dx) + Math.abs(dy) < 2) return;
      const base = getComputedStyle(el).transform;
      const spin = el.closest(".pit") ? `rotate(${(+el.dataset.c * 53) % 50 - 25}deg) scale(1.15)` : "";
      el.animate([{ transform: `translate(${dx}px,${dy}px) ${spin} ${base === "none" ? "" : base}` }, { transform: base }], { duration: flight, easing: "cubic-bezier(.2,.8,.2,1)" });
    });
    sweepTrick(before, prior, flight);
    const e = round.ev[S.step - 1];
    if (e && e.a < 68) popBubble(e.s, e.l, opts.bubble || 900);
  }
  $("move").textContent = `Move ${S.step} of ${round.ev.length}`;
  drawBoard(g, round, st, done, animate);
  drawThink(round);
}
function drawBoard(g, round, st, done, animate) {
  const box = $("board");
  const old = new Map();
  if (animate && !STILL) box.querySelectorAll("[data-k]").forEach((el) => old.set(el.dataset.k, el.getBoundingClientRect().top));
  const totals = round.scores.map((x, k) => x + (done ? round.result[k] : 0));
  const places = standing(totals);
  const order = [0, 1, 2, 3].sort((a, b) => totals[b] - totals[a] || a - b);
  const trump = st.trump == null ? "" : `, trump ${TRUMPS[st.trump]}`;
  const call = st.caller == null ? (st.top ? `Top bid ${esc(st.top)}` : round.fixed != null ? "Estimating" : st.phase === 0 ? "Dash Calls" : "Bidding") : callLine(g, round, st);
  const fin = done && round === g.rounds.at(-1) ? "Final, " : "";
  const rows = order.map((k) => {
    const pts = done && !round.skipped ? `<span class="${round.result[k] < 0 ? "neg" : ""}">${signed(round.result[k])}</span>` : "<span></span>";
    const est = st.est[k] ?? "–";
    return `<div class="b-row${st.turn === k && !done ? " turn" : ""}" data-k="${k}"><span><i class="k${places[k].index}"></i>${places[k].name}</span><span>${seatName(g, k)}</span><span>${est}</span><span>${st.won[k]}</span>${pts}<b>${totals[k].toLocaleString()}</b></div>`;
  }).join("");
  box.innerHTML = `<h4><span>${fin}Round ${round.no}</span><span>${call}${trump}${round.carry > 1 ? `, ×${round.carry}` : ""}</span></h4><div class="b-row head"><span>Standing</span><span>Seat</span><span>Est</span><span>Won</span><span>Round</span><span>Game</span></div>${rows}`;
  if (old.size) box.querySelectorAll("[data-k]").forEach((el) => {
    const was = old.get(el.dataset.k);
    const dy = was === undefined ? 0 : was - el.getBoundingClientRect().top;
    if (Math.abs(dy) > 1) el.animate([{ transform: `translateY(${dy}px)` }, { transform: "none" }], { duration: 450, easing: "cubic-bezier(.2,.9,.3,1)" });
  });
}
function seatName(g, k) {
  const net = g.net_seats.includes(k);
  const same = [0, 1, 2, 3].filter((j) => g.net_seats.includes(j) === net);
  return `${net ? "Model" : "Heuristic"} ${same.indexOf(k) + 1}`;
}
function pips(est, won) {
  if (est === null || est === undefined) return won ? `<span class="pips">${"<i class=on></i>".repeat(won)}</span>` : "";
  const made = Math.min(est, won), left = Math.max(est - won, 0), over = Math.max(won - est, 0);
  return `<span class="pips" title="Won ${won} of ${est}">${"<i class=on></i>".repeat(made)}${"<i></i>".repeat(left)}${"<i class=over></i>".repeat(over)}${est === 0 && !won ? "<small>0</small>" : ""}</span>`;
}
function callLine(g, round, st) {
  if (st.caller === null || st.caller === undefined) return round.fixed != null ? "Fixed trump: estimates" : "Estimates";
  const contract = st.top ?? `${st.est[st.caller]}${st.trump == null ? "" : TRUMPS[st.trump]}`;
  return `${seatName(g, st.caller)} called ${esc(contract)}`;
}
function totalLine(st) {
  const known = st.est.filter((x) => x !== null && x !== undefined);
  if (known.length < 4) return "";
  const t = known.reduce((a, b) => a + b, 0);
  return `Estimates ${t}, ${t > 13 ? "over" : "under"} by ${Math.abs(t - 13)}`;
}
function standing(scores) {
  return scores.map((x) => {
    const above = scores.filter((y) => y > x).length, tied = scores.filter((y) => y === x).length;
    return { index: above, name: POS_NAMES[above], tied: tied > 1 };
  });
}
const signed = (x) => `${x > 0 ? "+" : ""}${x.toLocaleString()}`;
function sheet(g, round, st) {
  const totals = round.scores.map((x, k) => x + round.result[k]);
  const places = standing(totals);
  const order = [0, 1, 2, 3].sort((a, b) => totals[b] - totals[a] || a - b);
  const last = round === g.rounds.at(-1);
  const title = round.skipped ? `Round ${round.no} skipped: nobody called` : `Round ${round.no}: ${callLine(g, round, st)}`;
  const rows = order.map((k) => `<tr><td><i class="dot k${places[k].index}"></i>${places[k].tied ? "Tied " : ""}${places[k].name}</td><td>${seatName(g, k)}</td><td class="n">${round.skipped ? "" : st.est[k]}</td><td class="n">${round.skipped ? "" : st.won[k]}</td><td class="n ${round.result[k] < 0 ? "neg" : ""}">${round.skipped ? "" : signed(round.result[k])}</td><td class="n"><b>${totals[k].toLocaleString()}</b></td></tr>`).join("");
  const notes = [];
  if (!round.skipped) notes.push(totalLine(st) + (round.fixed != null ? `, trump ${TRUMPS[round.fixed]} fixed` : ""));
  if (round.carry > 1) notes.push(`Stakes ×${round.carry}`);
  if (round.skipped) notes.push("The next round's stakes double");
  if (!round.skipped && st.est.every((x, k) => x !== st.won[k])) notes.push("Everyone missed their estimate, so nobody scores and the next round's stakes double");
  let whole = "";
  if (!last) {
    const fin = standing(g.scores);
    const best = [0, 1, 2, 3].sort((a, b) => g.scores[b] - g.scores[a] || a - b);
    whole = `<p class="whole">This version's whole game, ${g.rounds.length} rounds: ${best.map((k) => `${fin[k].name} ${seatName(g, k)} ${g.scores[k].toLocaleString()}`).join(", ")}</p>`;
  }
  return `<div class="sheet"><h4>${last ? "Final standings. " : ""}${title}</h4><table><thead><tr><th>${last ? "Final place" : "Standing"}</th><th>Seat</th><th class="n">Est</th><th class="n">Won</th><th class="n">Round</th><th class="n">Game</th></tr></thead><tbody>${rows}</tbody></table>${notes.length ? `<p>${esc(notes.join(". "))}.</p>` : ""}${whole}</div>`;
}
function sweepTrick(before, prior, flight) {
  if (!prior || prior.lastWin === null || !prior.trick.length) return;
  const felt = $("felt"), box = felt.getBoundingClientRect(), to = seatEl(prior.lastWin).getBoundingClientRect();
  prior.trick.forEach(([, c], i) => {
    const a = before.get(String(c));
    if (!a) return;
    const el = document.createElement("div");
    el.innerHTML = card(c); const f = el.firstChild; f.classList.add("flyer");
    f.style.left = `${a.left - box.left}px`; f.style.top = `${a.top - box.top}px`; felt.append(f);
    const dx = to.left + to.width / 2 - a.left - a.width / 2, dy = to.top + to.height / 2 - a.top - a.height / 2;
    f.animate([{ transform: "none", opacity: 1 }, { transform: `translate(${dx}px,${dy}px) rotate(${i * 12 - 18}deg) scale(.5)`, opacity: 0 }], { duration: flight * 1.5, easing: "cubic-bezier(.5,0,.8,.4)" }).onfinish = () => f.remove();
  });
}
function popBubble(k, text, ms) {
  const felt = $("felt"), box = felt.getBoundingClientRect(), r = seatEl(k).querySelector("h4").getBoundingClientRect();
  const el = document.createElement("div");
  el.className = "bubble"; el.textContent = text;
  el.style.left = `${Math.min(box.width - 90, Math.max(40, r.left - box.left + r.width / 2))}px`; el.style.top = `${Math.max(4, r.top - box.top - 4)}px`;
  felt.append(el);
  el.animate([{ transform: "translate(-50%,8px) scale(.4)", opacity: 0 }, { transform: "translate(-50%,-4px) scale(1.08)", opacity: 1, offset: 0.2 }, { transform: "translate(-50%,-4px) scale(1)", opacity: 1, offset: 0.8 }, { transform: "translate(-50%,-14px)", opacity: 0 }], { duration: ms, easing: "ease-out" }).onfinish = () => el.remove();
}
function drawThink(round) {
  const e = round.ev[S.step];
  if (!e) { $("think").innerHTML = `<p class="none">End of the round.</p>`; return; }
  if (!S.snap.game.net_seats.includes(e.s)) { $("think").innerHTML = `<p class="none">Heuristic is playing ${esc(e.l)}.</p>`; return; }
  if (!e.p) { $("think").innerHTML = `<p class="none">Only one legal move: ${esc(e.l)}.</p>`; return; }
  $("think").innerHTML = e.p.map(([a, x]) => `<div class="opt${a === e.a ? " chosen" : ""}"><span>${esc(label(a))}</span><div class="t"><i style="width:${x * 100}%"></i></div><b>${pct(x)}</b></div>`).join("");
}
function label(a) {
  if (a === 0) return "No Dash"; if (a === 1) return "Dash Call";
  if (a <= 51) return `${Math.floor((a - 2) / 5) + 4}${TRUMPS[(a - 2) % 5]}`;
  if (a === 52) return "With"; if (a === 53) return "Pass";
  if (a < 68) return `Estimate ${a - 54}`;
  const c = a - 68; return `${RANKS[c % 13] === "T" ? "10" : RANKS[c % 13]}${SUITS[Math.floor(c / 13)]}`;
}
function drawRoundChips() {
  const rounds = S.snap.game.rounds;
  $("rounds").innerHTML = rounds.map((r, i) => `<button type="button" class="${i === S.round ? "on" : ""}" data-r="${i}" aria-label="Round ${r.no}">${r.no}</button>`).join("");
}
$("rounds").onclick = (e) => {
  const b = e.target.closest("button"); if (!b || !S.snap) return;
  S.round = +b.dataset.r; S.step = 0; stopAuto(); drawRoundChips(); drawTable(false); drawRace();
};
const move = (d) => { if (!S.snap) return; const n = S.snap.game.rounds[S.round].ev.length; S.step = Math.max(0, Math.min(n, S.step + d)); drawTable(d > 0); };
$("next").onclick = () => { stopAuto(); move(1); };
$("prev").onclick = () => { stopAuto(); move(-1); };
let autoTimer = null;
function stopAuto() { S.auto = false; clearTimeout(autoTimer); $("auto").textContent = "Play round"; }
$("auto").onclick = () => {
  if (S.auto) return stopAuto();
  if (!S.snap) return;
  S.auto = true; $("auto").textContent = "Pause";
  if (S.step >= S.snap.game.rounds[S.round].ev.length) S.step = 0;
  const t = () => {
    if (!S.auto) return;
    if (S.step >= S.snap.game.rounds[S.round].ev.length) return stopAuto();
    move(1); autoTimer = setTimeout(t, 650);
  };
  t();
};
function drawRace() {
  const g = S.snap.game, w = 420, h = 150, pad = 4;
  const pts = g.rounds.map((r) => r.scores.map((x, k) => x + r.result[k]));
  const all = pts.flat(); const lo = Math.min(0, ...all), hi = Math.max(1, ...all);
  const X = (i) => pad + (i / Math.max(pts.length - 1, 1)) * (w - pad * 2), Y = (v) => h - pad - ((v - lo) / (hi - lo)) * (h - pad * 2);
  const seats = [0, 1, 2, 3].map((k) => `<path d="${pts.map((p, i) => `${i ? "L" : "M"}${X(i).toFixed(1)} ${Y(p[k]).toFixed(1)}`).join("")}" fill="none" stroke="${g.net_seats.includes(k) ? "#f2c14e" : "#7fb0ff"}" stroke-width="2" stroke-linejoin="round" ${k > 1 ? 'stroke-dasharray="5 4"' : ""}/>`).join("");
  $("race").setAttribute("viewBox", `0 0 ${w} ${h}`);
  $("race").innerHTML = `<line x1="0" x2="${w}" y1="${Y(0)}" y2="${Y(0)}" stroke="#3a4a9a"/><line x1="${X(S.round)}" x2="${X(S.round)}" y1="0" y2="${h}" stroke="#fbf6e9" opacity=".45"/>${seats}`;
}

// ---- behaviour, one small chart per measure
const MEASURES = [
  ["caller_made", "Caller makes its bid", "Higher means bids it can actually make", (x) => pct(x)],
  ["all_lose", "Everyone lost the round", "Should fall as estimates improve", (x) => pct(x)],
  ["skipped", "Rounds skipped", "Nobody called, so stakes double", (x) => pct(x)],
  ["dash", "Dash Calls", "Share of Dash decisions", (x) => pct(x)],
  ["with_", "With", "Share of auction moves", (x) => pct(x)],
  ["pass_", "Pass", "Share of auction moves", (x) => pct(x)],
  ["super_", "Super calls", "Bids at fixed-trump estimation", (x) => pct(x)],
  ["rounds_per_game", "Rounds per game", "18 with no replays, 38 at the cap", (x) => x.toFixed(1)],
  ["entropy", "Randomness left", "Near 0 would mean it stopped exploring", (x) => x.toFixed(2)],
  ["final_mse", "Error predicting final position", "0.52 means no idea; lower is better", (x) => x.toFixed(3)],
  ["pos_vs_start", "Reward against its first version", "0 is even, the best possible is +0.6", (x) => (x > 0 ? "+" : "") + x.toFixed(2)],
  ["per_s", "Decisions per second", "Drops if something else needs the GPU", (x) => Math.round(x).toLocaleString()],
];
function drawMultiples() {
  const rows = S.data.rows, it = S.data.versions[S.at]?.iteration ?? rows.at(-1)?.iteration ?? 0;
  const maxIt = rows.at(-1)?.iteration || 1;
  $("multiples").innerHTML = MEASURES.map(([key, name, sub, fmt]) => {
    const pts = rows.filter((r) => r[key] !== undefined && !(key === "rounds_per_game" && !r[key]));
    if (pts.length < 2) return "";
    const vals = pts.map((r) => r[key]);
    const lo = Math.min(...vals), hi = Math.max(...vals), span = hi - lo || 1;
    const X = (i) => (i / maxIt) * 300, Y = (v) => 60 - ((v - lo) / span) * 56;
    const d = pts.map((r, i) => `${i ? "L" : "M"}${X(r.iteration).toFixed(1)} ${Y(r[key]).toFixed(1)}`).join("");
    const near = pts.reduce((b, r) => (Math.abs(r.iteration - it) < Math.abs(b.iteration - it) ? r : b), pts[0]);
    return `<div class="mult"><h3><span>${name}</span><b>${fmt(near[key])}</b></h3><small>${sub}</small><svg viewBox="0 0 300 64" preserveAspectRatio="none" aria-label="${name} over training"><path d="${d}" fill="none" stroke="#f8e3a8" stroke-width="1.6" vector-effect="non-scaling-stroke"/><line x1="${X(it)}" x2="${X(it)}" y1="0" y2="64" stroke="#fbf6e9" opacity=".5" vector-effect="non-scaling-stroke"/></svg></div>`;
  }).join("");
}

// ---- arena checks
const OPPONENT_COLORS = ["#e0b83c", "#35a7e8", "#e8707e", "#6fbf3f"];
const CHECK_METRICS = [
  { key: "reward", low: "low", high: "high", name: "Weighted rank", sub: "King +1, Sub-King +0.2, Sub-Kooz −0.2, Kooz −1. Even is 0.", even: 0, fmt: (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2) },
  { key: "king", low: "king_low", high: "king_high", name: "King only", sub: "1 for King, 0 for anything else. Even is 25%.", even: 0.25, fmt: pct },
  { key: "delta", low: "delta_low", high: "delta_high", name: "Score delta per game", sub: "Our seats' mean score minus theirs. Even is 0.", even: 0, fmt: (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(0) },
];
function opponentName(o) { return o === "pimc" ? "PIMC" : o[0].toUpperCase() + o.slice(1); }
function checkChart(m, checks, opponents, maxIt) {
  const values = checks.flatMap((c) => [c[m.key], c[m.low], c[m.high]]).filter((v) => v != null).concat(m.even);
  const lo = Math.min(...values), hi = Math.max(...values), span = hi - lo || 1;
  const X = (i) => 12 + (i / maxIt) * 276, Y = (v) => 92 - ((v - lo) / span) * 84;
  const series = opponents.map((o, k) => {
    const pts = checks.filter((c) => c.opponent === o), color = OPPONENT_COLORS[k % OPPONENT_COLORS.length];
    const line = pts.length > 1 ? `<path d="${pts.map((c, i) => `${i ? "L" : "M"}${X(c.iteration).toFixed(1)} ${Y(c[m.key]).toFixed(1)}`).join("")}" fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round"/>` : "";
    const dots = pts.map((c) => {
      const whisker = c[m.low] != null ? `<line x1="${X(c.iteration)}" x2="${X(c.iteration)}" y1="${Y(c[m.low])}" y2="${Y(c[m.high])}" stroke="${color}" stroke-width="1.5" opacity=".55"/>` : "";
      return `<g class="check-dot">${whisker}<circle cx="${X(c.iteration)}" cy="${Y(c[m.key])}" r="4" fill="${color}" stroke="#17235e" stroke-width="2"/><circle cx="${X(c.iteration)}" cy="${Y(c[m.key])}" r="11" fill="transparent"><title>${opponentName(o)}, iteration ${c.iteration.toLocaleString()}: ${m.fmt(c[m.key])}${c[m.low] != null ? ` (${m.fmt(c[m.low])} to ${m.fmt(c[m.high])})` : ""}, sets ${c.won}-${c.lost}</title></circle></g>`;
    }).join("");
    return line + dots;
  }).join("");
  const latest = opponents.map((o) => { const pts = checks.filter((c) => c.opponent === o); return pts.length ? m.fmt(pts[pts.length - 1][m.key]) : ""; }).join(" / ");
  return `<div class="mult"><h3><span>${m.name}</span><b>${latest}</b></h3><small>${m.sub}</small><svg viewBox="0 0 300 100" role="img" aria-label="${m.name} at each check"><line x1="0" x2="300" y1="${Y(m.even)}" y2="${Y(m.even)}" stroke="#aab4e6" stroke-width="1" stroke-dasharray="3 4"/>${series}</svg></div>`;
}
function drawChecks() {
  const c = S.data.checks;
  if (!c.length) {
    $("check-legend").innerHTML = $("check-charts").innerHTML = "";
    $("checks").innerHTML = `<p class="empty">The first check runs 30 minutes into training.</p>`;
    return;
  }
  const opponents = [...new Set(c.map((x) => x.opponent))];
  const maxIt = Math.max(...c.map((x) => x.iteration));
  $("check-legend").innerHTML = opponents.map((o, k) => `<span><i style="background:${OPPONENT_COLORS[k % OPPONENT_COLORS.length]}"></i>vs ${opponentName(o)}</span>`).join("") + `<span class="check-even">Dashed line: an even match. Whiskers: 95% interval.</span>`;
  $("check-charts").innerHTML = CHECK_METRICS.map((m) => checkChart(m, c, opponents, maxIt)).join("");
  $("checks").innerHTML = c.map((x) => `<div class="check"><span>Iteration ${x.iteration.toLocaleString()} vs ${opponentName(x.opponent)}</span><div class="bar" role="img" aria-label="${x.shares.map((s, k) => `${POS_NAMES[k]} ${pct(s)}`).join(", ")}">${x.shares.map((s, k) => `<i style="width:${s * 100}%;background:${POS[k]}"></i>`).join("")}</div><span>${CHECK_METRICS[0].fmt(x.reward)}</span><span>King ${pct(x.king)}</span><span>${CHECK_METRICS[2].fmt(x.delta)}</span><span>${x.won}-${x.lost}</span></div>`).join("");
}

// ---- loading
async function loadRun(run, keepAt) {
  S.run = run;
  const res = await fetch(`/api/lab/${run}/summary`);
  S.data = await res.json();
  const latest = S.data.versions.length - 1;
  drawChecks();
  if (latest < 0) { clearView(`No versions were recorded for ${run}. Pick another run.`); drawMultiples(); return; }
  setEnabled();
  if (S.playing) { drawReel(); return; }
  go(keepAt && S.at >= 0 && S.at < latest ? S.at : latest);
}
(async () => {
  const runs = await (await fetch("/api/lab/runs")).json();
  $("run").innerHTML = runs.map((r) => `<option>${esc(r)}</option>`).join("");
  if (!runs.length) { $("iter-sub").textContent = "No training runs yet"; return; }
  $("run").value = runs.at(-1);
  $("run").onchange = () => { stopPlay(); stopAuto(); S.cache.clear(); S.at = -1; S.snap = null; S.round = 0; S.step = 0; $("rounds").innerHTML = ""; $("felt").innerHTML = ""; setEnabled(); loadRun($("run").value); };
  await loadRun($("run").value);
  setInterval(() => loadRun(S.run, true), 30000);
})();
