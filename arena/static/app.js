"use strict";
const $ = (html) => { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstChild; };
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const SUITS = ["♣", "♦", "♥", "♠"], RANKS = "23456789TJQKA";
const app = document.getElementById("app");

function cardHtml(c, cls = "") {
  const red = c >= 13 && c < 39 ? " red" : "";
  const r = RANKS[c % 13] === "T" ? "10" : RANKS[c % 13];
  return `<span class="card${red} ${cls}" data-card="${c}"><span>${r}</span><span class="s">${SUITS[Math.floor(c / 13)]}</span></span>`;
}

// ---------------------------------------------------------------- lobby
async function lobby() {
  const bots = await (await fetch("/api/bots")).json();
  const options = ['<option value="human">Person</option>']
    .concat(bots.map((b) => `<option value="${esc(b.spec)}">${esc(b.name)}${b.group === "anchor" ? " · " + esc(b.author) : ""}</option>`)).join("");
  const defaults = ["human", "hybrid", "sub:rl-hybrid", "heuristic"];
  app.innerHTML = `<h1>Estimation Table</h1><p class="muted">Seat people and bots in any mix. People get an invite link each; bots play here.</p>
  <div class="panel"><b>New table</b>${[0, 1, 2, 3].map((k) => `
    <div class="seatrow"><span class="muted">Seat ${k + 1}</span><select id="kind${k}">${options}</select>
    <input id="name${k}" placeholder="Name" maxlength="24" value="${k === 0 ? "Me" : ""}"></div>`).join("")}
    <div style="margin-top:10px"><button class="primary" id="create">Create table</button></div><div id="made" class="links"></div></div>
  <div class="panel"><b>Open tables</b><div id="list" class="small muted">loading…</div></div>`;
  [0, 1, 2, 3].forEach((k) => {
    const sel = document.getElementById("kind" + k), name = document.getElementById("name" + k);
    sel.value = bots.some((b) => b.spec === defaults[k]) || defaults[k] === "human" ? defaults[k] : "heuristic";
    const sync = () => { name.style.visibility = sel.value === "human" ? "visible" : "hidden"; };
    sel.onchange = sync; sync();
  });
  document.getElementById("create").onclick = async () => {
    const seats = [0, 1, 2, 3].map((k) => {
      const v = document.getElementById("kind" + k).value;
      return v === "human" ? { kind: "human", name: document.getElementById("name" + k).value } : { kind: "bot", spec: v };
    });
    const res = await (await fetch("/api/tables", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ seats }) })).json();
    const made = document.getElementById("made");
    if (res.error) { made.innerHTML = `<p class="err">${esc(res.error)}</p>`; return; }
    const people = res.seats.map((s, k) => [s, k]).filter(([s]) => s.link);
    made.innerHTML = `<p class="small muted">Send each person their own link. Anyone with the watch link sees the table, never the hands.</p>` +
      people.map(([s, k]) => `<div><b>Seat ${k + 1} · ${esc(s.name)}</b> <button data-copy="${esc(location.origin + s.link)}">Copy link</button> <a href="${esc(s.link)}">open</a></div>`).join("") +
      `<div><b>Watch</b> <button data-copy="${esc(location.origin + res.watch)}">Copy link</button> <a href="${esc(res.watch)}">open</a></div>`;
    made.querySelectorAll("[data-copy]").forEach((b) => (b.onclick = () => { navigator.clipboard?.writeText(b.dataset.copy); b.textContent = "Copied"; }));
    if (people.length === 1) location.href = people[0][0].link;
    loadList();
  };
  loadList();
}

async function loadList() {
  const rows = await (await fetch("/api/tables")).json();
  const el = document.getElementById("list");
  if (!el) return;
  el.innerHTML = rows.length ? `<table>${rows.map((r) => `<tr><td>${esc(r.names.join(" · "))}</td><td class="n">${r.over ? "finished" : "round " + r.round}</td><td class="n"><a href="/t/${esc(r.id)}">watch</a></td></tr>`).join("")}</table>` : "None yet.";
}

// ---------------------------------------------------------------- table
let state = null, ws = null, mySeat = null, lastError = "";

function connect(tid, token) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/${encodeURIComponent(tid)}`);
  ws.onopen = () => ws.send(JSON.stringify({ type: "hello", token: token || "" }));
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === "state") { state = msg.state; lastError = ""; render(); }
    if (msg.type === "error") { lastError = msg.error; render(); }
  };
  ws.onclose = () => setTimeout(() => connect(tid, token), 1500);
}

function send(obj) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }

function rel(seat) {
  const base = state.me ? state.me.seat : 0;
  return ["me", "left", "top", "right"][(seat - base + 4) % 4];
}

function plate(seat) {
  const s = state.seats[seat], est = state.estimates[seat];
  const role = [];
  if (state.dash.includes(seat)) role.push("Dash");
  if (state.caller === seat) role.push("Caller");
  const calls = state.bids.filter((b) => b.seat === seat), call = calls.length ? calls[calls.length - 1].label : "";
  const tally = est === null || est === undefined ? `won ${state.won[seat]}` : `${state.won[seat]} / ${est}`;
  const who = s.kind === "bot" ? `bot · ${esc(s.name)}` : `<span class="dot ${s.online ? "on" : ""}"></span>person`;
  return `<div class="plate p-${rel(seat)} ${state.turn === seat && !state.over ? "turn" : ""}">
    <div class="nm">${esc(s.kind === "bot" ? s.spec.replace("sub:", "") : s.name)}</div>
    <div class="small muted">${who}</div>
    <div class="tally">${tally} ${role.map((r) => `<span class="chip">${r}</span>`).join(" ")}</div>
    ${call && state.phase !== "play" ? `<div class="small">${esc(call)}</div>` : ""}
    <div class="score">${state.scores[seat]} pts · ${state.hands[seat]} cards</div></div>`;
}

function trickHtml() {
  let cards = state.trick, old = false, note = "";
  if (!cards.length && state.last_trick) { cards = state.last_trick.cards; old = true; note = `won by ${esc(nameOf(state.last_trick.winner))}`; }
  if (!cards.length) note = state.phase === "play" ? `${esc(nameOf(state.turn))} leads` : phaseText();
  const win = old ? state.last_trick.winner : null;
  return `<div class="trick ${old ? "old" : ""}">${cards.map(([s, c]) => `<div class="t-${rel(s)}">${cardHtml(c, s === win ? "win" : "")}</div>`).join("")}<div class="note">${note}</div></div>`;
}

function nameOf(seat) { if (seat === null || seat === undefined) return ""; const s = state.seats[seat]; return s.kind === "bot" ? s.spec.replace("sub:", "") : s.name; }

function phaseText() {
  return { dash: "Dash Calls", auction: "Bidding", estimate: "Estimates", play: "Playing", done: "Round over" }[state.phase] || state.phase;
}

function actionsHtml() {
  const me = state.me;
  if (!me) return `<p class="prompt muted">Watching${state.turn !== null && !state.over ? ` · ${esc(nameOf(state.turn))} to move` : ""}</p>`;
  if (state.over) return "";
  if (state.turn !== me.seat) return `<p class="prompt muted">${esc(nameOf(state.turn))} to move</p>`;
  if (state.phase === "play") return `<p class="prompt">Your turn: play a highlighted card</p>`;
  const legal = me.legal, bids = legal.filter((a) => a.id >= 2 && a.id <= 51), other = legal.filter((a) => a.id < 2 || a.id > 51);
  let grid = "";
  if (bids.length) {
    const ids = new Set(bids.map((a) => a.id)), counts = [];
    for (let n = 4; n <= 13; n++) if ([0, 1, 2, 3, 4].some((t) => ids.has(2 + (n - 4) * 5 + t))) counts.push(n);
    const head = `<span></span>${counts.map((n) => `<span class="h">${n}</span>`).join("")}`;
    const rows = ["♣", "♦", "♥", "♠", "NT"].map((sym, t) => `<span class="h">${sym}</span>` + counts.map((n) => {
      const id = 2 + (n - 4) * 5 + t;
      return ids.has(id) ? `<button data-act="${id}" title="${n} ${sym}">${n}${sym === "NT" ? "" : sym}</button>` : `<button disabled>·</button>`;
    }).join(""));
    grid = `<div class="bidgrid" style="grid-template-columns:auto repeat(${counts.length},minmax(30px,1fr))">${head}${rows.join("")}</div>`;
  }
  return `<p class="prompt">Your turn: ${phaseText()}</p>${grid}<div class="actions">${other.map((a) => `<button data-act="${a.id}">${esc(a.id === 53 && state.top_bidder === me.seat ? "Keep " + state.top_bid : a.label)}</button>`).join("")}</div>`;
}

function handHtml() {
  const me = state.me;
  if (!me) return "";
  const legal = new Set(me.legal.filter((a) => a.card !== null).map((a) => a.card));
  const myPlay = state.turn === me.seat && state.phase === "play";
  const sorted = [...me.hand].sort((a, b) => Math.floor(b / 13) - Math.floor(a / 13) || b - a);
  return `<div class="hand ${myPlay ? "" : "wait"}">${sorted.map((c) => cardHtml(c, myPlay ? (legal.has(c) ? "legal" : "off") : "")).join("")}</div>`;
}

function sideHtml() {
  const lr = state.last_round;
  const round = lr ? `<table><tr><th>Round ${lr.round}${lr.skipped ? " · skipped" : ""}</th><th class="n">est</th><th class="n">won</th><th class="n">pts</th></tr>${lr.seats.map((s, k) => `<tr><td>${esc(nameOf(k))}</td><td class="n">${s.estimate}</td><td class="n">${s.tricks}</td><td class="n ${s.score >= 0 ? "win" : ""}">${s.score > 0 ? "+" : ""}${s.score}</td></tr>`).join("")}</table>` : `<p class="small muted">No round finished yet.</p>`;
  const log = state.events.slice().reverse().map((e) => `<div><span class="muted">R${e.round}</span> ${esc(nameOf(e.seat))}: ${esc(e.label)}</div>`).join("");
  return `<div class="cols"><div class="panel">${round}</div><div class="panel"><b class="small">Moves</b><div class="log">${log || '<span class="muted small">none yet</span>'}</div></div></div>`;
}

function render() {
  if (!state) return;
  const over = state.over ? `<div class="panel" style="text-align:center"><b>Game over.</b> Winner: ${state.winners.map((w) => esc(nameOf(w))).join(", ")}
    ${state.me ? ' <button class="primary" id="again">New game</button>' : ""}</div>` : "";
  const bar = `<div class="bar"><b>Round ${state.round}</b><span>${phaseText()}</span>
    ${state.trump ? `<span class="chip">trump ${state.trump}</span>` : state.fixed_trump ? `<span class="chip">fixed ${state.fixed_trump}</span>` : ""}
    ${state.top_bid ? `<span class="chip">top bid ${esc(state.top_bid)} · ${esc(nameOf(state.top_bidder))}</span>` : ""}
    ${state.carry > 1 ? `<span class="chip">×${state.carry}</span>` : ""}<span class="muted small"><a href="/">lobby</a></span></div>`;
  app.innerHTML = `${bar}${over}<div class="board">${[0, 1, 2, 3].map(plate).join("")}${trickHtml()}</div>
    ${handHtml()}<div class="err">${esc(lastError)}</div>${actionsHtml()}${sideHtml()}`;
  app.querySelectorAll("[data-act]").forEach((b) => (b.onclick = () => send({ type: "act", action: Number(b.dataset.act) })));
  if (state.me && state.turn === state.me.seat && state.phase === "play") {
    const ids = new Map(state.me.legal.filter((a) => a.card !== null).map((a) => [a.card, a.id]));
    app.querySelectorAll(".hand .card.legal").forEach((el) => (el.onclick = () => send({ type: "act", action: ids.get(Number(el.dataset.card)) })));
  }
  const again = document.getElementById("again");
  if (again) again.onclick = () => send({ type: "new_game" });
}

const m = location.pathname.match(/^\/t\/([^/]+)/);
if (m) { connect(decodeURIComponent(m[1]), location.hash.slice(1)); app.innerHTML = '<p class="muted">Connecting…</p>'; }
else lobby();
