// WebMCP tools for every Estimation page, registered through Home UI (HomeUI.tools).
// They use the same API calls as the pages, so the same sign-in and server checks apply.
// The table page adds its own tools for playing a seat (see app.js).

async function whenHomeUI() {
  await customElements.whenDefined("home-bar");
  return window.HomeUI;
}

async function getJSON(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`${path} answered ${response.status}`);
  return response.json();
}

// Bots that talk to a paid AI service never sit at a table from a tool.
const isPaidBot = (spec) => /llm|claude|openai|codex/i.test(spec);

async function standings({ limit = 10 } = {}) {
  const board = await getJSON("/api/board");
  const ranked = board.standings.filter((row) => !row.provisional)
    .sort((a, b) => (b.elo ?? -1e9) - (a.elo ?? -1e9)).slice(0, limit);
  return {
    season: board.season?.number ?? null,
    standings: ranked.map((row, k) => ({ place: k + 1, bot: row.name, author: row.author,
      weighted_rank: row.elo, king_only: row.king_elo, score_delta: row.score_elo })),
    playing_now: board.playing.map((m) => `${m.name} vs ${m.opponent}`),
  };
}

async function openTables() {
  const rows = await getJSON("/api/tables");
  return rows.map((t) => ({ id: t.id, players: t.names, round: t.round, finished: t.over, watch: `/t/${t.id}` }));
}

async function createTable({ seats }) {
  const bots = await getJSON("/api/bots");
  const known = new Set(bots.map((b) => b.spec));
  const request = seats.map((seat) => {
    if (seat.bot) {
      if (isPaidBot(seat.bot) || !known.has(seat.bot)) throw new Error(`${seat.bot} can't sit at a table`);
      return { kind: "bot", spec: seat.bot };
    }
    return { kind: "human", name: seat.person || "Player" };
  });
  const response = await fetch("/api/tables", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ seats: request }) });
  const made = await response.json();
  if (made.error) throw new Error(made.error);
  return { watch: made.watch, people: made.seats.filter((s) => s.link).map((s) => ({ name: s.name, link: s.link })) };
}

whenHomeUI().then((HomeUI) => {
  HomeUI.tools.register({
    name: "estimation.get_standings",
    description: "Read the current bot standings on the live board, best first, with the matches playing now.",
    inputSchema: { type: "object", properties: {
      limit: { type: "integer", minimum: 1, maximum: 40, description: "How many ranked bots to return (default 10)." } } },
    readOnly: true,
    execute: standings,
  });
  HomeUI.tools.register({
    name: "estimation.list_tables",
    description: "List the open game tables with their players, round and a link to watch each one.",
    inputSchema: { type: "object", properties: {} },
    readOnly: true,
    execute: openTables,
  });
  HomeUI.tools.register({
    name: "estimation.list_bots",
    description: "List the bots that can take a seat at a new table.",
    inputSchema: { type: "object", properties: {} },
    readOnly: true,
    execute: async () => (await getJSON("/api/bots")).filter((b) => !isPaidBot(b.spec)).map((b) => ({ bot: b.spec, name: b.name, author: b.author })),
  });
  HomeUI.tools.register({
    name: "estimation.create_table",
    description: "Start a new table of four seats, each a person (who gets an invite link) or a bot from estimation.list_bots.",
    inputSchema: { type: "object", required: ["seats"], properties: {
      seats: { type: "array", minItems: 4, maxItems: 4, description: "The four seats in order.", items: { type: "object", properties: {
        person: { type: "string", maxLength: 24, description: "Name of the person for this seat." },
        bot: { type: "string", description: "A bot id from estimation.list_bots, instead of a person." } } } } } },
    consequential: true,
    execute: createTable,
  });
});
