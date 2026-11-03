import type { ReactElement } from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import data from "./data.json";
import { C, Caption, Card, EASE, Fade, Kicker, MONO, Stage, SUITS, useIn } from "./theme";

const T = 30; // crossfade frames between scenes

// 1 ─ title: four suits orbit slowly, the name settles in
const Title = () => {
  const f = useCurrentFrame();
  return (
    <Stage>
      {SUITS.map((s, i) => {
        const a = (f / 300) * Math.PI * 2 + (i * Math.PI) / 2;
        const t = useIn(6 + i * 8, 40);
        return (
          <div key={s} style={{ position: "absolute", left: 960 + Math.cos(a) * 640 - 40, top: 500 + Math.sin(a) * 330 - 50, fontSize: 84, opacity: 0.22 * t, color: i === 1 || i === 2 ? C.red : C.ink }}>{s}</div>
        );
      })}
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center" }}>
        <Fade at={20} dur={45} rise={24}><div style={{ fontSize: 132, fontWeight: 300, letterSpacing: 4 }}>Estimation Arena</div></Fade>
        <Fade at={60} dur={40}><div style={{ fontFamily: MONO, fontSize: 30, color: C.muted, letterSpacing: 8, marginTop: 24 }}>TEACHING BOTS A CARD GAME</div></Fade>
      </AbsoluteFill>
    </Stage>
  );
};

// 2 ─ the game: a hand fans out, three verbs
const HAND = [51, 49, 47, 45, 35, 30, 27, 25, 22, 12, 9, 6, 2];
const Game = () => (
  <Stage>
    <Kicker>the game</Kicker>
    <div style={{ position: "absolute", left: 0, right: 0, top: 330, display: "flex", justifyContent: "center" }}>
      {HAND.map((c, i) => {
        const t = useIn(10 + i * 4, 34);
        const rot = (i - 6) * 4.2 * t;
        return <Card key={c} c={c} w={104} style={{ marginLeft: i ? -34 : 0, opacity: t, transform: `translateY(${(1 - t) * 60 + Math.abs(i - 6) * 7}px) rotate(${rot}deg)` }} />;
      })}
    </div>
    <div style={{ position: "absolute", top: 650, width: "100%", display: "flex", justifyContent: "center", gap: 110, fontSize: 46, fontWeight: 300 }}>
      {["bid", "estimate", "win exactly that many"].map((w, i) => (
        <Fade key={w} at={78 + i * 22} dur={34}><span style={{ color: i === 2 ? C.amber : C.ink }}>{w}</span></Fade>
      ))}
    </div>
    <Caption at={150}>four players · thirteen tricks · no partners</Caption>
  </Stage>
);

// 3 ─ duplicate sets: one deal, the seats rotate around it
const Duplicate = () => {
  const f = useCurrentFrame();
  const turn = interpolate(f, [30, 70, 90, 130, 150, 190], [0, 1, 1, 2, 2, 3], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: EASE });
  const names = ["A", "B", "A", "B"];
  return (
    <Stage>
      <Kicker>fair matches</Kicker>
      <div style={{ position: "absolute", left: 960 - 60, top: 400 }}>
        {[0, 1, 2].map((k) => <div key={k} style={{ position: "absolute", left: k * 6, top: -k * 6, width: 120, height: 168, borderRadius: 12, background: "#1b2532", border: `2px solid ${C.blue}55` }} />)}
      </div>
      {[0, 1, 2, 3].map((s) => {
        const a = ((s + turn) * Math.PI) / 2 + Math.PI / 2;
        return (
          <div key={s} style={{ position: "absolute", left: 960 + Math.cos(a) * 300 - 48, top: 484 + Math.sin(a) * 300 - 48, width: 96, height: 96, borderRadius: 48, border: `2px solid ${s % 2 ? C.blue : C.amber}`, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: MONO, fontSize: 36, color: s % 2 ? C.blue : C.amber }}>{names[s]}</div>
        );
      })}
      <div style={{ position: "absolute", right: 170, top: 470, fontFamily: MONO, fontSize: 64, color: C.muted }}>× {1 + Math.round(turn)}</div>
      <Caption at={40}>same deal, every seat · luck cancels out</Caption>
    </Stage>
  );
};

// 4 ─ the ladder: our dot climbs past each anchor
const Ladder = () => {
  const f = useCurrentFrame();
  const rungs = data.rungs;
  const climb = interpolate(f, [20, 200], [0, rungs.length], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: EASE });
  const y = (i: number) => 820 - i * 170;
  return (
    <Stage>
      <Kicker>the climb</Kicker>
      <svg width="1920" height="1080" style={{ position: "absolute" }}>
        <line x1={760} y1={y(0) + 60} x2={760} y2={y(3) - 50} stroke={C.line} strokeWidth={3} />
        <line x1={1160} y1={y(0) + 60} x2={1160} y2={y(3) - 50} stroke={C.line} strokeWidth={3} />
        {rungs.map((_, i) => <line key={i} x1={760} y1={y(i)} x2={1160} y2={y(i)} stroke={climb > i + 0.5 ? ("tie" in rungs[i] ? C.muted : C.amber) : C.line} strokeWidth={3} />)}
        <circle cx={960} cy={y(0) + 60 - Math.min(climb, rungs.length - 0.6) * 170} r={14} fill={C.amber} />
      </svg>
      {rungs.map((r, i) => {
        const done = climb > i + 0.5 && (r.oursElo ?? 0) > 0;
        return (
          <div key={r.anchor}>
            <div style={{ position: "absolute", left: 330, width: 380, top: y(i) - 30, textAlign: "right", fontSize: 40, fontWeight: 300, color: C.muted }}>
              {r.anchor} <span style={{ fontFamily: MONO, fontSize: 26 }}>{r.anchorElo}</span>
            </div>
            <div style={{ position: "absolute", left: 1210, top: y(i) - 30, fontSize: 40, fontWeight: 400, opacity: done ? 1 : 0.15, color: "tie" in r ? C.ink : C.amber }}>
              {r.ours} <span style={{ fontFamily: MONO, fontSize: 26, color: "tie" in r ? C.muted : C.ink }}>{"tie" in r ? r.note : r.oursElo}</span>
            </div>
          </div>
        );
      })}
      <Caption at={200}>three rungs passed · the last one, a tie</Caption>
    </Stage>
  );
};

// 5 ─ search: imagined deals fill in, the means decide
const PATTERN = ["+--+-+--", "-+---+--", "++-+++-+"];
const Search = () => {
  const f = useCurrentFrame();
  return (
    <Stage>
      <Kicker>how it thinks</Kicker>
      {PATTERN.map((row, r) => (
        <div key={r} style={{ position: "absolute", left: 470, top: 300 + r * 150, display: "flex", alignItems: "center", gap: 18 }}>
          <Card c={[48, 44, 39][r]} w={70} style={{ marginRight: 30 }} />
          {row.split("").map((s, k) => {
            const t = useIn(20 + k * 9 + r * 3, 24);
            return <div key={k} style={{ width: 62, height: 62, borderRadius: 10, background: s === "+" ? C.green : C.red, opacity: 0.12 + 0.75 * t, transform: `scale(${0.85 + 0.15 * t})` }} />;
          })}
          {(() => {
            const t = useIn(120 + r * 6, 40);
            const w = (row.split("+").length - 1) * 38 * t;
            return <div style={{ marginLeft: 46, width: w, height: 26, borderRadius: 6, background: r === 2 && f > 170 ? C.amber : C.muted }} />;
          })()}
        </div>
      ))}
      <Fade at={22} style={{ position: "absolute", left: 610, top: 230, fontFamily: MONO, fontSize: 24, color: C.muted, letterSpacing: 4 }}>IMAGINED DEAL 1 … 8</Fade>
      <Fade at={176} style={{ position: "absolute", left: 1530, top: 615, fontSize: 40, color: C.amber }}>play</Fade>
      <Caption at={60}>guess the hidden hands · play each card out · keep the best</Caption>
    </Stage>
  );
};

// 6 ─ learning: the self-play loop and the curve it draws
const Learning = () => {
  const f = useCurrentFrame();
  const pts = data.learning;
  const t = useIn(30, 130);
  const max = Math.max(...pts);
  const path = pts.map((v, i) => `${i ? "L" : "M"}${1060 + i * 90} ${760 - (v / max) * 380}`).join(" ");
  const spin = (f / 180) * 360;
  return (
    <Stage>
      <Kicker>learning</Kicker>
      <svg width="1920" height="1080" style={{ position: "absolute" }}>
        <g transform={`rotate(${spin} 520 560)`}>
          <circle cx={520} cy={560} r={200} fill="none" stroke={C.line} strokeWidth={3} />
          <circle cx={520} cy={360} r={9} fill={C.blue} />
        </g>
        <path d={path} fill="none" stroke={C.amber} strokeWidth={5} strokeLinecap="round" pathLength={1} strokeDasharray={1} strokeDashoffset={1 - t} />
        <line x1={1040} y1={780} x2={1720} y2={780} stroke={C.line} strokeWidth={2} />
      </svg>
      {["play", "score", "update"].map((w, i) => {
        const a = -Math.PI / 2 + (i * 2 * Math.PI) / 3;
        return <Fade key={w} at={10 + i * 14} style={{ position: "absolute", left: 520 + Math.cos(a) * 280 - 80, top: 560 + Math.sin(a) * 280 - 28, width: 160, textAlign: "center", fontSize: 38, fontWeight: 300 }}>{w}</Fade>;
      })}
      <Fade at={150} style={{ position: "absolute", left: 1600, top: 300, fontFamily: MONO, fontSize: 32, color: C.amber }}>+{max} / round</Fade>
      <Caption at={70}>self-play on one home GPU</Caption>
    </Stage>
  );
};

// 7 ─ the live table: people and bots take seats
const SEATS = [{ n: "you", k: "person" }, { n: "Hybrid", k: "bot" }, { n: "a friend", k: "person" }, { n: "rl-hybrid", k: "bot" }];
const Table = () => (
  <Stage>
    <Kicker>play it</Kicker>
    <div style={{ position: "absolute", left: 790, top: 330, width: 340, height: 340, borderRadius: 200, border: `2px solid ${C.line}`, background: "#101820" }} />
    {SEATS.map((s, i) => {
      const a = Math.PI / 2 + (i * Math.PI) / 2;
      const t = useIn(16 + i * 18, 34);
      const person = s.k === "person";
      return (
        <div key={s.n} style={{ position: "absolute", left: 960 + Math.cos(a) * 330 - 120, top: 500 + Math.sin(a) * 260 - 44, width: 240, height: 88, borderRadius: 18, opacity: t, transform: `scale(${0.9 + 0.1 * t})`, border: `2px solid ${person ? C.green : C.blue}`, background: C.bg, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
          <div style={{ fontSize: 34 }}>{s.n}</div>
          <div style={{ fontFamily: MONO, fontSize: 18, color: person ? C.green : C.blue, letterSpacing: 3 }}>{s.k.toUpperCase()}</div>
        </div>
      );
    })}
    <Caption at={100}>people and bots · one table · any mix</Caption>
  </Stage>
);

// 8 ─ end card
const End = () => (
  <Stage>
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "center" }}>
      <Fade at={10} dur={40}><div style={{ fontSize: 88, fontWeight: 300, letterSpacing: 3 }}>Estimation Arena</div></Fade>
      <Fade at={40} dur={40}><div style={{ fontFamily: MONO, fontSize: 32, color: C.amber, marginTop: 28 }}>{data.url}</div></Fade>
      <Fade at={70} dur={40}><div style={{ fontSize: 28, color: C.muted, marginTop: 48 }}>engine and anchors by {data.engine}</div></Fade>
    </AbsoluteFill>
  </Stage>
);

const SCENES: [() => ReactElement, number][] = [[Title, 150], [Game, 200], [Duplicate, 210], [Ladder, 250], [Search, 220], [Learning, 200], [Table, 180], [End, 160]];
export const PROMO_FRAMES = SCENES.reduce((n, [, d]) => n + d, 0) - (SCENES.length - 1) * T;

export const Promo = () => (
  <TransitionSeries>
    {SCENES.flatMap(([Scene, d], i) => [
      <TransitionSeries.Sequence key={"s" + i} durationInFrames={d}><Scene /></TransitionSeries.Sequence>,
      ...(i < SCENES.length - 1 ? [<TransitionSeries.Transition key={"t" + i} presentation={fade()} timing={linearTiming({ durationInFrames: T })} />] : []),
    ])}
  </TransitionSeries>
);
