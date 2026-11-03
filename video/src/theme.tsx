import { AbsoluteFill, Easing, interpolate, useCurrentFrame } from "remotion";
import { loadFont as loadSans } from "@remotion/google-fonts/SpaceGrotesk";
import { loadFont as loadMono } from "@remotion/google-fonts/JetBrainsMono";
import type { CSSProperties, ReactNode } from "react";

export const SANS = loadSans("normal", { weights: ["300", "400", "500"], subsets: ["latin"] }).fontFamily;
export const MONO = loadMono("normal", { weights: ["400"], subsets: ["latin"] }).fontFamily;

export const C = {
  bg: "#0b0d10", ink: "#ece8df", muted: "#7d838c", line: "#1d2229", faint: "#14181d",
  amber: "#f0b429", blue: "#7fb0ff", green: "#5cc99c", red: "#e06c6c",
};
export const SUITS = ["♣", "♦", "♥", "♠"];
export const EASE = Easing.bezier(0.22, 1, 0.36, 1);

/** 0 → 1 over ``dur`` frames from ``start``, eased; never fast. */
export const useIn = (start: number, dur = 30) => {
  const f = useCurrentFrame();
  return interpolate(f, [start, start + dur], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: EASE });
};

export const Fade = ({ at, dur = 30, rise = 18, style, children }: { at: number; dur?: number; rise?: number; style?: CSSProperties; children: ReactNode }) => {
  const t = useIn(at, dur);
  return <div style={{ opacity: t, transform: `translateY(${(1 - t) * rise}px)`, ...style }}>{children}</div>;
};

/** The shared backdrop: ink black, a faint grid, a slow light drifting across. */
export const Stage = ({ children }: { children: ReactNode }) => {
  const f = useCurrentFrame();
  const x = 30 + 40 * Math.sin(f / 240);
  return (
    <AbsoluteFill style={{ background: C.bg, fontFamily: SANS, color: C.ink }}>
      <AbsoluteFill style={{ background: `radial-gradient(circle at ${x}% 35%, #1a2230 0%, transparent 55%)`, opacity: 0.9 }} />
      <svg width="1920" height="1080" style={{ position: "absolute", inset: 0, opacity: 0.5 }}>
        {Array.from({ length: 25 }, (_, i) => <line key={"v" + i} x1={i * 80} y1={0} x2={i * 80} y2={1080} stroke={C.faint} />)}
        {Array.from({ length: 14 }, (_, i) => <line key={"h" + i} x1={0} y1={i * 80} x2={1920} y2={i * 80} stroke={C.faint} />)}
      </svg>
      {children}
    </AbsoluteFill>
  );
};

export const Caption = ({ at, children }: { at: number; children: ReactNode }) => (
  <Fade at={at} dur={36} style={{ position: "absolute", bottom: 120, width: "100%", textAlign: "center", fontSize: 40, fontWeight: 300, letterSpacing: 1, color: C.ink }}>
    {children}
  </Fade>
);

export const Kicker = ({ children }: { children: ReactNode }) => (
  <div style={{ position: "absolute", top: 96, left: 120, fontFamily: MONO, fontSize: 22, letterSpacing: 6, color: C.muted, textTransform: "uppercase" }}>{children}</div>
);

export const Card = ({ c, w = 92, style }: { c: number; w?: number; style?: CSSProperties }) => {
  const red = c >= 13 && c < 39;
  const r = "23456789TJQKA"[c % 13].replace("T", "10");
  return (
    <div style={{ width: w, height: w * 1.4, borderRadius: w * 0.1, background: C.ink, color: red ? "#c8403c" : "#14171b", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", fontWeight: 500, fontSize: w * 0.34, lineHeight: 1.05, boxShadow: "0 18px 40px #0008", ...style }}>
      <span>{r}</span><span style={{ fontSize: w * 0.4 }}>{SUITS[Math.floor(c / 13)]}</span>
    </div>
  );
};
