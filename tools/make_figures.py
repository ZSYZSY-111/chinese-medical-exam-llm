#!/usr/bin/env python3
"""Render the README figures as dependency-free SVG (light and dark variants) from results/*.json.

    python tools/make_figures.py

Design rules: one axis per plot, thin marks, hairline solid grid, direct labels in text ink (never in the
series colour), a legend whenever there are two series, and a palette validated for colour-vision deficiency.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS, ASSETS = ROOT / "results", ROOT / "assets"
FONT = 'system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
              "s1": "#2a78d6", "s2": "#eb6834", "s1_soft": "#9ec5f4"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
             "s1": "#3987e5", "s2": "#d95926", "s1_soft": "#184f95"},
}


class Svg:
    def __init__(self, width, height, theme):
        self.w, self.h, self.t, self.parts = width, height, theme, []
        self.parts.append(f'<rect width="{width}" height="{height}" rx="10" fill="{theme["surface"]}"/>')

    def text(self, x, y, s, size=13, fill="ink", anchor="start", weight=400):
        s = str(s).replace("&", "&amp;").replace("<", "&lt;")
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{self.t[fill]}" text-anchor="{anchor}">{s}</text>')

    def line(self, x1, y1, x2, y2, stroke="grid", width=1, cap="butt"):
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{self.t[stroke]}" stroke-width="{width}" stroke-linecap="{cap}"/>')

    def polyline(self, pts, stroke, width=2, opacity=1.0):
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.parts.append(f'<polyline points="{d}" fill="none" stroke="{self.t[stroke]}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round" opacity="{opacity}"/>')

    def dot(self, x, y, fill, r=5):
        self.parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{self.t[fill]}" stroke="{self.t["surface"]}" stroke-width="2"/>')

    def bar(self, x, y, w, h, fill):
        """Column with a 4px rounded data end and a square baseline."""
        r = min(4, w / 2, h)
        self.parts.append(f'<path d="M{x:.1f},{y + h:.1f} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} V{y + h:.1f} Z" fill="{self.t[fill]}"/>')

    def legend(self, x, y, items):
        for label, colour, kind in items:
            if kind in ("line", "plain"):
                self.line(x, y - 4, x + 18, y - 4, stroke=colour, width=2, cap="round")
                if kind == "line":
                    self.dot(x + 9, y - 4, colour, r=4)
            else:
                self.parts.append(f'<rect x="{x}" y="{y - 10}" width="12" height="12" rx="3" fill="{self.t[colour]}"/>')
            self.text(x + (18 if kind == "box" else 26), y, label, size=12, fill="ink2")
            x += (18 if kind == "box" else 26) + 7.2 * len(label) + 22

    def save(self, path):
        body = "\n".join(self.parts)
        path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" font-family=\'{FONT}\'>\n{body}\n</svg>\n', encoding="utf-8")


def scale(v, lo, hi, a, b):
    return a + (v - lo) / (hi - lo) * (b - a)


def title(svg, heading, sub):
    svg.text(28, 38, heading, size=17, weight=600)
    svg.text(28, 60, sub, size=12.5, fill="ink2")


def fig_stages(theme, name):
    data = json.loads((RESULTS / "cmb_test.json").read_text(encoding="utf-8"))
    stages = [s for s in data["stages"] if s["key"] != "final"]
    labels = ["Qwen2.5-7B-Instruct, zero-shot", "Stage 1  SFT on decontaminated CMExam", "Stage 2  + option shuffling, LoRA averaging",
              "Stage 3  + decontaminated CMB-train"]
    deltas = ["", "+3.55", "+0.51", "+1.46"]
    svg = Svg(820, 330, theme)
    title(svg, "CMB-Exam test accuracy by training stage", "11,200 questions · official zero-shot prompt · exact letter-set match · bars show 95% Wilson intervals")
    left, right, top, row = 330, 770, 100, 50
    lo, hi = 77, 86
    for tick in range(77, 87):
        x = scale(tick, lo, hi, left, right)
        svg.line(x, top - 14, x, top + row * 3 + 22, stroke="grid")
        svg.text(x, top + row * 3 + 42, f"{tick}", size=11.5, fill="muted", anchor="middle")
    svg.text(right, top + row * 3 + 64, "accuracy (%)", size=11.5, fill="muted", anchor="end")
    for i, (stage, label) in enumerate(zip(stages, labels)):
        y = top + i * row
        svg.text(28, y + 4, label, size=13, fill="ink", weight=600 if i == 3 else 400)
        x0, x1 = (scale(v, lo, hi, left, right) for v in stage["ci95"])
        x = scale(stage["accuracy"], lo, hi, left, right)
        svg.line(x0, y, x1, y, stroke="s1_soft", width=6, cap="round")
        svg.dot(x, y, "s1", r=6)
        svg.text(x, y - 13, f'{stage["accuracy"]:.2f}', size=13, weight=600, anchor="middle")
        if deltas[i]:
            svg.text(x1 + 12, y + 4, deltas[i], size=12, fill="ink2")
    svg.save(ASSETS / f"stages_{name}.svg")


def fig_question_types(theme, name):
    data = json.loads((RESULTS / "cmb_test.json").read_text(encoding="utf-8"))
    stages = [s for s in data["stages"] if s["key"] != "final"]
    single = [s["by_question_type"]["单项选择题"]["accuracy"] for s in stages]
    multi = [s["by_question_type"]["多项选择题"]["accuracy"] for s in stages]
    svg = Svg(820, 380, theme)
    title(svg, "Where the gain comes from", "CMB-Exam test accuracy by question type · 9,999 single-answer and 1,190 multi-answer questions")
    left, right, top, bottom = 70, 640, 100, 300
    lo, hi = 40, 90
    for tick in range(40, 91, 10):
        y = scale(tick, lo, hi, bottom, top)
        svg.line(left, y, right, y, stroke="grid")
        svg.text(left - 10, y + 4, tick, size=11.5, fill="muted", anchor="end")
    xs = [scale(i, 0, 3, left + 30, right - 30) for i in range(4)]
    for x, label in zip(xs, ["zero-shot", "Stage 1", "Stage 2", "Stage 3"]):
        svg.text(x, bottom + 22, label, size=12, fill="ink2", anchor="middle")
    for values, colour, label in [(single, "s1", "single-answer"), (multi, "s2", "multi-answer")]:
        pts = [(x, scale(v, lo, hi, bottom, top)) for x, v in zip(xs, values)]
        svg.polyline(pts, colour)
        for x, y in pts:
            svg.dot(x, y, colour)
        svg.text(pts[-1][0] + 14, pts[-1][1] + 4, f"{values[-1]:.1f}  {label}", size=12.5, weight=600)
        svg.text(pts[0][0], pts[0][1] - 12, f"{values[0]:.1f}", size=12, fill="ink2", anchor="middle")
    svg.legend(left, 356, [("single-answer questions", "s1", "line"), ("multi-answer questions", "s2", "line")])
    svg.save(ASSETS / f"question_types_{name}.svg")


def fig_scale(theme, name):
    d = json.loads((RESULTS / "validation_analysis.json").read_text(encoding="utf-8"))["scale_curve"]
    xsv = d["x_fraction_of_cmb_train"]
    svg = Svg(820, 390, theme)
    title(svg, "Data-scale ablation", "Held-out validation set, 3,000 questions · every point is a full retrain with identical hyper-parameters")
    panels = [("Overall, official prompt", 60, 360, 79, 85, [(d["official_prompt_overall"], "s1", None)]),
              ("By question type, training prompt", 470, 770, 25, 90, [(d["training_prompt_single"], "s1", "single-answer"), (d["training_prompt_multi"], "s2", "multi-answer")])]
    top, bottom = 120, 300
    for heading, left, right, lo, hi, series in panels:
        svg.text(left, top - 22, heading, size=12.5, weight=600)
        step = 1 if hi - lo <= 10 else 10
        tick = lo if lo % step == 0 else lo + (step - lo % step)
        while tick <= hi:
            y = scale(tick, lo, hi, bottom, top)
            svg.line(left, y, right, y, stroke="grid")
            svg.text(left - 8, y + 4, tick, size=11.5, fill="muted", anchor="end")
            tick += step
        xs = [scale(v, 0, 1, left + 16, right - 16) for v in xsv]
        for x, v in zip(xs, xsv):
            svg.text(x, bottom + 22, f"{int(v * 100)}%", size=12, fill="ink2", anchor="middle")
        svg.text((left + right) / 2, bottom + 44, "share of CMB-train used", size=11.5, fill="muted", anchor="middle")
        for values, colour, label in series:
            pts = [(x, scale(v, lo, hi, bottom, top)) for x, v in zip(xs, values)]
            svg.polyline(pts, colour)
            for x, y in pts:
                svg.dot(x, y, colour)
            svg.text(pts[0][0], pts[0][1] - 12, f"{values[0]:.1f}", size=12, fill="ink2", anchor="middle")
            svg.text(pts[-1][0], pts[-1][1] - 12, f"{values[-1]:.1f}", size=12.5, weight=600, anchor="middle")
    svg.legend(470, 368, [("single-answer", "s1", "line"), ("multi-answer", "s2", "line")])
    svg.save(ASSETS / f"scale_ablation_{name}.svg")


def fig_format(theme, name):
    d = json.loads((RESULTS / "validation_analysis.json").read_text(encoding="utf-8"))["invalid_format_outputs_pct"]
    svg = Svg(820, 380, theme)
    title(svg, "Answers that can never be correct", "Share of validation outputs with the wrong number of letters for the question type")
    left, right, top, bottom = 70, 770, 100, 290
    lo, hi = 0, 50
    for tick in range(0, 51, 10):
        y = scale(tick, lo, hi, bottom, top)
        svg.line(left, y, right, y, stroke="axis" if tick == 0 else "grid")
        svg.text(left - 10, y + 4, f"{tick}%", size=11.5, fill="muted", anchor="end")
    labels = [["CMExam only,", "training prompt"], ["+ CMB-train,", "training prompt"], ["+ CMB-train,", "official prompt"], ["type-aligned", "training"]]
    band = (right - left) / 4
    for i, lines in enumerate(labels):
        cx = left + band * (i + 0.5)
        for k, (key, colour) in enumerate([("single_answer_questions_with_multi_letter_output", "s1"), ("multi_answer_questions_with_single_letter_output", "s2")]):
            v = d[key][i]
            x = cx - 26 + k * 28
            y = scale(v, lo, hi, bottom, top)
            if v > 0:
                svg.bar(x, y, 24, bottom - y, colour)
            svg.text(x + 12, (y if v > 0 else bottom) - 8, f"{v:g}%", size=12, weight=600 if v >= 10 else 400, anchor="middle")
        for j, line_text in enumerate(lines):
            svg.text(cx, bottom + 22 + j * 16, line_text, size=12, fill="ink2", anchor="middle")
    svg.legend(left, 360, [("single-answer question, several letters output", "s1", "box"), ("multi-answer question, one letter output", "s2", "box")])
    svg.save(ASSETS / f"format_errors_{name}.svg")


def fig_loss(theme, name):
    c = json.loads((RESULTS / "training_curves.json").read_text(encoding="utf-8"))["cmb_train"]
    svg = Svg(820, 360, theme)
    title(svg, "Stage 3 training run", "Qwen2.5-7B-Instruct, LoRA r=64, one epoch over 293,973 rows on a single RTX 5090 (about 6.5 hours)")
    left, right, top, bottom = 70, 770, 100, 280
    steps = c["max_steps"] or c["train_loss"][-1][0]
    lo, hi = 0.1, 0.4
    for tick in (0.1, 0.2, 0.3, 0.4):
        y = scale(tick, lo, hi, bottom, top)
        svg.line(left, y, right, y, stroke="grid")
        svg.text(left - 10, y + 4, f"{tick:.1f}", size=11.5, fill="muted", anchor="end")
    for tick in range(0, steps + 1, 2000):
        x = scale(tick, 0, steps, left, right)
        svg.text(x, bottom + 22, f"{tick:,}", size=11.5, fill="muted", anchor="middle")
    svg.text(right, bottom + 44, "optimizer step", size=11.5, fill="muted", anchor="end")
    train = [(scale(s, 0, steps, left, right), scale(min(max(v, lo), hi), lo, hi, bottom, top)) for s, v in c["train_loss"]]
    svg.polyline(train, "s1_soft", width=1.5)
    evals = [(scale(s, 0, steps, left, right), scale(v, lo, hi, bottom, top)) for s, v in c["eval_loss"]]
    svg.polyline(evals, "s2")
    for x, y in evals:
        svg.dot(x, y, "s2", r=4)
    svg.text(evals[-1][0] - 8, evals[-1][1] - 12, f'{c["eval_loss"][-1][1]:.3f}', size=12.5, weight=600, anchor="end")
    svg.legend(left, 340, [("training loss (logged every 20 steps)", "s1_soft", "plain"), ("validation loss", "s2", "line")])
    svg.save(ASSETS / f"training_loss_{name}.svg")


def fig_rl_arms(theme, name):
    """Rationale-mode validation accuracy of the SFT arms and the GRPO arms, with 95% Wilson intervals."""
    sft = json.loads((RESULTS / "segment_weighted_sft.json").read_text(encoding="utf-8"))
    rl = json.loads((RESULTS / "segment_credit_grpo.json").read_text(encoding="utf-8"))
    rows = [
        ("SFT · control (rationale LM loss only)", sft["rationale_mode"]["control"], "", 400),
        ("SFT · + auxiliary head predicting segment scores", sft["rationale_mode"]["predict"], f'{sft["paired_rationale_mode"]["predict_vs_control"]["delta"]:+.2f}', 400),
        ("SFT · segment scores as loss weights  (RL start)", sft["rationale_mode"]["weighted"], f'{sft["paired_rationale_mode"]["weighted_vs_control"]["delta"]:+.2f}', 400),
        ("GRPO · standard, 150 steps", rl["arms"]["g0_standard_grpo"], f'{rl["paired"]["g0_vs_start"]["delta"]:+.2f} vs RL start', 400),
        ("GRPO · segment credit, 150 steps", rl["arms"]["g2_segment_credit_grpo"], f'{rl["paired"]["g2_vs_g0"]["delta"]:+.2f} vs standard', 600),
    ]
    svg = Svg(940, 380, theme)
    title(svg, "Rationale-then-answer accuracy on the CMExam validation split", "6,657 questions · greedy · exact letter-set match · bars show 95% Wilson intervals · deltas are paired")
    left, right, top, row = 380, 790, 100, 50
    lo, hi = 72, 86
    for tick in range(72, 87, 2):
        x = scale(tick, lo, hi, left, right)
        svg.line(x, top - 14, x, top + row * 4 + 22, stroke="grid")
        svg.text(x, top + row * 4 + 42, f"{tick}", size=11.5, fill="muted", anchor="middle")
    svg.text(right, top + row * 4 + 64, "accuracy (%)", size=11.5, fill="muted", anchor="end")
    for i, (label, arm, delta, weight) in enumerate(rows):
        y = top + i * row
        colour = "s2" if i >= 3 else "s1"
        soft = "s1_soft"
        svg.text(28, y + 4, label, size=13, fill="ink", weight=weight)
        x0, x1 = (scale(v, lo, hi, left, right) for v in arm["ci95"])
        x = scale(arm["accuracy"], lo, hi, left, right)
        svg.line(x0, y, x1, y, stroke=soft, width=6, cap="round")
        svg.dot(x, y, colour, r=6)
        svg.text(x, y - 13, f'{arm["accuracy"]:.2f}', size=13, weight=600, anchor="middle")
        if delta:
            svg.text(x1 + 12, y + 4, delta, size=12, fill="ink2")
    svg.legend(28, top + row * 4 + 64, [("SFT arms", "s1", "line"), ("GRPO arms", "s2", "line")])
    svg.save(ASSETS / f"rl_arms_{name}.svg")


def fig_rl_dynamics(theme, name):
    """Share of completion tokens that carry a non-zero advantage, standard GRPO vs segment credit, over training."""
    rl = json.loads((RESULTS / "segment_credit_grpo.json").read_text(encoding="utf-8"))
    g0 = [(r["step"], 1.0 - r["frac_reward_zero_std"]) for r in rl["training_curves"]["g0"] if r["frac_reward_zero_std"] is not None]
    g2 = [(r["step"], r["token_frac_nonzero_adv"]) for r in rl["training_curves"]["g2"] if r["token_frac_nonzero_adv"] is not None]
    svg = Svg(860, 360, theme)
    title(svg, "How much of each training step carries a learning signal", "share of sampled tokens whose advantage is non-zero · 16 prompts × 8 samples per step · 150 steps")
    left, right, top, bottom = 70, 820, 90, 290
    for tick in range(0, 101, 25):
        y = scale(tick, 0, 100, bottom, top)
        svg.line(left, y, right, y, stroke="grid")
        svg.text(left - 10, y + 4, f"{tick}%", size=11.5, fill="muted", anchor="end")
    for tick in range(0, 151, 30):
        x = scale(tick, 0, 150, left, right)
        svg.text(x, bottom + 20, f"{tick}", size=11.5, fill="muted", anchor="middle")
    svg.text(right, bottom + 40, "training step", size=11.5, fill="muted", anchor="end")
    for series, colour in ((g0, "s1"), (g2, "s2")):
        pts = [(scale(s, 0, 150, left, right), scale(100 * v, 0, 100, bottom, top)) for s, v in series]
        svg.polyline(pts, colour, width=2)
        svg.dot(*pts[-1], colour, r=4)
    svg.text(scale(150, 0, 150, left, right) + 8, scale(100 * g2[-1][1], 0, 100, bottom, top) + 4, f"{100 * g2[-1][1]:.0f}%", size=12, fill="ink2")
    svg.text(scale(150, 0, 150, left, right) + 8, scale(100 * g0[-1][1], 0, 100, bottom, top) + 4, f"{100 * g0[-1][1]:.0f}%", size=12, fill="ink2")
    svg.legend(left, bottom + 40, [("standard GRPO (group not unanimous)", "s1", "plain"), ("segment credit (advantage + segment credit ≠ 0)", "s2", "plain")])
    svg.save(ASSETS / f"rl_dynamics_{name}.svg")


def main():
    ASSETS.mkdir(exist_ok=True)
    for name, theme in THEMES.items():
        for fig in (fig_stages, fig_question_types, fig_scale, fig_format, fig_loss, fig_rl_arms, fig_rl_dynamics):
            fig(theme, name)
    print("wrote", len(list(ASSETS.glob("*.svg"))), "SVG files to", ASSETS)


if __name__ == "__main__":
    main()
