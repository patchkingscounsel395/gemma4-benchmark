#!/usr/bin/env python3
"""Generate BENCHMARK_REPORT.md and benchmark_report.html from results JSON and config.yaml"""

import json
import os

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.yaml")

with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

RESULTS_FILE = os.path.join(SCRIPT_DIR, CONFIG["results_file"])
MD_FILE = os.path.join(SCRIPT_DIR, CONFIG["report_markdown"])
HTML_FILE = os.path.join(SCRIPT_DIR, CONFIG["report_html"])

with open(RESULTS_FILE) as f:
    DATA = json.load(f)

MODEL_GROUPS = CONFIG["model_groups"]
CTX_LABELS = {4096: "4k", 8192: "8k", 16384: "16k", 32768: "32k",
              65536: "64k", 131072: "128k", 262144: "256k"}


def get_rows(model_name):
    return sorted([r for r in DATA if r["model"] == model_name], key=lambda r: r["target_ctx"])


# ── Markdown Report ──────────────────────────────────────────────────────────
def gen_markdown():
    lines = []
    lines.append("# MLX Vision-Language Model Benchmark")
    lines.append("")
    lines.append("Decode throughput and memory usage across context lengths with and without TurboQuant KV cache quantization.")
    lines.append("")
    lines.append("**Hardware:** Apple M5 Max (128 GB unified memory)")
    lines.append("**Framework:** mlx-vlm 0.4.4, MLX 0.31.1")
    lines.append("**Context:** 4k to 256k tokens | Thinking: OFF")
    lines.append("**TurboQuant:** KV cache quantized to 2.5 bits")
    lines.append("")
    lines.append("See the [interactive HTML report](results/benchmark_report.html) for charts.")
    lines.append("")

    for group_name, model_names in MODEL_GROUPS.items():
        lines.append(f"## {group_name}")
        lines.append("")
        for mname in model_names:
            quant = mname.split()[-1]
            rows = get_rows(mname)
            if not rows:
                continue
            lines.append(f"### {quant}")
            lines.append("")
            lines.append("| Context | Prefill | Decode | Mem | TQ Prefill | TQ Decode | TQ Mem | Delta |")
            lines.append("|---------|---------|--------|-----|------------|-----------|--------|:-----:|")
            for r in rows:
                ctx = CTX_LABELS.get(r["target_ctx"], f"{r['target_ctx']//1000}k")
                d = r.get("decode", 0)
                td = r.get("tq_decode", 0)
                delta = ((td / d) - 1) * 100 if d > 0 and td > 0 else 0
                delta_s = f"**{delta:+.1f}%**" if abs(delta) > 5 else f"{delta:+.1f}%"
                lines.append(
                    f"| {ctx} | {r.get('prefill',0):,.0f} t/s | {d:.1f} t/s | {r.get('mem',0):.1f} GB "
                    f"| {r.get('tq_prefill',0):,.0f} t/s | {td:.1f} t/s | {r.get('tq_mem',0):.1f} GB | {delta_s} |"
                )
            lines.append("")

    lines.append("---")
    lines.append("*Benchmarked on Apple M5 Max (128 GB) with mlx-vlm 0.4.4. April 2026.*")

    os.makedirs(os.path.dirname(MD_FILE), exist_ok=True)
    with open(MD_FILE, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {MD_FILE} ({len(lines)} lines)")


# ── HTML Report with Charts ──────────────────────────────────────────────────
def gen_html():
    COLORS = {"4bit": "#58a6ff", "8bit": "#7ee787", "bf16": "#f0883e"}

    chart_data_js = {}
    for group_name, model_names in MODEL_GROUPS.items():
        datasets = []
        for mname in model_names:
            quant = mname.split()[-1]
            color = COLORS.get(quant, "#888")
            rows = get_rows(mname)
            if not rows:
                continue
            std_points = [{"x": r["target_ctx"], "y": r.get("decode", 0)} for r in rows]
            datasets.append({
                "label": f"{quant}", "data": std_points,
                "borderColor": color, "backgroundColor": color,
                "borderWidth": 2, "borderDash": [], "pointRadius": 3, "tension": 0.3,
            })
            tq_points = [{"x": r["target_ctx"], "y": r.get("tq_decode", 0)} for r in rows]
            datasets.append({
                "label": f"{quant} +TQ", "data": tq_points,
                "borderColor": color, "backgroundColor": color,
                "borderWidth": 2, "borderDash": [6, 3],
                "pointRadius": 3, "pointStyle": "triangle", "tension": 0.3,
            })
        chart_data_js[group_name] = datasets

    def table_html(group_name, model_names):
        h = []
        for mname in model_names:
            quant = mname.split()[-1]
            rows = get_rows(mname)
            if not rows:
                continue
            h.append(f'<h3>{quant}</h3>')
            h.append('<table><thead><tr>')
            h.append('<th>Context</th><th>Prefill</th><th>Decode</th><th>Mem</th>')
            h.append('<th class="section-start">TQ Prefill</th><th>TQ Decode</th><th>TQ Mem</th>')
            h.append('<th class="section-start">Delta</th></tr></thead><tbody>')
            for r in rows:
                ctx = CTX_LABELS.get(r["target_ctx"], f"{r['target_ctx']//1000}k")
                d = r.get("decode", 0)
                td = r.get("tq_decode", 0)
                delta = ((td / d) - 1) * 100 if d > 0 and td > 0 else 0
                dc = "positive" if delta > 5 else ("negative" if delta < -5 else "neutral")
                hl = ' class="highlight"' if delta > 10 else ""
                ds = f"<strong>{delta:+.1f}%</strong>" if abs(delta) > 5 else f"{delta:+.1f}%"
                h.append(f'<tr{hl}><td>{ctx}</td><td>{r.get("prefill",0):,.0f}</td><td>{d:.1f}</td><td>{r.get("mem",0):.1f} GB</td>')
                h.append(f'<td class="section-start">{r.get("tq_prefill",0):,.0f}</td><td>{td:.1f}</td><td>{r.get("tq_mem",0):.1f} GB</td>')
                h.append(f'<td class="section-start {dc}">{ds}</td></tr>')
            h.append('</tbody></table>')
        return "\n".join(h)

    sections_html = ""
    chart_canvases = ""
    for i, (group_name, model_names) in enumerate(MODEL_GROUPS.items()):
        sections_html += f'<h2>{group_name}</h2>\n'
        sections_html += f'<div class="chart-container"><canvas id="chart{i}"></canvas></div>\n'
        sections_html += table_html(group_name, model_names) + "\n"
        chart_canvases += f"""
new Chart(document.getElementById('chart{i}'), {{
  type: 'line',
  data: {{ datasets: {json.dumps(chart_data_js[group_name])} }},
  options: {{
    responsive: true,
    plugins: {{
      title: {{ display: true, text: '{group_name} — Decode tok/s vs Context', color: '#c9d1d9', font: {{ size: 14 }} }},
      legend: {{ labels: {{ color: '#8b949e', usePointStyle: true, font: {{ size: 11 }} }} }}
    }},
    scales: {{
      x: {{
        type: 'logarithmic',
        title: {{ display: true, text: 'Context (tokens)', color: '#8b949e' }},
        ticks: {{ color: '#8b949e', callback: function(v) {{ return (v/1000)+'k'; }} }},
        grid: {{ color: '#21262d' }}
      }},
      y: {{
        title: {{ display: true, text: 'Decode tok/s', color: '#8b949e' }},
        ticks: {{ color: '#8b949e' }},
        grid: {{ color: '#21262d' }}
      }}
    }}
  }}
}});
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MLX VLM Benchmark — Gemma 4 + Qwen 3.5</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: #0d1117; color: #e6edf3;
    max-width: 960px; margin: 0 auto; padding: 32px 24px; line-height: 1.6;
  }}
  h1 {{ font-size: 24px; margin-bottom: 6px; color: #fff; }}
  h2 {{ font-size: 20px; margin: 32px 0 12px; color: #58a6ff; border-bottom: 1px solid #21262d; padding-bottom: 6px; }}
  h3 {{ font-size: 14px; margin: 16px 0 6px; color: #c9d1d9; }}
  .subtitle {{ color: #8b949e; font-size: 14px; margin-bottom: 20px; }}
  .meta {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 24px; }}
  .meta span {{
    background: #161b22; border: 1px solid #30363d; border-radius: 6px;
    padding: 4px 12px; font-size: 12px; color: #8b949e; font-family: monospace;
  }}
  .chart-container {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin: 12px 0 20px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 8px 0 16px; font-size: 12px; font-family: 'SF Mono', monospace; }}
  thead th {{
    background: #161b22; color: #8b949e; font-weight: 600;
    padding: 6px 8px; text-align: right; border-bottom: 2px solid #30363d;
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;
  }}
  thead th:first-child {{ text-align: left; }}
  thead th.section-start {{ border-left: 2px solid #30363d; }}
  tbody td {{ padding: 4px 8px; text-align: right; border-bottom: 1px solid #21262d; }}
  tbody td:first-child {{ text-align: left; color: #c9d1d9; font-weight: 600; }}
  tbody td.section-start {{ border-left: 2px solid #30363d; }}
  tbody tr:hover {{ background: #161b22; }}
  .positive {{ color: #7ee787; }}
  .negative {{ color: #f85149; }}
  .neutral {{ color: #8b949e; }}
  .highlight {{ background: #122d1a; }}
  .footer {{
    margin-top: 32px; padding-top: 16px; border-top: 1px solid #21262d;
    font-size: 12px; color: #484f58; text-align: center;
  }}
</style>
</head>
<body>

<h1>MLX Vision-Language Model Benchmark</h1>
<p class="subtitle">Gemma 4 + Qwen 3.5 — Standard vs TurboQuant 2.5-bit KV Cache</p>

<div class="meta">
  <span>Apple M5 Max (128 GB)</span>
  <span>mlx-vlm 0.4.4</span>
  <span>Thinking: OFF</span>
  <span>Context: 4k-256k</span>
  <span>Solid = standard</span>
  <span>Dashed = TurboQuant 2.5</span>
</div>

{sections_html}

<div class="footer">
  Prefill &amp; decode in tokens/sec &bull; Apple M5 Max 128 GB &bull; mlx-vlm 0.4.4 &bull; MLX 0.31.1 &bull; April 2026
</div>

<script>
{chart_canvases}
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(HTML_FILE), exist_ok=True)
    with open(HTML_FILE, "w") as f:
        f.write(html)
    print(f"Wrote {HTML_FILE}")


if __name__ == "__main__":
    gen_markdown()
    gen_html()
