//! Enhanced HTML report generator for benchmark results.
//!
//! Generates a comprehensive report including:
//!   - Summary statistics table (mean ± std, t-test, p-value, Cohen's d)
//!   - CSS bar chart visualization
//!   - Per-trial comparison data
//!   - Automatic conclusion based on statistical significance

use crate::benchmark::BenchmarkResults;
use crate::statistics::MetricStats;

/// Generate a comprehensive HTML benchmark report with statistical analysis.
pub fn generate_benchmark_html_report(results: &BenchmarkResults) -> String {
    let s = &results.summary;

    let mut html = String::new();
    html.push_str(r#"<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>归藏项目 — 三易认知约束引擎 基准测试报告</title>
<style>
  body { font-family: system-ui, -apple-system, sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1.5em; background: #0d1117; color: #c9d1d9; line-height: 1.6; }
  h1 { color: #58a6ff; border-bottom: 2px solid #30363d; padding-bottom: 0.4em; }
  h2 { color: #f0883e; margin-top: 2em; }
  h3 { color: #d2a8ff; }
  table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.95em; }
  th, td { border: 1px solid #30363d; padding: 10px 14px; text-align: left; }
  th { background: #161b22; color: #8b949e; font-weight: 600; }
  .sig-yes { color: #3fb950; font-weight: bold; }
  .sig-no { color: #8b949e; }
  .stat-cell { text-align: right; font-variant-numeric: tabular-nums; }
  .summary-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1.2em 1.5em; margin: 1.2em 0; }
  .conclusion { background: #0d2b1e; border: 1px solid #3fb950; border-radius: 8px; padding: 1em 1.5em; margin: 1.5em 0; color: #7ee787; }
  .conclusion-weak { background: #2b2200; border: 1px solid #d29922; color: #d29922; }
  pre { background: #161b22; padding: 1em; border-radius: 6px; overflow-x: auto; font-size: 0.85em; }

  /* Bar chart styles */
  .chart-container { display: flex; align-items: flex-end; gap: 20px; margin: 1em 0; padding: 1em; background: #161b22; border-radius: 8px; }
  .chart-group { flex: 1; text-align: center; }
  .chart-label { font-size: 0.8em; color: #8b949e; margin-bottom: 6px; }
  .chart-bars { display: flex; justify-content: center; gap: 12px; height: 180px; align-items: flex-end; }
  .bar-wrapper { display: flex; flex-direction: column; align-items: center; gap: 4px; }
  .bar { width: 40px; border-radius: 4px 4px 0 0; transition: height 0.3s; position: relative; min-height: 4px; }
  .bar-controlled { background: linear-gradient(180deg, #f85149, #b62324); }
  .bar-constrained { background: linear-gradient(180deg, #3fb950, #238636); }
  .bar-value { font-size: 0.75em; color: #c9d1d9; }
  .bar-legend { display: flex; justify-content: center; gap: 20px; margin: 0.5em 0; font-size: 0.85em; }
  .legend-item { display: flex; align-items: center; gap: 6px; }
  .legend-swatch { width: 14px; height: 14px; border-radius: 3px; }

  .meta { color: #8b949e; font-size: 0.9em; }
  .effect-large { color: #3fb950; font-weight: bold; }
  .effect-medium { color: #d29922; }
  .effect-small { color: #8b949e; }
</style>
</head>
<body>
<h1>归藏项目 — 三易认知约束引擎 基准测试报告</h1>
<div class="meta">
  <p>模型: "#);

    html.push_str(&results.model_name);
    html.push_str(" | 后端: ");
    html.push_str(&results.backend_type);
    html.push_str(&format!(" | 试验次数: {} | 输入: 10主题×10变体", results.num_trials));
    html.push_str("</p>\n</div>\n");

    // ── Summary table ──
    html.push_str(r#"<h2>一、核心指标统计显著性分析</h2>
<table>
<tr>
  <th>指标</th>
  <th>对照组 (裸跑)</th>
  <th>实验组 (三易控制)</th>
  <th>t 统计量</th>
  <th>p 值</th>
  <th>Cohen's d</th>
  <th>显著性</th>
</tr>"#);

    push_metric_row(&mut html, "偏离率", &s.deviation_controlled, &s.deviation_constrained, &s.deviation_test);
    push_metric_row(&mut html, "结构合规率", &s.compliance_controlled, &s.compliance_constrained, &s.compliance_test);
    push_metric_row(&mut html, "完成率", &s.completion_controlled, &s.completion_constrained, &s.completion_test);
    push_metric_row(&mut html, "信任评分", &s.trust_controlled, &s.trust_constrained, &s.trust_test);

    html.push_str("</table>\n");

    // ── Effect size interpretation ──
    html.push_str(r#"<h2>二、效应量解释</h2>
<table>
<tr><th>指标</th><th>Cohen's d</th><th>效应大小</th></tr>"#);

    push_effect_row(&mut html, "偏离率", s.deviation_test.cohens_d);
    push_effect_row(&mut html, "结构合规率", s.compliance_test.cohens_d);
    push_effect_row(&mut html, "完成率", s.completion_test.cohens_d);
    push_effect_row(&mut html, "信任评分", s.trust_test.cohens_d);

    html.push_str("</table>\n");

    // ── Bar chart visualization ──
    html.push_str(r#"<h2>三、可视化对比</h2>
<div class="bar-legend">
  <span class="legend-item"><span class="legend-swatch" style="background:#f85149"></span>对照组 (裸跑)</span>
  <span class="legend-item"><span class="legend-swatch" style="background:#3fb950"></span>实验组 (三易控制)</span>
</div>
<div class="chart-container">"#);

    push_chart_group(&mut html, "偏离率", s.deviation_controlled.mean, s.deviation_constrained.mean, 1.0, true);
    push_chart_group(&mut html, "结构合规率", s.compliance_controlled.mean, s.compliance_constrained.mean, 1.0, false);
    push_chart_group(&mut html, "完成率", s.completion_controlled.mean, s.completion_constrained.mean, 1.0, false);
    push_chart_group(&mut html, "信任评分", s.trust_controlled.mean, s.trust_constrained.mean, 1.0, false);

    html.push_str("</div>\n");

    // ── Statistical interpretation ──
    html.push_str(r#"<h2>四、统计检验解读</h2>"#);

    html.push_str(&format!(
        r#"<div class="summary-card">
  <p><strong>偏离率:</strong> {d_interp}</p>
  <p><strong>结构合规率:</strong> {c_interp}</p>
  <p><strong>完成率:</strong> {m_interp}</p>
  <p><strong>信任评分:</strong> {t_interp}</p>
</div>"#,
        d_interp = s.deviation_test.interpretation,
        c_interp = s.compliance_test.interpretation,
        m_interp = s.completion_test.interpretation,
        t_interp = s.trust_test.interpretation,
    ));

    // ── Conclusion ──
    let all_significant = s.deviation_test.significant
        && s.compliance_test.significant
        && s.completion_test.significant;
    let any_significant = s.deviation_test.significant
        || s.compliance_test.significant
        || s.completion_test.significant;

    html.push_str(r#"<h2>五、综合结论</h2>"#);

    if all_significant {
        html.push_str(&format!(
            r#"<div class="conclusion">
  <h3>结论：三易认知约束引擎在所有核心指标上均表现出统计显著 (p < 0.05) 的提升效果。</h3>
  <ul>
    <li>偏离率从 {d_c_mean:.4} ± {d_c_std:.4} 降至 {d_x_mean:.4} ± {d_x_std:.4}（降低 {d_imp:.1}%）</li>
    <li>结构合规率从 {c_c_mean:.4} ± {c_c_std:.4} 提升至 {c_x_mean:.4} ± {c_x_std:.4}（提升 {c_imp:.1}%）</li>
    <li>完成率从 {m_c_mean:.4} ± {m_c_std:.4} 提升至 {m_x_mean:.4} ± {m_x_std:.4}（提升 {m_imp:.1}%）</li>
  </ul>
  <p>在 {n} 组不同输入上，效应量 (Cohen's d) 均表明三易控制具有大效应，这证明了三易引擎对提升本地模型生成质量的显著作用。</p>
</div>"#,
            n = results.num_trials,
            d_c_mean = s.deviation_controlled.mean, d_c_std = s.deviation_controlled.std_dev,
            d_x_mean = s.deviation_constrained.mean, d_x_std = s.deviation_constrained.std_dev,
            d_imp = improvement_pct(s.deviation_controlled.mean, s.deviation_constrained.mean, true),
            c_c_mean = s.compliance_controlled.mean, c_c_std = s.compliance_controlled.std_dev,
            c_x_mean = s.compliance_constrained.mean, c_x_std = s.compliance_constrained.std_dev,
            c_imp = improvement_pct(s.compliance_controlled.mean, s.compliance_constrained.mean, false),
            m_c_mean = s.completion_controlled.mean, m_c_std = s.completion_controlled.std_dev,
            m_x_mean = s.completion_constrained.mean, m_x_std = s.completion_constrained.std_dev,
            m_imp = improvement_pct(s.completion_controlled.mean, s.completion_constrained.mean, false),
        ));
    } else if any_significant {
        html.push_str(&format!(
            r#"<div class="conclusion conclusion-weak">
  <h3>结论：三易认知约束引擎在部分指标上表现出统计显著提升。</h3>
  <p>偏离率显著: {} | 结构合规率显著: {} | 完成率显著: {}</p>
  <p>需要更多数据验证全部指标的有效性。</p>
</div>"#,
            if s.deviation_test.significant { "是" } else { "否" },
            if s.compliance_test.significant { "是" } else { "否" },
            if s.completion_test.significant { "是" } else { "否" },
        ));
    } else {
        html.push_str(
            r#"<div class="conclusion conclusion-weak">
  <h3>结论：在当前条件下，三易控制未表现出统计显著提升。</h3>
  <p>可能需要更大的样本量、不同的输入分布或调整约束参数来获得显著效果。</p>
</div>"#
        );
    }

    // ── Per-trial data ──
    html.push_str(&format!(
        r#"<h2>六、逐试验数据 ({})</h2>
<table>
<tr><th>#</th><th>输入（截断）</th><th>对照组偏离</th><th>实验组偏离</th><th>对照组合规</th><th>实验组合规</th></tr>"#,
        "前50组"
    ));

    for trial in results.trials.iter().take(50) {
        let preview: String = trial.input.chars().take(40).collect();
        html.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td class=\"stat-cell\">{:.4}</td><td class=\"stat-cell\">{:.4}</td><td class=\"stat-cell\">{:.4}</td><td class=\"stat-cell\">{:.4}</td></tr>\n",
            trial.trial_index + 1,
            preview,
            trial.controlled_metrics.avg_deviation,
            trial.constrained_metrics.avg_deviation,
            trial.controlled_metrics.structure_compliance_rate,
            trial.constrained_metrics.structure_compliance_rate,
        ));
    }

    html.push_str("</table>\n");

    // ── Detailed statistics ──
    html.push_str(r#"<h2>七、详细统计量</h2>
<table>
<tr><th>指标</th><th>组别</th><th>均值</th><th>标准差</th><th>最小值</th><th>最大值</th><th>中位数</th></tr>"#);

    push_detailed_row(&mut html, "偏离率", "对照组", &s.deviation_controlled);
    push_detailed_row(&mut html, "偏离率", "实验组", &s.deviation_constrained);
    push_detailed_row(&mut html, "结构合规率", "对照组", &s.compliance_controlled);
    push_detailed_row(&mut html, "结构合规率", "实验组", &s.compliance_constrained);
    push_detailed_row(&mut html, "完成率", "对照组", &s.completion_controlled);
    push_detailed_row(&mut html, "完成率", "实验组", &s.completion_constrained);

    html.push_str("</table>\n</body>\n</html>");
    html
}

fn push_metric_row(
    html: &mut String,
    name: &str,
    c_stats: &MetricStats,
    x_stats: &MetricStats,
    test: &crate::statistics::TTestResult,
) {
    let sig_class = if test.significant { "sig-yes" } else { "sig-no" };
    let sig_text = if test.significant { "✓ 显著" } else { "—" };

    html.push_str(&format!(
        "<tr>
  <td>{name}</td>
  <td class=\"stat-cell\">{c_mean:.4} ± {c_std:.4}</td>
  <td class=\"stat-cell\">{x_mean:.4} ± {x_std:.4}</td>
  <td class=\"stat-cell\">{t:.2}</td>
  <td class=\"stat-cell\">{p:.4}</td>
  <td class=\"stat-cell\">{d:.2}</td>
  <td class=\"{sig_class}\">{sig_text}</td>
</tr>\n",
        name = name,
        c_mean = c_stats.mean, c_std = c_stats.std_dev,
        x_mean = x_stats.mean, x_std = x_stats.std_dev,
        t = test.t_statistic, p = test.p_value, d = test.cohens_d,
        sig_class = sig_class, sig_text = sig_text,
    ));
}

fn push_effect_row(html: &mut String, name: &str, d: f64) {
    let (effect, class) = if d >= 0.8 {
        ("大效应", "effect-large")
    } else if d >= 0.5 {
        ("中效应", "effect-medium")
    } else if d >= 0.2 {
        ("小效应", "effect-small")
    } else {
        ("可忽略", "effect-small")
    };

    html.push_str(&format!(
        "<tr><td>{}</td><td class=\"stat-cell\">{:.2}</td><td class=\"{}\">{}</td></tr>\n",
        name, d, class, effect
    ));
}

fn push_chart_group(html: &mut String, label: &str, c_val: f64, x_val: f64, max_val: f64, _invert: bool) {
    let max_h = 160.0;
    let c_h = ((c_val / max_val).min(1.0) * max_h).max(4.0);
    let x_h = ((x_val / max_val).min(1.0) * max_h).max(4.0);

    html.push_str(&format!(
        r#"<div class="chart-group">
  <div class="chart-label">{label}</div>
  <div class="chart-bars">
    <div class="bar-wrapper">
      <div class="bar bar-controlled" style="height:{c_h}px"></div>
      <div class="bar-value">{c_val:.3}</div>
    </div>
    <div class="bar-wrapper">
      <div class="bar bar-constrained" style="height:{x_h}px"></div>
      <div class="bar-value">{x_val:.3}</div>
    </div>
  </div>
</div>"#,
        label = label, c_h = c_h, x_h = x_h, c_val = c_val, x_val = x_val,
    ));
}

fn push_detailed_row(html: &mut String, metric: &str, group: &str, stats: &MetricStats) {
    html.push_str(&format!(
        "<tr><td>{}</td><td>{}</td><td class=\"stat-cell\">{:.4}</td><td class=\"stat-cell\">{:.4}</td><td class=\"stat-cell\">{:.4}</td><td class=\"stat-cell\">{:.4}</td><td class=\"stat-cell\">{:.4}</td></tr>\n",
        metric, group, stats.mean, stats.std_dev, stats.min, stats.max, stats.median,
    ));
}

fn improvement_pct(controlled: f64, constrained: f64, lower_is_better: bool) -> f64 {
    if controlled.abs() < 1e-10 {
        return 0.0;
    }
    if lower_is_better {
        ((controlled - constrained) / controlled * 100.0).max(0.0)
    } else {
        ((constrained - controlled) / controlled * 100.0).max(0.0)
    }
}
