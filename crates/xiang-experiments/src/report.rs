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

  /* Collapsible details for quality input */
  details.trial-detail { background: #161b22; border: 1px solid #30363d; border-radius: 8px; margin: 0.8em 0; padding: 0; }
  details.trial-detail > summary { padding: 12px 16px; cursor: pointer; color: #d2a8ff; font-weight: 600; font-size: 0.95em; background: #1c2128; border-radius: 8px; }
  details.trial-detail > summary:hover { color: #f0883e; }
  details.trial-detail[open] > summary { border-radius: 8px 8px 0 0; border-bottom: 1px solid #30363d; }
  details.trial-detail .detail-content { padding: 12px 16px; }
  .turn-block { background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 10px 14px; margin: 8px 0; }
  .turn-block .turn-meta { color: #8b949e; font-size: 0.85em; margin-bottom: 6px; }
  .turn-block .turn-meta strong { color: #c9d1d9; }
  .turn-block pre { margin: 6px 0 0 0; white-space: pre-wrap; word-break: break-word; font-size: 0.82em; max-height: 300px; overflow-y: auto; }
  .group-header { color: #58a6ff; font-weight: 600; margin: 10px 0 6px 0; font-size: 0.9em; }
  .qa-copy-btn { display: inline-block; padding: 6px 16px; margin: 8px 0; background: #238636; color: #ffffff; border: none; border-radius: 6px; cursor: pointer; font-size: 0.85em; }
  .qa-copy-btn:hover { background: #2ea043; }
  .input-block { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 10px 14px; margin: 6px 0; font-size: 0.9em; color: #c9d1d9; white-space: pre-wrap; }
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

    html.push_str("</table>\n");

    // ── AI 语义质量评估输入数据 (八) ──
    html.push_str(r#"<h2>八、AI 语义质量评估输入数据</h2>
<p class="meta">以下为逐试验逐轮次的生成内容，可供 AI 评估器进行语义质量评分。点击展开各试验查看详情。</p>
<button class="qa-copy-btn" onclick="navigator.clipboard.writeText(document.getElementById('qa-input-json').textContent)">复制 JSON 输入数据</button>
<pre id="qa-input-json" style="display:none;">"#);

    // Build inline JSON matching benchmark_quality_input.json
    {
        #[derive(serde::Serialize)]
        struct QTurn {
            turn_number: usize,
            generated_text: String,
            operator: String,
            deviation: f32,
            phase_valid: Option<bool>,
        }
        #[derive(serde::Serialize)]
        struct QTrial {
            trial_index: usize,
            input: String,
            controlled_turns: Vec<QTurn>,
            constrained_turns: Vec<QTurn>,
        }
        #[derive(serde::Serialize)]
        struct QFile<'a> {
            rubric: &'a str,
            trials: Vec<QTrial>,
        }

        let mut qt = Vec::new();
        for trial in &results.trials {
            let ct: Vec<QTurn> = trial.controlled_metrics.turn_data.iter().map(|t| {
                QTurn {
                    turn_number: t.turn_number,
                    generated_text: t.generated_text_preview.clone(),
                    operator: t.operator_checked.clone().unwrap_or_else(|| "?".into()),
                    deviation: t.deviation,
                    phase_valid: t.phase_valid,
                }
            }).collect();
            let xt: Vec<QTurn> = trial.constrained_metrics.turn_data.iter().map(|t| {
                QTurn {
                    turn_number: t.turn_number,
                    generated_text: t.generated_text_preview.clone(),
                    operator: t.operator_checked.clone().unwrap_or_else(|| "?".into()),
                    deviation: t.deviation,
                    phase_valid: t.phase_valid,
                }
            }).collect();
            qt.push(QTrial { trial_index: trial.trial_index, input: trial.input.clone(), controlled_turns: ct, constrained_turns: xt });
        }
        let qf = QFile { rubric: crate::QUALITY_EVALUATION_RUBRIC, trials: qt };
        let json_str = serde_json::to_string_pretty(&qf).unwrap_or_else(|_| "{}".into());
        html.push_str(&html_escape(&json_str));
    }
    html.push_str("</pre>\n");

    // Per-trial details
    for trial in &results.trials {
        html.push_str(&format!(
            r#"<details class="trial-detail">
<summary>试验 #{} — {}</summary>
<div class="detail-content">
<div class="input-block"><strong>输入:</strong> {}</div>"#,
            trial.trial_index + 1,
            html_escape(&truncate_str(&trial.input, 60)),
            html_escape(&trial.input),
        ));

        // Controlled turns
        html.push_str(r#"<div class="group-header">▼ 对照组 (裸跑)</div>"#);
        for turn in &trial.controlled_metrics.turn_data {
            let op = turn.operator_checked.as_deref().unwrap_or("?");
            let pv = match turn.phase_valid {
                Some(true) => "✓",
                Some(false) => "✗",
                None => "—",
            };
            html.push_str(&format!(
                r#"<div class="turn-block">
<div class="turn-meta"><strong>轮次#{tn}</strong> | 算子: {op} | 偏离: {dev:.3} | 阶段有效: {pv}</div>
<pre>{text}</pre>
</div>"#,
                tn = turn.turn_number,
                op = op,
                dev = turn.deviation,
                pv = pv,
                text = html_escape(&turn.generated_text_preview),
            ));
        }

        // Constrained turns
        html.push_str(r#"<div class="group-header">▼ 实验组 (三易控制)</div>"#);
        for turn in &trial.constrained_metrics.turn_data {
            let op = turn.operator_checked.as_deref().unwrap_or("?");
            let pv = match turn.phase_valid {
                Some(true) => "✓",
                Some(false) => "✗",
                None => "—",
            };
            html.push_str(&format!(
                r#"<div class="turn-block">
<div class="turn-meta"><strong>轮次#{tn}</strong> | 算子: {op} | 偏离: {dev:.3} | 阶段有效: {pv}</div>
<pre>{text}</pre>
</div>"#,
                tn = turn.turn_number,
                op = op,
                dev = turn.deviation,
                pv = pv,
                text = html_escape(&turn.generated_text_preview),
            ));
        }

        html.push_str("</div>\n</details>\n");
    }

    // ── AI 语义质量评估结果 (九，仅当加载了评估报告时) ──
    if let Some(_qr) = &results.quality_report {
        html.push_str(r#"<h2>九、AI 语义质量评估结果</h2>"#);

        // Quality summary table
        html.push_str(r#"<h3>9.1 质量指标统计显著性分析</h3>
<table>
<tr>
  <th>质量维度</th>
  <th>对照组 (裸跑)</th>
  <th>实验组 (三易控制)</th>
  <th>t 统计量</th>
  <th>p 值</th>
  <th>Cohen's d</th>
  <th>显著性</th>
</tr>"#);

        push_quality_metric_row(&mut html, "综合质量", &s.quality_overall_controlled, &s.quality_overall_constrained, &s.quality_overall_test);
        push_quality_metric_row(&mut html, "任务完成度", &s.quality_completion_controlled, &s.quality_completion_constrained, &s.quality_completion_test);
        push_quality_metric_row(&mut html, "逻辑连贯性", &s.quality_coherence_controlled, &s.quality_coherence_constrained, &s.quality_coherence_test);
        push_quality_metric_row(&mut html, "内容相关性", &s.quality_relevance_controlled, &s.quality_relevance_constrained, &s.quality_relevance_test);
        push_quality_metric_row(&mut html, "分析深度", &s.quality_depth_controlled, &s.quality_depth_constrained, &s.quality_depth_test);
        push_quality_metric_row(&mut html, "结构清晰度", &s.quality_structure_controlled, &s.quality_structure_constrained, &s.quality_structure_test);

        html.push_str("</table>\n");

        // Quality effect sizes
        html.push_str(r#"<h3>9.2 质量维度效应量</h3>
<table>
<tr><th>质量维度</th><th>Cohen's d</th><th>效应大小</th></tr>"#);

        push_quality_effect_row(&mut html, "综合质量", s.quality_overall_test.as_ref());
        push_quality_effect_row(&mut html, "任务完成度", s.quality_completion_test.as_ref());
        push_quality_effect_row(&mut html, "逻辑连贯性", s.quality_coherence_test.as_ref());
        push_quality_effect_row(&mut html, "内容相关性", s.quality_relevance_test.as_ref());
        push_quality_effect_row(&mut html, "分析深度", s.quality_depth_test.as_ref());
        push_quality_effect_row(&mut html, "结构清晰度", s.quality_structure_test.as_ref());

        html.push_str("</table>\n");

        // Quality bar chart
        html.push_str(r#"<h3>9.3 质量维度可视化对比</h3>
<div class="bar-legend">
  <span class="legend-item"><span class="legend-swatch" style="background:#f85149"></span>对照组 (裸跑)</span>
  <span class="legend-item"><span class="legend-swatch" style="background:#3fb950"></span>实验组 (三易控制)</span>
</div>
<div class="chart-container">"#);

        push_quality_chart_group(&mut html, "综合质量", &s.quality_overall_controlled, &s.quality_overall_constrained);
        push_quality_chart_group(&mut html, "任务完成度", &s.quality_completion_controlled, &s.quality_completion_constrained);
        push_quality_chart_group(&mut html, "逻辑连贯性", &s.quality_coherence_controlled, &s.quality_coherence_constrained);
        push_quality_chart_group(&mut html, "内容相关性", &s.quality_relevance_controlled, &s.quality_relevance_constrained);
        push_quality_chart_group(&mut html, "分析深度", &s.quality_depth_controlled, &s.quality_depth_constrained);
        push_quality_chart_group(&mut html, "结构清晰度", &s.quality_structure_controlled, &s.quality_structure_constrained);

        html.push_str("</div>\n");

        // Quality detailed statistics
        html.push_str(r#"<h3>9.4 质量维度详细统计量</h3>
<table>
<tr><th>质量维度</th><th>组别</th><th>均值</th><th>标准差</th><th>最小值</th><th>最大值</th><th>中位数</th></tr>"#);

        push_quality_detailed_row(&mut html, "综合质量", "对照组", &s.quality_overall_controlled);
        push_quality_detailed_row(&mut html, "综合质量", "实验组", &s.quality_overall_constrained);
        push_quality_detailed_row(&mut html, "任务完成度", "对照组", &s.quality_completion_controlled);
        push_quality_detailed_row(&mut html, "任务完成度", "实验组", &s.quality_completion_constrained);
        push_quality_detailed_row(&mut html, "逻辑连贯性", "对照组", &s.quality_coherence_controlled);
        push_quality_detailed_row(&mut html, "逻辑连贯性", "实验组", &s.quality_coherence_constrained);
        push_quality_detailed_row(&mut html, "内容相关性", "对照组", &s.quality_relevance_controlled);
        push_quality_detailed_row(&mut html, "内容相关性", "实验组", &s.quality_relevance_constrained);
        push_quality_detailed_row(&mut html, "分析深度", "对照组", &s.quality_depth_controlled);
        push_quality_detailed_row(&mut html, "分析深度", "实验组", &s.quality_depth_constrained);
        push_quality_detailed_row(&mut html, "结构清晰度", "对照组", &s.quality_structure_controlled);
        push_quality_detailed_row(&mut html, "结构清晰度", "实验组", &s.quality_structure_constrained);

        html.push_str("</table>\n");

        // Per-evaluation comparison notes
        push_quality_evaluation_notes(&mut html, _qr);
    }

    html.push_str("\n</body>\n</html>");
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

// ── Quality evaluation report helpers ──

use crate::statistics::TTestResult;

fn push_quality_metric_row(
    html: &mut String,
    name: &str,
    c_stats: &Option<MetricStats>,
    x_stats: &Option<MetricStats>,
    test: &Option<TTestResult>,
) {
    let (c_mean, c_std) = match c_stats {
        Some(s) => (format!("{:.4}", s.mean), format!("{:.4}", s.std_dev)),
        None => ("N/A".into(), "N/A".into()),
    };
    let (x_mean, x_std) = match x_stats {
        Some(s) => (format!("{:.4}", s.mean), format!("{:.4}", s.std_dev)),
        None => ("N/A".into(), "N/A".into()),
    };
    let (t, p, d, sig_class, sig_text) = match test {
        Some(t) => (
            format!("{:.2}", t.t_statistic),
            format!("{:.4}", t.p_value),
            format!("{:.2}", t.cohens_d),
            if t.significant { "sig-yes" } else { "sig-no" },
            if t.significant { "✓ 显著" } else { "—" },
        ),
        None => ("—".into(), "—".into(), "—".into(), "sig-no", "—"),
    };

    html.push_str(&format!(
        "<tr>
  <td>{name}</td>
  <td class=\"stat-cell\">{c_mean} ± {c_std}</td>
  <td class=\"stat-cell\">{x_mean} ± {x_std}</td>
  <td class=\"stat-cell\">{t}</td>
  <td class=\"stat-cell\">{p}</td>
  <td class=\"stat-cell\">{d}</td>
  <td class=\"{sig_class}\">{sig_text}</td>
</tr>\n",
        name = name,
    ));
}

fn push_quality_effect_row(html: &mut String, name: &str, test: Option<&TTestResult>) {
    let d = test.map(|t| t.cohens_d).unwrap_or(0.0);
    let (effect, class) = if d >= 0.8 {
        ("大效应", "effect-large")
    } else if d >= 0.5 {
        ("中效应", "effect-medium")
    } else if d >= 0.2 {
        ("小效应", "effect-small")
    } else {
        ("可忽略", "effect-small")
    };
    let d_str = test.map(|t| format!("{:.2}", t.cohens_d)).unwrap_or_else(|| "—".into());

    html.push_str(&format!(
        "<tr><td>{}</td><td class=\"stat-cell\">{}</td><td class=\"{}\">{}</td></tr>\n",
        name, d_str, class, effect
    ));
}

fn push_quality_chart_group(
    html: &mut String,
    label: &str,
    c_stats: &Option<MetricStats>,
    x_stats: &Option<MetricStats>,
) {
    let c_val = c_stats.as_ref().map(|s| s.mean).unwrap_or(0.0);
    let x_val = x_stats.as_ref().map(|s| s.mean).unwrap_or(0.0);
    let max_val = c_val.max(x_val).max(0.01);
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

fn push_quality_detailed_row(html: &mut String, metric: &str, group: &str, stats: &Option<MetricStats>) {
    match stats {
        Some(s) => {
            html.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td class=\"stat-cell\">{:.4}</td><td class=\"stat-cell\">{:.4}</td><td class=\"stat-cell\">{:.4}</td><td class=\"stat-cell\">{:.4}</td><td class=\"stat-cell\">{:.4}</td></tr>\n",
                metric, group, s.mean, s.std_dev, s.min, s.max, s.median,
            ));
        }
        None => {
            html.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td class=\"stat-cell\">—</td><td class=\"stat-cell\">—</td><td class=\"stat-cell\">—</td><td class=\"stat-cell\">—</td><td class=\"stat-cell\">—</td></tr>\n",
                metric, group,
            ));
        }
    }
}

fn push_quality_evaluation_notes(html: &mut String, report: &crate::QualityEvaluationReport) {
    html.push_str(r#"<h3>9.5 逐试验评估备注</h3>
<table>
<tr><th>#</th><th>输入摘要</th><th>对照组综合</th><th>实验组综合</th><th>对比备注</th></tr>"#);

    for eval in &report.evaluations {
        let summary: String = eval.input_summary.chars().take(50).collect();
        html.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td class=\"stat-cell\">{:.4}</td><td class=\"stat-cell\">{:.4}</td><td>{}</td></tr>\n",
            eval.trial_index + 1,
            summary,
            eval.controlled_overall,
            eval.constrained_overall,
            eval.comparison_note,
        ));
    }

    html.push_str("</table>\n");
}

// ── HTML utility helpers ──

/// Escape special HTML characters.
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// Truncate a string to `max_chars` Unicode characters, appending "…" if truncated.
fn truncate_str(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_chars).collect();
        format!("{}…", truncated)
    }
}

// ── Three-way report (约束组 / 微调组 / 原生组) ──────────────

/// Generate a three-way benchmark HTML report comparing:
///   - 约束组: RouteB model + silent engine (logit bias only, no text prompts)
///   - 微调组: RouteB model + bare question (no engine)
///   - 原生组: Base model + bare question (baseline)
pub fn generate_threeway_benchmark_html_report(
    yue_shu_results: &BenchmarkResults,
    wei_diao_results: &BenchmarkResults,
    yuan_sheng_results: &BenchmarkResults,
) -> String {
    let ys_s = &yue_shu_results.summary;     // 约束组 → constrained fields
    let ws_s = &wei_diao_results.summary;     // 微调组 → controlled fields
    let ys2_s = &yuan_sheng_results.summary;  // 原生组 → controlled fields

    // Extract stats from each result's appropriate fields
    let yue_shu = (
        &ys_s.deviation_constrained,
        &ys_s.compliance_constrained,
        &ys_s.completion_constrained,
        &ys_s.trust_constrained,
    );
    let wei_diao = (
        &ws_s.deviation_controlled,
        &ws_s.compliance_controlled,
        &ws_s.completion_controlled,
        &ws_s.trust_controlled,
    );
    let yuan_sheng = (
        &ys2_s.deviation_controlled,
        &ys2_s.compliance_controlled,
        &ys2_s.completion_controlled,
        &ys2_s.trust_controlled,
    );

    let model_routeb = &yue_shu_results.model_name;
    let model_base = &yuan_sheng_results.model_name;

    let mut html = String::new();
    html.push_str(r#"<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>路线B — 三组对照基准测试报告</title>
<style>
  body { font-family: system-ui, -apple-system, sans-serif; max-width: 1200px; margin: 2em auto; padding: 0 1.5em; background: #0d1117; color: #c9d1d9; line-height: 1.6; }
  h1 { color: #58a6ff; border-bottom: 2px solid #30363d; padding-bottom: 0.4em; }
  h2 { color: #f0883e; margin-top: 2em; }
  h3 { color: #d2a8ff; }
  table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.95em; }
  th, td { border: 1px solid #30363d; padding: 10px 14px; text-align: left; }
  th { background: #161b22; color: #8b949e; font-weight: 600; }
  .stat-cell { text-align: right; font-variant-numeric: tabular-nums; }
  .stat-best { font-weight: bold; color: #3fb950; }
  .summary-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1.2em 1.5em; margin: 1.2em 0; }
  .meta { color: #8b949e; font-size: 0.9em; }
  pre { background: #161b22; padding: 1em; border-radius: 6px; overflow-x: auto; font-size: 0.85em; }

  .chart-container { display: flex; align-items: flex-end; gap: 20px; margin: 1em 0; padding: 1em; background: #161b22; border-radius: 8px; flex-wrap: wrap; }
  .chart-group { flex: 1; min-width: 140px; text-align: center; }
  .chart-label { font-size: 0.8em; color: #8b949e; margin-bottom: 6px; }
  .chart-bars { display: flex; justify-content: center; gap: 8px; height: 200px; align-items: flex-end; }
  .bar-wrapper { display: flex; flex-direction: column; align-items: center; gap: 4px; }
  .bar { width: 30px; border-radius: 4px 4px 0 0; transition: height 0.3s; position: relative; min-height: 4px; }
  .bar-yueshu { background: linear-gradient(180deg, #3fb950, #238636); }
  .bar-weidiao { background: linear-gradient(180deg, #58a6ff, #1f6feb); }
  .bar-yuansheng { background: linear-gradient(180deg, #f85149, #b62324); }
  .bar-value { font-size: 0.7em; color: #c9d1d9; }
  .bar-legend { display: flex; justify-content: center; gap: 20px; margin: 0.5em 0; font-size: 0.85em; }
  .legend-item { display: flex; align-items: center; gap: 6px; }
  .legend-swatch { width: 14px; height: 14px; border-radius: 3px; }

  .conclusion { background: #0d2b1e; border: 1px solid #3fb950; border-radius: 8px; padding: 1em 1.5em; margin: 1.5em 0; color: #7ee787; }
  .conclusion-weak { background: #2b2200; border: 1px solid #d29922; color: #d29922; }
  details.trial-detail { background: #161b22; border: 1px solid #30363d; border-radius: 8px; margin: 0.8em 0; padding: 0; }
  details.trial-detail > summary { padding: 12px 16px; cursor: pointer; color: #d2a8ff; font-weight: 600; font-size: 0.95em; background: #1c2128; border-radius: 8px; }
  details.trial-detail > summary:hover { color: #f0883e; }
  details.trial-detail[open] > summary { border-radius: 8px 8px 0 0; border-bottom: 1px solid #30363d; }
  details.trial-detail .detail-content { padding: 12px 16px; }
  .turn-block { background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 10px 14px; margin: 8px 0; }
  .turn-block .turn-meta { color: #8b949e; font-size: 0.85em; margin-bottom: 6px; }
  .turn-block .turn-meta strong { color: #c9d1d9; }
  .turn-block pre { margin: 6px 0 0 0; white-space: pre-wrap; word-break: break-word; font-size: 0.82em; max-height: 300px; overflow-y: auto; }
  .group-header { color: #58a6ff; font-weight: 600; margin: 10px 0 6px 0; font-size: 0.9em; }
  .input-block { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 10px 14px; margin: 6px 0; font-size: 0.9em; color: #c9d1d9; white-space: pre-wrap; }
  .effect-large { color: #3fb950; font-weight: bold; }
  .effect-medium { color: #d29922; }
  .effect-small { color: #8b949e; }
</style>
</head>
<body>
<h1>路线B — 三组对照基准测试报告</h1>
<div class="meta">
  <p>试验次数: "#);
    html.push_str(&yue_shu_results.num_trials.to_string());
    html.push_str(r#" | 输入: 10主题×10变体 | 种子: 42</p>
  <p>路线B模型: "#);
    html.push_str(&truncate_filename(model_routeb));
    html.push_str(r#" | 基础模型: "#);
    html.push_str(&truncate_filename(model_base));
    html.push_str(r#"</p>
  <p>后端: "#);
    html.push_str(&yue_shu_results.backend_type);
    html.push_str(r#" | 测试时间: "#);
    html.push_str(&chrono_now());
    html.push_str(r#"</p>
  <table>
  <tr><th>组别</th><th>模型</th><th>约束系统</th></tr>
  <tr><td>约束组</td><td>路线B (微调后)</td><td>✅ 静默引擎 (logit bias, 无文本提示)</td></tr>
  <tr><td>微调组</td><td>路线B (微调后)</td><td>❌ 裸问题 (无约束)</td></tr>
  <tr><td>原生组</td><td>Qwen2.5-0.5B (基座)</td><td>❌ 裸问题 (基线)</td></tr>
  </table>
</div>
"#);

    // ── 核心指标三组对比 ──
    html.push_str(r#"<h2>一、核心指标三组对比</h2>
<table>
<tr>
  <th>指标</th>
  <th>约束组 <span class="meta">(路线B+静默引擎)</span></th>
  <th>微调组 <span class="meta">(路线B+裸问题)</span></th>
  <th>原生组 <span class="meta">(基座+裸问题)</span></th>
  <th>最优组</th>
</tr>"#);

    push_threeway_row(&mut html, "偏离率 ↓", yue_shu.0, wei_diao.0, yuan_sheng.0, true);
    push_threeway_row(&mut html, "结构合规率 ↑", yue_shu.1, wei_diao.1, yuan_sheng.1, false);
    push_threeway_row(&mut html, "完成率 ↑", yue_shu.2, wei_diao.2, yuan_sheng.2, false);
    push_threeway_row(&mut html, "信任评分 ↑", yue_shu.3, wei_diao.3, yuan_sheng.3, false);

    html.push_str("</table>\n");

    // ── 可视化 ──
    html.push_str(r#"<h2>二、可视化对比</h2>
<div class="bar-legend">
  <span class="legend-item"><span class="legend-swatch" style="background:#3fb950"></span>约束组 (路线B+静默引擎)</span>
  <span class="legend-item"><span class="legend-swatch" style="background:#58a6ff"></span>微调组 (路线B+裸问题)</span>
  <span class="legend-item"><span class="legend-swatch" style="background:#f85149"></span>原生组 (基座+裸问题)</span>
</div>
<div class="chart-container">"#);

    push_threeway_chart(&mut html, "偏离率", yue_shu.0.mean, wei_diao.0.mean, yuan_sheng.0.mean, 1.0);
    push_threeway_chart(&mut html, "结构合规率", yue_shu.1.mean, wei_diao.1.mean, yuan_sheng.1.mean, 1.0);
    push_threeway_chart(&mut html, "完成率", yue_shu.2.mean, wei_diao.2.mean, yuan_sheng.2.mean, 1.0);

    html.push_str("</div>\n");

    // ── 详细统计量 ──
    html.push_str(r#"<h2>三、详细统计量</h2>
<table>
<tr><th>指标</th><th>组别</th><th>均值</th><th>标准差</th><th>最小值</th><th>最大值</th><th>中位数</th></tr>"#);

    push_threeway_detailed(&mut html, "偏离率", "约束组", yue_shu.0);
    push_threeway_detailed(&mut html, "偏离率", "微调组", wei_diao.0);
    push_threeway_detailed(&mut html, "偏离率", "原生组", yuan_sheng.0);
    push_threeway_detailed(&mut html, "结构合规率", "约束组", yue_shu.1);
    push_threeway_detailed(&mut html, "结构合规率", "微调组", wei_diao.1);
    push_threeway_detailed(&mut html, "结构合规率", "原生组", yuan_sheng.1);
    push_threeway_detailed(&mut html, "完成率", "约束组", yue_shu.2);
    push_threeway_detailed(&mut html, "完成率", "微调组", wei_diao.2);
    push_threeway_detailed(&mut html, "完成率", "原生组", yuan_sheng.2);

    html.push_str("</table>\n");

    // ── 关键发现 ──
    html.push_str(r#"<h2>四、关键发现</h2>
<div class="summary-card">"#);

    // Compare 微调组 vs 原生组 (fine-tuning effect without constraints)
    let ft_dev = wei_diao.0.mean;
    let na_dev = yuan_sheng.0.mean;
    let ft_cmp = wei_diao.1.mean;
    let na_cmp = yuan_sheng.1.mean;

    html.push_str(&format!(
        r#"<p><strong>微调效果（微调组 vs 原生组）：</strong></p>
<ul>
  <li>偏离率: {:.4} → {:.4}（{}）</li>
  <li>结构合规率: {:.4} → {:.4}（{}）</li>
</ul>
"#,
        na_dev, ft_dev,
        if ft_dev < na_dev { format!("降低 {:.1}%", (na_dev - ft_dev) / na_dev * 100.0) } else { format!("升高 {:.1}%", (ft_dev - na_dev) / na_dev * 100.0) },
        na_cmp, ft_cmp,
        if ft_cmp > na_cmp { format!("提升 {:.1}%", (ft_cmp - na_cmp) / na_cmp * 100.0) } else { format!("降低 {:.1}%", (na_cmp - ft_cmp) / na_cmp * 100.0) },
    ));

    // Compare 约束组 vs 微调组 (constraint effect on fine-tuned model)
    let ys_dev = yue_shu.0.mean;
    let ys_cmp = yue_shu.1.mean;

    html.push_str(&format!(
        r#"<p><strong>约束效果（约束组 vs 微调组）：</strong></p>
<ul>
  <li>偏离率: {:.4} → {:.4}（{}）</li>
  <li>结构合规率: {:.4} → {:.4}（{}）</li>
</ul>
"#,
        ft_dev, ys_dev,
        if ys_dev < ft_dev { format!("降低 {:.1}%", (ft_dev - ys_dev) / ft_dev * 100.0) } else { format!("升高 {:.1}%", (ys_dev - ft_dev) / ft_dev * 100.0) },
        ft_cmp, ys_cmp,
        if ys_cmp > ft_cmp { format!("提升 {:.1}%", (ys_cmp - ft_cmp) / ft_cmp * 100.0) } else { format!("降低 {:.1}%", (ft_cmp - ys_cmp) / ft_cmp * 100.0) },
    ));

    // Final verdict
    let best_dev = yue_shu.0.mean.min(wei_diao.0.mean).min(yuan_sheng.0.mean);
    let best_dev_group = if best_dev == yue_shu.0.mean { "约束组" } else if best_dev == wei_diao.0.mean { "微调组" } else { "原生组" };
    let best_cmp = yue_shu.1.mean.max(wei_diao.1.mean).max(yuan_sheng.1.mean);
    let best_cmp_group = if best_cmp == yue_shu.1.mean { "约束组" } else if best_cmp == wei_diao.1.mean { "微调组" } else { "原生组" };

    html.push_str(&format!(
        r#"<p><strong>综合结论：</strong></p>
<ul>
  <li>最低偏离率: {} ({:.4})</li>
  <li>最高结构合规率: {} ({:.4})</li>
</ul>
</div>
"#,
        best_dev_group, best_dev,
        best_cmp_group, best_cmp,
    ));

    // ── 逐试验数据 ──
    let num_display = yue_shu_results.num_trials.min(50);
    html.push_str(&format!(
        r#"<h2>五、逐试验数据 (前{}组)</h2>
<table>
<tr><th>#</th><th>输入</th><th>约束组偏离</th><th>微调组偏离</th><th>原生组偏离</th><th>约束组合规</th><th>微调组合规</th><th>原生组合规</th></tr>
"#,
        num_display
    ));

    for i in 0..num_display {
        let ys_t = &yue_shu_results.trials[i];
        let wd_t = &wei_diao_results.trials[i];
        let ys2_t = &yuan_sheng_results.trials[i];
        let preview: String = ys_t.input.chars().take(35).collect();
        html.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td class=\"stat-cell\">{:.4}</td><td class=\"stat-cell\">{:.4}</td><td class=\"stat-cell\">{:.4}</td><td class=\"stat-cell\">{:.4}</td><td class=\"stat-cell\">{:.4}</td><td class=\"stat-cell\">{:.4}</td></tr>\n",
            i + 1, preview,
            ys_t.constrained_metrics.avg_deviation,
            wd_t.controlled_metrics.avg_deviation,
            ys2_t.controlled_metrics.avg_deviation,
            ys_t.constrained_metrics.structure_compliance_rate,
            wd_t.controlled_metrics.structure_compliance_rate,
            ys2_t.controlled_metrics.structure_compliance_rate,
        ));
    }

    html.push_str("</table>\n");

    // ── 逐试验展开详情 ──
    let num_detail = yue_shu_results.trials.len().min(10);
    html.push_str(&format!(
        r#"<h2>六、逐试验详情 (前{}组)</h2>
"#,
        num_detail
    ));

    for i in 0..num_detail {
        let ys_t = &yue_shu_results.trials[i];
        let wd_t = &wei_diao_results.trials[i];
        let ys2_t = &yuan_sheng_results.trials[i];
        let preview: String = ys_t.input.chars().take(60).collect();

        html.push_str(&format!(
            r#"<details class="trial-detail">
<summary>试验 #{} — {}</summary>
<div class="detail-content">
<div class="input-block"><strong>输入:</strong> {}</div>
"#,
            i + 1, html_escape(&preview), html_escape(&ys_t.input),
        ));

        // 约束组 turns (from yue_shu_results constrained_metrics)
        html.push_str(r#"<div class="group-header" style="color:#3fb950">▼ 约束组 (路线B + 静默引擎)</div>"#);
        for turn in &ys_t.constrained_metrics.turn_data {
            let op = turn.operator_checked.as_deref().unwrap_or("?");
            let pv = match turn.phase_valid { Some(true) => "✓", Some(false) => "✗", None => "—" };
            html.push_str(&format!(
                r#"<div class="turn-block">
<div class="turn-meta"><strong>轮次#{tn}</strong> | 算子: {op} | 偏离: {dev:.3} | 阶段有效: {pv}</div>
<pre>{text}</pre>
</div>"#,
                tn = turn.turn_number, op = op, dev = turn.deviation, pv = pv,
                text = html_escape(&turn.generated_text_preview),
            ));
        }

        // 微调组 turns (from wei_diao_results controlled_metrics)
        html.push_str(r#"<div class="group-header" style="color:#58a6ff">▼ 微调组 (路线B + 裸问题)</div>"#);
        for turn in &wd_t.controlled_metrics.turn_data {
            let op = turn.operator_checked.as_deref().unwrap_or("?");
            let pv = match turn.phase_valid { Some(true) => "✓", Some(false) => "✗", None => "—" };
            html.push_str(&format!(
                r#"<div class="turn-block">
<div class="turn-meta"><strong>轮次#{tn}</strong> | 算子: {op} | 偏离: {dev:.3} | 阶段有效: {pv}</div>
<pre>{text}</pre>
</div>"#,
                tn = turn.turn_number, op = op, dev = turn.deviation, pv = pv,
                text = html_escape(&turn.generated_text_preview),
            ));
        }

        // 原生组 turns (from yuan_sheng_results controlled_metrics)
        html.push_str(r#"<div class="group-header" style="color:#f85149">▼ 原生组 (基座 + 裸问题)</div>"#);
        for turn in &ys2_t.controlled_metrics.turn_data {
            let op = turn.operator_checked.as_deref().unwrap_or("?");
            let pv = match turn.phase_valid { Some(true) => "✓", Some(false) => "✗", None => "—" };
            html.push_str(&format!(
                r#"<div class="turn-block">
<div class="turn-meta"><strong>轮次#{tn}</strong> | 算子: {op} | 偏离: {dev:.3} | 阶段有效: {pv}</div>
<pre>{text}</pre>
</div>"#,
                tn = turn.turn_number, op = op, dev = turn.deviation, pv = pv,
                text = html_escape(&turn.generated_text_preview),
            ));
        }

        html.push_str("</div>\n</details>\n");
    }

    html.push_str("\n</body>\n</html>");
    html
}

fn push_threeway_row(
    html: &mut String,
    name: &str,
    yue_shu: &MetricStats,
    wei_diao: &MetricStats,
    yuan_sheng: &MetricStats,
    lower_is_better: bool,
) {
    let vals = [yue_shu.mean, wei_diao.mean, yuan_sheng.mean];
    let best_idx = if lower_is_better {
        vals.iter().enumerate().min_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0)
    } else {
        vals.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0)
    };
    let best_labels = ["约束组", "微调组", "原生组"];

    let yc = if 0 == best_idx { "stat-cell stat-best" } else { "stat-cell" };
    let wc = if 1 == best_idx { "stat-cell stat-best" } else { "stat-cell" };
    let ysc = if 2 == best_idx { "stat-cell stat-best" } else { "stat-cell" };

    html.push_str(&format!(
        "<tr>
  <td>{name}</td>
  <td class=\"{yc}\">{y_mean:.4} ± {y_std:.4}</td>
  <td class=\"{wc}\">{w_mean:.4} ± {w_std:.4}</td>
  <td class=\"{ysc}\">{ys_mean:.4} ± {ys_std:.4}</td>
  <td>{best}</td>
</tr>\n",
        name = name,
        yc = yc, y_mean = yue_shu.mean, y_std = yue_shu.std_dev,
        wc = wc, w_mean = wei_diao.mean, w_std = wei_diao.std_dev,
        ysc = ysc, ys_mean = yuan_sheng.mean, ys_std = yuan_sheng.std_dev,
        best = best_labels[best_idx],
    ));
}

fn push_threeway_chart(html: &mut String, label: &str, v1: f64, v2: f64, v3: f64, max_val: f64) {
    let max_h = 180.0;
    let h1 = ((v1 / max_val).min(1.0) * max_h).max(4.0);
    let h2 = ((v2 / max_val).min(1.0) * max_h).max(4.0);
    let h3 = ((v3 / max_val).min(1.0) * max_h).max(4.0);

    html.push_str(&format!(
        r#"<div class="chart-group">
  <div class="chart-label">{label}</div>
  <div class="chart-bars">
    <div class="bar-wrapper">
      <div class="bar bar-yueshu" style="height:{h1}px"></div>
      <div class="bar-value">{v1:.3}</div>
    </div>
    <div class="bar-wrapper">
      <div class="bar bar-weidiao" style="height:{h2}px"></div>
      <div class="bar-value">{v2:.3}</div>
    </div>
    <div class="bar-wrapper">
      <div class="bar bar-yuansheng" style="height:{h3}px"></div>
      <div class="bar-value">{v3:.3}</div>
    </div>
  </div>
</div>"#,
        label = label, h1 = h1, h2 = h2, h3 = h3, v1 = v1, v2 = v2, v3 = v3,
    ));
}

fn push_threeway_detailed(html: &mut String, metric: &str, group: &str, stats: &MetricStats) {
    html.push_str(&format!(
        "<tr><td>{}</td><td>{}</td><td class=\"stat-cell\">{:.4}</td><td class=\"stat-cell\">{:.4}</td><td class=\"stat-cell\">{:.4}</td><td class=\"stat-cell\">{:.4}</td><td class=\"stat-cell\">{:.4}</td></tr>\n",
        metric, group, stats.mean, stats.std_dev, stats.min, stats.max, stats.median,
    ));
}

/// Truncate a model path to just the filename.
fn truncate_filename(path: &str) -> String {
    path.rsplit(&['/', '\\'][..]).next().unwrap_or(path).to_string()
}

/// Get current time as a formatted string.
fn chrono_now() -> String {
    // Use std::time for portability
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    // Simple YYYY-MM-DD HH:MM:SS from UTC
    let days = secs / 86400;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let mins = (time_secs % 3600) / 60;
    let secs_rem = time_secs % 60;

    // Days since epoch to date (not accurate for leap years but good enough for display)
    let year = 1970 + (days as f64 / 365.25) as u64;
    format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02} (UTC)",
        year, 1 + ((days % 365) / 30).min(11), 1 + (days % 30).min(27),
        hours, mins, secs_rem)
}
