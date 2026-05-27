//! Statistical analysis module for benchmark results.
//!
//! Implements Welch's t-test, Cohen's d effect size, and descriptive statistics.
//! Zero external dependencies — all math implemented from first principles.

use serde::Serialize;

/// Descriptive statistics for a metric across trials.
#[derive(Debug, Clone, Serialize)]
pub struct MetricStats {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub n: usize,
}

/// Result of Welch's independent two-sample t-test.
#[derive(Debug, Clone, Serialize)]
pub struct TTestResult {
    /// t-statistic
    pub t_statistic: f64,
    /// Two-tailed p-value
    pub p_value: f64,
    /// Cohen's d effect size
    pub cohens_d: f64,
    /// Whether the difference is statistically significant (p < 0.05)
    pub significant: bool,
    /// Interpretation in Chinese
    pub interpretation: String,
}

/// Compute descriptive statistics for a slice of values.
pub fn compute_metric_stats(values: &[f64]) -> MetricStats {
    let n = values.len();
    if n == 0 {
        return MetricStats {
            mean: 0.0, std_dev: 0.0, min: 0.0, max: 0.0, median: 0.0, n: 0,
        };
    }

    let sum: f64 = values.iter().sum();
    let mean = sum / n as f64;

    let var: f64 = values.iter()
        .map(|v| (v - mean) * (v - mean))
        .sum::<f64>() / (n - 1) as f64; // Bessel's correction
    let std_dev = var.sqrt();

    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Median
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    };

    MetricStats { mean, std_dev, min, max, median, n }
}

/// Compute Cohen's d effect size (pooled standard deviation).
///
/// d = |mean_a - mean_b| / sqrt((var_a + var_b) / 2)
///
/// Interpretation:
///   d < 0.2: negligible
///   0.2 <= d < 0.5: small
///   0.5 <= d < 0.8: medium
///   d >= 0.8: large
pub fn cohens_d(a: &[f64], b: &[f64]) -> f64 {
    let n_a = a.len() as f64;
    let n_b = b.len() as f64;
    if n_a < 2.0 || n_b < 2.0 {
        return 0.0;
    }

    let mean_a = a.iter().sum::<f64>() / n_a;
    let mean_b = b.iter().sum::<f64>() / n_b;

    let var_a = a.iter().map(|v| (v - mean_a) * (v - mean_a)).sum::<f64>() / (n_a - 1.0);
    let var_b = b.iter().map(|v| (v - mean_b) * (v - mean_b)).sum::<f64>() / (n_b - 1.0);

    let pooled_sd = ((var_a + var_b) / 2.0).sqrt();
    if pooled_sd < 1e-10 {
        return 0.0;
    }

    (mean_a - mean_b).abs() / pooled_sd
}

/// Welch's independent two-sample t-test (unequal variance).
///
/// t = (mean_a - mean_b) / sqrt(var_a/n_a + var_b/n_b)
///
/// Degrees of freedom (Welch-Satterthwaite approximation):
///   df = (var_a/n_a + var_b/n_b)^2 /
///        ((var_a/n_a)^2/(n_a-1) + (var_b/n_b)^2/(n_b-1))
///
/// P-value is computed via the Student's t CDF using the incomplete beta function.
pub fn welch_t_test(a: &[f64], b: &[f64]) -> TTestResult {
    let n_a = a.len() as f64;
    let n_b = b.len() as f64;

    if n_a < 2.0 || n_b < 2.0 {
        return TTestResult {
            t_statistic: 0.0, p_value: 1.0, cohens_d: 0.0,
            significant: false,
            interpretation: "样本量不足，无法进行统计检验".into(),
        };
    }

    let mean_a = a.iter().sum::<f64>() / n_a;
    let mean_b = b.iter().sum::<f64>() / n_b;

    let var_a = a.iter().map(|v| (v - mean_a) * (v - mean_a)).sum::<f64>() / (n_a - 1.0);
    let var_b = b.iter().map(|v| (v - mean_b) * (v - mean_b)).sum::<f64>() / (n_b - 1.0);

    let se_a = var_a / n_a;
    let se_b = var_b / n_b;
    let se_diff = (se_a + se_b).sqrt();

    if se_diff < 1e-10 {
        // Zero variance in both groups — means are effectively identical
        return TTestResult {
            t_statistic: 0.0, p_value: 1.0, cohens_d: 0.0,
            significant: false,
            interpretation: "两组无差异（方差为零）".into(),
        };
    }

    let t = (mean_a - mean_b) / se_diff;

    // Welch-Satterthwaite degrees of freedom
    let num = (se_a + se_b) * (se_a + se_b);
    let denom = (se_a * se_a) / (n_a - 1.0) + (se_b * se_b) / (n_b - 1.0);
    let df = if denom < 1e-10 { n_a + n_b - 2.0 } else { num / denom };

    // Two-tailed p-value via Student's t CDF
    let p = 2.0 * student_t_sf(t.abs(), df);

    let d = cohens_d(a, b);

    let significant = p < 0.05;

    let interpretation = if p < 0.001 {
        format!("极显著 (p < 0.001)，三易控制效果非常明确")
    } else if p < 0.01 {
        format!("高度显著 (p < 0.01)，三易控制效果明确")
    } else if p < 0.05 {
        format!("显著 (p < 0.05)，三易控制有统计学意义")
    } else {
        format!("不显著 (p = {:.4})，需要更多数据验证", p)
    };

    TTestResult { t_statistic: t, p_value: p, cohens_d: d, significant, interpretation }
}

/// Survival function of Student's t-distribution: P(T > t) for t >= 0.
///
/// Uses the relationship between the t CDF and the regularized incomplete beta function.
/// F(t) = 1 - 0.5 * I_{df/(df+t^2)}(df/2, 1/2)
fn student_t_sf(t: f64, df: f64) -> f64 {
    if t <= 0.0 {
        return 0.5;
    }
    let x = df / (df + t * t);
    0.5 * regularized_beta(df / 2.0, 0.5, x)
}

/// Regularized incomplete beta function I_x(a, b) using continued fraction representation.
///
/// This is the Lentz algorithm adapted for the incomplete beta function.
fn regularized_beta(a: f64, b: f64, x: f64) -> f64 {
    if x < 0.0 || x > 1.0 {
        return if x > 1.0 { 1.0 } else { 0.0 };
    }
    if x == 0.0 || x == 1.0 {
        return x;
    }

    // Use the continued fraction representation for I_x(a, b)
    let ln_beta = ln_beta_func(a, b);
    let front = (x.ln() * a + (1.0 - x).ln() * b - ln_beta).exp() / a;

    // Lentz continued fraction
    let mut f = 1.0;
    let mut c = 1.0;
    let mut d = 0.0;
    let max_iter = 200;

    for m in 1..=max_iter {
        let mf = m as f64;

        // Even step
        let numer = -(a + mf) * (a + b + mf) * x / ((a + 2.0 * mf) * (a + 2.0 * mf - 1.0));
        d = 1.0 + numer * d;
        if d.abs() < 1e-30 { d = 1e-30; }
        c = 1.0 + numer / c;
        if c.abs() < 1e-30 { c = 1e-30; }
        d = 1.0 / d;
        f *= c * d;

        // Odd step
        let numer = mf * (b - mf) * x / ((a + 2.0 * mf - 1.0) * (a + 2.0 * mf));
        d = 1.0 + numer * d;
        if d.abs() < 1e-30 { d = 1e-30; }
        c = 1.0 + numer / c;
        if c.abs() < 1e-30 { c = 1e-30; }
        d = 1.0 / d;
        let del = c * d;
        f *= del;

        if (del - 1.0).abs() < 1e-12 {
            break;
        }
    }

    let result = front * (f - 1.0);
    result.clamp(0.0, 1.0)
}

/// Natural log of the Beta function: ln(B(a, b)) = ln(G(a) * G(b) / G(a+b))
fn ln_beta_func(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Stirling's approximation for ln(Gamma(x))
fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    // Stirling series
    let z = x;
    let ln_sqrt_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
    (z - 0.5) * z.ln() - z + ln_sqrt_2pi
        + 1.0 / (12.0 * z)
        - 1.0 / (360.0 * z * z * z)
        + 1.0 / (1260.0 * z * z * z * z * z)
}

// ── Benchmark Summary Types ──────────────────────────────────

use crate::benchmark::BenchmarkTrial;

/// Statistical summary of all benchmark trials.
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkSummary {
    pub deviation_controlled: MetricStats,
    pub deviation_constrained: MetricStats,
    pub deviation_test: TTestResult,

    pub compliance_controlled: MetricStats,
    pub compliance_constrained: MetricStats,
    pub compliance_test: TTestResult,

    pub completion_controlled: MetricStats,
    pub completion_constrained: MetricStats,
    pub completion_test: TTestResult,

    pub trust_controlled: MetricStats,
    pub trust_constrained: MetricStats,
    pub trust_test: TTestResult,

    // ── AI 语义质量评估 (optional) ──
    pub quality_overall_controlled: Option<MetricStats>,
    pub quality_overall_constrained: Option<MetricStats>,
    pub quality_overall_test: Option<TTestResult>,
    pub quality_completion_controlled: Option<MetricStats>,
    pub quality_completion_constrained: Option<MetricStats>,
    pub quality_completion_test: Option<TTestResult>,
    pub quality_coherence_controlled: Option<MetricStats>,
    pub quality_coherence_constrained: Option<MetricStats>,
    pub quality_coherence_test: Option<TTestResult>,
    pub quality_relevance_controlled: Option<MetricStats>,
    pub quality_relevance_constrained: Option<MetricStats>,
    pub quality_relevance_test: Option<TTestResult>,
    pub quality_depth_controlled: Option<MetricStats>,
    pub quality_depth_constrained: Option<MetricStats>,
    pub quality_depth_test: Option<TTestResult>,
    pub quality_structure_controlled: Option<MetricStats>,
    pub quality_structure_constrained: Option<MetricStats>,
    pub quality_structure_test: Option<TTestResult>,
}

/// Compute the full benchmark summary from all trials.
/// Optionally includes quality evaluation statistics if `quality_report` is provided.
pub fn compute_benchmark_summary(
    trials: &[BenchmarkTrial],
    quality_report: Option<&crate::QualityEvaluationReport>,
) -> BenchmarkSummary {
    let _n = trials.len();

    // Extract metric vectors
    let dev_c: Vec<f64> = trials.iter().map(|t| t.controlled_metrics.avg_deviation as f64).collect();
    let dev_x: Vec<f64> = trials.iter().map(|t| t.constrained_metrics.avg_deviation as f64).collect();

    let comp_c: Vec<f64> = trials.iter().map(|t| t.controlled_metrics.structure_compliance_rate as f64).collect();
    let comp_x: Vec<f64> = trials.iter().map(|t| t.constrained_metrics.structure_compliance_rate as f64).collect();

    let done_c: Vec<f64> = trials.iter().map(|t| t.controlled_metrics.completion_rate as f64).collect();
    let done_x: Vec<f64> = trials.iter().map(|t| t.constrained_metrics.completion_rate as f64).collect();

    let trust_c: Vec<f64> = trials.iter().map(|t| t.controlled_metrics.trust_score as f64).collect();
    let trust_x: Vec<f64> = trials.iter().map(|t| t.constrained_metrics.trust_score as f64).collect();

    BenchmarkSummary {
        deviation_controlled: compute_metric_stats(&dev_c),
        deviation_constrained: compute_metric_stats(&dev_x),
        deviation_test: welch_t_test(&dev_c, &dev_x),

        compliance_controlled: compute_metric_stats(&comp_c),
        compliance_constrained: compute_metric_stats(&comp_x),
        compliance_test: welch_t_test(&comp_c, &comp_x),

        completion_controlled: compute_metric_stats(&done_c),
        completion_constrained: compute_metric_stats(&done_x),
        completion_test: welch_t_test(&done_c, &done_x),

        trust_controlled: compute_metric_stats(&trust_c),
        trust_constrained: compute_metric_stats(&trust_x),
        trust_test: welch_t_test(&trust_c, &trust_x),

        quality_overall_controlled: quality_report.map(quality_ctrl_avg),
        quality_overall_constrained: quality_report.map(quality_expr_avg),
        quality_overall_test: quality_report.map(quality_ttest),
        quality_completion_controlled: quality_report.map(|qr| dim_avg(qr, |e| &e.controlled_per_turn, |s| s.task_completion)),
        quality_completion_constrained: quality_report.map(|qr| dim_avg(qr, |e| &e.constrained_per_turn, |s| s.task_completion)),
        quality_completion_test: quality_report.map(|qr| dim_ttest(qr, |s| s.task_completion)),
        quality_coherence_controlled: quality_report.map(|qr| dim_avg(qr, |e| &e.controlled_per_turn, |s| s.logical_coherence)),
        quality_coherence_constrained: quality_report.map(|qr| dim_avg(qr, |e| &e.constrained_per_turn, |s| s.logical_coherence)),
        quality_coherence_test: quality_report.map(|qr| dim_ttest(qr, |s| s.logical_coherence)),
        quality_relevance_controlled: quality_report.map(|qr| dim_avg(qr, |e| &e.controlled_per_turn, |s| s.content_relevance)),
        quality_relevance_constrained: quality_report.map(|qr| dim_avg(qr, |e| &e.constrained_per_turn, |s| s.content_relevance)),
        quality_relevance_test: quality_report.map(|qr| dim_ttest(qr, |s| s.content_relevance)),
        quality_depth_controlled: quality_report.map(|qr| dim_avg(qr, |e| &e.controlled_per_turn, |s| s.analysis_depth)),
        quality_depth_constrained: quality_report.map(|qr| dim_avg(qr, |e| &e.constrained_per_turn, |s| s.analysis_depth)),
        quality_depth_test: quality_report.map(|qr| dim_ttest(qr, |s| s.analysis_depth)),
        quality_structure_controlled: quality_report.map(|qr| dim_avg(qr, |e| &e.controlled_per_turn, |s| s.structural_clarity)),
        quality_structure_constrained: quality_report.map(|qr| dim_avg(qr, |e| &e.constrained_per_turn, |s| s.structural_clarity)),
        quality_structure_test: quality_report.map(|qr| dim_ttest(qr, |s| s.structural_clarity)),
    }
}

/// Controlled-group overall quality average.
fn quality_ctrl_avg(report: &crate::QualityEvaluationReport) -> MetricStats {
    let vals: Vec<f64> = report.evaluations.iter().map(|e| e.controlled_overall as f64).collect();
    compute_metric_stats(&vals)
}

/// Constrained-group overall quality average.
fn quality_expr_avg(report: &crate::QualityEvaluationReport) -> MetricStats {
    let vals: Vec<f64> = report.evaluations.iter().map(|e| e.constrained_overall as f64).collect();
    compute_metric_stats(&vals)
}

/// Overall quality t-test.
fn quality_ttest(report: &crate::QualityEvaluationReport) -> TTestResult {
    let ctrl: Vec<f64> = report.evaluations.iter().map(|e| e.controlled_overall as f64).collect();
    let expr: Vec<f64> = report.evaluations.iter().map(|e| e.constrained_overall as f64).collect();
    welch_t_test(&ctrl, &expr)
}

/// Average a per-dimension score across all turns in all trials.
fn dim_avg<F, G>(report: &crate::QualityEvaluationReport, group: F, dim: G) -> MetricStats
where
    F: Fn(&crate::QualityEvaluation) -> &Vec<crate::QualityScores>,
    G: Fn(&crate::QualityScores) -> f32,
{
    let vals: Vec<f64> = report.evaluations.iter()
        .flat_map(|e| group(e).iter().map(|s| dim(s) as f64))
        .collect();
    compute_metric_stats(&vals)
}

/// Welch t-test on a per-dimension score between controlled and constrained.
fn dim_ttest<G>(report: &crate::QualityEvaluationReport, dim: G) -> TTestResult
where
    G: Fn(&crate::QualityScores) -> f32 + Copy,
{
    let ctrl: Vec<f64> = report.evaluations.iter()
        .flat_map(|e| e.controlled_per_turn.iter().map(|s| dim(s) as f64))
        .collect();
    let expr: Vec<f64> = report.evaluations.iter()
        .flat_map(|e| e.constrained_per_turn.iter().map(|s| dim(s) as f64))
        .collect();
    welch_t_test(&ctrl, &expr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_and_std() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = compute_metric_stats(&values);
        assert!((stats.mean - 3.0).abs() < 0.001);
        assert!((stats.std_dev - 1.5811).abs() < 0.01);
        assert_eq!(stats.n, 5);
    }

    #[test]
    fn test_median_odd() {
        let values = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let stats = compute_metric_stats(&values);
        assert!((stats.median - 3.0).abs() < 0.001);
        assert!((stats.min - 1.0).abs() < 0.001);
        assert!((stats.max - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_median_even() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let stats = compute_metric_stats(&values);
        assert!((stats.median - 2.5).abs() < 0.001);
    }

    #[test]
    fn test_empty_stats() {
        let stats = compute_metric_stats(&[]);
        assert_eq!(stats.n, 0);
        assert_eq!(stats.mean, 0.0);
    }

    #[test]
    fn test_cohens_d_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let d = cohens_d(&a, &b);
        assert!(d.abs() < 0.001);
    }

    #[test]
    fn test_cohens_d_different() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![6.0, 7.0, 8.0, 9.0, 10.0];
        let d = cohens_d(&a, &b);
        // Mean diff = 5, pooled SD ~ 1.58 each, d ~ 3.16
        assert!(d > 2.5 && d < 4.0, "d = {}", d);
    }

    #[test]
    fn test_welch_t_test_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = welch_t_test(&a, &b);
        assert!(!result.significant);
        assert!(result.p_value > 0.05);
    }

    #[test]
    fn test_welch_t_test_very_different() {
        // Generate two clearly different distributions
        let a: Vec<f64> = (0..30).map(|_| 0.3 + rand_simple() * 0.1).collect();
        let b: Vec<f64> = (0..30).map(|_| 0.7 + rand_simple() * 0.1).collect();
        let result = welch_t_test(&a, &b);
        assert!(result.significant, "p_value = {}", result.p_value);
        assert!(result.p_value < 0.001, "p_value = {}", result.p_value);
    }

    #[test]
    fn test_welch_small_sample() {
        let result = welch_t_test(&[1.0], &[2.0]);
        assert!(!result.significant);
        assert_eq!(result.p_value, 1.0);
    }

    /// Simple deterministic pseudo-random in [0, 1) for test data
    fn rand_simple() -> f64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(12345);
        let val = COUNTER.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
            Some(v.wrapping_mul(6364136223846793005).wrapping_add(1))
        }).unwrap_or(12345);
        // Extract high bits for hash-like randomness
        let x = val.wrapping_mul(0x9E3779B97F4A7C15);
        ((x >> 33) & 0xFFFFFFFF) as f64 / 0xFFFFFFFFu64 as f64
    }
}
