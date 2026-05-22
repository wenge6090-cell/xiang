/// 阴仪阶段协议验证器 (Yin Protocol Checker)
///
/// 阴仪**不是**语义理解器，而是**规则匹配器**。
/// 它检查阳仪的输出是否符合当前算子的形式规范（阶段约束），
/// 而非计算语义偏离度。
///
/// 设计原则（来自归藏设计笔记 §2.2）:
///   - 不编码语义向量（实时路径中）
///   - 不计算余弦相似度
///   - 不判断"是否跑题"
///   - 只检查形式是否符合算子阶段规范

use regex::Regex;

// ─── 算子规则 ──────────────────────────────────────────

/// 单个算子的阶段协议规则。
#[derive(Debug, Clone)]
pub struct OperatorRule {
    /// 必须包含的正则模式（至少命中一个即可）
    pub must_contain_patterns: Vec<String>,
    /// 禁止出现的正则模式
    pub must_not_contain: Vec<String>,
    /// 最小输出长度 (char count)
    pub min_length_chars: Option<usize>,
    /// 最大输出长度 (char count)
    pub max_length_chars: Option<usize>,
    /// 结构要求
    pub structure: Option<OutputStructure>,
}

/// 输出结构约束。
#[derive(Debug, Clone, PartialEq)]
pub enum OutputStructure {
    /// 宽松段落式（允许自然段落）
    Loose,
    /// 单一方向声明（只能有一条主线）
    SinglePath,
    /// 编号列表（必须包含序号）
    NumberedList,
}

/// 阶段检查结果。
#[derive(Debug, Clone)]
pub struct RuleResult {
    /// 是否通过
    pub is_valid: bool,
    /// 违规项列表
    pub violations: Vec<String>,
}

// ─── 阴仪检查器 ────────────────────────────────────────

/// 阴仪：阶段协议验证器。
///
/// 对四个生成型算子（生/动/长/育）输出进行形式规范检查。
/// 检查依据是算子阶段规则，而非语义理解。
pub struct YinProtocolChecker {
    rules: std::collections::HashMap<String, OperatorRule>,
}

impl YinProtocolChecker {
    /// 使用默认规则库创建阴仪检查器。
    pub fn new() -> Self {
        YinProtocolChecker {
            rules: Self::default_rules(),
        }
    }

    /// 使用自定义规则创建。
    pub fn with_rules(rules: std::collections::HashMap<String, OperatorRule>) -> Self {
        YinProtocolChecker { rules }
    }

    /// 检查算子输出是否符合阶段协议。
    ///
    /// - `operator`: 算子名 ("生" | "动" | "长" | "育")
    /// - `text`: 算子输出的文本（已去除标记）
    ///
    /// 返回 `RuleResult`，其中 `is_valid` 表示是否通过所有规则。
    pub fn check(&self, operator: &str, text: &str) -> RuleResult {
        let mut violations = Vec::new();

        match self.rules.get(operator) {
            None => {
                // 未知算子，不做检查
                return RuleResult {
                    is_valid: true,
                    violations,
                };
            }
            Some(rule) => {
                // 1. 检查必须包含的模式
                let has_required = if rule.must_contain_patterns.is_empty() {
                    true
                } else {
                    rule.must_contain_patterns.iter().any(|pattern| {
                        match Regex::new(pattern) {
                            Ok(re) => re.is_match(text),
                            Err(_) => false,
                        }
                    })
                };

                if !has_required {
                    violations.push(format!(
                        "缺少必要模式: 必须至少包含以下之一: {}",
                        rule.must_contain_patterns.join(", ")
                    ));
                }

                // 2. 检查禁止的模式
                for pattern in &rule.must_not_contain {
                    if let Ok(re) = Regex::new(pattern) {
                        if re.is_match(text) {
                            violations.push(format!("包含禁止模式: {pattern}"));
                        }
                    }
                }

                // 3. 检查长度
                let text_len = text.chars().count();
                if let Some(min) = rule.min_length_chars {
                    if text_len < min {
                        violations.push(format!(
                            "输出过短: {text_len} chars, 需要至少 {min} chars"
                        ));
                    }
                }
                if let Some(max) = rule.max_length_chars {
                    if text_len > max {
                        violations.push(format!(
                            "输出过长: {text_len} chars, 需要最多 {max} chars"
                        ));
                    }
                }

                // 4. 检查结构要求
                if let Some(ref structure) = rule.structure {
                    match structure {
                        OutputStructure::SinglePath => {
                            // 检测"此外/另一方面"等发散词（表示多路径）
                            let diverge_patterns = [
                                r"此外", r"另一方面", r"还可以", r"另外",
                                r"顺便", r"同时.*也可以",
                            ];
                            for p in &diverge_patterns {
                                if let Ok(re) = Regex::new(p) {
                                    if re.is_match(text) {
                                        violations.push(format!(
                                            "结构违规(SinglePath): 检测到发散词 '{p}'"
                                        ));
                                        break;
                                    }
                                }
                            }
                        }
                        OutputStructure::NumberedList => {
                            // 检测是否包含编号结构
                            let numbered = Regex::new(r"(\d+[\.\)、]|第.+步|步骤\d+|子任务\d+)").ok();
                            let has_numbering = numbered
                                .as_ref()
                                .map(|re| re.is_match(text))
                                .unwrap_or(false);
                            if !has_numbering {
                                violations.push(
                                    "结构违规(NumberedList): 缺少编号/步骤结构".to_string()
                                );
                            }
                        }
                        OutputStructure::Loose => {
                            // 不检查，允许任何段落格式
                        }
                    }
                }
            }
        }

        RuleResult {
            is_valid: violations.is_empty(),
            violations,
        }
    }

    /// 获取算子规则。
    pub fn get_rule(&self, operator: &str) -> Option<&OperatorRule> {
        self.rules.get(operator)
    }

    /// 注册或替换算子规则。
    pub fn set_rule(&mut self, operator: String, rule: OperatorRule) {
        self.rules.insert(operator, rule);
    }

    // ─── 默认规则库 ──────────────────────────────────

    fn default_rules() -> std::collections::HashMap<String, OperatorRule> {
        let mut rules = std::collections::HashMap::new();

        // 生算子：起念探索
        // 输出以试探性语言为主，必须包含开放性疑问或假设
        rules.insert(
            "生".to_string(),
            OperatorRule {
                must_contain_patterns: vec![
                    r"(也许|可能|或许|考虑|可以|一种思路|从.*?入手|值得探讨)".to_string(),
                    r"(\?|吗$|呢$|如何|怎样|是否)".to_string(),
                ],
                must_not_contain: vec![
                    r"(第一步|第二步|子任务\d|1\.|2\.)".to_string(),
                    r"(因此|所以|最终|综上所述|我们应该)".to_string(),
                ],
                min_length_chars: Some(10),
                max_length_chars: Some(500),
                structure: None,
            },
        );

        // 动算子：发散联想
        // 输出以列举、联想、多角度扩展为主
        rules.insert(
            "动".to_string(),
            OperatorRule {
                must_contain_patterns: vec![
                    r"(此外|另一方面|还可以|引申|相关|扩展|另外|同时)".to_string(),
                ],
                must_not_contain: vec![
                    r"(最终|结论|我们应该|最佳方案|确定)".to_string(),
                ],
                min_length_chars: None,
                max_length_chars: None,
                structure: Some(OutputStructure::Loose),
            },
        );

        // 长算子：明确方向
        // 输出收敛，选择一个明确的分析方向
        rules.insert(
            "长".to_string(),
            OperatorRule {
                must_contain_patterns: vec![
                    r"(重点|聚焦|选择|方向|深入|沿着|主要)".to_string(),
                ],
                must_not_contain: vec![
                    r"(也许|可能|或许|另一个思路|另一方面|还可以)".to_string(),
                ],
                min_length_chars: None,
                max_length_chars: None,
                structure: Some(OutputStructure::SinglePath),
            },
        );

        // 育算子：方案分解
        // 输出结构化的子任务列表，确定性描述
        rules.insert(
            "育".to_string(),
            OperatorRule {
                must_contain_patterns: vec![
                    r"(第一步|第二步|子任务|1\.|2\.|步骤|首先.*然后|包括)".to_string(),
                ],
                must_not_contain: vec![
                    r"(也许|可能|或许|可以考虑)".to_string(),
                ],
                min_length_chars: None,
                max_length_chars: None,
                structure: Some(OutputStructure::NumberedList),
            },
        );

        rules
    }
}

impl Default for YinProtocolChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ─── 控制型算子（不产生文本，不需要检查） ─────────────
// 归、杀、止、藏 — 由 CangVM 内部执行，不通过阴仪

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sheng_valid_exploratory() {
        let checker = YinProtocolChecker::new();
        let text = "也许我们可以从系统架构入手分析这个问题？需要进一步考察数据流的走向如何影响整体性能。";
        let result = checker.check("生", text);
        assert!(result.is_valid, "Expected valid, got: {:?}", result.violations);
    }

    #[test]
    fn test_sheng_reject_structured() {
        let checker = YinProtocolChecker::new();
        let text = "第一步，分析需求。第二步，设计方案。因此我们应该采用A方案。";
        let result = checker.check("生", text);
        assert!(!result.is_valid, "Should reject structured output in 生");
    }

    #[test]
    fn test_sheng_too_short() {
        let checker = YinProtocolChecker::new();
        let text = "好的。";
        let result = checker.check("生", text);
        assert!(!result.is_valid, "Should reject too short");
    }

    #[test]
    fn test_dong_valid_divergent() {
        let checker = YinProtocolChecker::new();
        let text = "此外还可以考虑性能维度。另一方面，安全性也是需要关注的扩展方向。";
        let result = checker.check("动", text);
        assert!(result.is_valid, "Expected valid, got: {:?}", result.violations);
    }

    #[test]
    fn test_dong_reject_conclusive() {
        let checker = YinProtocolChecker::new();
        let text = "最终我们应该选择方案A。这是最佳方案。";
        let result = checker.check("动", text);
        assert!(!result.is_valid, "Should reject conclusive tone in 动");
    }

    #[test]
    fn test_zhang_valid_focused() {
        let checker = YinProtocolChecker::new();
        let text = "我们聚焦在用户认证这条路径上，深入分析OAuth2.0的实现细节。";
        let result = checker.check("长", text);
        assert!(result.is_valid, "Expected valid, got: {:?}", result.violations);
    }

    #[test]
    fn test_zhang_reject_divergent() {
        let checker = YinProtocolChecker::new();
        let text = "也许我们可以这样，也许也可以那样，另一个思路是...";
        let result = checker.check("长", text);
        assert!(!result.is_valid, "Should reject divergent language in 长");
    }

    #[test]
    fn test_yu_valid_structured() {
        let checker = YinProtocolChecker::new();
        let text = "第一步，部署数据库。第二步，配置API网关。步骤3，集成前端。包括单元测试。";
        let result = checker.check("育", text);
        assert!(result.is_valid, "Expected valid, got: {:?}", result.violations);
    }

    #[test]
    fn test_yu_reject_vague() {
        let checker = YinProtocolChecker::new();
        let text = "也许可以考虑部署数据库，或许也可以先配置网关。";
        let result = checker.check("育", text);
        assert!(!result.is_valid, "Should reject vague language in 育");
    }

    #[test]
    fn test_unknown_operator_passes() {
        let checker = YinProtocolChecker::new();
        let text = "任意文本";
        let result = checker.check("归", text);
        assert!(result.is_valid, "Unknown operator should pass");
    }

    #[test]
    fn test_custom_rules() {
        let mut rules = std::collections::HashMap::new();
        rules.insert(
            "生".to_string(),
            OperatorRule {
                must_contain_patterns: vec![],
                must_not_contain: vec![r"危险".to_string()],
                min_length_chars: None,
                max_length_chars: Some(50),
                structure: None,
            },
        );
        let checker = YinProtocolChecker::with_rules(rules);
        assert!(checker.check("生", "安全的内容").is_valid);
        assert!(!checker.check("生", "这是危险的内容").is_valid);
    }
}
