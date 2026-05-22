/// 项目工作上下文 — 跨轮代谢的项目记忆
///
/// 从多轮对话中蒸馏有效信息，排出噪音。
/// 只保留可执行的"决策/洞察"级别的信息，不保存低层状态转换。
///
/// **代谢机制**：
///   - **强化保留**：高奖励+结构阶段（育/长）的决策 → importance 高 → 优先保留
///   - **排出噪音**：探索/调试阶段（生/动/杀）→ importance 低 → 超出容量后优先丢弃
///   - **始终锚定**：项目目标永远保留，不受容量限制
///
/// **与 CangSea 的关系**：
///   - CangSea 是内部 Hebbian 记忆（状态→状态 转换权重）——仅供引擎内部使用
///   - ProjectContext 是外部语义记忆（人类可读决策列表）——注入到系统提示词
///   - CangSea 不注入给 LLM，只在 ShanVM 生成 forces 时参考
#[derive(Debug, Clone)]
pub struct ProjectContext {
    /// 项目目标（始终保留，不受容量限制，注入时置顶）
    pub goal: String,
    /// 累积决策，按轮次排序
    pub decisions: Vec<DecisionEntry>,
    /// 最大保留决策数（超出后按 importance 淘汰最低的）
    pub max_retain: usize,
}

/// 一条蒸馏后的决策/洞察。
#[derive(Debug, Clone)]
pub struct DecisionEntry {
    /// 重要性分数 [0, 1]，由 reward × operator_weight 计算
    pub importance: f32,
    /// 产生的轮次
    pub round: u32,
    /// 决策/洞察内容（人类可读文本）
    pub content: String,
}

impl ProjectContext {
    /// 从项目目标创建空上下文。
    pub fn new(goal: String, max_retain: usize) -> Self {
        ProjectContext {
            goal,
            decisions: Vec::with_capacity(max_retain),
            max_retain,
        }
    }

    /// 添加一条决策。
    ///
    /// `operator` — 当前算子名 ("生"/"动"/"长"/"育"/"杀")
    /// `reward`   — 本轮的奖励信号 [-1, 1]
    /// `content`  — 蒸馏后的决策内容
    ///
    /// 重要性计算公式：
    ///   importance = |reward| × operator_weight
    ///   其中 operator_weight: 育=1.0, 长=0.8, 生/动=0.4, 杀=0.15
    ///
    /// 负向 reward 的决策直接丢弃（不保留）。
    pub fn add_decision(
        &mut self,
        round: u32,
        operator: &str,
        reward: f32,
        content: String,
    ) {
        // 只保留正向经验
        if reward <= 0.0 {
            return;
        }

        let op_weight = match operator {
            "育" => 1.0,   // 结构化输出最具价值
            "长" => 0.8,   // 聚焦分析有价值
            "动" => 0.5,   // 发散有一定价值
            "生" => 0.4,   // 探索价值较低
            _ => 0.3,
        };

        let importance = reward.abs().min(1.0) * op_weight;

        self.decisions.push(DecisionEntry {
            importance,
            round,
            content,
        });

        // 按轮次排序
        self.decisions.sort_by_key(|d| d.round);

        // 超出容量 → 淘汰 importance 最低的
        self.prune();
    }

    /// 淘汰低重要性决策直至回到容量限制内。
    fn prune(&mut self) {
        while self.decisions.len() > self.max_retain {
            if let Some(pos) = self
                .decisions
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.importance
                        .partial_cmp(&b.importance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
            {
                self.decisions.remove(pos);
            } else {
                break;
            }
        }
    }

    /// 构建注入用的项目上下文段落。
    ///
    /// 包含项目目标 + 已累积的关键决策。
    /// 决策按轮次排列，带有重要性指示。
    pub fn section(&self) -> String {
        let mut s = String::new();

        // ── 项目目标（始终置顶）──
        s.push_str("【项目目标 · 始终锚定】\n");
        s.push_str(&format!("  {}\n", self.goal));

        if !self.decisions.is_empty() {
            s.push_str("\n【已积累关键决策 · 跨轮蒸馏】\n");

            for entry in &self.decisions {
                // importance 越高 → 标记越强
                let mark = if entry.importance >= 0.7 {
                    "★"
                } else if entry.importance >= 0.4 {
                    "✦"
                } else {
                    "·"
                };
                s.push_str(&format!(
                    "  [R{:02}] {} {}\n",
                    entry.round, mark, entry.content
                ));
            }
        } else {
            s.push_str("\n【项目上下文】尚未积累有效决策，持续收集信息。\n");
        }

        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_decision_accumulation() {
        let mut ctx = ProjectContext::new("构建全栈Web应用".into(), 10);

        ctx.add_decision(1, "育", 0.8, "使用actix-web框架".into());
        ctx.add_decision(2, "长", 0.6, "模块化分层架构".into());
        ctx.add_decision(3, "生", 0.4, "初步探索API设计".into());

        assert_eq!(ctx.decisions.len(), 3);

        // 育(0.8) 应该比 生(0.4) 重要
        let d1 = &ctx.decisions[0];
        let d3 = &ctx.decisions[2];
        assert!(d1.importance > d3.importance,
            "育算子决策应比生算子更受重视: {:.2} vs {:.2}",
            d1.importance, d3.importance);
    }

    #[test]
    fn test_negative_reward_dropped() {
        let mut ctx = ProjectContext::new("测试".into(), 10);
        ctx.add_decision(1, "杀", -0.6, "失败的调试".into());
        assert_eq!(ctx.decisions.len(), 0);
    }

    #[test]
    fn test_capacity_pruning() {
        let mut ctx = ProjectContext::new("测试".into(), 3);

        ctx.add_decision(1, "育", 0.9, "重要决策A".into());   // imp=0.90
        ctx.add_decision(2, "育", 0.8, "重要决策B".into());   // imp=0.80
        ctx.add_decision(3, "生", 0.3, "弱决策C".into());     // imp=0.12
        ctx.add_decision(4, "育", 0.95, "最重要D".into());    // imp=0.95

        assert_eq!(ctx.decisions.len(), 3);
        // "弱决策C"(0.12) 应被淘汰，"最重要D"(0.95) 应保留
        assert!(ctx.decisions.iter().any(|d| d.round == 4),
            "最高重要性决策应保留");
        assert!(!ctx.decisions.iter().any(|d| d.round == 3),
            "低重要性决策应被淘汰");
    }

    #[test]
    fn test_section_output_format() {
        let mut ctx = ProjectContext::new("构建Web应用".into(), 10);
        ctx.add_decision(1, "育", 0.9, "使用Rust后端".into());
        ctx.add_decision(2, "生", 0.3, "初步设计前端".into());

        let section = ctx.section();
        assert!(section.contains("项目目标"));
        assert!(section.contains("构建Web应用"));
        assert!(section.contains("[R01]"));
        assert!(section.contains("[R02]"));
        assert!(section.contains("Rust后端"));
        assert!(section.contains("★"));  // 高重要性有星标
    }
}
