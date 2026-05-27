/// 三易约束体系全文与动态提示词构建器
///
/// 路线A v4.0 实验提示词策略：
/// - 动态提示词分为"策略指令"（告诉LLM该做什么）和"约束状态"（描述引擎状态）
/// - 策略指令包含：算子行动指导 + 周易卦象行为前缀 + 连山FangWei策略指导
/// - 真正让约束系统深入到推理过程，而非仅仅标注状态
use crate::FangWei;

/// 三易约束体系全文
///
/// 在未微调模型上等价于"微调知识注入"。
/// 两组均注入此文本，使其相互抵消，不作为实验变量。
pub const SAN_YI_CONSTRAINT_FULL_TEXT: &str = r#"【三易约束体系说明】

你运行在一个名为「三易」的认知约束框架中。三易由三台状态机组成，在推理时对你的思维过程施加结构化引导。

一、归藏引擎 —— 意识循环（八气算子）
你的输出将按四个生成算子循环推进：
- 生（起念探索）：提出开放性问题或假设，使用试探词（也许/可能/如何/是否）
- 动（发散联想）：从多个角度扩展思考，使用发散词（此外/另一方面/还可以）
- 长（收敛聚焦）：选择一条分析路径深入，使用聚焦词（重点/沿着/深入）
- 育（方案分解）：输出结构化的子任务列表，使用编号结构（第一步/1.）

每个算子阶段都应体现对应的思维特征。

二、周易引擎 —— 认知姿态（八卦体系）
系统会根据上下文动态切换你的认知姿态：
- 乾/创造（温度1.2）：创造性发散
- 兑/表达（温度0.9）：输出交付
- 离/明照（温度0.5）：审视反思
- 震/启动（温度1.0）：快速激发
- 巽/渗透（温度0.7）：细致分析
- 坎/破局（温度1.1）：突破障碍
- 艮/止定（温度0.3）：保守收窄
- 坤/承载（温度0.6）：稳定执行

温度越低，输出越确定保守；温度越高，输出越多样创造。

三、连山引擎 —— 障碍导航（偏离度 > 0.5 时触发）
当你的思维偏离核心目标时，系统会触发障碍导航，给出六种方向之一：
继续 → 强力突破 → 迂回绕行 → 拆解分解 → 升级上报 → 放弃

四、阴仪协议
你的输出将受到正则规则的形式检查：
- 生阶段：禁止出现「第一步」「因此」等结构词
- 动阶段：禁止出现「最终」「应该」等结论词
- 长阶段：禁止出现「也许」「另一个思路」等发散词
- 育阶段：禁止出现「也许」「可能」等模糊词

五、偏离度
系统通过汉明距离计算你的思维偏离度（0.0-1.0）：
- 偏离度 < 0.3：思维在目标轨道上
- 偏离度 0.3-0.7：思维有偏移趋势
- 偏离度 > 0.7：思维严重偏离，触发强制纠正

请根据以上约束体系理解并配合系统的引导。
"#;

/// 将 FangWei 策略映射为可执行的行动指令。
/// 不是简单地标注"当前策略=XX"，而是告诉LLM在策略下应该怎么做。
pub fn fangwei_strategic_guidance(fw: FangWei) -> &'static str {
    match fw {
        FangWei::Continue => "当前方向可行。保持现有分析策略，持续推进。",
        FangWei::PushThrough => "遇到理解障碍时不要绕行。强化论证力度，使用具体证据和逻辑推导推进。",
        FangWei::NavigateAround => "当前路径受阻。尝试换一个角度或方法解决同一个问题，不要在原路径上继续。",
        FangWei::WaitGather => "信息不足。先收集更多上下文或补充细节，再继续分析。",
        FangWei::Decompose => "问题过于复杂。请将当前问题拆解为可独立处理的子问题，逐一分析。",
        FangWei::Escalate => "多次尝试未果。标记此问题为需要外部介入，输出当前所有已尝试的路径和失败原因。",
        FangWei::Abort => "此路径不可行。终止当前方向，回到最基础的定义重新开始。",
    }
}

/// 将算子阶段映射为可执行的行动指导。
/// 告诉LLM在当前算子阶段应该采用怎样的思维表达方式。
pub fn operator_actionable_guidance(op: &str) -> &'static str {
    match op {
        "生" => "当前处于探索阶段。请提出开放性问题或假设，使用试探性表达（也许/可能/如何）。不要急于下结论。",
        "动" => "当前处于发散阶段。请从多个角度扩展思考，使用发散词（此外/另一方面）。避免过早聚焦到单一方向。",
        "长" => "当前处于收敛阶段。请选择一条最有希望的分析路径深入，使用聚焦结构（重点/沿着/深入）。避免发散到新方向。",
        "育" => "当前处于结构化阶段。请输出具体的步骤、子任务或方案框架，使用编号结构（第一步/1.）。不要停留在抽象讨论。",
        _ => "保持当前分析节奏，按照问题具体要求推进。",
    }
}

/// 三易动态提示词构建器
///
/// 将三引擎运行时状态编译为主行动指令 + 约束状态两部分自然语言。
/// 仅注入约束实验组，对照组不注入。
pub struct DynamicPromptState {
    /// 当前算子名称（生/动/长/育）
    pub operator: String,
    /// 当前算子对应的阶段描述
    pub phase_description: String,
    /// 当前周易姿态名称
    pub posture_name: String,
    /// 当前周易姿态描述
    pub posture_description: String,
    /// 当前温度值
    pub temperature: f32,
    /// 当前偏离度
    pub deviation: f32,
    /// 偏离度描述
    pub deviation_description: String,
    /// 连山决策方向描述（如 "已激活 | 再甲 → 强力突破"）
    pub shan_decision: Option<String>,
    /// 连山策略的可执行指导指令
    pub fangwei_guidance: Option<String>,
    /// 算子阶段的行动指导
    pub operator_guidance: String,
    /// 周易卦象的行为前缀提示
    pub zhou_prompt_prefix: String,
}

impl DynamicPromptState {
    /// 从引擎状态创建动态提示词状态
    pub fn new(
        operator: &str,
        posture_name: &str,
        posture_description: &str,
        temperature: f32,
        deviation: f32,
        shan_decision: Option<&str>,
        fangwei_guidance: Option<String>,
        zhou_prompt_prefix: &str,
    ) -> Self {
        let phase_desc = match operator {
            "生" => "起念探索",
            "动" => "发散联想",
            "长" => "收敛聚焦",
            "育" => "方案分解",
            _ => "未知阶段",
        };

        let dev_desc = if deviation < 0.3 {
            "聚焦良好"
        } else if deviation < 0.7 {
            "保持聚焦"
        } else {
            "需要回归焦点"
        };

        let operator_guidance = operator_actionable_guidance(operator).to_string();

        Self {
            operator: operator.to_string(),
            phase_description: phase_desc.to_string(),
            posture_name: posture_name.to_string(),
            posture_description: posture_description.to_string(),
            temperature,
            deviation,
            deviation_description: dev_desc.to_string(),
            shan_decision: shan_decision.map(|s| s.to_string()),
            fangwei_guidance,
            operator_guidance,
            zhou_prompt_prefix: zhou_prompt_prefix.to_string(),
        }
    }
}

/// 将动态提示词状态编译为自然语言文本。
///
/// 输出格式（v4.0）：
///   【策略指令】 → 算子指导 + 卦象前缀 + 连山指导（直接告诉LLM该做什么）
///   【当前约束状态】 → 归藏/周易/偏离度/连山状态（提供上下文）
///
/// 设计原则：指令在前（actionable），状态在后（contextual）。
pub fn build_sanyi_dynamic_prompt(state: &DynamicPromptState) -> String {
    let mut parts = Vec::new();

    // ── 策略指令段：告诉LLM该做什么 ──
    let mut guidance_parts = vec![state.operator_guidance.clone()];

    // 周易卦象行为前缀
    if !state.zhou_prompt_prefix.is_empty() {
        guidance_parts.push(state.zhou_prompt_prefix.clone());
    }

    // 连山策略指导（如有）
    if let Some(ref fw_guidance) = state.fangwei_guidance {
        guidance_parts.push(fw_guidance.clone());
    }

    parts.push(format!("【策略指令】\n{}", guidance_parts.join("\n\n")));

    // ── 约束状态段：引擎当前状态描述 ──
    parts.push(format!(
        "【当前约束状态】\n\
         归藏算子：{}（{}）\n\
         周易姿态：{}（{}）\n\
         采样温度：{:.1}\n\
         偏离度：{:.2}（{}）",
        state.operator,
        state.phase_description,
        state.posture_name,
        state.posture_description,
        state.temperature,
        state.deviation,
        state.deviation_description,
    ));

    // 连山导航状态（如有）
    if let Some(ref decision) = state.shan_decision {
        parts.push(format!("连山导航：{}", decision));
    }

    parts.join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_text_not_empty() {
        assert!(!SAN_YI_CONSTRAINT_FULL_TEXT.is_empty());
        assert!(SAN_YI_CONSTRAINT_FULL_TEXT.contains("三易"));
        assert!(SAN_YI_CONSTRAINT_FULL_TEXT.contains("归藏"));
        assert!(SAN_YI_CONSTRAINT_FULL_TEXT.contains("连山"));
        assert!(SAN_YI_CONSTRAINT_FULL_TEXT.contains("周易"));
        assert!(SAN_YI_CONSTRAINT_FULL_TEXT.contains("阴仪"));
    }

    #[test]
    fn test_fangwei_guidance_all_variants() {
        // Verify all FangWei variants produce non-empty guidance
        let all = [
            FangWei::Continue, FangWei::PushThrough, FangWei::NavigateAround,
            FangWei::WaitGather, FangWei::Decompose, FangWei::Escalate, FangWei::Abort,
        ];
        for fw in &all {
            let g = fangwei_strategic_guidance(*fw);
            assert!(!g.is_empty(), "FangWei {:?} should have guidance", fw);
        }
    }

    #[test]
    fn test_operator_guidance_all_phases() {
        for op in &["生", "动", "长", "育"] {
            let g = operator_actionable_guidance(op);
            assert!(!g.is_empty(), "Operator {} should have guidance", op);
        }
    }

    #[test]
    fn test_dynamic_prompt_sheng() {
        let state = DynamicPromptState::new(
            "生", "震", "启动", 1.0, 0.15, None,
            None, "",
        );
        let prompt = build_sanyi_dynamic_prompt(&state);
        assert!(prompt.contains("【策略指令】"));
        assert!(prompt.contains("【当前约束状态】"));
        assert!(prompt.contains("生"));
        assert!(prompt.contains("起念探索"));
        assert!(prompt.contains("震"));
        assert!(prompt.contains("启动"));
        assert!(prompt.contains("聚焦良好"));
        // Should contain operator guidance
        assert!(prompt.contains("探索阶段"));
    }

    #[test]
    fn test_dynamic_prompt_yu_with_shan() {
        let state = DynamicPromptState::new(
            "育", "离", "明照", 0.5, 0.75,
            Some("已激活 | 再甲 → 强力突破"),
            Some(fangwei_strategic_guidance(FangWei::PushThrough).to_string()),
            "以明辨洞察的方式分析。逐层拆解，追根溯源。",
        );
        let prompt = build_sanyi_dynamic_prompt(&state);
        assert!(prompt.contains("育"));
        assert!(prompt.contains("方案分解"));
        assert!(prompt.contains("离"));
        assert!(prompt.contains("明照"));
        assert!(prompt.contains("需要回归焦点"));
        assert!(prompt.contains("强力突破"));
        assert!(prompt.contains("强化论证力度"));
        assert!(prompt.contains("明辨洞察"));
    }

    #[test]
    fn test_dynamic_prompt_variation() {
        // 验证不同状态产生不同提示词
        let s1 = DynamicPromptState::new("生", "乾", "创造", 1.2, 0.1, None, None, "");
        let s2 = DynamicPromptState::new("育", "艮", "止定", 0.3, 0.9, Some("放弃"),
            Some(fangwei_strategic_guidance(FangWei::Abort).to_string()), "");
        let p1 = build_sanyi_dynamic_prompt(&s1);
        let p2 = build_sanyi_dynamic_prompt(&s2);
        assert_ne!(p1, p2, "不同状态应产生不同提示词");
    }

    #[test]
    fn test_dynamic_prompt_with_zhou_prefix() {
        let state = DynamicPromptState::new(
            "生", "震", "启动", 1.0, 0.1, None, None,
            "以创造性思维展开回答。大胆假设，积极构建。",
        );
        let prompt = build_sanyi_dynamic_prompt(&state);
        assert!(prompt.contains("创造性思维"));
        assert!(prompt.contains("大胆假设"));
    }
}
