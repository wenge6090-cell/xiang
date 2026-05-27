/// 三易语义动态注入器
///
/// v3.1 实验提示词策略：
/// - 约束实验组：系统提示词 = 三易约束体系全文 + 三易动态提示词 + 项目上下文
/// - 对照组裸奔：系统提示词 = 三易约束体系全文（仅体系全文，无动态提示词）
///
/// 引擎层控制（Logit-Bias、Temperature）对模型无声生效。
/// 提示词层注入体系全文模拟微调知识，动态提示词作为实验变量。

use xiang_core::{SAN_YI_CONSTRAINT_FULL_TEXT, DynamicPromptState, build_sanyi_dynamic_prompt};

/// 注入状态
pub struct InjectionState {
    /// 项目目标与已积累的关键决策（由 ProjectContext::section() 生成）
    pub project_context: Option<String>,
    /// 三易引擎动态状态（仅约束实验组有值，对照组为 None）
    pub engine_state: Option<EngineHint>,
}

/// 三易引擎运行时状态快照（用于构建动态提示词）
pub struct EngineHint {
    /// 当前算子名称（生/动/长/育）
    pub operator: String,
    /// 当前周易姿态名（乾/坤/震/巽/坎/离/艮/兑）
    pub posture_name: String,
    /// 当前周易姿态描述（创造/承载/启动/渗透/破局/明照/止定/表达）
    pub posture_description: String,
    /// 当前温度值
    pub temperature: f32,
    /// 当前偏离度
    pub deviation: f32,
    /// 连山导航决策描述（仅偏离度 > 0.5 时有值）
    pub shan_decision: Option<String>,
}

/// 构建 ChatML 格式 system prompt。
///
/// v3.1 策略：
/// - 始终注入三易约束体系全文（模拟微调知识）
/// - engine_state 有值时 → 追加动态提示词（实验组）
/// - engine_state 为 None → 仅体系全文 + 项目上下文（对照组 / 兼容模式）
/// - project_context 有值时 → 追加项目上下文
pub fn build_injection(state: &InjectionState) -> String {
    let mut body = String::from(SAN_YI_CONSTRAINT_FULL_TEXT);

    // ── 动态提示词（仅约束实验组注入）──
    if let Some(ref hint) = state.engine_state {
        let dynamic_state = DynamicPromptState::new(
            &hint.operator,
            &hint.posture_name,
            &hint.posture_description,
            hint.temperature,
            hint.deviation,
            hint.shan_decision.as_deref(),
            None,   // fangwei_guidance (experiment-only)
            "",     // zhou_prompt_prefix (experiment-only)
        );
        let dynamic = build_sanyi_dynamic_prompt(&dynamic_state);
        body.push_str("\n\n");
        body.push_str(&dynamic);
    }

    // ── 项目上下文 ──
    if let Some(ref ctx) = state.project_context {
        body.push_str("\n\n");
        body.push_str(ctx);
    }

    format!(
        "<|im_start|>system\n\
         {}\n\
         <|im_end|>",
        body
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_controlled_group_no_engine_state() {
        // 对照组：仅体系全文，无动态提示词
        let state = InjectionState {
            project_context: None,
            engine_state: None,
        };
        let prompt = build_injection(&state);
        assert!(prompt.contains("三易约束体系说明"));
        assert!(prompt.contains("归藏引擎"));
        assert!(prompt.contains("周易引擎"));
        assert!(prompt.contains("连山引擎"));
        assert!(prompt.starts_with("<|im_start|>system"));
        assert!(prompt.ends_with("<|im_end|>"));
        // 对照组不应包含动态提示词
        assert!(!prompt.contains("【当前约束状态】"));
    }

    #[test]
    fn test_constrained_group_with_engine_state() {
        // 实验组：体系全文 + 动态提示词
        let state = InjectionState {
            project_context: None,
            engine_state: Some(EngineHint {
                operator: "生".to_string(),
                posture_name: "震".to_string(),
                posture_description: "启动".to_string(),
                temperature: 1.0,
                deviation: 0.15,
                shan_decision: None,
            }),
        };
        let prompt = build_injection(&state);
        assert!(prompt.contains("三易约束体系说明"));
        assert!(prompt.contains("【当前约束状态】"));
        assert!(prompt.contains("生"));
        assert!(prompt.contains("起念探索"));
        assert!(prompt.contains("震"));
        assert!(prompt.contains("启动"));
        assert!(prompt.contains("1.0"));
        assert!(prompt.contains("0.15"));
    }

    #[test]
    fn test_constrained_group_with_project_context() {
        // 实验组：体系全文 + 动态提示词 + 项目上下文
        let state = InjectionState {
            project_context: Some(String::from(
                "【项目目标】\n  构建全栈Web应用\n\n\
                 【已积累关键决策】\n  [R01] ★ 使用actix-web"
            )),
            engine_state: Some(EngineHint {
                operator: "育".to_string(),
                posture_name: "离".to_string(),
                posture_description: "明照".to_string(),
                temperature: 0.5,
                deviation: 0.6,
                shan_decision: Some("强力突破".to_string()),
            }),
        };
        let prompt = build_injection(&state);
        assert!(prompt.contains("三易约束体系说明"));
        assert!(prompt.contains("【当前约束状态】"));
        assert!(prompt.contains("项目目标"));
        assert!(prompt.contains("actix-web"));
        // 确认包含动态状态
        assert!(prompt.contains("育"));
        assert!(prompt.contains("方案分解"));
        assert!(prompt.contains("离"));
        assert!(prompt.contains("明照"));
        assert!(prompt.contains("强力突破"));
    }

    #[test]
    fn test_dynamic_prompt_variation() {
        // 验证不同引擎状态产生不同的提示词
        let s1 = InjectionState {
            project_context: None,
            engine_state: Some(EngineHint {
                operator: "生".to_string(),
                posture_name: "乾".to_string(),
                posture_description: "创造".to_string(),
                temperature: 1.2,
                deviation: 0.1,
                shan_decision: None,
            }),
        };
        let s2 = InjectionState {
            project_context: None,
            engine_state: Some(EngineHint {
                operator: "育".to_string(),
                posture_name: "艮".to_string(),
                posture_description: "止定".to_string(),
                temperature: 0.3,
                deviation: 0.9,
                shan_decision: Some("放弃".to_string()),
            }),
        };
        let p1 = build_injection(&s1);
        let p2 = build_injection(&s2);
        assert_ne!(p1, p2, "不同引擎状态应产生不同的系统提示词");
    }
}
