/// 三易语义动态注入器
///
/// 引擎层控制（Logit-Bias、Temperature）对模型无声生效。
/// 系统提示词只放真正有价值的信息：项目目标 + 已积累的关键决策。

/// 注入状态——只保留项目上下文。
/// 算子/姿态/偏离/连山导航已从提示词中移除，
/// 改为通过 Logit-Bias + Temperature 在引擎层无声控制。
pub struct InjectionState {
    /// 项目目标与已积累的关键决策（由 ProjectContext::section() 生成）
    pub project_context: Option<String>,
}

/// 构建极简 ChatML 格式 system prompt。
///
/// 只包含：项目上下文 + 一行角色标识。
/// 引擎层的算子约束、姿态调控、偏离度信息
/// 全部由 Logit-Bias + Temperature 在采样层生效，无需提示词。
pub fn build_injection(state: &InjectionState) -> String {
    let proj = if let Some(ref ctx) = state.project_context {
        ctx.clone()
    } else {
        String::from("通用AI编程助手。")
    };

    format!(
        "<|im_start|>system\n\
         {}\n\
         <|im_end|>",
        proj
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_injection_minimal() {
        let state = InjectionState {
            project_context: None,
        };
        let prompt = build_injection(&state);
        assert!(prompt.contains("通用AI编程助手"));
        assert!(prompt.starts_with("<|im_start|>system"));
        assert!(prompt.ends_with("<|im_end|>"));
        // 不再包含引擎内部状态
        assert!(!prompt.contains("探索阶段"));
        assert!(!prompt.contains("姿态"));
        assert!(!prompt.contains("偏离"));
        assert!(!prompt.contains("连山"));
        assert!(!prompt.contains("藏海"));
    }

    #[test]
    fn test_build_injection_with_project_context() {
        let state = InjectionState {
            project_context: Some(String::from(
                "【项目目标 · 始终锚定】\n  构建全栈Web应用\n\n\
                 【已积累关键决策】\n  [R01] ★ 使用actix-web作为后端框架\n  [R02] ★ PostgreSQL数据库"
            )),
        };
        let prompt = build_injection(&state);
        assert!(prompt.contains("构建全栈Web应用"));
        assert!(prompt.contains("actix-web"));
        assert!(prompt.contains("PostgreSQL"));
        assert!(prompt.contains("[R01]"));
        assert!(prompt.contains("[R02]"));
        // 不包含引擎噪音
        assert!(!prompt.contains("探索阶段"));
        assert!(!prompt.contains("偏差"));
        assert!(!prompt.contains("连山"));
    }
}
