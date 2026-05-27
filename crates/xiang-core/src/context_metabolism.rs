/// 上下文新陈代谢系统
///
/// 路线A v3.3 核心组件：从每轮 LLM 输出中提取高质量内容片段，
/// 压缩为稠密摘要注入下一轮系统提示词，替代原始历史对话的线性膨胀。
///
/// v3.3 改进（实机测试驱动）：
///   - 段落感知句子拆分：先 \\n\\n 再 \\n 再标点，避免 Markdown 行碎片
///   - Markdown 脱敏：匹配前剥离 **、*、#、列表前缀
///   - 生算子策略扩展：增加定义/核心/总结关键词
///   - 句子换行拼接：join_within_limit 用 \\n 而非空格
///
/// **与 ProjectContext 的分工**：
///   - ProjectContext：决策级语义记忆（"使用actix-web框架"），跨轮蒸馏
///   - ContextMetabolism：输出内容级上下文（LLM 回答的关键摘要），每轮提取

/// 代谢质量等级
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetabolismQuality {
    /// 高质量：偏离度 < 0.3，阶段合规 → 较长摘要
    High,
    /// 中等质量：偏离度 < 0.7，阶段合规 → 中等摘要
    Medium,
    /// 低质量：偏离度 ≥ 0.7 或阶段违规 → 极简摘要（不丢弃）
    Low,
}

/// 一条代谢后的内容条目
#[derive(Debug, Clone)]
pub struct MetabolismEntry {
    /// 产生的轮次
    pub turn: usize,
    /// 当前算子名
    pub operator: String,
    /// 偏离度
    pub deviation: f32,
    /// 质量等级
    pub quality: MetabolismQuality,
    /// 提取的内容摘要
    pub snippet: String,
}

/// 上下文新陈代谢器
///
/// 维护一个紧凑的输出历史缓冲区，用质量过滤和容量限制
/// 保证注入给 LLM 的上下文始终高稠密。
#[derive(Debug, Clone)]
pub struct ContextMetabolism {
    /// 代谢条目（按轮次排序，最新在末尾）
    entries: Vec<MetabolismEntry>,
    /// 最大保留字符总数（超出后淘汰旧条目）
    max_chars: usize,
    /// 当前总字符数
    total_chars: usize,
}

impl ContextMetabolism {
    /// 创建代谢器。
    ///
    /// `max_chars` 控制缓冲区总容量上限。
    /// 对于 48K 上下文，建议 8000-10000 字符，
    /// 确保加上系统提示词和当前对话仍在窗口内。
    pub fn new(max_chars: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_chars,
            total_chars: 0,
        }
    }

    /// 投喂一轮输出。每轮都会产生一条代谢条目（不再因偏离度高而丢弃）。
    /// 
    /// 入口处自动剥离英文内容（`<think>` 块 + ASCII 主导行），
    /// 防止未微调模型的原生英文推理污染后续轮次的上下文。
    pub fn feed(
        &mut self,
        turn: usize,
        operator: &str,
        deviation: f32,
        phase_valid: bool,
        output_text: &str,
    ) {
        let quality = self.assess_quality(deviation, phase_valid);

        let output_text = self.strip_english(output_text);
        let output_text = self.strip_system_prompt_echo(&output_text);

        let snippet = self.extract_by_operator(&output_text, operator, &quality);

        if snippet.is_empty() {
            return;
        }

        let entry = MetabolismEntry {
            turn,
            operator: operator.to_string(),
            deviation,
            quality,
            snippet,
        };

        self.total_chars += entry.snippet.chars().count();
        self.entries.push(entry);

        self.evict_old();
    }

    /// 构建注入用的代谢上下文段落。
    pub fn section(&self) -> String {
        if self.entries.is_empty() {
            return String::new();
        }

        let mut s = String::from("【上下文代谢 · 已提取关键内容】\n");
        s.push_str("前序轮次分析摘要（◆高信度 ◇中信度 ·低信度）：\n\n");

        for entry in &self.entries {
            let quality_mark = match entry.quality {
                MetabolismQuality::High => "◆",
                MetabolismQuality::Medium => "◇",
                MetabolismQuality::Low => "·",
            };

            s.push_str(&format!(
                "{} [{}{}] (偏离度:{:.2})\n{}\n\n",
                quality_mark,
                entry.operator,
                entry.turn,
                entry.deviation,
                entry.snippet,
            ));
        }

        s
    }

    /// 清空所有代谢条目。
    pub fn clear(&mut self) {
        self.entries.clear();
        self.total_chars = 0;
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    // ── 内部方法 ──

    fn assess_quality(&self, deviation: f32, phase_valid: bool) -> MetabolismQuality {
        if !phase_valid {
            return MetabolismQuality::Low;
        }
        if deviation < 0.3 {
            MetabolismQuality::High
        } else if deviation < 0.7 {
            MetabolismQuality::Medium
        } else {
            MetabolismQuality::Low
        }
    }

    /// 剥离英文内容，防止未微调模型的原生英文推理污染代谢上下文。
    ///
    /// 两层过滤：
    ///   1. 移除 `<think>...</think>` 块（含首尾标签）
    ///   2. 移除 ASCII 字母占比 > 50% 的行
    fn strip_english(&self, text: &str) -> String {
        // 第一层：移除 <think>...</think> 块
        let mut result = text.to_string();
        while let (Some(start), Some(end)) = (
            result.to_lowercase().find("<think>"),
            result.to_lowercase().find("</think>"),
        ) {
            if end > start {
                let end = end + "</think>".len();
                result.replace_range(start..end, "");
            } else {
                break;
            }
        }

        // 第二层：按行过滤英文主导行
        let lines: Vec<&str> = result.lines().collect();
        let mut filtered = String::with_capacity(result.len());
        for line in &lines {
            if self.is_english_dominant(line) {
                continue;
            }
            filtered.push_str(line);
            filtered.push('\n');
        }

        while filtered.ends_with("\n\n") {
            filtered.pop();
        }

        filtered.trim().to_string()
    }

    /// 判断一行是否以英文为主（ASCII 字母占比 > 50%）。
    fn is_english_dominant(&self, line: &str) -> bool {
        let chars: Vec<char> = line.chars().filter(|c| !c.is_whitespace()).collect();
        if chars.is_empty() {
            return false;
        }
        let alpha_count = chars.iter().filter(|c| c.is_ascii_alphabetic()).count();
        alpha_count > chars.len() / 2
    }

    /// 剥离 system prompt 回显内容，防止三易约束体系文本泄漏到代谢摘要中。
    ///
    /// 匹配特征行：三易约束体系说明中的算子定义、引擎说明、协议条款等。
    /// 这些行在模型输出中作为回显出现时，应被滤除。
    fn strip_system_prompt_echo(&self, text: &str) -> String {
        let echo_patterns: &[&str] = &[
            "三易约束体系说明",
            "三易由三台状态机组成",
            "归藏引擎 —— 意识循环",
            "周易引擎 —— 认知姿态",
            "连山引擎 —— 障碍导航",
            "你的输出将按四个生成算子循环推进",
            "生（起念探索）：",
            "动（发散联想）：",
            "长（收敛聚焦）：",
            "育（方案分解）：",
            "系统会根据上下文动态切换你的认知姿态",
            "当你的思维偏离核心目标时",
            "你的输出将受到正则规则的形式检查",
            "一、归藏引擎",
            "二、周易引擎",
            "三、连山引擎",
            "四、阴仪协议",
            "五、偏离度",
            "八卦体系",
            "八气算子",
            "温度越低，输出越确定保守",
            "请根据以上约束体系理解并配合系统的引导",
            "【当前约束状态】",
            "归藏算子：",
            "周易姿态：",
            "连山导航：",
            "偏离度：",
            "你运行在一个名为",
        ];

        let mut result = String::with_capacity(text.len());
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                result.push('\n');
                continue;
            }
            let is_echo = echo_patterns.iter().any(|pat| trimmed.contains(pat));
            if !is_echo {
                result.push_str(line);
                result.push('\n');
            }
        }
        result.trim().to_string()
    }

    fn extract_by_operator(&self, text: &str, operator: &str, quality: &MetabolismQuality) -> String {
        let max_chars = match quality {
            MetabolismQuality::High => 300,
            MetabolismQuality::Medium => 180,
            MetabolismQuality::Low => 100,
        };

        let text = text.trim();
        if text.is_empty() {
            return String::new();
        }

        match operator {
            "生" => self.extract_sheng(text, max_chars),
            "动" => self.extract_dong(text, max_chars),
            "长" => self.extract_chang(text, max_chars),
            "育" => self.extract_yu(text, max_chars),
            _ => self.truncate_at_sentence(text, max_chars),
        }
    }

    // ── 文本预处理 ──

    /// 剥离 Markdown 格式化字符，返回纯文本用于关键词匹配。
    fn clean_text(&self, text: &str) -> String {
        let mut cleaned = String::with_capacity(text.len());
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            // 双星号 **bold**
            if c == '*' && i + 1 < chars.len() && chars[i + 1] == '*' {
                i += 2;
                continue;
            }
            // 单星号 *italic* 或列表标记
            if c == '*' {
                i += 1;
                continue;
            }
            // 井号标题
            if c == '#' && (i == 0 || (i > 0 && chars[i - 1] == '\n')) {
                while i < chars.len() && chars[i] == '#' {
                    i += 1;
                }
                if i < chars.len() && chars[i] == ' ' {
                    i += 1;
                }
                continue;
            }
            // 反引号
            if c == '`' {
                i += 1;
                continue;
            }
            // 下划线
            if c == '_' {
                i += 1;
                continue;
            }
            // 表格竖线
            if c == '|' {
                i += 1;
                continue;
            }
            // > 引用
            if c == '>' && (i == 0 || (i > 0 && chars[i - 1] == '\n')) {
                i += 1;
                if i < chars.len() && chars[i] == ' ' {
                    i += 1;
                }
                continue;
            }
            cleaned.push(c);
            i += 1;
        }
        cleaned
    }

    /// 段落感知的句子拆分。
    ///
    /// 策略：
    ///   1. 先按双换行 \\n\\n 拆段落
    ///   2. 每个段落内按单换行和句末标点（。？！）拆分
    ///   3. 过滤纯符号/纯空白片段
    fn split_sentences<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let mut result = Vec::new();

        // Step 1: 按段落拆分
        let paragraphs: Vec<&str> = if text.contains("\n\n") {
            text.split("\n\n").collect()
        } else {
            vec![text]
        };

        for para in paragraphs {
            let para = para.trim();
            if para.is_empty() {
                continue;
            }

            // Step 2: 段落内拆分句子
            let mut start = 0;
            for (i, c) in para.char_indices() {
                if matches!(c, '。' | '？' | '！') {
                    let end = i + c.len_utf8();
                    let s = para[start..end].trim();
                    if !s.is_empty() && !self.is_noise_line(s) {
                        result.push(s);
                    }
                    start = end;
                } else if c == '\n' {
                    // 单换行也拆分（Markdown 列表项）
                    let s = para[start..i].trim();
                    if !s.is_empty() && !self.is_noise_line(s) {
                        result.push(s);
                    }
                    start = i + 1;
                }
            }

            // 最后一段
            let remainder = para[start..].trim();
            if !remainder.is_empty() && !self.is_noise_line(remainder) {
                result.push(remainder);
            }
        }

        // 如果拆分后为空，回退到整段
        if result.is_empty() {
            let first = text.trim();
            if !first.is_empty() {
                result.push(first);
            }
        }

        result
    }

    /// 判断是否为噪音行（纯符号、极短无意义片段）
    fn is_noise_line(&self, s: &str) -> bool {
        let s = s.trim();
        if s.len() < 3 {
            return true;
        }
        // 中文标点开头的续行碎片
        if s.starts_with('，') || s.starts_with('。') || s.starts_with('）') || s.starts_with('】') {
            return true;
        }
        // 纯标点/空白
        if s.chars().all(|c| c.is_ascii_punctuation() || c.is_whitespace()) {
            return true;
        }
        // 纯 Markdown 格式残留
        if s == "**" || s == "*" || s == "#" || s == "##" || s == "---" {
            return true;
        }
        false
    }

    // ── 算子专用提取策略 ──

    /// 生算子：提取探索方向、核心问题、定义性陈述。
    ///
    /// 策略：
    ///   1. 问句（？结尾、或含"如何/怎么/什么"的陈述性疑问）
    ///   2. 探索/可能性表达（也许/可能/值得/方向/假设/探索/尝试）
    ///   3. 核心/定义性句子（是/作为/定义/本质/意味着）
    ///   4. 兜底：取前 2 句
    fn extract_sheng(&self, text: &str, max_chars: usize) -> String {
        let sentences = self.split_sentences(text);
        let cleaned = self.clean_text(text);
        let clean_sents = self.split_sentences(&cleaned);
        let mut extracted: Vec<&str> = Vec::new();

        // Pass 1: 问句
        let question_kw = ["？", "?", "如何", "怎么", "什么才是", "能否"];
        for (orig, clean) in sentences.iter().zip(clean_sents.iter().chain(std::iter::repeat(&""))) {
            let check = if clean.is_empty() { orig } else { clean };
            if question_kw.iter().any(|kw| check.contains(kw)) {
                extracted.push(*orig);
            }
        }

        // Pass 2: 可能性表达
        let explore_kw = ["也许", "可能", "值得", "方向", "假设", "探索", "尝试", "潜在"];
        for (orig, clean) in sentences.iter().zip(clean_sents.iter().chain(std::iter::repeat(&""))) {
            if extracted.contains(orig) { continue; }
            let check = if clean.is_empty() { orig } else { clean };
            if explore_kw.iter().any(|kw| check.contains(kw)) {
                extracted.push(*orig);
            }
        }

        // Pass 3: 核心/定义（复合关键词，避免单"是"匹配过多）
        let define_kw = ["核心是", "本质是", "定义是", "意味着", "关键在于", "主要是", "实质是", "第一步", "成为"];
        for (orig, clean) in sentences.iter().zip(clean_sents.iter().chain(std::iter::repeat(&""))) {
            if extracted.contains(orig) { continue; }
            let check = if clean.is_empty() { orig } else { clean };
            if define_kw.iter().any(|kw| check.contains(kw)) {
                extracted.push(*orig);
            }
        }

        if extracted.is_empty() {
            // 兜底：取前 2 句
            return self.join_within_limit(
                &sentences.iter().take(2).copied().collect::<Vec<_>>(),
                max_chars,
            );
        }

        self.join_within_limit(&extracted, max_chars)
    }

    /// 动算子：提取关键发现、矛盾、突破点。
    fn extract_dong(&self, text: &str, max_chars: usize) -> String {
        let sentences = self.split_sentences(text);
        let cleaned = self.clean_text(text);
        let clean_sents = self.split_sentences(&cleaned);
        let mut extracted: Vec<&str> = Vec::new();

        // Pass 1: 转折/对比
        let contrast_kw = ["但是", "然而", "却", "反而", "不过", "与此相对", "另一方面", "不同"];
        for (orig, clean) in sentences.iter().zip(clean_sents.iter().chain(std::iter::repeat(&""))) {
            let check = if clean.is_empty() { orig } else { clean };
            if contrast_kw.iter().any(|kw| check.contains(kw)) {
                extracted.push(*orig);
            }
        }

        // Pass 2: 关键发现
        let finding_kw = ["发现", "关键", "突破", "核心", "重要", "值得关注", "不同寻常", "指标", "瓶颈"];
        for (orig, clean) in sentences.iter().zip(clean_sents.iter().chain(std::iter::repeat(&""))) {
            if extracted.contains(orig) { continue; }
            let check = if clean.is_empty() { orig } else { clean };
            if finding_kw.iter().any(|kw| check.contains(kw)) {
                extracted.push(*orig);
            }
        }

        if extracted.is_empty() {
            // 首句 + 末句兜底
            let mut fallback = Vec::new();
            if let Some(first) = sentences.first() {
                fallback.push(*first);
            }
            if sentences.len() > 1 {
                if let Some(last) = sentences.last() {
                    if !fallback.contains(last) {
                        fallback.push(*last);
                    }
                }
            }
            return self.join_within_limit(&fallback, max_chars);
        }

        self.join_within_limit(&extracted, max_chars)
    }

    /// 长算子：提取推理链、证据、深层分析。
    fn extract_chang(&self, text: &str, max_chars: usize) -> String {
        let sentences = self.split_sentences(text);
        let cleaned = self.clean_text(text);
        let clean_sents = self.split_sentences(&cleaned);
        let mut extracted: Vec<&str> = Vec::new();

        // Pass 1: 因果推理
        let cause_kw = ["因为", "因此", "所以", "由此可见", "这说明", "归根到底", "导致"];
        for (orig, clean) in sentences.iter().zip(clean_sents.iter().chain(std::iter::repeat(&""))) {
            let check = if clean.is_empty() { orig } else { clean };
            if cause_kw.iter().any(|kw| check.contains(kw)) {
                extracted.push(*orig);
            }
        }

        // Pass 2: 递进分析
        let depth_kw = ["进一步", "深入", "实际上", "本质", "深层", "根本"];
        for (orig, clean) in sentences.iter().zip(clean_sents.iter().chain(std::iter::repeat(&""))) {
            if extracted.contains(orig) { continue; }
            let check = if clean.is_empty() { orig } else { clean };
            if depth_kw.iter().any(|kw| check.contains(kw)) {
                extracted.push(*orig);
            }
        }

        // Pass 3: 论据/数据
        let evidence_kw = ["根据", "数据", "证据", "表明", "反映", "体现", "显示"];
        for (orig, clean) in sentences.iter().zip(clean_sents.iter().chain(std::iter::repeat(&""))) {
            if extracted.contains(orig) { continue; }
            let check = if clean.is_empty() { orig } else { clean };
            if evidence_kw.iter().any(|kw| check.contains(kw)) {
                extracted.push(*orig);
            }
        }

        if extracted.is_empty() {
            let mut fallback = Vec::new();
            if let Some(first) = sentences.first() {
                fallback.push(*first);
            }
            for s in sentences.iter().rev() {
                if !fallback.contains(s) && (s.contains("因此") || s.contains("所以") || s.contains("可见")) {
                    fallback.push(*s);
                    break;
                }
            }
            return self.join_within_limit(&fallback, max_chars);
        }

        self.join_within_limit(&extracted, max_chars)
    }

    /// 育算子：提取决策结论、编号列表、行动方案。
    fn extract_yu(&self, text: &str, max_chars: usize) -> String {
        let sentences = self.split_sentences(text);
        let mut extracted: Vec<&str> = Vec::new();

        // Pass 1: 编号列表项（从原始行检测，保留 Markdown 格式以提高可读性）
        let lines: Vec<&str> = text.lines().collect();
        for line in &lines {
            let trimmed = line.trim();
            if trimmed.starts_with(|c: char| c.is_ascii_digit())
                || trimmed.starts_with("第")
                || trimmed.starts_with("- ")
                || trimmed.starts_with("• ")
                || trimmed.starts_with("→ ")
                || trimmed.starts_with("* ")
            {
                let cleaned_line = self.clean_text(trimmed);
                if cleaned_line.len() >= 5 {
                    extracted.push(trimmed);
                }
            }
        }

        // Pass 2: 决策/建议句
        let decision_kw = ["决定", "建议", "选择", "采用", "方案", "推荐", "最终", "综上", "结论"];
        for s in &sentences {
            if !extracted.contains(s) {
                let s_clean = self.clean_text(s);
                if decision_kw.iter().any(|kw| s_clean.contains(kw)) {
                    extracted.push(s);
                }
            }
        }

        // Pass 3: 末尾结论句
        if extracted.len() <= 1 {
            let tail_start = sentences.len().saturating_sub(3);
            let tail_kw = ["因此", "所以", "综上", "总之", "最终", "建议", "结论"];
            for s in &sentences[tail_start..] {
                if !extracted.contains(s) {
                    let s_clean = self.clean_text(s);
                    if tail_kw.iter().any(|kw| s_clean.contains(kw)) {
                        extracted.push(s);
                    }
                }
            }
        }

        if extracted.is_empty() {
            return self.join_within_limit(
                &sentences.iter().take(3).copied().collect::<Vec<_>>(),
                max_chars,
            );
        }

        self.join_within_limit(&extracted, max_chars)
    }

    // ── 通用工具方法 ──

    /// 将提取的句子拼接为多行摘要，自动去重，确保不超过字符上限。
    fn join_within_limit(&self, sentences: &[&str], max_chars: usize) -> String {
        let mut result = String::new();
        let mut included: Vec<&str> = Vec::new();

        for s in sentences {
            let s = s.trim();
            if s.is_empty() {
                continue;
            }
            // 去重：跳过与已包含句子高度重复的（>70% 字符重叠）
            if self.is_duplicate(s, &included) {
                continue;
            }
            let sep = if result.is_empty() { 0 } else { 1 }; // \n
            if result.chars().count() + sep + s.chars().count() > max_chars {
                if result.is_empty() {
                    return self.truncate_at_sentence(s, max_chars);
                }
                break;
            }
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str(s);
            included.push(s);
        }

        if result.is_empty() && !sentences.is_empty() {
            return self.truncate_at_sentence(sentences[0], max_chars);
        }

        result
    }

    /// 检查句子是否与已包含集合中的某条高度重复。
    fn is_duplicate(&self, s: &str, included: &[&str]) -> bool {
        if s.len() < 10 {
            return false; // 太短不判重
        }
        for existing in included {
            if existing.len() < 10 {
                continue;
            }
            // 取较短者的长度作为基准
            let min_len = s.len().min(existing.len());
            let overlap = s.chars().zip(existing.chars())
                .take(min_len)
                .filter(|(a, b)| a == b)
                .count();
            if overlap as f64 / min_len as f64 > 0.7 {
                return true;
            }
        }
        false
    }

    /// 在句末标点处截断，保持语义完整。
    fn truncate_at_sentence(&self, text: &str, max_chars: usize) -> String {
        let text = text.trim();
        if text.chars().count() <= max_chars {
            return text.to_string();
        }

        let truncated: String = text.chars().take(max_chars).collect();
        for (i, c) in truncated.char_indices().rev() {
            if matches!(c, '。' | '？' | '！' | '\n' | '…') {
                return truncated[..i + c.len_utf8()].to_string();
            }
        }
        truncated
    }

    /// 淘汰旧条目直至总字符数在容量限制内。
    fn evict_old(&mut self) {
        while self.total_chars > self.max_chars && !self.entries.is_empty() {
            let removed = self.entries.remove(0);
            self.total_chars = self.total_chars.saturating_sub(removed.snippet.chars().count());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── 文本预处理测试 ──

    #[test]
    fn test_clean_text_strips_markdown() {
        let m = ContextMetabolism::new(5000);
        let input = "**关键发现**：这是一条*重要*信息。";
        let cleaned = m.clean_text(input);
        assert!(!cleaned.contains("**"), "应剥离 **: {}", cleaned);
        assert!(!cleaned.contains('*'), "应剥离 *: {}", cleaned);
        assert!(cleaned.contains("关键发现"), "应保留文本: {}", cleaned);
        assert!(cleaned.contains("重要"), "应保留内文: {}", cleaned);
    }

    #[test]
    fn test_clean_text_strips_headers() {
        let m = ContextMetabolism::new(5000);
        let input = "## 分析结果\n这是内容。";
        let cleaned = m.clean_text(input);
        assert!(!cleaned.contains("##"), "应剥离标题标记: {}", cleaned);
        assert!(cleaned.contains("分析结果"), "应保留标题文本: {}", cleaned);
    }

    #[test]
    fn test_split_paragraph_aware() {
        let m = ContextMetabolism::new(5000);
        let text = "第一段句子一。第一段句子二。\n\n第二段句子一？第二段句子二！";
        let sentences = m.split_sentences(text);
        assert!(sentences.len() >= 4, "应拆分 >=4 句: {:?}", sentences);
        assert!(sentences.iter().any(|s| s.contains("第一段句子一")));
        assert!(sentences.iter().any(|s| s.contains("第二段句子一")));
    }

    #[test]
    fn test_split_filters_noise_lines() {
        let m = ContextMetabolism::new(5000);
        let text = "**\n有效内容。\n*\n##\n另一个有效句子。";
        let sentences = m.split_sentences(text);
        assert!(sentences.len() <= 3, "应过滤噪音行: {:?}", sentences);
        assert!(sentences.iter().any(|s| s.contains("有效内容")));
    }

    // ── 生算子测试 ──

    #[test]
    fn test_sheng_extracts_questions_and_definition() {
        let mut m = ContextMetabolism::new(5000);
        // 模拟模型 Markdown 输出
        let output = "\
**自我管理的第一步，是成为自己的教练。**
这意味着你既是学生，也是导师。

*   **行动**：写下你的愿景，并不断追问：如何定义理想中的自己？
*   **痛点**：没有指导意味着容易迷失方向，容易随波逐流。
*   也许可以尝试每日复盘作为自我纠偏机制。";

        m.feed(1, "生", 0.15, true, output);

        assert_eq!(m.len(), 1);
        let section = m.section();
        assert!(section.contains("◆"), "高质量应有 ◆");
        // 应包含核心定义句
        assert!(section.contains("自我管理") || section.contains("教练"), "应含定义句");
        // 应包含问句
        assert!(section.contains("如何定义") || section.contains("理想"), "应含问句");
    }

    #[test]
    fn test_sheng_markdown_model_output() {
        let mut m = ContextMetabolism::new(5000);
        // 真实模型输出的 Markdown 格式
        let output = "\
**自我管理的本质是成为自己的导师。**

*   **行动**：写下你的愿景，追问：这个愿景如何定义理想中的自己？
*   **痛点**：缺乏指导意味着容易迷失方向。
*   **突破**：关键在于建立内部反馈循环。";

        m.feed(1, "生", 0.1, true, output);
        assert_eq!(m.len(), 1);

        let section = m.section();
        // 不应出现纯 Markdown 噪音
        assert!(!section.contains("**\n"), "不应有裸 ** 标记");
    }

    // ── 动算子测试 ──

    #[test]
    fn test_dong_extracts_findings_with_markdown() {
        let mut m = ContextMetabolism::new(5000);
        let output = "\
常规方法可以完成任务。
**但是**在实际运行中，我们发现了一个**关键瓶颈**：数据库连接池在高峰期耗尽。
然而这并不是唯一的性能问题。";

        m.feed(2, "动", 0.4, true, output);

        assert_eq!(m.len(), 1);
        let section = m.section();
        assert!(section.contains("◇"), "中等质量应有 ◇");
        // 关键词匹配应在脱敏后生效
        assert!(section.contains("关键瓶颈") || section.contains("但是"));
    }

    // ── 长算子测试 ──

    #[test]
    fn test_chang_extracts_reasoning_chain() {
        let mut m = ContextMetabolism::new(5000);
        let output = "\
初步分析显示延迟增加。
进一步排查发现，数据库连接池耗尽是因为慢查询未优化。
因此需要同时调整连接池大小和SQL索引。";

        m.feed(3, "长", 0.35, true, output);

        assert_eq!(m.len(), 1);
        let section = m.section();
        assert!(section.contains("进一步") || section.contains("因此") || section.contains("因为"));
    }

    // ── 育算子测试 ──

    #[test]
    fn test_yu_extracts_structured_list() {
        let mut m = ContextMetabolism::new(5000);
        let output = "\
## 部署方案

第一步，配置数据库连接池。
第二步，设置API网关限流。
第三步，部署前端静态资源。
第四步，运行集成测试。

综上，**建议采用蓝绿部署策略**降低风险。";

        m.feed(4, "育", 0.1, true, output);

        assert_eq!(m.len(), 1);
        let section = m.section();
        assert!(section.contains("第一步"), "应含列表项");
        assert!(section.contains("蓝绿部署"), "应含决策建议");
    }

    // ── 基础功能回归测试 ──

    #[test]
    fn test_low_quality_still_extracts() {
        let mut m = ContextMetabolism::new(5000);
        m.feed(5, "长", 0.85, false, "内容偏离主题，但最后一句话应该回到正轨。");

        assert_eq!(m.len(), 1, "低质量输出应保留条目");
        assert!(m.section().contains("·"), "低质量应有 · 标记");
    }

    #[test]
    fn test_capacity_eviction() {
        let mut m = ContextMetabolism::new(200);
        m.feed(1, "生", 0.1, true, &"一。".repeat(30));
        assert!(m.len() >= 1);
        m.feed(2, "动", 0.2, true, &"二。".repeat(30));
        m.feed(3, "长", 0.3, true, &"三。".repeat(30));
        assert!(m.total_chars <= 200,
            "总字符数应 <= 200，实际 {}", m.total_chars);
    }

    #[test]
    fn test_clear() {
        let mut m = ContextMetabolism::new(5000);
        m.feed(1, "生", 0.2, true, "测试内容。");
        assert!(!m.is_empty());
        m.clear();
        assert!(m.is_empty());
        assert_eq!(m.total_chars, 0);
    }

    #[test]
    fn test_truncate_at_sentence() {
        let m = ContextMetabolism::new(5000);
        let text = "这是第一句话。这是第二句话。这是第三句话还没说完";
        let result = m.truncate_at_sentence(text, 8);
        assert!(result.ends_with('。'), "应在句末截断: '{}'", result);
    }

    #[test]
    fn test_quality_markers_all_levels() {
        let mut m = ContextMetabolism::new(5000);
        m.feed(1, "育", 0.1, true, "高质量：第一步，确定架构选型。");
        m.feed(2, "长", 0.5, true, "中等：因此需要深入排查数据库层。");
        m.feed(3, "动", 0.9, false, "低质量：完全偏离了分析方向。");

        let section = m.section();
        assert!(section.contains("◆"), "应有 ◆ 标记");
        assert!(section.contains("◇"), "应有 ◇ 标记");
        assert!(section.contains("·"), "应有 · 标记");
    }

    #[test]
    fn test_all_rounds_present_no_discard() {
        let mut m = ContextMetabolism::new(10000);
        let inputs = [
            ("生", 0.15, true, "从架构角度分析？微服务也许合适。"),
            ("动", 0.80, false, "偏离了主题，但发现数据库是关键瓶颈。"),
            ("长", 0.45, true, "进一步分析因此需要深入SQL优化。"),
            ("育", 0.20, true, "建议采用：第一步索引优化。第二步读写分离。"),
            ("生", 0.65, true, "也许缓存策略值得重新评估？"),
            ("动", 0.10, true, "关键突破：异步处理可降低延迟。但是需要改架构。"),
            ("长", 0.95, false, "分析完全走偏，但因此决定回到基线。"),
            ("育", 0.30, true, "最终方案：蓝绿部署 + 灰度发布。"),
            ("生", 0.55, true, "新方向？可能考虑事件溯源架构。"),
            ("动", 0.75, true, "低质量问题，发现了一条关键信息。"),
        ];
        for (i, (op, dev, phase, text)) in inputs.iter().enumerate() {
            m.feed(i + 1, op, *dev, *phase, text);
        }
        assert_eq!(m.len(), 10, "每轮都应产生代谢条目，实际 {}", m.len());
    }

    #[test]
    fn test_join_uses_newlines() {
        let m = ContextMetabolism::new(5000);
        let sentences = vec!["第一句。", "第二句。", "第三句。"];
        let result = m.join_within_limit(&sentences, 500);
        assert!(result.contains('\n'), "应用换行分割: '{}'", result);
        assert_eq!(result.lines().count(), 3, "应有3行");
    }

    // ── 英文过滤测试 ──

    #[test]
    fn test_strip_think_blocks() {
        let m = ContextMetabolism::new(5000);
        let input = "<think>\nThinking Process:\n1. Analyze the request.\n</think>\n这是中文内容。";
        let result = m.strip_english(input);
        assert!(!result.contains("Thinking"), "应剥离 think 块内英文: '{}'", result);
        assert!(!result.contains("<think>"), "应剥离标签: '{}'", result);
        assert!(result.contains("这是中文内容"), "应保留中文: '{}'", result);
    }

    #[test]
    fn test_strip_english_dominant_lines() {
        let m = ContextMetabolism::new(5000);
        let input = "这是第一句中文。\nWait, I need to ensure this is correct.\n继续第二句中文。";
        let result = m.strip_english(input);
        assert!(result.contains("这是第一句中文"), "应保留中文: '{}'", result);
        assert!(result.contains("继续第二句中文"), "应保留中文: '{}'", result);
        assert!(!result.contains("Wait"), "应剥离英文行: '{}'", result);
    }

    #[test]
    fn test_strip_english_preserves_technical_terms() {
        let m = ContextMetabolism::new(5000);
        let input = "至此，SQL优化方案已经明确。下一步需要调整OAuth2.0的JWT过期策略。";
        let result = m.strip_english(input);
        assert!(result.contains("SQL优化"), "应保留技术术语: '{}'", result);
        assert!(result.contains("OAuth2.0"), "应保留技术术语: '{}'", result);
        assert!(result.contains("JWT"), "应保留技术术语: '{}'", result);
    }

    #[test]
    fn test_strip_english_in_metabolism_feed() {
        let mut m = ContextMetabolism::new(5000);
        let output = "<think>\nThinking Process:\n</think>\n从架构角度分析？微服务也许合适。";
        m.feed(1, "生", 0.15, true, output);
        assert_eq!(m.len(), 1, "应产生代谢条目");
        let section = m.section();
        assert!(!section.contains("Thinking"), "代谢结果不应含英文: '{}'", section);
        assert!(!section.contains("think"), "代谢结果不应含标签: '{}'", section);
        assert!(section.contains("微服务") || section.contains("架构"), "应保留中文: '{}'", section);
    }

    #[test]
    fn test_strip_english_handles_qwen35_output() {
        let mut m = ContextMetabolism::new(5000);
        let output = "\
<think>
Thinking Process:

1.  **Analyze the Request:**
    *   **Task:** Multi-turn deep analysis.
    *   **Constraint 1:** Only complete one phase per turn.
</think>

【阶段1：步骤A-假设审查】

本阶段的核心任务是：基于第一性原理，彻底拆解并质疑当前数据库中所有被视为\"理所当然\"的假设。";
        m.feed(1, "生", 0.15, true, output);
        assert_eq!(m.len(), 1);
        let section = m.section();
        assert!(!section.contains("Thinking"), "不应含英文: '{}'", section);
        assert!(!section.contains("Analyze"), "不应含英文: '{}'", section);
        assert!(!section.contains("Constraint"), "不应含英文: '{}'", section);
        assert!(section.contains("第一性原理"), "应保留中文: '{}'", section);
        assert!(section.contains("假设"), "应保留中文: '{}'", section);
    }
}
