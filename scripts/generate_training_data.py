"""
象·路线B 训练数据生成器
生成多领域、多卦象、多算子阶段的干净训练数据。
每个样本的 instruction 包含完整的约束体系说明 + 当前卦象/算子/任务，
output 是符合该约束阶段的结构化回复。
"""
import json
import random
import os
from pathlib import Path

OUTPUT_DIR = Path(r"C:\xing\training_data_v2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════
# 约束体系模板（精简版，每条 instruction 嵌入）
# ═══════════════════════════════════════════════════════

CONSTRAINT_HEADER = """你运行在「三易」认知约束框架中。

【归藏引擎 — 八气算子】
你的输出按四个生成算子循环推进：
- 生（起念探索）：使用试探词（也许/可能/如何/是否），开放提问
- 动（发散联想）：使用发散词（此外/另一方面/还可以），多角度扩展
- 长（收敛聚焦）：使用聚焦词（重点/沿着/深入），选一条路径深入
- 育（方案分解）：使用编号结构（第一步/1.），输出结构化子任务

每个算子阶段需体现对应思维特征。"""

INSTRUCTION_STARTS = [
    "在三易约束框架中，请按照当前算子阶段完成以下任务",
    "请遵循约束体系，在此阶段输出符合规范的内容",
    "基于三易框架，开始本轮生成",
]

# ═══════════════════════════════════════════════════════
# 卦象 × 算子 × 领域 模板
# ═══════════════════════════════════════════════════════

# 八卦认知姿态
BAGUA_STATES = [
    {"name": "乾", "desc": "创造", "temp": 1.2, "style": "大胆假设、创新突破、不设限制"},
    {"name": "兑", "desc": "表达", "temp": 0.9, "style": "清晰输出、直接交付、语言精炼"},
    {"name": "离", "desc": "明照", "temp": 0.5, "style": "审视反思、质疑验证、找出漏洞"},
    {"name": "震", "desc": "启动", "temp": 1.0, "style": "快速激发、果断起步、不拘泥"},
    {"name": "巽", "desc": "渗透", "temp": 0.7, "style": "细致分析、层层深入、不留死角"},
    {"name": "坎", "desc": "破局", "temp": 1.1, "style": "突破障碍、另辟蹊径、不走寻常路"},
    {"name": "艮", "desc": "止定", "temp": 0.3, "style": "保守收窄、只输出确定结论、不冒险"},
    {"name": "坤", "desc": "承载", "temp": 0.6, "style": "稳定执行、按部就班、扎实可靠"},
]

# 八气算子阶段
OPERATOR_PHASES = [
    {
        "name": "生",
        "desc": "起念探索",
        "guide": "提出开放性问题或假设。使用试探词（也许/可能/如何/是否）。不要给出确定结论，只做探索性思考。",
        "forbidden": "禁止使用「第一步」「因此」「最终结论」「应该」。保持开放性。",
    },
    {
        "name": "动",
        "desc": "发散联想",
        "guide": "从多个角度扩展思考。使用发散词（此外/另一方面/还可以/换个角度看）。列举多种可能性，不急于收敛。",
        "forbidden": "禁止使用「最终」「应该」「正确方案是」。展开思维，不做结论。",
    },
    {
        "name": "长",
        "desc": "收敛聚焦",
        "guide": "从多角度中选择一条最有价值的路径深入。使用聚焦词（重点/沿着/深入/聚焦于）。排除其他选项，锁定一条主线。",
        "forbidden": "禁止使用「也许」「另一个思路」「还可以」。做了选择就不再回头。",
    },
    {
        "name": "育",
        "desc": "方案分解",
        "guide": "将选定的路径分解为具体可执行的步骤。使用编号结构（第一步/第二步/1./2.）。每步都应该是确定性的、可操作的。",
        "forbidden": "禁止使用「也许」「可能」「或者」。每个步骤必须明确具体。",
    },
]

# 任务领域
DOMAINS = {
    "项目管理": [
        "如何在一个月内完成通常需要三个月的产品交付？",
        "资源被砍了40%，但交付标准不变，如何应对？",
        "项目中途核心成员离职，如何在不受影响的前提下继续推进？",
    ],
    "机械设计": [
        "设计一款低成本、高可靠性的家用机械手爪，预算只有行业标准的1/3",
        "现有液冷管路振动导致焊缝开裂，如何在不大改结构的前提下解决？",
        "凸轮从动件磨损过快，分析可能的根因并提出改进方向",
    ],
    "编程调试": [
        "一个系统在高并发下偶发性崩溃，日志显示大量超时但无明显错误，如何定位？",
        "旧系统需要重构，但代码库长达十万行且无测试，如何安全推进？",
        "API 响应时间从 50ms 恶化到 800ms，且波动极大，分析可能原因",
    ],
    "生活规划": [
        "想转行进入一个完全陌生的领域，但现有工作占用了全部精力，如何规划？",
        "大城市高收入和小城市高生活质量之间如何权衡？",
        "如何利用碎片时间系统性地学习一门新技能？",
    ],
    "哲学思辨": [
        "如果人类的所有知识都可以被 AI 瞬间掌握，人类学习的意义是什么？",
        "在数据驱动的世界里，直觉和经验的价值是否被低估了？",
        "技术进步的终点是解放人类还是异化人类？",
    ],
}


def make_constraint_state(bagua, operator):
    """生成当前约束状态的描述文本"""
    return (
        f"\n【当前约束状态】\n"
        f"周易姿态：{bagua['name']}（{bagua['desc']}） | 采样温度：{bagua['temp']}\n"
        f"归藏算子：{operator['name']}（{operator['desc']}）\n"
        f"输出引导：{operator['guide']}\n"
        f"阶段约束：{operator['forbidden']}\n"
    )


def generate_instruction(domain, task, bagua, operator):
    """组装完整的 instruction"""
    lines = [CONSTRAINT_HEADER, make_constraint_state(bagua, operator)]
    lines.append(f"\n---")
    lines.append(f"领域：{domain}")
    lines.append(f"任务：{task}")
    lines.append(f"\n当前处于「{operator['name']}」阶段，{operator['desc']}。")
    lines.append(random.choice(INSTRUCTION_STARTS))
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════
# 高质量输出模板（每个算子 × 卦象 = 不同风格）
# 这些是模板，生成时注入具体的任务上下文
# ═══════════════════════════════════════════════════════


def build_sheng_output(task, bagua):
    """生（起念探索）阶段的输出"""
    templates = {
        "乾": f"关于「{task}」，一个值得探索的方向是：我们是否可以从第一性原理出发，不依赖现有框架来审视这个问题？\n\n也许这意味着不要先入为主地把这当成一个已有答案的问题，而是把它看作一片未曾踏足的领域。一个可能的起点是追问：这个问题的核心矛盾是什么？在什么条件下它会自动消失？\n\n如何着手？目前可以暂不做任何限制——把所有可能性都先摆上桌。",
        "坤": f"面对「{task}」，稳妥的切入方式是先看清楚我们已经掌握了什么、还不知道什么。\n\n可能的关键在于理解约束条件的真实边界——不是别人告诉我们的边界，而是物理上不可逾越的边界。也许可以先画一张地图，标记已知区域和未知区域。\n\n如何验证边界是否真实？或许可以逐一试探每个约束，看它到底是硬性条件还是主观预设。",
        "离": f"对于「{task}」，比起急于给出方案，或许先审视这个问题本身的表述是否存在误导。\n\n是否存在一些隐含的前提假设被当作了事实？也许第一步应该是识别出这些前提，然后逐一检验它们是否真的成立。如果不是，问题的框架可能需要重新调整。\n\n如何判断问题本身是否存在陷阱？可以问自己：如果这个问题不存在，我们此刻会在做什么？",
    }
    base = templates.get(bagua["name"], templates["坤"])
    # 确保使用试探词
    if "可能" not in base and "也许" not in base and "如何" not in base and "是否" not in base:
        base = f"关于「{task}」，也许可以从一个不同的角度切入。如何重新审视这个问题的边界？可能的切入点是什么？目前不需要急于给出答案，把探索空间打开即可。"
    return base


def build_dong_output(task, bagua):
    """动（发散联想）阶段的输出"""
    templates = {
        "乾": f"围绕「{task}」，可以从多个维度展开思考：\n\n**维度一：技术维度**——是否存在新技术或新工具可以彻底改变问题的约束条件？此外，现有技术是否有被低估的应用方式？\n\n**维度二：结构维度**——换个角度看，问题的结构是否可以重新排列？是否可以通过改变流程顺序来绕过瓶颈？\n\n**维度三：人的维度**——另一方面，决策者是否掌握足够信息？团队是否具备应对方案所需的技能？\n\n**维度四：时间维度**——还可以考虑：这个问题在不同时间尺度下是否呈现出不同的性质？短期压力和长期趋势可能指向完全不同的方向。",
        "巽": f"深入分析「{task}」，需要从不同层面逐步展开：\n\n**表层**：直观的、显而易见的原因和方案是什么？这些通常被最先想到——但此外还有没有更深的东西？\n\n**中层**：在表面之下的系统结构层面——问题的各个部分如何相互影响？是否存在正反馈循环或缓冲机制？\n\n**深层**：再往下走——支配这个问题的底层原则或物理规律是什么？换个角度看，这些规律是否有例外？\n\n**关联层**：此外还需考虑——这个问题和其他看似无关的领域是否存在类似的模式？是否可以借用不同行业的解决方案？",
    }
    base = templates.get(bagua["name"],
        f"对于「{task}」，可以从不同角度展开：\n\n**角度一**：从最直接的方向看——此外还有没有迂回的方法？\n\n**角度二**：换个角度看，如果把约束条件当作变量而非定值，会有什么不同的思路？\n\n**角度三**：另一方面，如果完全不设限制地思考，理想的终极方案是什么？那个方案里有没有可以被降级保留的核心？")
    return base


def build_chang_output(task, bagua):
    """长（收敛聚焦）阶段的输出"""
    templates = {
        "艮": f"基于对「{task}」的多角度发散，现在需要做出明确的收敛选择。\n\n**选定路径**：聚焦于从约束条件本身入手——深入分析每一个约束是否真实存在。这是最保守但最可靠的方向，因为如果连边界都没有确定，任何方案都可能是空中楼阁。\n\n**排除项**：不再考虑引入新技术（成本不可控）、不扩大团队（短期不可行）、不降低交付标准（与目标冲突）。\n\n**深入方向**：重点放在约束条件的重新定义和优先级重排上。沿着这条主线，下一步是量化每个约束的弹性和刚性。",
        "离": f"审查了当前所有可能的方向后，做出以下收敛判断。\n\n**淘汰逻辑**：那些依赖外部条件变化的方向（如等项目资金增加）被排除——因为它们不在控制范围内。那些需要大量前期投入的方向也被搁置——因为与资源受限的前提矛盾。\n\n**选定方向**：聚焦于内部流程优化——深入分析那些不依赖外部输入、可以通过局部改进产生放大效应的环节。\n\n**收敛结论**：重点不是在资源不足时硬撑，而是重新定义「最小可行交付」，把精力集中在最有杠杆效应的环节上。",
    }
    base = templates.get(bagua["name"],
        f"经过发散思考，现在需要聚焦于「{task}」的最优路径。\n\n**选择逻辑**：在所有方向中，重点选取同时满足「可行」「高效」「可验证」三个条件的那一条。其他方向虽然也有价值，但在当前约束下不具竞争力。\n\n**核心路径**：深入挖掘——不是寻找更多选项，而是把选定路径的每一步都想透彻。\n\n**下一步**：沿着这条路径，开始结构化分解。")
    return base


def build_yu_output(task, bagua):
    """育（方案分解）阶段的输出"""
    templates = {
        "坤": f"基于对「{task}」的收敛分析，将选定路径分解为具体执行步骤：\n\n**第一步：信息收集与约束确认**\n- 列出所有明确的和隐含的约束条件\n- 逐一核实每个约束是否为硬性限制\n- 量化每个约束的可变动范围（弹性系数）\n\n**第二步：关键路径识别**\n- 绘制从现状到目标的完整路径图\n- 标记最窄处的瓶颈节点\n- 评估每个节点的可并行度\n\n**第三步：资源重分配方案**\n- 将现有资源按瓶颈优先级重新分配\n- 非瓶颈节点采用最低保障策略\n- 预留 15% 资源作为突发事件的缓冲\n\n**第四步：执行与监控**\n- 设定每阶段的检查点和质量标准\n- 建立偏离预警机制（超过 20% 偏差触发复盘）\n- 每完成一个阶段，重新评估后续阶段的资源需求",
        "坎": f"针对「{task}」，不按常规思路分解——要用破局的方式把路径变成可执行的步骤：\n\n**第一步：反向推演**\n从终点往前倒推：如果要让目标一定达成，在最后一步之前必须满足什么条件？继续倒推到现状——中间每一步的依赖关系一目了然。\n\n**第二步：识别杠杆点**\n不是所有步骤都同等重要。找出那个「投入最少、撬动最大」的一步——把它排在最前面执行。\n\n**第三步：绕过典型陷阱**\n列出在类似任务中最常见的三种失败模式。为每一种预设逃生路线——不等问题发生就预先封堵。\n\n**第四步：最小验证闭环**\n不等全流程跑通。先做最小的切面——只验证从第一步到第三步的可行性。通过后再扩展。",
    }
    base = templates.get(bagua["name"],
        f"将「{task}」的分析结果分解为可执行方案：\n\n第一步：梳理现状——明确当前状态与目标状态之间的差距，列出所有影响变量。\n第二步：制定执行计划——将大任务拆分成 3-5 个子任务，每个子任务有明确的完成标准。\n第三步：资源配置——为每个子任务分配对应的资源（时间/人力/预算）。\n第四步：风险预案——预判每个子任务可能的失败点，制定应对措施。")
    return base


# 输出构建函数的调度表
OUTPUT_BUILDERS = {
    "生": build_sheng_output,
    "动": build_dong_output,
    "长": build_chang_output,
    "育": build_yu_output,
}


# ═══════════════════════════════════════════════════════
# 主生成逻辑
# ═══════════════════════════════════════════════════════

def generate_dataset(filepath, target_count=200):
    """生成训练数据集"""
    samples = []
    all_combos = []

    # 收集所有 (domain, task, bagua, operator) 组合
    for domain, tasks in DOMAINS.items():
        for task in tasks:
            for bagua in BAGUA_STATES:
                for operator in OPERATOR_PHASES:
                    all_combos.append((domain, task, bagua, operator))

    # 去重用的 key 集合
    seen = set()
    # 需要的总组合数 = 领域5 × 任务3 × 卦象8 × 算子4 = 480
    # 打乱后采样 target_count 个
    random.shuffle(all_combos)

    for combo in all_combos:
        if len(samples) >= target_count:
            break
        domain, task, bagua, operator = combo
        # 去重 key: domain + task + bagua_name + operator_name
        dedup_key = f"{domain}|{task}|{bagua['name']}|{operator['name']}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        instruction = generate_instruction(domain, task, bagua, operator)
        builder = OUTPUT_BUILDERS.get(operator["name"])
        if builder is None:
            continue
        output = builder(task, bagua)

        # 基本质量检查
        if len(output) < 50:
            continue
        # 生阶段不出现禁止词
        if operator["name"] == "生" and ("第一步" in output or "因此" in output[:80]):
            continue
        # 动阶段不出现结论词
        if operator["name"] == "动" and ("最终" in output[:100] or "应该" in output[:100]):
            continue
        # 长阶段不出现发散词
        if operator["name"] == "长" and ("另一个思路" in output[:100] or "还可以" in output[:100]):
            continue
        # 育阶段不出现模糊词（核心步骤部分）
        if operator["name"] == "育":
            # 只在后半部分（步骤区域）检查
            step_section = output[len(output)//2:]
            if "也许" in step_section or "可能" in step_section[:100]:
                continue

        sample = {"instruction": instruction, "output": output}
        samples.append(sample)

    print(f"生成 {len(samples)} 条训练数据 (目标: {target_count})")

    # 统计
    phases = {}
    gua = {}
    for s in samples:
        inst = s["instruction"]
        for op in OPERATOR_PHASES:
            if f"归藏算子：{op['name']}" in inst:
                phases[op["name"]] = phases.get(op["name"], 0) + 1
        for bg in BAGUA_STATES:
            if f"周易姿态：{bg['name']}" in inst:
                gua[bg["name"]] = gua.get(bg["name"], 0) + 1

    print(f"算子分布: {phases}")
    print(f"卦象分布: {gua}")
    print(f"领域数: {len(DOMAINS)}, 卦象数: {len(BAGUA_STATES)}, 算子数: {len(OPERATOR_PHASES)}")

    # 写入
    with open(filepath, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"已保存到: {filepath}")
    return len(samples)


if __name__ == "__main__":
    random.seed(42)

    # 生成主训练集
    main_path = OUTPUT_DIR / "training_data_v2.jsonl"
    count = generate_dataset(str(main_path), target_count=250)
    print(f"\n✅ 完成！共生成 {count} 条训练数据")
    print(f"   输出目录: {OUTPUT_DIR}")
