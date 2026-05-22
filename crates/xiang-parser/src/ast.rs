/// AST node types for the 象语言 (XiangLang) DSL.

use xiang_core::{Gua, FangWei, SixQi, SixJia, SanYuan, Bagua};
use std::fmt;

/// Top-level program AST.
#[derive(Debug, Clone)]
pub struct Program {
    pub agents: Vec<AgentDef>,
    pub statements: Vec<Stmt>,
}

/// Agent definition: `定义 名称: ...`
#[derive(Debug, Clone)]
pub struct AgentDef {
    pub name: String,
    pub body: Vec<Stmt>,
}

/// Statement types.
#[derive(Debug, Clone)]
pub enum Stmt {
    /// Variable declaration: `卦 X = ...`, `数 X = ...`, `位 X = ...`, `言 X = ...`, `布尔 X = ...`
    VarDecl {
        var_type: VarType,
        name: String,
        value: Expr,
    },
    /// Assignment: `赋 X = ...`
    Assign {
        target: String,
        value: Expr,
    },
    /// Eight-qi operator (归藏层)
    QiOp(QiOp),
    /// Lianshan operator (连山层)
    LianShanOp(LianShanOp),
    /// Zhouyi operator (周易层)
    ZhouYiOp(ZhouYiOp),
    /// Conditional: `若 条件: ... [否则: ...]`
    Cond {
        condition: Condition,
        then_branch: Vec<Stmt>,
        else_branch: Option<Vec<Stmt>>,
    },
    /// Loop: `周天 [N]: ...` or `当 条件: ...`
    Loop(LoopStmt),
    /// I/O: `感 ...` or `发 ...`
    Io(IoStmt),
    /// Honesty verification: `言 "..." 诚于: ...`
    Chengshi(ChengshiStmt),
    /// CangSea operation: `存/取/增权/减权/取权`
    CangSeaOp(CangSeaOp),
}

/// Variable type for declarations.
#[derive(Debug, Clone, PartialEq)]
pub enum VarType {
    Gua,
    Num,
    Bit,
    Yan,
    Bool,
}

/// Eight-qi operator variants.
#[derive(Debug, Clone)]
pub enum QiOp {
    Sheng,
    Dong,
    Zhang,
    Yu,
    Sha { target: Option<Gua> },
    Zhi { mask: Gua },
    Gui,
    Cang { label: CangLabel },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CangLabel {
    Good,
    Evil,
}

/// Lianshan operator variants.
#[derive(Debug, Clone)]
pub enum LianShanOp {
    Gen,                     // 艮
    Qi { season: SixQi },    // 气 春/夏/秋/冬
    Jia { level: SixJia },   // 甲 初/再/三
    Yuan { fresh: SanYuan }, // 元 新/陈
    Zhi {                     // 峙: thrust/drag lines
        push_forces: Vec<String>,
        resist_forces: Vec<String>,
    },
    Jue { direction: FangWei }, // 决: 7 directions
}

/// Zhouyi operator variants.
#[derive(Debug, Clone)]
pub enum ZhouYiOp {
    Pose { bagua: Bagua },     // 姿 卦名
    Shengke {                  // 生克: generation/restraint rules
        rules: Vec<ShengkeRule>,
    },
}

#[derive(Debug, Clone)]
pub enum ShengkeRule {
    Sheng { from: Bagua, to: Bagua }, // from → to (生)
    Ke { from: Bagua, to: Bagua },    // from 克 to (克)
}

/// Loop statement.
#[derive(Debug, Clone)]
pub enum LoopStmt {
    ZhouTian { cycles: Option<u32>, body: Vec<Stmt> },
    While { condition: Condition, body: Vec<Stmt> },
}

/// I/O statement.
#[derive(Debug, Clone)]
pub enum IoStmt {
    /// `感 "字符串"` or `感 天时`
    GAN { source: String },
    /// `发 "字符串"` or `发 戒 卦`
    FA { message: String },
}

/// Honesty verification statement: `言 "text" 诚于: conditions...`
#[derive(Debug, Clone)]
pub struct ChengshiStmt {
    pub declaration: String,          // the 言 text
    pub declared_gua: Option<Gua>,    // optional declared gua
    pub conditions: Vec<ChengshiCondition>,
}

#[derive(Debug, Clone)]
pub enum ChengshiCondition {
    GuaCondition(Gua),
    DeviationCondition { cmp: CmpOp, value: f32 },
    NoShaYi,          // 无杀意
    LoadCondition { cmp: CmpOp, value: f32 },  // 载重 < N
}

/// Condition expression.
#[derive(Debug, Clone)]
pub enum Condition {
    Deviation { cmp: CmpOp, value: f32 },
    BitEq { index: u8, value: bool },
    CangSeaHasExperience,
    CangSeaNoExperience,
    CurrentGuaIs(Bagua),
    PatternMatch { expr: Box<Expr>, pattern: Gua },
}

/// Comparison operator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CmpOp {
    Gt,
    Lt,
    Eq,
    Ge,
    Le,
}

/// Expression types.
#[derive(Debug, Clone)]
pub enum Expr {
    GuaLit(Gua),
    NumLit(f32),
    StrLit(String),
    BoolLit(bool),
    Var(String),
    GANResult,         // result of 感
}

impl Expr {
    /// Try to interpret the expression as a Gua.
    pub fn as_gua(&self) -> Option<Gua> {
        match self {
            Expr::GuaLit(g) => Some(*g),
            _ => None,
        }
    }

    /// Try to interpret the expression as a number.
    pub fn as_num(&self) -> Option<f32> {
        match self {
            Expr::NumLit(n) => Some(*n),
            _ => None,
        }
    }

    /// Try to interpret as a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Expr::StrLit(s) => Some(s),
            _ => None,
        }
    }

    /// Try to interpret as a boolean.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Expr::BoolLit(b) => Some(*b),
            _ => None,
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::GuaLit(g) => write!(f, "{}", g),
            Expr::NumLit(n) => write!(f, "{}", n),
            Expr::StrLit(s) => write!(f, "\"{}\"", s),
            Expr::BoolLit(b) => write!(f, "{}", b),
            Expr::Var(s) => write!(f, "{}", s),
            Expr::GANResult => write!(f, "感"),
        }
    }
}

/// CangSea operation.
#[derive(Debug, Clone)]
pub enum CangSeaOp {
    Store {            // 存 经验: 卦 G, 标签 L
        gua: Gua,
        label: CangLabel,
    },
    Retrieve {         // 取 类似: 卦 G, 相似度 > N
        gua: Gua,
        similarity_threshold: f32,
    },
    Reinforce {        // 增权 从 A 至 B 权重 N
        from: Gua,
        to: Gua,
        weight: f32,
    },
    Weaken {           // 减权 从 A 至 B 权重 N
        from: Gua,
        to: Gua,
        weight: f32,
    },
    GetWeight {        // 卦 X = 取权 从 A
        target_var: String,
        from: Gua,
    },
}
