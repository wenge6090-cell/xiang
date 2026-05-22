/// Token types for the 象语言 (XiangLang) DSL.
///
/// All tokens are Chinese keywords, following the spec.
/// Only literal values and punctuation use non-Chinese representations.

use xiang_core::Gua;
use std::fmt;

/// Span in source code for error reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub line: usize,
    pub col: usize,
    pub offset: usize,
    pub len: usize,
}

impl Span {
    pub fn new(line: usize, col: usize, offset: usize, len: usize) -> Self {
        Span { line, col, offset, len }
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}, col {}", self.line + 1, self.col + 1)
    }
}

/// All token types in the XiangLang grammar.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // ── 八气算子 (归藏层) ──
    生,
    动,
    长,
    育,
    杀,
    止,
    归,
    藏,

    // ── 连山算子 ──
    艮,
    气,
    甲,
    元,
    峙,
    决,

    // ── 周易算子 ──
    姿,
    生克,

    // ── 控制流 ──
    若,
    否则,
    周天,
    当,
    择,
    重开,
    关闭,

    // ── 赋值/变量 ──
    赋,

    // ── I/O ──
    感,
    发,

    // ── 藏海操作 ──
    存,
    取,
    增权,
    减权,
    取权,

    // ── 诚实验证 ──
    言,
    诚于,

    // ── 声明 ──
    定义,

    // ── 字面量 ──
    GuaPat(Gua),     // 卦模式字面量，如 111000
    StrLit(String),  // 字符串字面量 "text"
    NumLit(f32),     // 数字字面量 0.7, 3
    Ident(String),   // 标识符（中文名）

    // ── 条件/关键字 ──
    偏离度,
    某位,
    当前卦,
    藏海有经验,
    藏海无经验,
    似,
    载重,
    尝试,

    // ── 布尔/标签 ──
    真,
    假,
    是,
    否,
    善,
    恶,

    // ── 比较符 ──
    Gt,    // >
    Lt,    // <
    Eq,    // ==
    Ge,    // >=
    Le,    // <=

    // ── 标点 ──
    Colon,     // :
    Dot,       // .
    Comma,     // ,
    Arrow,     // →
    Squiggle,  // ~ (used for 似)
    OpenParen, // (
    CloseParen,// )

    // ── 缩进 / 专用 ──
    Indent,
    Dedent,
    Newline,

    // ── 特殊 ──
    Eof,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::生 => write!(f, "生"),
            Token::动 => write!(f, "动"),
            Token::长 => write!(f, "长"),
            Token::育 => write!(f, "育"),
            Token::杀 => write!(f, "杀"),
            Token::止 => write!(f, "止"),
            Token::归 => write!(f, "归"),
            Token::藏 => write!(f, "藏"),
            Token::艮 => write!(f, "艮"),
            Token::气 => write!(f, "气"),
            Token::甲 => write!(f, "甲"),
            Token::元 => write!(f, "元"),
            Token::峙 => write!(f, "峙"),
            Token::决 => write!(f, "决"),
            Token::姿 => write!(f, "姿"),
            Token::生克 => write!(f, "生克"),
            Token::若 => write!(f, "若"),
            Token::否则 => write!(f, "否则"),
            Token::周天 => write!(f, "周天"),
            Token::当 => write!(f, "当"),
            Token::择 => write!(f, "择"),
            Token::重开 => write!(f, "重开"),
            Token::关闭 => write!(f, "关闭"),
            Token::赋 => write!(f, "赋"),
            Token::感 => write!(f, "感"),
            Token::发 => write!(f, "发"),
            Token::存 => write!(f, "存"),
            Token::取 => write!(f, "取"),
            Token::增权 => write!(f, "增权"),
            Token::减权 => write!(f, "减权"),
            Token::取权 => write!(f, "取权"),
            Token::言 => write!(f, "言"),
            Token::诚于 => write!(f, "诚于"),
            Token::定义 => write!(f, "定义"),
            Token::GuaPat(g) => write!(f, "{}", g),
            Token::StrLit(s) => write!(f, "\"{}\"", s),
            Token::NumLit(n) => write!(f, "{}", n),
            Token::Ident(s) => write!(f, "{}", s),
            Token::偏离度 => write!(f, "偏离度"),
            Token::某位 => write!(f, "某位"),
            Token::当前卦 => write!(f, "当前卦"),
            Token::藏海有经验 => write!(f, "藏海有经验"),
            Token::藏海无经验 => write!(f, "藏海无经验"),
            Token::似 => write!(f, "似"),
            Token::载重 => write!(f, "载重"),
            Token::尝试 => write!(f, "尝试"),
            Token::真 => write!(f, "真"),
            Token::假 => write!(f, "假"),
            Token::是 => write!(f, "是"),
            Token::否 => write!(f, "否"),
            Token::善 => write!(f, "善"),
            Token::恶 => write!(f, "恶"),
            Token::Gt => write!(f, ">"),
            Token::Lt => write!(f, "<"),
            Token::Eq => write!(f, "=="),
            Token::Ge => write!(f, ">="),
            Token::Le => write!(f, "<="),
            Token::Colon => write!(f, ":"),
            Token::Dot => write!(f, "."),
            Token::Comma => write!(f, ","),
            Token::Arrow => write!(f, "→"),
            Token::Squiggle => write!(f, "~"),
            Token::OpenParen => write!(f, "("),
            Token::CloseParen => write!(f, ")"),
            Token::Indent => write!(f, "<INDENT>"),
            Token::Dedent => write!(f, "<DEDENT>"),
            Token::Newline => write!(f, "<NL>"),
            Token::Eof => write!(f, "<EOF>"),
        }
    }
}
