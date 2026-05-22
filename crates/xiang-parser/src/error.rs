/// Parse errors for the 象语言 frontend.

use crate::token::{Span, Token};
use std::fmt;

#[derive(Debug, Clone)]
pub struct ParseError {
    pub span: Span,
    pub kind: ParseErrorKind,
}

#[derive(Debug, Clone)]
pub enum ParseErrorKind {
    LexError(String),
    UnexpectedToken { expected: String, found: Token },
    UnexpectedEof { expected: String },
    InvalidGua(String),
    InvalidNumber(String),
    UnclosedString,
    InvalidIndentation,
    DuplicateAgent(String),
    UnknownVariable(String),
    Other(String),
}

impl ParseError {
    pub fn new(span: Span, kind: ParseErrorKind) -> Self {
        ParseError { span, kind }
    }

    pub fn lex(span: Span, msg: impl Into<String>) -> Self {
        ParseError { span, kind: ParseErrorKind::LexError(msg.into()) }
    }

    pub fn unexpected(span: Span, expected: impl Into<String>, found: Token) -> Self {
        ParseError { span, kind: ParseErrorKind::UnexpectedToken { expected: expected.into(), found } }
    }

    pub fn eof(span: Span, expected: impl Into<String>) -> Self {
        ParseError { span, kind: ParseErrorKind::UnexpectedEof { expected: expected.into() } }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: ", self.span)?;
        match &self.kind {
            ParseErrorKind::LexError(msg) => write!(f, "词法错误: {msg}"),
            ParseErrorKind::UnexpectedToken { expected, found } => {
                write!(f, "语法错误: 期望 {expected}, 实为 {found}")
            }
            ParseErrorKind::UnexpectedEof { expected } => {
                write!(f, "语法错误: 期望 {expected}, 文件意外结束")
            }
            ParseErrorKind::InvalidGua(s) => write!(f, "无效卦值: {s}"),
            ParseErrorKind::InvalidNumber(s) => write!(f, "无效数字: {s}"),
            ParseErrorKind::UnclosedString => write!(f, "未闭合的字符串"),
            ParseErrorKind::InvalidIndentation => write!(f, "缩进错误"),
            ParseErrorKind::DuplicateAgent(s) => write!(f, "重复的Agent定义: {s}"),
            ParseErrorKind::UnknownVariable(s) => write!(f, "未知变量: {s}"),
            ParseErrorKind::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for ParseError {}
