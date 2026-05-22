pub mod token;
pub mod ast;
pub mod lexer;
pub mod parser;
pub mod error;

use lexer::Lexer;
use parser::Parser;
use error::ParseError;
use ast::Program;

/// Parse a XiangLang source file into an AST.
pub fn parse(source: &str) -> Result<Program, Vec<ParseError>> {
    let mut lexer = Lexer::new(source);
    let tokens: Vec<(token::Token, token::Span)> = {
        let (tokens_with_span, lex_errors) = lexer.tokenize();
        let mut combined = Vec::new();
        for t in tokens_with_span {
            combined.push((t.token, t.span));
        }
        if !lex_errors.is_empty() {
            return Err(lex_errors);
        }
        combined
    };

    let mut parser = Parser::new(tokens);
    parser.parse()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_program() {
        let source = "\n定义 思维向导:\n    生\n    动\n    归\n";
        let result = parse(source);
        assert!(result.is_ok(), "Parse error: {:?}", result.err());
        let program = result.unwrap();
        assert_eq!(program.agents.len(), 1);
        assert_eq!(program.agents[0].name, "思维向导");
        assert_eq!(program.agents[0].body.len(), 3);
    }

    #[test]
    fn test_parse_all_qi_ops() {
        let source = "\n定义 测试:\n    生\n    动\n    长\n    育\n    杀 110\n    止 101101\n    归\n    藏 善\n    藏 恶\n";
        let result = parse(source);
        assert!(result.is_ok(), "Parse error: {:?}", result.err());
        let program = result.unwrap();
        assert_eq!(program.agents[0].body.len(), 9);
    }

    #[test]
    fn test_parse_conditional() {
        let source = "\n定义 测试:\n    若 偏离度 > 0.7:\n        杀 110\n        藏 恶\n    否则:\n        藏 善\n";
        let result = parse(source);
        assert!(result.is_ok(), "Parse error: {:?}", result.err());
    }

    #[test]
    fn test_parse_loop() {
        let source = "\n定义 测试:\n    周天 3:\n        生\n        归\n";
        let result = parse(source);
        assert!(result.is_ok(), "Parse error: {:?}", result.err());
    }

    #[test]
    fn test_parse_var_decl() {
        let source = "\n定义 测试:\n    数 阈值 = 0.8\n    卦 我的状态 = 000000\n    言 念头 = \"hello\"\n    布尔 停止 = 假\n";
        let result = parse(source);
        assert!(result.is_ok(), "Parse error: {:?}", result.err());
    }

    #[test]
    fn test_parse_zhouyi() {
        let source = "\n定义 测试:\n    姿 乾\n    生克:\n        震 → 离\n        兑 克 震\n";
        let result = parse(source);
        assert!(result.is_ok(), "Parse error: {:?}", result.err());
    }

    #[test]
    fn test_parse_lianshan() {
        let source = "\n定义 测试:\n    艮\n    气 春\n    甲 初\n    元 新\n    峙:\n        推力: \"前进\"\n        阻力: \"后退\"\n    决: navigate_around\n";
        let result = parse(source);
        assert!(result.is_ok(), "Parse error: {:?}", result.err());
    }

    #[test]
    fn test_parse_io() {
        let source = "\n定义 测试:\n    感 \"传感器\"\n    发 \"输出消息\"\n";
        let result = parse(source);
        assert!(result.is_ok(), "Parse error: {:?}", result.err());
    }

    #[test]
    fn test_parse_complex_nested() {
        let source = "
定义 思维向导:
    卦 我的状态 = 000000
    卦 问题焦点 = 111111
    数 过热阈值 = 0.8
    周天:
        藏
        生
        动
        若 偏离度 > 0.9:
            杀 110
            藏 恶
        长
        育
        若 偏离度 > 0.7:
            杀 110
        止 101101
        感 \"温度传感器\"
        若 偏离度 < 0.3:
            藏 善
        否则:
            藏 恶
";
        let result = parse(source);
        assert!(result.is_ok(), "Parse error: {:?}", result.err());
    }

    #[test]
    fn test_parse_error_missing_colon() {
        let source = "\n定义 思维向导\n    生\n";
        let result = parse(source);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_empty() {
        let result = parse("");
        assert!(result.is_ok());
    }
}
