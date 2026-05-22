/// UTF-8 lexer for 象语言 (XiangLang) source files.
///
/// Hand-written state machine that scans multi-byte UTF-8 sequences,
/// matches Chinese keywords, and tracks Python-style indentation.

use crate::token::{Span, Token};
use crate::error::{ParseError, ParseErrorKind};
use xiang_core::Gua;

/// The lexer produces tokens with spans.
#[derive(Debug, Clone)]
pub struct TokenWithSpan {
    pub token: Token,
    pub span: Span,
}

/// Lexer state machine.
pub struct Lexer {
    /// Source code as bytes
    source: Vec<u8>,
    /// Current byte offset
    pos: usize,
    /// Current line (0-indexed)
    line: usize,
    /// Current column in line (0-indexed)
    col: usize,
    /// Indentation stack (tracks indentation levels)
    indent_stack: Vec<usize>,
    /// Whether we're at the start of a line (for indentation tracking)
    at_line_start: bool,
    /// Whether we've emitted Eof
    done: bool,
    /// Pending DEDENT tokens
    pending_dedents: usize,
    /// Collected errors
    errors: Vec<ParseError>,
}

impl Lexer {
    pub fn new(source: &str) -> Self {
        Lexer {
            source: source.as_bytes().to_vec(),
            pos: 0,
            line: 0,
            col: 0,
            indent_stack: vec![0],
            at_line_start: true,
            done: false,
            pending_dedents: 0,
            errors: Vec::new(),
        }
    }

    /// Collect all tokens, returning a Vec or errors.
    pub fn tokenize(&mut self) -> (Vec<TokenWithSpan>, Vec<ParseError>) {
        let mut tokens = Vec::new();
        while let Some(result) = self.next_token() {
            match result {
                Ok(t) => tokens.push(t),
                Err(e) => self.errors.push(e),
            }
        }
        (tokens, std::mem::take(&mut self.errors))
    }

    fn next_token(&mut self) -> Option<Result<TokenWithSpan, ParseError>> {
        if self.done {
            return None;
        }
        if self.pending_dedents > 0 {
            self.pending_dedents -= 1;
            let span = Span::new(self.line, self.col, self.pos, 0);
            return Some(Ok(TokenWithSpan { token: Token::Dedent, span }));
        }
        self.skip_whitespace_and_comments();
        if self.done {
            return None;
        }
        if self.at_line_start {
            return self.handle_indentation();
        }
        self.read_token()
    }

    /// Peek at current byte, or return None if EOF.
    fn peek(&self) -> Option<u8> {
        self.source.get(self.pos).copied()
    }

    /// Peek at next N bytes as str, or return None if not enough remaining.
    fn peek_str(&self, n: usize) -> Option<&str> {
        if self.pos + n <= self.source.len() {
            std::str::from_utf8(&self.source[self.pos..self.pos + n]).ok()
        } else {
            None
        }
    }

    /// Advance by one byte.
    fn advance(&mut self) -> Option<u8> {
        let b = self.source.get(self.pos).copied()?;
        self.pos += 1;
        if b == b'\n' {
            self.line += 1;
            self.col = 0;
            self.at_line_start = true;
        } else {
            self.col += 1;
        }
        Some(b)
    }

    /// Advance by N bytes.
    fn advance_n(&mut self, n: usize) {
        for _ in 0..n {
            self.advance();
        }
    }

    /// Current span starting at current position.
    fn span_start(&self) -> Span {
        Span::new(self.line, self.col, self.pos, 0)
    }

    /// Finish a span: end position is current pos.
    fn span_end(&self, start: &Span) -> Span {
        Span::new(start.line, start.col, start.offset, self.pos - start.offset)
    }

    /// Skip spaces, tabs, and comments (# to end of line).
    /// Breaks on newline so indentation can be handled.
    fn skip_whitespace_and_comments(&mut self) {
        loop {
            match self.peek() {
                Some(b' ') | Some(b'\t') => {
                    self.advance();
                }
                Some(b'\n') => {
                    self.advance();
                    break; // Stop here — handle_indentation will process the new line
                }
                Some(b'#') => {
                    // Skip to end of line
                    while let Some(b) = self.peek() {
                        if b == b'\n' { break; }
                        self.advance();
                    }
                }
                Some(b'\r') => {
                    self.advance();
                }
                _ => break,
            }
        }
    }

    /// Handle indentation at the start of a line.
    fn handle_indentation(&mut self) -> Option<Result<TokenWithSpan, ParseError>> {
        let mut indent = 0;
        let start_pos = self.pos;
        let start_line = self.line;
        let start_col = 0;

        // Count spaces at start of line
        while let Some(b) = self.peek() {
            match b {
                b' ' => { indent += 1; self.advance(); }
                b'\t' => { indent += 4; self.advance(); } // Tab = 4 spaces
                b'\n' => {
                    // Empty line: skip it, stay at line_start
                    self.advance();
                    indent = 0;
                }
                b'\r' => { self.advance(); }
                b'#' => {
                    // Comment line: skip to end, treat as empty
                    while let Some(b) = self.peek() {
                        if b == b'\n' { break; }
                        self.advance();
                    }
                    indent = 0;
                }
                _ => break,
            }
        }

        // EOF with only whitespace
        if self.peek().is_none() {
            // Emit DEDENTs to close all indentation
            self.indent_stack.truncate(1);
            if self.indent_stack.len() > 1 {
                // Shouldn't happen since we truncated to 1
            }
            // Emit DEDENTs
            while self.indent_stack.len() > 1 {
                self.indent_stack.pop();
                // Can't emit here via the regular mechanism - just add to pending
                self.pending_dedents += 1;
            }
            self.at_line_start = false;
            self.done = true;
            let span = Span::new(start_line, start_col, start_pos, 0);
            return Some(Ok(TokenWithSpan { token: Token::Eof, span }));
        }

        self.at_line_start = false;

        let current_indent = *self.indent_stack.last().unwrap_or(&0);

        if indent > current_indent {
            // Indent
            self.indent_stack.push(indent);
            let span = Span::new(start_line, start_col, start_pos, indent);
            Some(Ok(TokenWithSpan { token: Token::Indent, span }))
        } else if indent < current_indent {
            // Dedent: pop until we find a matching level
            while indent < *self.indent_stack.last().unwrap_or(&0) {
                self.indent_stack.pop();
                self.pending_dedents += 1;
            }
            if indent != *self.indent_stack.last().unwrap_or(&0) {
                let span = Span::new(start_line, start_col, start_pos, indent);
                return Some(Err(ParseError::new(span, ParseErrorKind::InvalidIndentation)));
            }
            // Return Indent/Dedent(if pending) + then read the actual token
            if self.pending_dedents > 0 {
                self.pending_dedents -= 1;
                let span = Span::new(start_line, start_col, start_pos, indent);
                Some(Ok(TokenWithSpan { token: Token::Dedent, span }))
            } else {
                self.read_token()
            }
        } else {
            // Same indent level
            self.read_token()
        }
    }

    /// Read the next actual token.
    fn read_token(&mut self) -> Option<Result<TokenWithSpan, ParseError>> {
        let start = self.span_start();

        // EOF
        if self.peek().is_none() {
            // Emit remaining DEDENTs
            while self.indent_stack.len() > 1 {
                self.indent_stack.pop();
                self.pending_dedents += 1;
            }
            if self.pending_dedents > 0 {
                self.pending_dedents -= 1;
                return Some(Ok(TokenWithSpan { token: Token::Dedent, span: start }));
            }
            self.done = true;
            return Some(Ok(TokenWithSpan { token: Token::Eof, span: start }));
        }

        // Newline: emit Newline token
        if self.peek() == Some(b'\n') {
            self.advance();
            self.at_line_start = true;
            return Some(Ok(TokenWithSpan { token: Token::Newline, span: self.span_end(&start) }));
        }

        // Punctuation
        match self.peek() {
            Some(b':') => { self.advance(); return self.ok(Token::Colon, &start); }
            Some(b'.') => { self.advance(); return self.ok(Token::Dot, &start); }
            Some(b',') => { self.advance(); return self.ok(Token::Comma, &start); }
            Some(b'(') => { self.advance(); return self.ok(Token::OpenParen, &start); }
            Some(b')') => { self.advance(); return self.ok(Token::CloseParen, &start); }
            Some(b'~') => { self.advance(); return self.ok(Token::Squiggle, &start); }
            _ => {}
        }

        // Multi-char punctuation (must check before single char)
        if self.peek_str(2) == Some("==") {
            self.advance_n(2);
            return self.ok(Token::Eq, &start);
        }
        if self.peek_str(2) == Some(">=") {
            self.advance_n(2);
            return self.ok(Token::Ge, &start);
        }
        if self.peek_str(2) == Some("<=") {
            self.advance_n(2);
            return self.ok(Token::Le, &start);
        }
        if self.peek_str(3) == Some("→") {
            self.advance_n(3);
            return self.ok(Token::Arrow, &start);
        }
        // Single char punctuation
        if self.peek() == Some(b'=') { self.advance(); return self.ok(Token::Eq, &start); }
        if self.peek() == Some(b'>') { self.advance(); return self.ok(Token::Gt, &start); }
        if self.peek() == Some(b'<') { self.advance(); return self.ok(Token::Lt, &start); }

        // String literal: "text"
        if self.peek() == Some(b'"') {
            return self.read_string(&start);
        }

        // Gua pattern: sequence of 0/1 chars (3 or 6) — check BEFORE number
        if self.peek().map_or(false, |b| b == b'0' || b == b'1') {
            if let Some(result) = self.try_read_gua_pattern(&start) {
                return Some(result);
            }
        }

        // Number literal: digits or minus
        if self.peek().map_or(false, |b| b.is_ascii_digit() || b == b'-') {
            return self.read_number(&start);
        }

        // Chinese keyword or identifier
        let b = self.peek()?;
        if b >= 0x80 || b == b'_' || b.is_ascii_alphabetic() {
            return self.read_keyword_or_ident(&start);
        }

        // Unknown character
        self.advance();
        let span = self.span_end(&start);
        Some(Err(ParseError::lex(span, format!("无法识别的字符: '{}'", b as char))))
    }

    fn ok(&self, token: Token, start: &Span) -> Option<Result<TokenWithSpan, ParseError>> {
        Some(Ok(TokenWithSpan { token, span: self.span_end(start) }))
    }

    /// Read a string literal: "text"
    fn read_string(&mut self, start: &Span) -> Option<Result<TokenWithSpan, ParseError>> {
        self.advance(); // skip opening "
        let mut content = String::new();
        loop {
            match self.peek() {
                None => {
                    let span = self.span_end(start);
                    return Some(Err(ParseError::new(span, ParseErrorKind::UnclosedString)));
                }
                Some(b'"') => {
                    self.advance(); // skip closing "
                    let span = self.span_end(start);
                    return Some(Ok(TokenWithSpan { token: Token::StrLit(content), span }));
                }
                Some(b'\n') => {
                    let span = self.span_end(start);
                    return Some(Err(ParseError::new(span, ParseErrorKind::UnclosedString)));
                }
                Some(b) => {
                    content.push(b as char);
                    self.advance();
                }
            }
        }
    }

    /// Read a number literal.
    fn read_number(&mut self, start: &Span) -> Option<Result<TokenWithSpan, ParseError>> {
        let mut num_str = String::new();
        let mut has_dot = false;

        // Optional leading minus
        if self.peek() == Some(b'-') {
            num_str.push('-');
            self.advance();
        }

        while let Some(b) = self.peek() {
            if b.is_ascii_digit() {
                num_str.push(b as char);
                self.advance();
            } else if b == b'.' && !has_dot {
                has_dot = true;
                num_str.push('.');
                self.advance();
            } else {
                break;
            }
        }

        match num_str.parse::<f32>() {
            Ok(n) => {
                let span = self.span_end(start);
                Some(Ok(TokenWithSpan { token: Token::NumLit(n), span }))
            }
            Err(_) => {
                let span = self.span_end(start);
                Some(Err(ParseError::new(span, ParseErrorKind::InvalidNumber(num_str))))
            }
        }
    }

    /// Try to read a gua pattern literal (3 or 6 binary digits).
    fn try_read_gua_pattern(&mut self, start: &Span) -> Option<Result<TokenWithSpan, ParseError>> {
        let saved_pos = self.pos;
        let saved_line = self.line;
        let saved_col = self.col;
        let mut bits = String::new();

        loop {
            match self.peek() {
                Some(b'0') | Some(b'1') => {
                    bits.push(self.advance().unwrap() as char);
                }
                _ => break,
            }
        }

        // Must be exactly 3 or 6 bits, and followed by whitespace/punctuation/eof
        if (bits.len() == 3 || bits.len() == 6) && self.is_token_boundary() {
            // Parse as binary: fill to 6 bits (3-bit patterns go to lower trigram)
            let val: u8 = if bits.len() == 3 {
                u8::from_str_radix(&bits, 2).unwrap()
            } else {
                u8::from_str_radix(&bits, 2).unwrap()
            };
            if let Some(gua) = Gua::try_from_u8(val) {
                let span = self.span_end(start);
                return Some(Ok(TokenWithSpan { token: Token::GuaPat(gua), span }));
            }
        }

        // Not a gua pattern: restore position
        self.pos = saved_pos;
        self.line = saved_line;
        self.col = saved_col;
        None
    }

    /// Check if current position is at a token boundary (whitespace, punctuation, EOF).
    fn is_token_boundary(&self) -> bool {
        match self.peek() {
            None => true,
            Some(b' ') | Some(b'\t') | Some(b'\n') | Some(b'\r') => true,
            Some(b':') | Some(b',') | Some(b'.') | Some(b'(') | Some(b')') => true,
            Some(b'>') | Some(b'<') | Some(b'=') | Some(b'~') => true,
            _ => false,
        }
    }

    /// Read a Chinese keyword or identifier.
    fn read_keyword_or_ident(&mut self, start: &Span) -> Option<Result<TokenWithSpan, ParseError>> {
        // Read the complete UTF-8 multi-byte sequence(s) for a Chinese word.
        let mut word = String::new();
        loop {
            match self.peek() {
                None => break,
                Some(b) if b >= 0x80 => {
                    // Multi-byte UTF-8 sequence: read lead byte and continuation bytes
                    let len = utf8_byte_len(b);
                    if self.pos + len <= self.source.len() {
                        if let Ok(s) = std::str::from_utf8(&self.source[self.pos..self.pos + len]) {
                            let ch = s.chars().next().unwrap();
                            word.push(ch);
                            self.advance_n(len);
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                Some(b) if b.is_ascii_alphabetic() || b == b'_' => {
                    word.push(b as char);
                    self.advance();
                }
                _ => break,
            }
        }

        if word.is_empty() {
            let span = self.span_end(start);
            return Some(Err(ParseError::lex(span, "空 token")));
        }

        // Match against keywords
        let token = match_keyword(&word);

        let span = self.span_end(start);
        Some(Ok(TokenWithSpan { token, span }))
    }
}

/// Return the number of bytes in a UTF-8 sequence starting with byte `b`.
fn utf8_byte_len(b: u8) -> usize {
    if b & 0x80 == 0 { 1 }
    else if b & 0xE0 == 0xC0 { 2 }
    else if b & 0xF0 == 0xE0 { 3 }
    else if b & 0xF8 == 0xF0 { 4 }
    else { 1 } // invalid, treat as single byte
}

/// Match a Chinese word against all keywords. Returns Ident if no keyword matches.
fn match_keyword(word: &str) -> Token {
    match word {
        // 八气算子
        "生" => Token::生,
        "动" => Token::动,
        "长" => Token::长,
        "育" => Token::育,
        "杀" => Token::杀,
        "止" => Token::止,
        "归" => Token::归,
        "藏" => Token::藏,

        // 连山算子
        "艮" => Token::艮,
        "气" => Token::气,
        "甲" => Token::甲,
        "元" => Token::元,
        "峙" => Token::峙,
        "决" => Token::决,

        // 周易算子
        "姿" => Token::姿,
        "生克" => Token::生克,

        // 控制流
        "若" => Token::若,
        "否则" => Token::否则,
        "周天" => Token::周天,
        "当" => Token::当,
        "择" => Token::择,
        "重开" => Token::重开,
        "关闭" => Token::关闭,

        // 赋值
        "赋" => Token::赋,

        // I/O
        "感" => Token::感,
        "发" => Token::发,

        // 藏海
        "存" => Token::存,
        "取" => Token::取,
        "增权" => Token::增权,
        "减权" => Token::减权,
        "取权" => Token::取权,

        // 诚实验证
        "言" => Token::言,
        "诚于" => Token::诚于,

        // 声明
        "定义" => Token::定义,

        // 条件
        "偏离度" => Token::偏离度,
        "某位" => Token::某位,
        "当前卦" => Token::当前卦,
        "藏海有经验" => Token::藏海有经验,
        "藏海无经验" => Token::藏海无经验,
        "似" => Token::似,
        "载重" => Token::载重,
        "尝试" => Token::尝试,

        // 布尔/标签
        "真" => Token::真,
        "假" => Token::假,
        "是" => Token::是,
        "否" => Token::否,
        "善" => Token::善,
        "恶" => Token::恶,

        // 八卦名（作为标识符）
        _ => Token::Ident(word.to_string()),
    }
}
