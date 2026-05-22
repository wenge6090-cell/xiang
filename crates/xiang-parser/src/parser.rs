/// Recursive descent parser for 象语言 (XiangLang).
///
/// Hand-written parser that consumes a token stream and produces an AST.

use crate::token::{Span, Token};
use crate::ast::*;
use crate::error::{ParseError, ParseErrorKind};
use xiang_core::{Gua, Bagua, SixQi, SixJia, SanYuan, FangWei};
use std::iter::Peekable;
use std::vec::IntoIter;

pub struct Parser {
    tokens: Peekable<IntoIter<(Token, Span)>>,
    errors: Vec<ParseError>,
}

impl Parser {
    pub fn new(tokens: Vec<(Token, Span)>) -> Self {
        let filtered: Vec<_> = tokens
            .into_iter()
            .filter(|(t, _)| !matches!(t, Token::Newline))
            .collect();
        Parser {
            tokens: filtered.into_iter().peekable(),
            errors: Vec::new(),
        }
    }

    pub fn parse(&mut self) -> Result<Program, Vec<ParseError>> {
        let program = self.parse_program();
        if self.errors.is_empty() {
            program.map_err(|e| vec![e])
        } else {
            if let Err(e) = &program {
                self.errors.push(e.clone());
            }
            Err(std::mem::take(&mut self.errors))
        }
    }

    // ── Token stream helpers ───────────────────────────────

    fn peek(&mut self) -> Option<&(Token, Span)> {
        self.tokens.peek()
    }

    fn next(&mut self) -> Option<(Token, Span)> {
        self.tokens.next()
    }

    fn span(&mut self) -> Span {
        match self.peek() {
            Some((_, s)) => *s,
            None => Span::new(0, 0, 0, 0),
        }
    }

    fn peek_token_kind(&mut self) -> Option<Token> {
        self.peek().map(|(t, _)| t.clone())
    }

    fn peek_is_ident(&mut self, kw: &str) -> bool {
        matches!(self.peek(), Some((Token::Ident(s), _)) if s == kw)
    }

    // ── Program ────────────────────────────────────────────

    fn parse_program(&mut self) -> Result<Program, ParseError> {
        let mut agents = Vec::new();
        let mut statements = Vec::new();

        loop {
            match self.peek() {
                None => break,
                Some((Token::定义, _)) => {
                    agents.push(self.parse_agent_def()?);
                }
                Some((Token::Eof, _)) => break,
                _ => {
                    statements.push(self.parse_stmt()?);
                }
            }
        }

        Ok(Program { agents, statements })
    }

    // ── Agent Definition ───────────────────────────────────

    fn parse_agent_def(&mut self) -> Result<AgentDef, ParseError> {
        let (_, span) = self.next().unwrap(); // 定义
        let (name_token, name_span) = self.next()
            .ok_or_else(|| ParseError::eof(span, "Agent名称"))?;
        let name = match name_token {
            Token::Ident(s) => s,
            t => return Err(ParseError::unexpected(name_span, "Agent名称", t)),
        };
        self.expect(Token::Colon)?;

        self.expect(Token::Indent)?;
        let mut body = Vec::new();
        loop {
            match self.peek() {
                Some((Token::Dedent, _)) | Some((Token::Eof, _)) | None => break,
                _ => body.push(self.parse_stmt()?),
            }
        }
        if matches!(self.peek(), Some((Token::Dedent, _))) {
            self.next();
        }

        Ok(AgentDef { name, body })
    }

    // ── Statement dispatch ─────────────────────────────────

    fn parse_stmt(&mut self) -> Result<Stmt, ParseError> {
        // Check for var type keywords appearing as Ident or Token::言
        match self.peek() {
            Some((Token::Ident(kw), _)) => match kw.as_str() {
                "卦" | "数" | "位" | "布尔" => return self.parse_var_decl(),
                "赋" => return self.parse_assign(),
                _ => {}
            }
            Some((Token::言, _)) => return self.parse_var_decl_with_yan(),
            _ => {}
        }

        match self.peek_token_kind() {
            // 八气算子
            Some(Token::生) | Some(Token::动) | Some(Token::长)
            | Some(Token::育) | Some(Token::杀) | Some(Token::止)
            | Some(Token::归) | Some(Token::藏) => self.parse_qi_op(),
            // 连山算子
            Some(Token::艮) | Some(Token::气) | Some(Token::甲)
            | Some(Token::元) | Some(Token::峙) | Some(Token::决) => self.parse_lianshan_op(),
            // 周易算子
            Some(Token::姿) | Some(Token::生克) => self.parse_zhouyi_op(),
            // 控制流
            Some(Token::若) => self.parse_cond(),
            Some(Token::周天) | Some(Token::当) => self.parse_loop(),
            // I/O
            Some(Token::感) | Some(Token::发) => self.parse_io(),
            // 藏海操作
            Some(Token::存) | Some(Token::取) | Some(Token::增权)
            | Some(Token::减权) | Some(Token::取权) => self.parse_cang_sea_op(),
            Some(Token::Dedent) => {
                self.next();
                Err(ParseError::new(self.span(), ParseErrorKind::InvalidIndentation))
            }
            Some(Token::Eof) => Err(ParseError::eof(self.span(), "语句")),
            None => Err(ParseError::eof(self.span(), "语句")),
            Some(t) => {
                let (_, span) = self.next().unwrap();
                Err(ParseError::unexpected(span, "语句开始", t))
            }
        }
    }

    // ── Variable Declaration / Assignment ──────────────────

    fn parse_var_decl(&mut self) -> Result<Stmt, ParseError> {
        let (type_tok, type_span) = self.next().unwrap();
        let type_str = match &type_tok {
            Token::Ident(s) => s.clone(),
            _ => unreachable!(),
        };
        let var_type = match type_str.as_str() {
            "卦" => VarType::Gua,
            "数" => VarType::Num,
            "位" => VarType::Bit,
            "言" => VarType::Yan,
            "布尔" => VarType::Bool,
            _ => return Err(ParseError::unexpected(type_span, "类型 (卦/数/位/言/布尔)", type_tok)),
        };

        let (name_tok, name_span) = self.next()
            .ok_or_else(|| ParseError::eof(type_span, "变量名"))?;
        let name = match name_tok {
            Token::Ident(s) => s,
            t => return Err(ParseError::unexpected(name_span, "变量名", t)),
        };

        self.expect(Token::Eq)?;
        let value = self.parse_expr()?;
        Ok(Stmt::VarDecl { var_type, name, value })
    }

    /// Parse var decl when the type keyword is Token::言 (not Token::Ident("言"))
    fn parse_var_decl_with_yan(&mut self) -> Result<Stmt, ParseError> {
        self.next(); // consume Token::言
        let (name_tok, name_span) = self.next()
            .ok_or_else(|| ParseError::eof(self.span(), "变量名"))?;
        let name = match name_tok {
            Token::Ident(s) => s,
            t => return Err(ParseError::unexpected(name_span, "变量名", t)),
        };
        self.expect(Token::Eq)?;
        let value = self.parse_expr()?;
        Ok(Stmt::VarDecl { var_type: VarType::Yan, name, value })
    }

    fn parse_assign(&mut self) -> Result<Stmt, ParseError> {
        let (_, span) = self.next().unwrap(); // 赋
        let (name_tok, name_span) = self.next()
            .ok_or_else(|| ParseError::eof(span, "变量名"))?;
        let target = match name_tok {
            Token::Ident(s) => s,
            t => return Err(ParseError::unexpected(name_span, "变量名", t)),
        };
        self.expect(Token::Eq)?;
        let value = self.parse_expr()?;
        Ok(Stmt::Assign { target, value })
    }

    // ── Eight-Qi Operators ─────────────────────────────────

    fn parse_qi_op(&mut self) -> Result<Stmt, ParseError> {
        let (tok, span) = self.next().unwrap();
        let qi_op = match tok {
            Token::生 => QiOp::Sheng,
            Token::动 => QiOp::Dong,
            Token::长 => QiOp::Zhang,
            Token::育 => QiOp::Yu,
            Token::杀 => QiOp::Sha { target: self.try_parse_gua_pat() },
            Token::止 => QiOp::Zhi { mask: self.parse_gua_pat()? },
            Token::归 => QiOp::Gui,
            Token::藏 => QiOp::Cang { label: self.try_parse_cang_label().unwrap_or(CangLabel::Good) },
            t => return Err(ParseError::unexpected(span, "八气算子", t)),
        };
        Ok(Stmt::QiOp(qi_op))
    }

    fn try_parse_gua_pat(&mut self) -> Option<Gua> {
        match self.peek() {
            Some((Token::GuaPat(g), _)) => {
                let g = *g;
                self.next();
                Some(g)
            }
            _ => None,
        }
    }

    fn parse_gua_pat(&mut self) -> Result<Gua, ParseError> {
        let (tok, span) = self.next()
            .ok_or_else(|| ParseError::eof(self.span(), "卦模式"))?;
        match tok {
            Token::GuaPat(g) => Ok(g),
            t => Err(ParseError::unexpected(span, "卦模式 (如 111000)", t)),
        }
    }

    fn try_parse_cang_label(&mut self) -> Option<CangLabel> {
        match self.peek() {
            Some((Token::善, _)) => { self.next(); Some(CangLabel::Good) }
            Some((Token::恶, _)) => { self.next(); Some(CangLabel::Evil) }
            _ => None,
        }
    }

    // ── Lianshan Operators ─────────────────────────────────

    fn parse_lianshan_op(&mut self) -> Result<Stmt, ParseError> {
        let (tok, _span) = self.next().unwrap();
        let op = match tok {
            Token::艮 => LianShanOp::Gen,
            Token::气 => LianShanOp::Qi { season: self.parse_season()? },
            Token::甲 => LianShanOp::Jia { level: self.parse_jia_level()? },
            Token::元 => LianShanOp::Yuan { fresh: self.parse_yuan()? },
            Token::峙 => {
                self.expect(Token::Colon)?;
                let (push, resist) = self.parse_zhi_forces()?;
                LianShanOp::Zhi { push_forces: push, resist_forces: resist }
            }
            Token::决 => {
                self.expect(Token::Colon)?;
                LianShanOp::Jue { direction: self.parse_direction()? }
            }
            t => return Err(ParseError::unexpected(self.span(), "连山算子", t)),
        };
        Ok(Stmt::LianShanOp(op))
    }

    fn parse_season(&mut self) -> Result<SixQi, ParseError> {
        let (tok, span) = self.next().ok_or_else(|| ParseError::eof(self.span(), "季节"))?;
        match tok {
            Token::Ident(s) => match s.as_str() {
                "春" => Ok(SixQi::春),
                "夏" => Ok(SixQi::夏),
                "秋" => Ok(SixQi::秋),
                "冬" => Ok(SixQi::冬),
                _ => Err(ParseError::unexpected(span, "春/夏/秋/冬", Token::Ident(s))),
            }
            t => Err(ParseError::unexpected(span, "春/夏/秋/冬", t)),
        }
    }

    fn parse_jia_level(&mut self) -> Result<SixJia, ParseError> {
        let (tok, span) = self.next().ok_or_else(|| ParseError::eof(self.span(), "甲"))?;
        match tok {
            Token::Ident(s) => match s.as_str() {
                "初" => Ok(SixJia::初),
                "再" => Ok(SixJia::再),
                "三" => Ok(SixJia::三),
                _ => Err(ParseError::unexpected(span, "初/再/三", Token::Ident(s))),
            }
            t => Err(ParseError::unexpected(span, "初/再/三", t)),
        }
    }

    fn parse_yuan(&mut self) -> Result<SanYuan, ParseError> {
        let (tok, span) = self.next().ok_or_else(|| ParseError::eof(self.span(), "元"))?;
        match tok {
            Token::Ident(s) => match s.as_str() {
                "新" => Ok(SanYuan::新),
                "陈" => Ok(SanYuan::陈),
                _ => Err(ParseError::unexpected(span, "新/陈", Token::Ident(s))),
            }
            t => Err(ParseError::unexpected(span, "新/陈", t)),
        }
    }

    fn parse_direction(&mut self) -> Result<FangWei, ParseError> {
        let (tok, span) = self.next().ok_or_else(|| ParseError::eof(self.span(), "方位"))?;
        match tok {
            Token::Ident(s) => match s.as_str() {
                "continue" => Ok(FangWei::Continue),
                "push_through" => Ok(FangWei::PushThrough),
                "navigate_around" => Ok(FangWei::NavigateAround),
                "wait_gather" => Ok(FangWei::WaitGather),
                "decompose" => Ok(FangWei::Decompose),
                "escalate" => Ok(FangWei::Escalate),
                "abort" => Ok(FangWei::Abort),
                _ => Err(ParseError::unexpected(span, "方位", Token::Ident(s))),
            }
            t => Err(ParseError::unexpected(span, "方位", t)),
        }
    }

    fn parse_zhi_forces(&mut self) -> Result<(Vec<String>, Vec<String>), ParseError> {
        // Consume optional Indent (forces may be on a new indented line)
        if matches!(self.peek(), Some((Token::Indent, _))) {
            self.next();
        }
        let mut push_forces = Vec::new();
        let mut resist_forces = Vec::new();
        while self.peek_is_ident("推力") || self.peek_is_ident("阻力") {
            let is_push = self.peek_is_ident("推力");
            self.next(); // consume 推力 or 阻力
            self.expect(Token::Colon)?;
            let (tok, _) = self.next()
                .ok_or_else(|| ParseError::eof(self.span(), "描述"))?;
            if let Token::StrLit(s) = tok {
                if is_push { push_forces.push(s); }
                else { resist_forces.push(s); }
            }
        }
        // Consume optional Dedent
        if matches!(self.peek(), Some((Token::Dedent, _))) {
            self.next();
        }
        Ok((push_forces, resist_forces))
    }

    // ── Zhouyi Operators ───────────────────────────────────

    fn parse_zhouyi_op(&mut self) -> Result<Stmt, ParseError> {
        let (tok, _span) = self.next().unwrap();
        match tok {
            Token::姿 => Ok(Stmt::ZhouYiOp(ZhouYiOp::Pose { bagua: self.parse_bagua_name()? })),
            Token::生克 => {
                self.expect(Token::Colon)?;
                Ok(Stmt::ZhouYiOp(ZhouYiOp::Shengke { rules: self.parse_shengke_rules()? }))
            }
            t => Err(ParseError::unexpected(self.span(), "姿 or 生克", t)),
        }
    }

    fn parse_bagua_name(&mut self) -> Result<Bagua, ParseError> {
        let (tok, span) = self.next().ok_or_else(|| ParseError::eof(self.span(), "八卦名"))?;
        match tok {
            Token::Ident(s) => match s.as_str() {
                "乾" => Ok(Bagua::乾), "兑" => Ok(Bagua::兑),
                "离" => Ok(Bagua::离), "震" => Ok(Bagua::震),
                "巽" => Ok(Bagua::巽), "坎" => Ok(Bagua::坎),
                "艮" => Ok(Bagua::艮), "坤" => Ok(Bagua::坤),
                _ => Err(ParseError::unexpected(span, "八卦名", Token::Ident(s))),
            }
            t => Err(ParseError::unexpected(span, "八卦名", t)),
        }
    }

    fn parse_shengke_rules(&mut self) -> Result<Vec<ShengkeRule>, ParseError> {
        // Consume optional Indent (rules may be indented)
        if matches!(self.peek(), Some((Token::Indent, _))) {
            self.next();
        }
        let mut rules = Vec::new();
        loop {
            // Peek ahead: is next token a Bagua name?
            let is_bagua = matches!(self.peek(), Some((Token::Ident(s), _)) if {
                matches!(s.as_str(), "乾"|"兑"|"离"|"震"|"巽"|"坎"|"艮"|"坤")
            });
            if !is_bagua { break; }

            let from = self.parse_bagua_name()?;
            match self.peek_token_kind() {
                Some(Token::Arrow) => {
                    self.next();
                    let to = self.parse_bagua_name()?;
                    rules.push(ShengkeRule::Sheng { from, to });
                }
                Some(Token::Ident(kw)) if kw == "克" => {
                    self.next();
                    let to = self.parse_bagua_name()?;
                    rules.push(ShengkeRule::Ke { from, to });
                }
                _ => break,
            }
        }
        if rules.is_empty() {
            return Err(ParseError::eof(self.span(), "生克规则"));
        }
        // Consume optional Dedent
        if matches!(self.peek(), Some((Token::Dedent, _))) {
            self.next();
        }
        Ok(rules)
    }

    // ── Conditional ────────────────────────────────────────

    fn parse_cond(&mut self) -> Result<Stmt, ParseError> {
        self.next(); // consume 若
        let condition = self.parse_condition()?;
        self.expect(Token::Colon)?;
        let then_branch = self.parse_block()?;
        let else_branch = if matches!(self.peek(), Some((Token::否则, _))) {
            self.next(); // consume 否则
            self.expect(Token::Colon)?;
            Some(self.parse_block()?)
        } else {
            None
        };
        Ok(Stmt::Cond { condition, then_branch, else_branch })
    }

    fn parse_condition(&mut self) -> Result<Condition, ParseError> {
        let (tok, span) = self.next().ok_or_else(|| ParseError::eof(self.span(), "条件"))?;
        match tok {
            Token::偏离度 => {
                let cmp = self.parse_cmp_op()?;
                let (val_tok, val_span) = self.next()
                    .ok_or_else(|| ParseError::eof(span, "数值"))?;
                let value = match val_tok {
                    Token::NumLit(n) => n,
                    t => return Err(ParseError::unexpected(val_span, "数值", t)),
                };
                Ok(Condition::Deviation { cmp, value })
            }
            Token::某位 => {
                self.expect(Token::Eq)?;
                let (val_tok, val_span) = self.next().ok_or_else(|| ParseError::eof(span, "位值"))?;
                match val_tok {
                    Token::NumLit(n) => Ok(Condition::BitEq { index: 0, value: n == 1.0 }),
                    t => Err(ParseError::unexpected(val_span, "0或1", t)),
                }
            }
            Token::藏海有经验 => Ok(Condition::CangSeaHasExperience),
            Token::藏海无经验 => Ok(Condition::CangSeaNoExperience),
            Token::当前卦 => {
                self.expect(Token::Eq)?;
                Ok(Condition::CurrentGuaIs(self.parse_bagua_name()?))
            }
            t => Err(ParseError::unexpected(span, "条件表达式", t)),
        }
    }

    fn parse_cmp_op(&mut self) -> Result<CmpOp, ParseError> {
        let (tok, span) = self.next().ok_or_else(|| ParseError::eof(self.span(), "比较运算符"))?;
        match tok {
            Token::Gt => Ok(CmpOp::Gt),
            Token::Lt => Ok(CmpOp::Lt),
            Token::Eq => Ok(CmpOp::Eq),
            Token::Ge => Ok(CmpOp::Ge),
            Token::Le => Ok(CmpOp::Le),
            t => Err(ParseError::unexpected(span, "> < == >= <=", t)),
        }
    }

    // ── Block ──────────────────────────────────────────────

    fn parse_block(&mut self) -> Result<Vec<Stmt>, ParseError> {
        if matches!(self.peek(), Some((Token::Indent, _))) {
            self.next();
            let mut stmts = Vec::new();
            loop {
                match self.peek_token_kind() {
                    Some(Token::Dedent) | Some(Token::Eof) | None => break,
                    _ => stmts.push(self.parse_stmt()?),
                }
            }
            if matches!(self.peek(), Some((Token::Dedent, _))) {
                self.next();
            }
            Ok(stmts)
        } else {
            let stmt = self.parse_stmt()?;
            Ok(vec![stmt])
        }
    }

    // ── Loop ───────────────────────────────────────────────

    fn parse_loop(&mut self) -> Result<Stmt, ParseError> {
        let (tok, _span) = self.next().unwrap();
        match tok {
            Token::周天 => {
                let cycles = match self.peek() {
                    Some((Token::NumLit(n), _)) => {
                        let n = *n;
                        self.next();
                        Some(n as u32)
                    }
                    _ => None,
                };
                self.expect(Token::Colon)?;
                Ok(Stmt::Loop(LoopStmt::ZhouTian { cycles, body: self.parse_block()? }))
            }
            Token::当 => {
                let condition = self.parse_condition()?;
                self.expect(Token::Colon)?;
                Ok(Stmt::Loop(LoopStmt::While { condition, body: self.parse_block()? }))
            }
            t => Err(ParseError::unexpected(self.span(), "周天 or 当", t)),
        }
    }

    // ── I/O ────────────────────────────────────────────────

    fn parse_io(&mut self) -> Result<Stmt, ParseError> {
        let (tok, _span) = self.next().unwrap();
        match tok {
            Token::感 => Ok(Stmt::Io(IoStmt::GAN { source: self.parse_string_or_ident()? })),
            Token::发 => Ok(Stmt::Io(IoStmt::FA { message: self.parse_string_or_ident()? })),
            t => Err(ParseError::unexpected(self.span(), "感 or 发", t)),
        }
    }

    fn parse_string_or_ident(&mut self) -> Result<String, ParseError> {
        let (tok, span) = self.next().ok_or_else(|| ParseError::eof(self.span(), "字符串或标识符"))?;
        match tok {
            Token::StrLit(s) => Ok(s),
            Token::Ident(s) => Ok(s),
            t => Err(ParseError::unexpected(span, "字符串或标识符", t)),
        }
    }

    // ── CangSea Operations ─────────────────────────────────

    fn parse_cang_sea_op(&mut self) -> Result<Stmt, ParseError> {
        let (tok, span) = self.next().unwrap();
        match tok {
            Token::存 => {
                self.expect_ident("经验")?;
                self.expect(Token::Colon)?;
                self.expect_ident("卦")?;
                let gua = self.parse_gua_pat()?;
                self.expect(Token::Comma)?;
                self.expect_ident("标签")?;
                let label = self.try_parse_cang_label()
                    .ok_or_else(|| ParseError::eof(span, "标签 (善/恶)"))?;
                Ok(Stmt::CangSeaOp(CangSeaOp::Store { gua, label }))
            }
            Token::取 => {
                self.expect_ident("类似")?;
                self.expect(Token::Colon)?;
                self.expect_ident("卦")?;
                let gua = self.parse_gua_pat()?;
                self.expect(Token::Comma)?;
                self.expect_ident("相似度")?;
                self.expect(Token::Gt)?;
                let (val_tok, val_span) = self.next()
                    .ok_or_else(|| ParseError::eof(span, "相似度阈值"))?;
                let similarity_threshold = match val_tok {
                    Token::NumLit(n) => n,
                    t => return Err(ParseError::unexpected(val_span, "数字", t)),
                };
                Ok(Stmt::CangSeaOp(CangSeaOp::Retrieve { gua, similarity_threshold }))
            }
            Token::增权 => {
                self.expect_ident("从")?;
                let from = self.parse_gua_pat()?;
                self.expect_ident("至")?;
                let to = self.parse_gua_pat()?;
                self.expect_ident("权重")?;
                let (val_tok, val_span) = self.next()
                    .ok_or_else(|| ParseError::eof(span, "权重值"))?;
                let weight = match val_tok {
                    Token::NumLit(n) => n,
                    t => return Err(ParseError::unexpected(val_span, "数字", t)),
                };
                Ok(Stmt::CangSeaOp(CangSeaOp::Reinforce { from, to, weight }))
            }
            Token::减权 => {
                self.expect_ident("从")?;
                let from = self.parse_gua_pat()?;
                self.expect_ident("至")?;
                let to = self.parse_gua_pat()?;
                self.expect_ident("权重")?;
                let (val_tok, val_span) = self.next()
                    .ok_or_else(|| ParseError::eof(span, "权重值"))?;
                let weight = match val_tok {
                    Token::NumLit(n) => n,
                    t => return Err(ParseError::unexpected(val_span, "数字", t)),
                };
                Ok(Stmt::CangSeaOp(CangSeaOp::Weaken { from, to, weight }))
            }
            _ => Err(ParseError::unexpected(span, "藏海操作", tok)),
        }
    }

    // ── Expression ─────────────────────────────────────────

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        let (tok, span) = self.next().ok_or_else(|| ParseError::eof(self.span(), "表达式"))?;
        match tok {
            Token::GuaPat(g) => Ok(Expr::GuaLit(g)),
            Token::NumLit(n) => Ok(Expr::NumLit(n)),
            Token::StrLit(s) => Ok(Expr::StrLit(s)),
            Token::真 => Ok(Expr::BoolLit(true)),
            Token::假 => Ok(Expr::BoolLit(false)),
            Token::Ident(s) => Ok(Expr::Var(s)),
            Token::感 => Ok(Expr::GANResult),
            t => Err(ParseError::unexpected(span, "表达式", t)),
        }
    }

    // ── Helpers ────────────────────────────────────────────

    fn expect(&mut self, expected: Token) -> Result<(), ParseError> {
        match self.next() {
            Some((t, _span)) if t == expected => Ok(()),
            Some((t, span)) => Err(ParseError::unexpected(span, &format!("{expected}"), t)),
            None => Err(ParseError::eof(self.span(), &format!("{expected}"))),
        }
    }

    fn expect_ident(&mut self, expected: &str) -> Result<(), ParseError> {
        let (tok, span) = self.next().ok_or_else(|| ParseError::eof(self.span(), expected))?;
        match tok {
            Token::Ident(s) if s == expected => Ok(()),
            t => Err(ParseError::unexpected(span, expected, t)),
        }
    }
}
