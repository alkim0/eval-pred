use super::exec::{ExecStats, RunContext};
use super::funcs::FUNC_MAP;
use super::parser::ParseContext;
use super::table::{FileCol, Table};
use super::utils;
use chrono::{DateTime, Duration, Utc};
use log::debug;
use regex::Regex;
use sqlparser::ast;
use std::cmp;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

macro_rules! do_number_op {
    ($xcol:expr, $ycol:expr, $op:tt) => {
        //if let DBVal::Bool(_) | DBVal::Str(_) | DBVal::Null = $x {
        //    panic!("x is not a number type");
        //} else if let DBVal::Bool(_) | DBVal::Str(_) | DBVal::Null = $y {
        //    panic!("y is not a number type");
        if $xcol.len() == 0 || $ycol.len() == 0 {
            DBCol::Int(vec![])
        } else if !$xcol.is_number_type() || !$ycol.is_number_type() {
            panic!("x or y is not a number type, instead they are ({:?}, {:?})", $xcol.data_type(), $ycol.data_type());
        } else if let (DBCol::Int(xvals), DBCol::Int(yvals)) = ($xcol, $ycol) {
            DBCol::Int(xvals.iter().zip(yvals.iter()).map(|(x, y)| x $op y).collect())
        } else if let (DBCol::Long(xvals), DBCol::Long(yvals)) = ($xcol, $ycol) {
            DBCol::Long(xvals.iter().zip(yvals.iter()).map(|(x, y)| x $op y).collect())
        } else if let (DBCol::Float(xvals), DBCol::Float(yvals)) = ($xcol, $ycol) {
            DBCol::Float(xvals.iter().zip(yvals.iter()).map(|(x, y)| x $op y).collect())
        } else if $xcol.is_integral_type() && $ycol.is_integral_type() {
            DBCol::Long($xcol.iter_as_long().zip($ycol.iter_as_long()).map(|(x, y)| x $op y).collect())
        } else {
            DBCol::Double($xcol.double_iter().zip($ycol.double_iter()).map(|(x, y)| x $op y).collect())
        }
    }
}

macro_rules! do_plus_op {
    ($xcol:expr, $ycol:expr, $op:tt) => {
        if $xcol.len() == 0 || $ycol.len() == 0 {
            DBCol::Int(vec![])
        } else if let (DBCol::DateTime(xvals), DBCol::Duration(yvals))= ($xcol, $ycol) {
            DBCol::DateTime(xvals.iter().zip(yvals.iter()).map(|(x, y)| *x + *y).collect())
        } else if let (DBCol::DateTime(xvals), DBCol::Str(yvals)) = ($xcol, $ycol) {
            let duration = utils::parse_duration(&yvals[0]);
            DBCol::DateTime(xvals.iter().map(|x| *x + duration).collect())
        } else {
            do_number_op!($xcol, $ycol, +)
        }
    }
}

macro_rules! do_minus_op {
    ($xcol:expr, $ycol:expr) => {
        if $xcol.len() == 0 || $ycol.len() == 0 {
            DBCol::Int(vec![])
        } else if let (DBCol::DateTime(xvals), DBCol::DateTime(yvals))= ($xcol, $ycol) {
            DBCol::Duration(xvals.iter().zip(yvals.iter()).map(|(x, y)| *x - *y).collect())
        } else if let (DBCol::DateTime(xvals), DBCol::Duration(yvals))= ($xcol, $ycol) {
            DBCol::DateTime(xvals.iter().zip(yvals.iter()).map(|(x, y)| *x - *y).collect())
        } else if let (DBCol::DateTime(xvals), DBCol::Str(yvals)) = ($xcol, $ycol) {
            let duration = utils::parse_duration(&yvals[0]);
            DBCol::DateTime(xvals.iter().map(|x| *x - duration).collect())
        } else {
            do_number_op!($xcol, $ycol, -)
        }
    }
}

macro_rules! do_integral_op {
    ($xcol:expr, $ycol:expr, $op:tt) => {
        if $xcol.len() == 0 || $ycol.len() == 0 {
            DBCol::Int(vec![])
        } else if !$xcol.is_integral_type() || !$ycol.is_integral_type() {
            panic!("x or y is not a integral type");
        } else if let (DBCol::Int(xvals), DBCol::Int(yvals)) = ($xcol, $ycol) {
            DBCol::Int(xvals.iter().zip(yvals.iter()).map(|(x,y)| x $op y).collect())
        } else {
            DBCol::Long($xcol.iter_as_long().zip($ycol.iter_as_long()).map(|(x,y)| x $op y).collect())
        }
    };
}

macro_rules! do_bool_op {
    ($xcol:expr, $ycol:expr, $op:tt) => {
        if $xcol.len() == 0 || $ycol.len() == 0 {
            DBCol::Bool(vec![])
        } else if let (DBCol::Str(xvals),DBCol::Str(yvals)) = ($xcol, $ycol) {
            DBCol::Bool(xvals.iter().zip(yvals.iter()).map(|(x, y)| x $op y).collect())
        } else if let (DBCol::Int(xvals), DBCol::Int(yvals)) = ($xcol, $ycol) {
            DBCol::Bool(xvals.iter().zip(yvals.iter()).map(|(x, y)| x $op y).collect())
        } else if let (DBCol::Long(xvals), DBCol::Long(yvals)) = ($xcol, $ycol) {
            DBCol::Bool(xvals.iter().zip(yvals.iter()).map(|(x, y)| x $op y).collect())
        } else if let (DBCol::Float(xvals), DBCol::Float(yvals)) = ($xcol, $ycol) {
            DBCol::Bool(xvals.iter().zip(yvals.iter()).map(|(x, y)| x $op y).collect())
        } else if let (DBCol::Bool(xvals), DBCol::Bool(yvals)) = ($xcol, $ycol) {
            DBCol::Bool(xvals.iter().zip(yvals.iter()).map(|(x, y)| x $op y).collect())
        } else if let (DBCol::DateTime(xvals), DBCol::DateTime(yvals)) = ($xcol, $ycol) {
            DBCol::Bool(xvals.iter().zip(yvals.iter()).map(|(x, y)| x $op y).collect())
        } else if let (DBCol::DateTime(xvals), DBCol::Str(yvals)) = ($xcol, $ycol) {
            let dt = utils::parse_datetime(&yvals[0]);
            DBCol::Bool(xvals.iter().map(|x| x $op &dt).collect())
        } else if let (DBCol::Duration(xvals), DBCol::Duration(yvals)) = ($xcol, $ycol) {
            DBCol::Bool(xvals.iter().zip(yvals.iter()).map(|(x, y)| x $op y).collect())
        } else if let (DBCol::Duration(xvals), DBCol::Str(yvals)) = ($xcol, $ycol) {
            let duration = utils::parse_duration(&yvals[0]);
            DBCol::Bool(xvals.iter().map(|x| x $op &duration).collect())
        } else if $xcol.is_integral_type() && $ycol.is_integral_type() {
            DBCol::Bool($xcol.iter_as_long().zip($ycol.iter_as_long()).map(|(x, y)| x $op y).collect())
        } else if $xcol.is_number_type() && $ycol.is_number_type() {
            DBCol::Bool($xcol.double_iter().zip($ycol.double_iter()).map(|(x, y)| x $op y).collect())
        } else {
            panic!("bool comparison between weird types example ({:?}, {:?})", $xcol.any(), $ycol.any());
        }
    };
}

#[derive(Debug)]
pub enum DataType {
    Str,
    Int,
    Long,
    Float,
    Double,
    Bool,
    DateTime,
    Duration,
}

// A vector of db values. All values are expressed as DBcols. Singleton values are vectors of size
// 1.
#[derive(Debug, PartialEq, Clone)]
pub enum DBCol {
    Int(Vec<i32>),
    Long(Vec<i64>),
    Bool(Vec<bool>),
    Float(Vec<f32>),
    Double(Vec<f64>),
    Str(Vec<String>),
    DateTime(Vec<DateTime<Utc>>),
    Duration(Vec<Duration>),
}

// XXX This will implement equals for floats, but this is always a tricky concept; if you use
// floats for hash map keys, you should reevaluate your life
impl Eq for DBCol {}

impl Hash for DBCol {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            DBCol::Float(vals) => {
                for f in vals {
                    f.to_bits().hash(state);
                }
            }
            DBCol::Double(vals) => {
                for d in vals {
                    d.to_bits().hash(state);
                }
            }
            DBCol::Int(vals) => {
                vals.hash(state);
            }
            DBCol::Long(vals) => {
                vals.hash(state);
            }
            DBCol::Bool(vals) => {
                vals.hash(state);
            }
            DBCol::Str(vals) => {
                vals.hash(state);
            }
            DBCol::DateTime(vals) => {
                vals.hash(state);
            }
            DBCol::Duration(vals) => {
                vals.hash(state);
            }
        }
    }
}

impl DBCol {
    pub fn len(&self) -> usize {
        use DBCol::*;
        match self {
            Int(vals) => vals.len(),
            Long(vals) => vals.len(),
            Bool(vals) => vals.len(),
            Float(vals) => vals.len(),
            Double(vals) => vals.len(),
            Str(vals) => vals.len(),
            DateTime(vals) => vals.len(),
            Duration(vals) => vals.len(),
        }
    }

    pub fn data_type(&self) -> DataType {
        use DBCol::*;
        match self {
            Int(_) => DataType::Int,
            Long(_) => DataType::Long,
            Float(_) => DataType::Float,
            Double(_) => DataType::Double,
            Bool(_) => DataType::Bool,
            Str(_) => DataType::Str,
            DateTime(_) => DataType::DateTime,
            Duration(_) => DataType::Duration,
        }
    }

    fn is_number_type(&self) -> bool {
        use DBCol::*;
        match self {
            Int(_) | Long(_) | Float(_) | Double(_) => true,
            _ => false,
        }
    }

    fn is_integral_type(&self) -> bool {
        use DBCol::*;
        match self {
            Int(_) | Long(_) => true,
            _ => false,
        }
    }

    pub fn iter_as_long<'a>(&'a self) -> Box<dyn Iterator<Item = i64> + 'a> {
        match self {
            DBCol::Int(vals) => Box::new(vals.iter().map(|v| *v as i64)),
            DBCol::Long(vals) => Box::new(vals.iter().cloned()),
            _ => {
                panic!("Not of integral type");
            }
        }
    }

    pub fn double_iter<'a>(&'a self) -> Box<dyn Iterator<Item = f64> + 'a> {
        match self {
            DBCol::Int(vals) => Box::new(vals.iter().map(|v| *v as f64)),
            DBCol::Long(vals) => Box::new(vals.iter().map(|v| *v as f64)),
            DBCol::Float(vals) => Box::new(vals.iter().map(|v| *v as f64)),
            DBCol::Double(vals) => Box::new(vals.iter().cloned()),
            _ => {
                panic!("Not of number type");
            }
        }
    }

    // `num` is the size we want the resulting length to be.
    pub fn repeat(&mut self, num: usize) {
        match self {
            DBCol::Int(vals) => {
                assert_eq!(vals.len(), 1);
                vals.append(&mut vec![vals[0]; num - 1]);
            }
            DBCol::Long(vals) => {
                assert_eq!(vals.len(), 1);
                vals.append(&mut vec![vals[0]; num - 1]);
            }
            DBCol::Float(vals) => {
                assert_eq!(vals.len(), 1);
                vals.append(&mut vec![vals[0]; num - 1]);
            }
            DBCol::Double(vals) => {
                assert_eq!(vals.len(), 1);
                vals.append(&mut vec![vals[0]; num - 1]);
            }
            DBCol::Str(vals) => {
                assert_eq!(vals.len(), 1);
                vals.append(&mut (0..(num - 1)).map(|_| vals[0].clone()).collect());
            }
            DBCol::Bool(vals) => {
                assert_eq!(vals.len(), 1);
                vals.append(&mut vec![vals[0]; num - 1]);
            }
            DBCol::DateTime(vals) => {
                assert_eq!(vals.len(), 1);
                vals.append(&mut vec![vals[0]; num - 1]);
            }
            DBCol::Duration(vals) => {
                assert_eq!(vals.len(), 1);
                vals.append(&mut vec![vals[0]; num - 1]);
            }
        }
    }

    fn clear(&mut self) {
        match self {
            DBCol::Int(vals) => {
                vals.clear();
            }
            DBCol::Long(vals) => {
                vals.clear();
            }
            DBCol::Float(vals) => {
                vals.clear();
            }
            DBCol::Double(vals) => {
                vals.clear();
            }
            DBCol::Str(vals) => {
                vals.clear();
            }
            DBCol::Bool(vals) => {
                vals.clear();
            }
            DBCol::DateTime(vals) => {
                vals.clear();
            }
            DBCol::Duration(vals) => {
                vals.clear();
            }
        }
    }

    // Returns a sample element from the column
    pub fn any(&self) -> Option<Box<dyn fmt::Debug>> {
        if self.len() == 0 {
            None
        } else {
            match self {
                DBCol::Int(vals) => Some(Box::new(vals[0])),
                DBCol::Long(vals) => Some(Box::new(vals[0])),
                DBCol::Float(vals) => Some(Box::new(vals[0])),
                DBCol::Double(vals) => Some(Box::new(vals[0])),
                DBCol::Str(vals) => Some(Box::new(vals[0].clone())),
                DBCol::Bool(vals) => Some(Box::new(vals[0])),
                DBCol::DateTime(vals) => Some(Box::new(vals[0])),
                DBCol::Duration(vals) => Some(Box::new(vals[0])),
            }
        }
    }

    // Returns a few samples from the column:
    pub fn some(&self, num: usize) -> Vec<Box<dyn fmt::Debug>> {
        let num = cmp::min(num, self.len());
        match self {
            DBCol::Int(vals) => vals[..num]
                .iter()
                .map(|x| Box::new(*x) as Box<dyn fmt::Debug>)
                .collect(),
            DBCol::Long(vals) => vals[..num]
                .iter()
                .map(|x| Box::new(*x) as Box<dyn fmt::Debug>)
                .collect(),
            DBCol::Float(vals) => vals[..num]
                .iter()
                .map(|x| Box::new(*x) as Box<dyn fmt::Debug>)
                .collect(),
            DBCol::Double(vals) => vals[..num]
                .iter()
                .map(|x| Box::new(*x) as Box<dyn fmt::Debug>)
                .collect(),
            DBCol::Str(vals) => vals[..num]
                .iter()
                .map(|x| Box::new(x.clone()) as Box<dyn fmt::Debug>)
                .collect(),
            DBCol::Bool(vals) => vals[..num]
                .iter()
                .map(|x| Box::new(*x) as Box<dyn fmt::Debug>)
                .collect(),
            DBCol::DateTime(vals) => vals[..num]
                .iter()
                .map(|x| Box::new(*x) as Box<dyn fmt::Debug>)
                .collect(),
            DBCol::Duration(vals) => vals[..num]
                .iter()
                .map(|x| Box::new(*x) as Box<dyn fmt::Debug>)
                .collect(),
        }
    }
}

// The result of evaluating an `Expr` on a table.
// By default, all DBCols are stored in a hash map, keyed based on the groups. If there are is no
// grouping in place (e.g., no group by), then everything is stored in the default key (an empty
// Vec).
// There should always be at least one key in DBResult (e.g., the default key).
#[derive(Debug)]
pub struct DBResult {
    pub cols: HashMap<Vec<DBCol>, DBCol>,
    // The table that this result was evaluated against
    pub ref_table: Option<Rc<dyn Table>>,
}

impl PartialEq for DBResult {
    // TODO Check the ref_table as well
    fn eq(&self, other: &Self) -> bool {
        self.cols == other.cols
    }
}

impl DBResult {
    pub fn is_ungrouped(&self) -> bool {
        self.cols.len() == 1 && self.cols.iter().next().unwrap().0.is_empty()
    }

    pub fn group(&mut self, groups: &Vec<Vec<DBCol>>) {
        assert_eq!(1, self.cols.len());
        let main_col_entry = self.cols.iter().next().unwrap();
        assert!(main_col_entry.0.is_empty());
        assert_eq!(
            main_col_entry.1.len(),
            groups.len(),
            "\n{:?}\n**\n{:?}",
            main_col_entry.1,
            groups
        );

        macro_rules! entry_default_push {
            ($entry:expr, $val:expr, $type:path) => {
                let mut entry = $entry.or_insert($type(vec![]));
                if let $type(ref mut entry_vals) = &mut entry {
                    entry_vals.push($val);
                } else {
                    panic!("blah");
                }
            };
        }

        let mut grouped_cols = HashMap::new();
        for i in 0..main_col_entry.1.len() {
            let entry = grouped_cols.entry(groups[i].clone());
            match main_col_entry.1 {
                DBCol::Int(vals) => {
                    entry_default_push!(entry, vals[i], DBCol::Int);
                }
                DBCol::Long(vals) => {
                    entry_default_push!(entry, vals[i], DBCol::Long);
                }
                DBCol::Float(vals) => {
                    entry_default_push!(entry, vals[i], DBCol::Float);
                }
                DBCol::Double(vals) => {
                    entry_default_push!(entry, vals[i], DBCol::Double);
                }
                DBCol::Str(vals) => {
                    entry_default_push!(entry, vals[i].clone(), DBCol::Str);
                }
                DBCol::Bool(vals) => {
                    entry_default_push!(entry, vals[i], DBCol::Bool);
                }
                DBCol::DateTime(vals) => {
                    entry_default_push!(entry, vals[i], DBCol::DateTime);
                }
                DBCol::Duration(vals) => {
                    entry_default_push!(entry, vals[i], DBCol::Duration);
                }
            }
        }
        self.cols = grouped_cols;
    }

    pub fn len(&self) -> usize {
        self.cols.iter().map(|(_, col)| col.len()).sum()
    }
}
//    fn is_empty(&self) -> bool {
//        self.cols.iter().all(|(_, v)| v.0.is_empty())
//    }
//}

// Here, `ref_table` refers to the table with which the expression is evaluated iin  respect to. For
// The reason `ref_tables` might have multiple i.e.,
// example, the table may be different between expressions in the predicate and the join condition.
// We equate the following Expr to a predicate atom; a small basic subexpression of the predicate
// which evaluates to true/false, we assume the predicate is composed of ANDs and ORs of these
#[derive(Debug, Clone)]
pub enum Expr {
    ColRef {
        col: Rc<FileCol>,
        // This ref_table probably points to the original FileTable
        ref_table: Rc<dyn Table>,
    },
    Wildcard {
        ref_table: Rc<dyn Table>,
    },
    IsNull(Box<Expr>),
    IsNotNull(Box<Expr>),
    BinaryOp {
        left: Box<Expr>,
        right: Box<Expr>,
        op: BinaryOperator,
    },
    UnaryOp {
        expr: Box<Expr>,
        op: UnaryOperator,
    },
    Nested(Box<Expr>),
    Value(DBCol),
    Function {
        name: String,
        args: Vec<Box<Expr>>,
    },
    Case {
        cond: Box<Expr>,
        then: Box<Expr>,
        else_: Option<Box<Expr>>,
    },
}

#[derive(Debug, Clone)]
pub enum BinaryOperator {
    Plus,
    Minus,
    Multiply,
    Divide,
    Modulus,
    Gt,
    Lt,
    GtEq,
    LtEq,
    Eq,
    NotEq,
    Like,
    NotLike,
}

macro_rules! make_binary_ops {
    ( $expr:expr, $context:expr, $($op:ident),*) => {
        if let ast::Expr::BinaryOp { left, op, right } = $expr {
            match op {
                ast::BinaryOperator::And | ast::BinaryOperator::Or => {
                    Err(ExprCreateError::ExprHasAndOr)
                },
                $(
                    ast::BinaryOperator::$op => {
                        let left = Expr::new(left, $context)?;
                        let right = Expr::new(right, $context)?;
                        Ok(Expr::BinaryOp {
                            left: Box::new(left),
                            op: BinaryOperator::$op,
                            right: Box::new(right),
                        })
                    },
                )*
            }
        } else {
            panic!("wtf");
        }
    };
}

macro_rules! make_unary_ops {
    ($expr:expr, $context:expr, $($op:ident),*) => {
        if let ast::Expr::UnaryOp { expr, op } = $expr {
            match op {
                $(
                    ast::UnaryOperator::$op => {
                        Ok(Expr::UnaryOp {
                            op: UnaryOperator::$op,
                            expr: Box::new(Expr::new(expr, $context)?),
                        })
                    },
                )*
            }
        } else {
            panic!("wtf");
        }
    };
}

#[derive(Debug, Clone)]
pub enum UnaryOperator {
    Plus,
    Minus,
    Not,
}

#[derive(Debug)]
pub enum ExprCreateError {
    ColDoesNotExist,
    ExprHasAndOr,
    UnimplementedFunc(String),
}

impl Expr {
    pub fn new(ast_expr: &ast::Expr, context: &ParseContext) -> Result<Self, ExprCreateError> {
        match ast_expr {
            ast::Expr::Identifier(ident) => {
                let result = context
                    .ref_table
                    .find_col(ident, None)
                    .and_then(|col| Some((col, &context.ref_table)))
                    .or_else(|| {
                        for table in &context.other_tables {
                            let col = table.find_col(ident, None);
                            if let Some(col) = col {
                                return Some((col, table));
                            }
                        }
                        None
                    })
                    .and_then(|(col, table)| {
                        Some(Expr::ColRef {
                            col: col.clone(),
                            ref_table: table.clone(),
                        })
                    })
                    .ok_or(ExprCreateError::ColDoesNotExist);
                result
            }
            ast::Expr::Wildcard => Ok(Expr::Wildcard {
                ref_table: context.ref_table.clone(),
            }),
            ast::Expr::QualifiedWildcard(idents) => {
                assert!(idents.len() == 1);
                Ok(Expr::Wildcard {
                    ref_table: context
                        .aliases
                        .get(&idents[0])
                        .unwrap_or(&context.ref_table)
                        .clone(),
                })
            }
            ast::Expr::CompoundIdentifier(idents) => {
                assert!(idents.len() == 2);

                fn try_find_col<'a>(
                    table: &'a Rc<dyn Table>,
                    col_name: &str,
                    table_name: &str,
                ) -> Option<(Rc<FileCol>, &'a Rc<dyn Table>)> {
                    table
                        .find_col(col_name, Some(table_name))
                        .and_then(|col| Some((col, table)))
                }

                let result = context
                    .aliases
                    .get(&idents[0])
                    .and_then(|table| try_find_col(table, &idents[1], &idents[0]))
                    .or_else(|| {
                        if context.ref_table.contains_subtable(&idents[0]) {
                            try_find_col(&context.ref_table, &idents[1], &idents[0])
                        } else {
                            for table in &context.other_tables {
                                let col = table.find_col(&idents[1], Some(&idents[0]));
                                if let Some(col) = col {
                                    return Some((col, table));
                                }
                            }
                            None
                        }
                    })
                    .and_then(|(col, table)| {
                        Some(Expr::ColRef {
                            col,
                            ref_table: table.clone(),
                        })
                    })
                    .ok_or(ExprCreateError::ColDoesNotExist);
                result
            }
            ast::Expr::IsNull(subexpr) => Ok(Expr::IsNull(Box::new(Expr::new(subexpr, context)?))),
            ast::Expr::IsNotNull(subexpr) => {
                Ok(Expr::IsNotNull(Box::new(Expr::new(subexpr, context)?)))
            }
            ast::Expr::Value(val) => match val {
                ast::Value::Long(long) => Ok(Expr::Value(DBCol::Long(vec![*long as i64]))),
                ast::Value::Double(double) => {
                    Ok(Expr::Value(DBCol::Double(vec![double.into_inner()])))
                }
                ast::Value::SingleQuotedString(string) => {
                    Ok(Expr::Value(DBCol::Str(vec![string.to_string()])))
                }
                ast::Value::Boolean(boolean) => Ok(Expr::Value(DBCol::Bool(vec![*boolean]))),
                _ => {
                    panic!("Unknown type for val {:?}", val);
                }
            },
            ast::Expr::BinaryOp { .. } => make_binary_ops!(
                ast_expr, context, Plus, Minus, Multiply, Divide, Modulus, Gt, Lt, GtEq, LtEq, Eq,
                NotEq, Like, NotLike
            ),
            ast::Expr::UnaryOp { .. } => make_unary_ops!(ast_expr, context, Plus, Minus, Not),
            ast::Expr::Nested(expr) => Expr::new(expr, context),
            ast::Expr::Function(ast::Function { name, args, .. }) => {
                let name = name.to_string();
                //if name == "timezone" || name == "date_trunc" {
                //    return Err(ExprCreateError::UnimplementedFunc(name.to_string()));
                //}
                match &name[..] {
                    "extract_" | "timezone" | "array_agg" => {
                        return Err(ExprCreateError::UnimplementedFunc(name.to_string()));
                    }
                    _ => {}
                }

                let mut parsed_args = Vec::new();
                for arg in args {
                    let e = Expr::new(arg, context)?;
                    parsed_args.push(Box::new(e));
                }

                if name == "json_path_lookup" {
                    Ok(*parsed_args.swap_remove(0))
                } else if name == "round" {
                    Ok(Expr::Function {
                        name,
                        args: vec![parsed_args.swap_remove(0)],
                    })
                } else {
                    Ok(Expr::Function {
                        name,
                        args: parsed_args,
                    })
                }
            }
            ast::Expr::Case {
                conditions,
                results,
                else_result,
                ..
            } => {
                assert!(conditions.len() == 1, "more than 1 case condition");
                assert!(results.len() == 1, "more than 1 case result");
                let cond = Expr::new(&conditions[0], context)?;
                let then = Expr::new(&results[0], context)?;
                let else_ = else_result
                    .as_ref()
                    .and_then(|e| Some(Expr::new(e, context)))
                    .transpose()?;
                Ok(Expr::Case {
                    cond: Box::new(cond),
                    then: Box::new(then),
                    else_: else_.and_then(|e| Some(Box::new(e))),
                })
            }
            // TODO Implement in list/between in future
            ast::Expr::InList { .. } | ast::Expr::Between { .. } => {
                Ok(Expr::Value(DBCol::Bool(vec![true])))
            }
            _ => {
                panic!("Expression not supported {:?}", ast_expr);
            }
        }
    }

    fn eval_binary_op(
        &self,
        left: &Box<Expr>,
        right: &Box<Expr>,
        op: &BinaryOperator,
        run_context: &RunContext,
        exec_stats: &mut ExecStats,
    ) -> DBResult {
        let mut left_result = left.eval(run_context, exec_stats);
        let mut right_result = right.eval(run_context, exec_stats);

        for (group, col) in left_result.cols.iter_mut() {
            let other_col = right_result
                .cols
                .get_mut(group)
                .expect("other col doesn't have some group that col has");
            if col.len() < other_col.len() && col.len() == 1 {
                col.repeat(other_col.len());
            } else if col.len() > other_col.len() && other_col.len() == 1 {
                other_col.repeat(col.len());
            } else if col.len() == 0 || other_col.len() == 0 {
                // Handled in the do_*_op macros.
            } else if col.len() != other_col.len() {
                panic!(
                    "col and other_col are not same length\ncol: {:?}\n**\nother_col: {:?}",
                    col, other_col
                );
            }

            *col = match op {
                BinaryOperator::Plus => do_plus_op!(&col, &other_col, +),
                BinaryOperator::Minus => do_minus_op!(&col, &other_col),
                BinaryOperator::Multiply => do_number_op!(&col, &other_col, *),
                BinaryOperator::Divide => do_number_op!(&col, &other_col, /),
                BinaryOperator::Modulus => do_integral_op!(&col, &other_col, %),
                BinaryOperator::Gt => do_bool_op!(&col, &other_col, >),
                BinaryOperator::Lt => do_bool_op!(&col, &other_col, <),
                BinaryOperator::GtEq => do_bool_op!(&col, &other_col, >=),
                BinaryOperator::LtEq => do_bool_op!(&col, &other_col, <=),
                BinaryOperator::Eq => do_bool_op!(&col, &other_col, ==),
                BinaryOperator::NotEq => do_bool_op!(&col, &other_col, !=),
                BinaryOperator::Like => {
                    if let (DBCol::Str(xvals), DBCol::Str(yvals)) = (&col, other_col) {
                        DBCol::Bool(
                            xvals
                                .iter()
                                .zip(yvals.iter())
                                .map(|(x, y)| {
                                    let re = Regex::new(&utils::sql_pattern_to_regex(y)).unwrap();
                                    re.is_match(x)
                                })
                                .collect(),
                        )
                    } else {
                        panic!("one of the types is not a string for like");
                    }
                }
                BinaryOperator::NotLike => {
                    if let (DBCol::Str(xvals), DBCol::Str(yvals)) = (&col, other_col) {
                        DBCol::Bool(
                            xvals
                                .iter()
                                .zip(yvals.iter())
                                .map(|(x, y)| {
                                    let re = Regex::new(&utils::sql_pattern_to_regex(y)).unwrap();
                                    re.is_match(x)
                                })
                                .collect(),
                        )
                    } else {
                        panic!("one of the types is not a string for not like");
                    }
                }
            }
        }
        left_result
    }

    fn eval_unary_op(
        &self,
        expr: &Box<Expr>,
        op: &UnaryOperator,
        run_context: &RunContext,
        exec_stats: &mut ExecStats,
    ) -> DBResult {
        let mut result = expr.eval(run_context, exec_stats);
        for (_, col) in result.cols.iter_mut() {
            match op {
                UnaryOperator::Plus => {}
                UnaryOperator::Minus => {
                    *col = match col {
                        DBCol::Int(vals) => DBCol::Int(vals.iter().map(|x| -*x).collect()),
                        DBCol::Long(vals) => DBCol::Long(vals.iter().map(|x| -*x).collect()),
                        DBCol::Float(vals) => DBCol::Float(vals.iter().map(|x| -*x).collect()),
                        DBCol::Double(vals) => DBCol::Double(vals.iter().map(|x| -*x).collect()),
                        _ => {
                            panic!("minus of non-number");
                        }
                    };
                }
                UnaryOperator::Not => {
                    *col = match col {
                        DBCol::Bool(vals) => DBCol::Bool(vals.iter().map(|x| !*x).collect()),
                        _ => {
                            panic!("val is not bool type for NOT");
                        }
                    };
                }
            }
        }
        result
    }

    // TODO Implement cases - right now it just returns the then result.
    // TODO Redo this logic so you evaluate the case col first, then build bitmaps out of yes/no,
    // and then fetch then/else based on those values
    #[allow(unreachable_code, unused_variables)]
    fn eval_case(
        &self,
        cond: &Box<Expr>,
        then: &Box<Expr>,
        else_: &Option<Box<Expr>>,
        run_context: &RunContext,
        exec_stats: &mut ExecStats,
    ) -> DBResult {
        return then.eval(run_context, exec_stats);

        macro_rules! do_case_then_else {
            ($cond_vals:expr, $then_vals:expr, $else_vals:expr) => {
                $cond_vals
                    .iter()
                    .enumerate()
                    .map(|(i, cond_val)| {
                        if *cond_val {
                            $then_vals[i].clone()
                        } else {
                            $else_vals[i].clone()
                        }
                    })
                    .collect()
            };
        }

        // TODO - FINISH HERE
        unreachable!();
        let mut cond_result = cond.eval(run_context, exec_stats);
        let then_result = then.eval(run_context, exec_stats);
        assert!(else_.is_some());
        let else_result = else_.as_ref().unwrap().eval(run_context, exec_stats);
        for (group, col) in cond_result.cols.iter_mut() {
            let cond_vals;
            if let DBCol::Bool(cond) = col {
                cond_vals = cond
            } else {
                panic!("cond is not of bool type");
            }
            let then_col = &then_result.cols[group];
            let else_col = &else_result.cols[group];
            *col = if then_col.is_number_type() && else_col.is_number_type() {
                match (then_col, else_col) {
                    (DBCol::Int(then_vals), DBCol::Int(else_vals)) => {
                        DBCol::Int(do_case_then_else!(cond_vals, then_vals, else_vals))
                    }
                    (DBCol::Long(then_vals), DBCol::Long(else_vals)) => {
                        DBCol::Long(do_case_then_else!(cond_vals, then_vals, else_vals))
                    }
                    (DBCol::Float(then_vals), DBCol::Float(else_vals)) => {
                        DBCol::Float(do_case_then_else!(cond_vals, then_vals, else_vals))
                    }
                    _ => {
                        if then_col.is_integral_type() && else_col.is_integral_type() {
                            let then_vals: Vec<i64> = then_col.iter_as_long().collect();
                            let else_vals: Vec<i64> = else_col.iter_as_long().collect();
                            DBCol::Long(do_case_then_else!(cond_vals, then_vals, else_vals))
                        } else {
                            let then_vals: Vec<f64> = then_col.double_iter().collect();
                            let else_vals: Vec<f64> = else_col.double_iter().collect();
                            DBCol::Double(do_case_then_else!(cond_vals, then_vals, else_vals))
                        }
                    }
                }
            } else if let (DBCol::Str(then_vals), DBCol::Str(else_vals)) = (then_col, else_col) {
                DBCol::Str(do_case_then_else!(cond_vals, then_vals, else_vals))
            } else if let (DBCol::Bool(then_vals), DBCol::Bool(else_vals)) = (then_col, else_col) {
                DBCol::Bool(do_case_then_else!(cond_vals, then_vals, else_vals))
            } else {
                panic!("then and else are incompatible");
            }
        }
        cond_result
    }

    // XXX We  might want this if we reorder the table join order
    // pub fn set_ref_table

    pub fn eval(&self, run_context: &RunContext, exec_stats: &mut ExecStats) -> DBResult {
        debug!("[Expr] Evaluating {}", self);

        match self {
            Expr::ColRef { col, ref_table } => {
                if run_context.ref_table.is_some() {
                    col.read(
                        run_context,
                        exec_stats,
                        run_context.ref_table.as_ref().unwrap(),
                    )
                } else {
                    col.read(run_context, exec_stats, ref_table)
                }
            }
            Expr::Wildcard { ref_table } => {
                let all_cols = ref_table.all_cols();
                let mut all_cols = all_cols.clone();
                all_cols.sort_by_key(|col| col.name().to_string());
                let col = all_cols
                    .iter()
                    .next()
                    .expect(&format!("Table has no cols {}", ref_table.name())[..]);
                col.read(run_context, exec_stats, ref_table)
            }
            // FIXME - For now, let's not worry about nulls
            Expr::IsNull(subexpr) => {
                let mut result = subexpr.eval(run_context, exec_stats);
                for (_, col) in result.cols.iter_mut() {
                    *col = DBCol::Bool(vec![false; col.len()]);
                }
                result
            }
            // FIXME - For now, let's not worry about nulls
            Expr::IsNotNull(subexpr) => {
                let mut result = subexpr.eval(run_context, exec_stats);
                for (_, col) in result.cols.iter_mut() {
                    *col = DBCol::Bool(vec![true; col.len()]);
                }
                result
            }
            Expr::BinaryOp { left, right, op } => {
                self.eval_binary_op(left, right, op, run_context, exec_stats)
            }
            Expr::UnaryOp { expr, op } => self.eval_unary_op(expr, op, run_context, exec_stats),
            Expr::Nested(expr) => expr.eval(run_context, exec_stats),
            Expr::Value(val) => {
                if run_context.groups.is_empty() {
                    DBResult {
                        cols: vec![(vec![], val.clone())].into_iter().collect(),
                        ref_table: None,
                    }
                } else {
                    DBResult {
                        cols: run_context
                            .groups
                            .iter()
                            .map(|g| (g.clone(), val.clone()))
                            .collect(),
                        ref_table: None,
                    }
                }
            }
            Expr::Function { name, args } => {
                // XXX We make an assumption that aggregate functions only appear in the projection
                // (i.e., no with clause for now)
                let arg_results = args
                    .iter()
                    .map(|e| e.eval(run_context, exec_stats))
                    .collect();
                //let arg_results = args
                //    .iter()
                //    .map(|e| {
                //        let mut result = e.eval(run_context);
                //        if !run_context.groups.is_empty() && result.is_ungrouped() {
                //            result.group(&run_context.groups);
                //        }
                //        result
                //    })
                //    .collect();
                if !FUNC_MAP.contains_key(&name[..]) {
                    panic!("Have not implemented func {}", name);
                }
                FUNC_MAP[&name[..]](arg_results)
            }
            Expr::Case { cond, then, else_ } => {
                self.eval_case(cond, then, else_, run_context, exec_stats)
            }
        }
    }

    // Get all columns referenced in this expression.
    pub fn get_all_cols(&self) -> Vec<Rc<FileCol>> {
        match self {
            Expr::ColRef { col, .. } => vec![col.clone()],
            Expr::IsNull(subexpr)
            | Expr::IsNotNull(subexpr)
            | Expr::Nested(subexpr)
            | Expr::UnaryOp { expr: subexpr, .. } => subexpr.get_all_cols(),
            Expr::Wildcard { .. } => {
                panic!("No");
            }
            Expr::BinaryOp { left, right, .. } => {
                let mut left_cols = left.get_all_cols();
                let mut right_cols = right.get_all_cols();
                left_cols.append(&mut right_cols);
                left_cols
            }
            Expr::Value(_) => vec![],
            Expr::Function { args, .. } => {
                let mut ret = vec![];
                for arg in args {
                    ret.append(&mut arg.get_all_cols());
                }
                ret
            }
            Expr::Case { cond, then, else_ } => {
                let mut ret = cond.get_all_cols();
                ret.append(&mut then.get_all_cols());
                if else_.is_some() {
                    ret.append(&mut else_.as_ref().unwrap().get_all_cols());
                }
                ret
            }
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::ColRef { col, .. } => {
                write!(f, "{}.{}", col.table.upgrade().unwrap().name(), col.name())
            }
            Expr::Wildcard { .. } => write!(f, "*"),
            Expr::IsNull(subexpr) => write!(f, "{} IS NULL", subexpr),
            Expr::IsNotNull(subexpr) => write!(f, "{} IS NOT NULL", subexpr),
            Expr::BinaryOp { left, right, op } => {
                let op = match op {
                    BinaryOperator::Plus => "+",
                    BinaryOperator::Minus => "-",
                    BinaryOperator::Multiply => "*",
                    BinaryOperator::Divide => "/",
                    BinaryOperator::Modulus => "%",
                    BinaryOperator::Gt => ">",
                    BinaryOperator::Lt => "<",
                    BinaryOperator::GtEq => ">=",
                    BinaryOperator::LtEq => "<=",
                    BinaryOperator::Eq => "=",
                    BinaryOperator::NotEq => "!=",
                    BinaryOperator::Like => "LIKE",
                    BinaryOperator::NotLike => "NOT LIKE",
                };
                write!(f, "{} {} {}", left, op, right)
            }
            Expr::UnaryOp { expr, op } => {
                let op = match op {
                    UnaryOperator::Plus => "+",
                    UnaryOperator::Minus => "-",
                    UnaryOperator::Not => "NOT ",
                };
                write!(f, "{}{}", op, expr)
            }
            Expr::Nested(subexpr) => write!(f, "({})", subexpr),
            Expr::Value(col) => match col {
                DBCol::Str(vals) => write!(f, "'{}'", vals[0]),
                _ => write!(f, "{:?}", col.any().unwrap()),
            },
            Expr::Function { name, args } => {
                let args: Vec<String> = args.iter().map(|arg| arg.to_string()).collect();
                write!(f, "{}({})", name, args.join(", "))
            }
            Expr::Case { cond, then, else_ } => {
                let else_ = else_
                    .as_ref()
                    .and_then(|e| Some(format!(" ELSE {}", e)))
                    .unwrap_or("".to_string());
                write!(f, "CASE WHEN {} THEN {}{} END", cond, then, else_)
            }
        }
    }
}
