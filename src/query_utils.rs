use super::exec::{ExecStats, RunContext};
use super::expr::{DBCol, DBResult};
use super::parser::PredNode;
use super::table::{Id, Table};
use chrono::{DateTime, Duration, Utc};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

pub fn estimate_selectivities(node: &PredNode, selectivities: &mut HashMap<String, f64>) {
    match node {
        PredNode::AndNode(children) => {
            for child in children {
                estimate_selectivities(child, selectivities);
            }
        }
        PredNode::OrNode(children) => {
            for child in children {
                estimate_selectivities(child, selectivities);
            }
        }
        PredNode::PredAtomNode(node) => {
            if selectivities.contains_key(&node.expr.to_string()) {
                return;
            }

            let cols = node.expr.get_all_cols();
            assert!(
                cols.iter()
                    .map(|col| col.table.upgrade().unwrap().id())
                    .collect::<Vec<Id>>()
                    .windows(2)
                    .all(|w| w[0] == w[1]),
                "There is more than one table in this predicate: {:?}",
                cols.iter()
                    .map(|col| col.table.upgrade().unwrap().name().to_string())
                    .collect::<Vec<String>>()
            );

            if cols.is_empty() {
                return;
            }

            let first_col = cols.iter().next().unwrap();
            let table = first_col.table.upgrade().unwrap();
            let mut exec_stats = ExecStats::new();
            let result = node.expr.eval(
                &RunContext {
                    index: None,
                    groups: vec![],
                    ref_table: Some(table.clone()),
                    exec_params: Default::default(),
                },
                &mut exec_stats,
            );
            assert_eq!(result.cols.len(), 1);
            let col = result.cols.values().next().unwrap();
            if let DBCol::Bool(vals) = col {
                let selec = (vals.iter().filter(|v| **v).collect::<Vec<&bool>>().len() as f64)
                    / vals.len() as f64;
                selectivities.insert(node.expr.to_string(), selec);
            } else {
                panic!("Non-bool result? (e.g., {:?})", col.some(10));
            }
        }
    }
}

#[derive(Debug, PartialOrd, PartialEq, Clone)]
pub enum DBVal {
    Int(i32),
    Long(i64),
    Float(f32),
    Double(f64),
    Str(String),
    Bool(bool),
    DateTime(DateTime<Utc>),
    Duration(Duration),
}

impl Eq for DBVal {}

impl Hash for DBVal {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            DBVal::Float(f) => {
                f.to_bits().hash(state);
            }
            DBVal::Double(d) => {
                d.to_bits().hash(state);
            }
            DBVal::Int(i) => {
                i.hash(state);
            }
            DBVal::Long(l) => {
                l.hash(state);
            }
            DBVal::Bool(b) => {
                b.hash(state);
            }
            DBVal::Str(s) => {
                s.hash(state);
            }
            DBVal::DateTime(d) => {
                d.hash(state);
            }
            DBVal::Duration(d) => {
                d.hash(state);
            }
        }
    }
}

pub type ResultSet = HashMap<Vec<DBVal>, HashSet<Vec<DBVal>>>;

pub fn print_result_set(rs: &ResultSet, tag: &str) {
    println!("{}:", tag);
    let mut keys: Vec<&Vec<DBVal>> = rs.keys().collect();
    keys.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    for key in keys {
        println!("    {:?}: {{", key);
        let mut vals: Vec<&Vec<DBVal>> = rs[key].iter().collect();
        vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        for val in vals {
            println!("        {:?}", val);
        }
        println!("    }}");
    }
}

// Returns as a hash map of rows
pub fn process_dbresults(results: Vec<DBResult>) -> ResultSet {
    assert!(!results.is_empty());

    let mut parsed_results: HashMap<Vec<DBVal>, HashSet<Vec<DBVal>>> = HashMap::new();
    let groups = results[0].cols.keys();
    //for (i, result) in results.iter().enumerate() {
    //    eprintln!("{} {:?}", i, result);
    //}
    for group in groups {
        let mut rows = HashSet::new();
        let col = results[0].cols.get(group).unwrap();
        for i in 0..col.len() {
            let mut row = vec![];
            for result in &results {
                let col = result.cols.get(group).expect("Col didn't have group");
                row.push(match col {
                    DBCol::Int(vals) => DBVal::Int(vals[i]),
                    DBCol::Long(vals) => DBVal::Long(vals[i]),
                    DBCol::Float(vals) => DBVal::Float(vals[i]),
                    DBCol::Double(vals) => DBVal::Double(vals[i]),
                    DBCol::Str(vals) => DBVal::Str(vals[i].clone()),
                    DBCol::Bool(vals) => DBVal::Bool(vals[i]),
                    DBCol::DateTime(vals) => DBVal::DateTime(vals[i]),
                    DBCol::Duration(vals) => DBVal::Duration(vals[i]),
                });
            }
            rows.insert(row);
        }
        let group = group
            .iter()
            .map(|col| match col {
                DBCol::Int(vals) => DBVal::Int(vals[0]),
                DBCol::Long(vals) => DBVal::Long(vals[0]),
                DBCol::Float(vals) => DBVal::Float(vals[0]),
                DBCol::Double(vals) => DBVal::Double(vals[0]),
                DBCol::Str(vals) => DBVal::Str(vals[0].clone()),
                DBCol::Bool(vals) => DBVal::Bool(vals[0]),
                DBCol::DateTime(vals) => DBVal::DateTime(vals[0]),
                DBCol::Duration(vals) => DBVal::Duration(vals[0]),
            })
            .collect();
        parsed_results.insert(group, rows);
    }
    parsed_results
}
