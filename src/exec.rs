use super::db::DB;
use super::expr::{DBCol, DBResult};
use super::parser::Query;
use super::table::Table;
use log::debug;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use std::cmp;
use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Instant;

// `index` stores the set of indices which are considered valid in the current context
pub struct RunContext {
    // This should be in reference to the `ref_table` found in `TableNode`.
    // If this is None, we want the entire range of values.
    pub index: Option<RoaringBitmap>,
    pub groups: Vec<Vec<DBCol>>,
    // XXX Passing a `ref_table` table will make the expressions evaluate according to this table. This
    // should only be used in special debugging circumstances.
    pub ref_table: Option<Rc<dyn Table>>,
    pub exec_params: ExecParams,
}

pub struct Executor<'a> {
    db: &'a DB,
    selectivities: Rc<HashMap<String, f64>>,
    costs: Rc<HashMap<String, f64>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ApproxOptType {
    NoApproxOpt,
    OnePredLookahead,
    WeightedPartial,
    Byp,
    Tdacb,
    BDC,
    BDCWithBestD,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ExecParams {
    pub disable_or_opt: bool,
    pub approx_opt_type: ApproxOptType,
    pub extra_data_retrieval_latency: u32,
    // XXX this may or may not be able to calculated beforehand based on how often the type of join
    // changes left/inner
    pub special_case_one_multi_table_clause: bool, // If we have a case-2 query, and only one of the clauses is multi-table, we can special case it to make it just perform conjoin/disjoin after propagating all the other clauses
    pub dont_use_centroids: bool, // This is equivalent special casing every multi-table clause
    pub include_debug_info: bool,
    pub check_plan_only: bool, // used to check ordering of bdc plans
}

impl Default for ExecParams {
    fn default() -> Self {
        Self {
            disable_or_opt: false,
            approx_opt_type: ApproxOptType::NoApproxOpt,
            extra_data_retrieval_latency: 0,
            special_case_one_multi_table_clause: false,
            dont_use_centroids: false,
            include_debug_info: false,
            check_plan_only: false,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ExecStats {
    pub total_time_ms: u128,
    pub pred_only_time_ms: u128,
    pub num_preds_evaled: u128,
    pub num_plans_considered: u128,
    pub plan_time_ms: u128,
    pub num_bufs_read: u128,
    pub num_joined_records: u128,
    pub num_final_records: u128,
    pub num_synthesized_records: u128, // includes all intermediate records which are created that may not be as a result in the conjoin/disjoin functions
}

impl ExecStats {
    pub fn new() -> Self {
        ExecStats {
            total_time_ms: 0,
            pred_only_time_ms: 0,
            num_preds_evaled: 0,
            num_plans_considered: 0,
            plan_time_ms: 0,
            num_bufs_read: 0,
            num_joined_records: 0,
            num_final_records: 0,
            num_synthesized_records: 0,
        }
    }
}

pub fn read_selectivities(path: &PathBuf) -> HashMap<String, f64> {
    let mut selectivities = HashMap::new();
    let selec_file = BufReader::new(File::open(path).unwrap());
    for line in selec_file.lines() {
        let line = line.unwrap();
        let vals: Vec<&str> = line.trim().splitn(2, ',').collect();
        let (selec, expr) = (vals[0], vals[1]);
        selectivities.insert(expr.to_string(), selec.parse::<f64>().unwrap());
    }
    selectivities
}

impl<'a> Executor<'a> {
    pub fn new(
        db: &'a DB,
        selectivities: Option<HashMap<String, f64>>,
        costs: Option<HashMap<String, f64>>,
    ) -> Self {
        let selectivities = Rc::new(selectivities.unwrap_or(HashMap::new()));
        let costs = Rc::new(costs.unwrap_or(HashMap::new()));
        Executor {
            db,
            selectivities,
            costs,
        }
    }

    pub fn run(
        &mut self,
        mut query: Query,
        exec_params: &ExecParams,
        exec_stats: &mut ExecStats,
    ) -> Vec<DBResult> {
        let total_time_beg = Instant::now();

        debug!("PRINTING QUERY\n{}", query);

        query
            .filter
            .as_mut()
            .and_then(|node| Some(node.set_selec_map(&self.selectivities, &self.costs)));

        let mut run_context = RunContext {
            index: None,
            groups: vec![],
            ref_table: None,
            exec_params: exec_params.clone(),
        };

        debug!("EVALUATING JOIN");
        query
            .table
            .table
            .eval_join(exec_params, exec_stats, query.filter.as_ref());
        let data_num = query.table.table.len();
        debug!("EVALUATING FILTER");
        let now = Instant::now();
        run_context.index = query.filter.as_mut().and_then(|pred| {
            Some(pred.eval(&(0..data_num as u32).collect(), exec_params, exec_stats))
        });
        exec_stats.pred_only_time_ms = now.elapsed().as_millis();

        if let Some(index) = &run_context.index {
            if index.is_empty() {
                exec_stats.num_final_records = 0;
                return vec![];
            }
        }

        debug!("EVALUATING GROUP BY");
        run_context.groups = query.group_by.eval(&run_context, exec_stats);
        debug!(
            "EVALED GROUP BYS:\n{}",
            run_context.groups[..cmp::min(10, run_context.groups.len())]
                .iter()
                .map(|groups| format!(
                    "{}({})",
                    " ".repeat(8),
                    groups
                        .iter()
                        .map(|col| format!("{:?}", col.any()))
                        .collect::<Vec<String>>()
                        .join(", ")
                ))
                .collect::<Vec<String>>()
                .join("\n")
        );
        debug!("EVALUATING PROJECTION");
        let results = query.projection.eval(&run_context, exec_stats);
        exec_stats.total_time_ms = total_time_beg.elapsed().as_millis();
        exec_stats.num_final_records = results[0].len() as u128;
        results
    }
}
