use chameleon::{
    config::READ_BUF_SIZE, ApproxOptType, ExecParams, ExecStats, Executor, Parser, DB,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;
use structopt::StructOpt;
use uuid::Uuid;

const DROP_CACHES_CMD: &str = "drop_caches";

const EXP_LOG_VERSION: &str = "0.1";

const DEFAULT_SLEEP_TIME: u64 = 1;

#[derive(Debug, StructOpt, Clone)]
#[structopt(name = "run-forest-exp", about = "Chameleon DB system.")]
pub struct Opt {
    #[structopt(short, long)]
    debug: bool,

    #[structopt(long)]
    no_bestd: bool,

    #[structopt(short = "t", default_value = "1")]
    num_trials: u32,

    #[structopt(parse(from_os_str))]
    queries_dir: PathBuf,

    #[structopt(parse(from_os_str))]
    db_path: PathBuf,

    #[structopt(parse(from_os_str))]
    output_path: PathBuf,
}

fn drop_caches() {
    Command::new("sh")
        .arg("-c")
        .arg(DROP_CACHES_CMD)
        .output()
        .expect("Failed to drop caches");
}

#[derive(Serialize, Deserialize)]
struct ExpItem {
    id: Uuid,
    timestamp: DateTime<Utc>,
    data: ExpData,
    description: String,
    version: String,
    no_results: bool,
}

#[derive(Serialize, Deserialize)]
struct ExpData {
    alg: String,
    pred_expr: String,
    depth: u32,
    sleep_time: u64,
    read_buf_size: usize,
    db_path: String,
    num_preds: usize,
    uniform_cost: bool,
    selectivities: Vec<(String, f64)>,
    costs: Vec<(String, u32)>,
    trials: Vec<TrialData>,
}

#[derive(Serialize, Deserialize)]
struct TrialData {
    total_runtime_ms: u128,
    pred_only_time_ms: u128,
    num_preds_evaled: u128,
    plan_time_ms: u128,
}

fn run_exp(opt: &Opt, params: ExpParams) -> ExpItem {
    let mut trials = vec![];

    let db = DB::new(&opt.db_path);
    let mut exec = Executor::new(
        &db,
        Some(params.selectivities.clone()),
        Some(
            params
                .costs
                .iter()
                .map(|(k, v)| (k.clone(), *v as f64))
                .collect(),
        ),
    );
    let parser = Parser::new(&db);
    let query_str = format!("select area_1 from forest where ({});", params.pred_expr);

    let bdc_result;

    drop_caches();
    let now = Instant::now();
    let query = parser.parse(&query_str, &Default::default());
    if query.is_err() {
        panic!("Cannot parse query: {}", query_str);
    } else {
        let mut exec_params: ExecParams = Default::default();
        exec_params.approx_opt_type = if opt.no_bestd {
            ApproxOptType::BDC
        } else {
            ApproxOptType::BDCWithBestD
        };
        let mut exec_stats = ExecStats::new();
        bdc_result = exec.run(query.unwrap(), &exec_params, &mut exec_stats);
        trials.push(TrialData {
            total_runtime_ms: now.elapsed().as_millis(),
            pred_only_time_ms: exec_stats.pred_only_time_ms,
            plan_time_ms: exec_stats.plan_time_ms,
            num_preds_evaled: exec_stats.num_preds_evaled,
        });
    }

    let query = parser.parse(&query_str, &Default::default());
    if query.is_err() {
        panic!("Cannot parse query: {}", query_str);
    } else {
        let exec_params: ExecParams = Default::default();
        let mut exec_stats = ExecStats::new();
        let result = exec.run(query.unwrap(), &exec_params, &mut exec_stats);
        assert_eq!(result, bdc_result);
    }

    ExpItem {
        id: Uuid::new_v4(),
        timestamp: Utc::now(),
        description: "BDC".to_string(),
        version: EXP_LOG_VERSION.to_string(),
        data: ExpData {
            alg: if opt.no_bestd {
                "bdc".to_string()
            } else {
                "bdc-bestd".to_string()
            },
            pred_expr: params.pred_expr.to_string(),
            depth: params.depth,
            sleep_time: DEFAULT_SLEEP_TIME,
            read_buf_size: READ_BUF_SIZE,
            db_path: opt.db_path.to_str().unwrap().to_string(),
            num_preds: params.selectivities.len(),
            uniform_cost: params.uniform_cost,
            selectivities: params.selectivities.into_iter().collect(),
            costs: params.costs.into_iter().collect(),
            trials,
        },
        no_results: false,
    }
}

struct ExpParams {
    pred_expr: String,
    depth: u32,
    selectivities: HashMap<String, f64>,
    costs: HashMap<String, u32>,
    uniform_cost: bool,
}

#[derive(Serialize, Deserialize)]
struct QueryItem {
    depth: u32,
    uniform_cost: bool,
    pred_exprs: String,
    selectivities: Vec<(String, f64)>,
    costs: Vec<(String, u32)>,
}

fn read_queries(
    queries_dir: &Path,
    uniform_cost: bool,
    depth: u32,
) -> Result<Vec<QueryItem>, Box<dyn Error>> {
    let path = queries_dir.join(format!(
        "queries-{}-depth{}.json",
        if uniform_cost { "uniform" } else { "varcost" },
        depth
    ));
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let parsed = serde_json::from_reader(reader)?;
    Ok(parsed)
}

fn main() {
    let opt = Opt::from_args();

    let mut exp_log = vec![];
    for depth in [2, 3, 4] {
        for uniform_cost in [true, false] {
            let queries = read_queries(&opt.queries_dir, uniform_cost, depth).unwrap();
            for (i, query) in queries.into_iter().enumerate() {
                print!("Starting depth: {depth} uniform_cost: {uniform_cost} query: {i} ... ");
                io::stdout().flush().unwrap();

                let item = run_exp(
                    &opt,
                    ExpParams {
                        pred_expr: query.pred_exprs,
                        depth,
                        uniform_cost,
                        selectivities: query.selectivities.into_iter().collect(),
                        costs: query.costs.into_iter().collect(),
                    },
                );

                exp_log.push(item);
                let writer = File::create(&opt.output_path)
                    .ok()
                    .map(|file| BufWriter::new(file))
                    .unwrap();
                serde_json::to_writer_pretty(writer, &exp_log).unwrap();

                println!("Done!");
            }
        }
    }
}
