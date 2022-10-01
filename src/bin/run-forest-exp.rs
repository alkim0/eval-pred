use chameleon::config::READ_BUF_SIZE;
use chameleon::{
    ApproxOptType, DBCol, ExecParams, ExecStats, Executor, Id, Parser, PredNode, RunContext, DB,
};
use chrono::{DateTime, Local, Utc};
use log::debug;
use rand::{seq::SliceRandom, Rng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;
use structopt::StructOpt;
use uuid::Uuid;

const DROP_CACHES_CMD: &str = "drop_caches";

const EXP_LOG_VERSION: &str = "0.1";

const DEFAULT_SLEEP_TIME: u64 = 1;
const DEFAULT_SELECTIVITY: f32 = 0.2;

#[derive(Debug, StructOpt, Clone)]
#[structopt(name = "run-forest-exp", about = "Chameleon DB system.")]
pub struct Opt {
    #[structopt(short, long)]
    debug: bool,

    #[structopt(short = "t", default_value = "1")]
    num_trials: u32,

    #[structopt(short = "n", default_value = "1")]
    num_queries: u32,

    //#[structopt(short, long)]
    //print_results: bool,
    //#[structopt(short, long)]
    //uniform_cost: bool,
    #[structopt(long)]
    no_tdacb: bool,

    #[structopt(long)]
    dont_skip_no_results: bool,

    #[structopt(
        short,
        long = "preds",
        parse(from_os_str),
        default_value = "data/forest-preds"
    )]
    preds_path: PathBuf,

    //#[structopt(long)]
    //selectivity_only: bool,

    //#[structopt(long)]
    //sleep_time_only: bool,
    #[structopt(short, long, parse(from_os_str))]
    output_path: Option<PathBuf>,

    #[structopt(parse(from_os_str))]
    db_path: PathBuf,
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
enum ExpData {
    ForestData(ForestData),
    NoData,
}

#[derive(Serialize, Deserialize)]
struct ForestData {
    opt_runtimes_ms: Vec<u128>,
    no_opt_runtimes_ms: Vec<u128>,
    approx_opt_runtimes_ms: Vec<u128>,
    tdacb_runtimes_ms: Vec<u128>,
    pred_only_opt_runtimes_ms: Vec<u128>,
    pred_only_no_opt_runtimes_ms: Vec<u128>,
    pred_only_approx_opt_runtimes_ms: Vec<u128>,
    pred_tdacb_runtimes_ms: Vec<u128>,
    opt_num_preds_evaled: Vec<u128>,
    no_opt_num_preds_evaled: Vec<u128>,
    approx_opt_num_preds_evaled: Vec<u128>,
    tdacb_num_preds_evaled: Vec<u128>,
    tdacb_plan_times: Vec<u128>,
    sleep_time: u64,
    read_buf_size: usize,
    db_path: String,
    pred_exprs: Vec<String>,
    depths: Vec<u32>,
    num_preds: Vec<usize>,
    selectivities: Vec<Vec<(String, f64)>>,
    uniform_cost: bool,
    costs: Vec<Vec<(String, u32)>>,
}

fn run_exp(
    opt: &Opt,
    pred_expr: &str,
    selectivities: HashMap<String, f64>,
    depth: u32,
    costs: HashMap<String, u32>,
    no_tdacb: bool,
    uniform_cost: bool,
) -> ExpItem {
    let mut opt_times = vec![];
    let mut no_opt_times = vec![];
    let mut approx_opt_times = vec![];
    let mut tdacb_times = vec![];
    let mut pred_opt_times = vec![];
    let mut pred_no_opt_times = vec![];
    let mut pred_approx_opt_times = vec![];
    let mut pred_tdacb_times = vec![];
    let mut opt_num_preds_evaled = vec![];
    let mut no_opt_num_preds_evaled = vec![];
    let mut approx_opt_num_preds_evaled = vec![];
    let mut tdacb_num_preds_evaled = vec![];
    let mut tdacb_plan_times = vec![];
    let mut pred_exprs = vec![];
    let mut select_vecs = vec![];
    let mut depths = vec![];
    let mut num_preds = vec![];
    let mut cost_vecs = vec![];

    for _ in 0..opt.num_trials {
        pred_exprs.push(pred_expr.to_string());
        let mut selec_vec = selectivities
            .iter()
            .map(|(expr, s)| (expr.clone(), *s))
            .collect::<Vec<(String, f64)>>();
        selec_vec.sort_by_key(|(expr, _)| expr.clone());
        select_vecs.push(selec_vec);
        let mut cost_vec = costs
            .iter()
            .map(|(expr, c)| (expr.clone(), *c))
            .collect::<Vec<(String, u32)>>();
        cost_vec.sort_by_key(|(expr, _)| expr.clone());
        cost_vecs.push(cost_vec);

        depths.push(depth);
        num_preds.push(selectivities.len());

        let no_opt_result;
        let opt_result;
        let approx_result;
        let mut tdacb_result = None;

        let db = DB::new(&opt.db_path);
        let mut exec = Executor::new(
            &db,
            Some(selectivities.clone()),
            Some(costs.iter().map(|(k, v)| (k.clone(), *v as f64)).collect()),
        );
        let parser = Parser::new(&db);

        let query_str = format!("select area_1 from forest where ({});", pred_expr);

        drop_caches();
        let now = Instant::now();
        debug!("*** PARSING QUERY ***");
        let query = parser.parse(&query_str, &Default::default());
        if query.is_err() {
            panic!("Cannot parse query: {}", query_str);
        } else {
            let mut exec_params: ExecParams = Default::default();
            exec_params.disable_or_opt = true;
            let mut exec_stats = ExecStats::new();
            no_opt_result = exec.run(query.unwrap(), &exec_params, &mut exec_stats);
            if !opt.dont_skip_no_results && exec_stats.num_final_records == 0 {
                return ExpItem {
                    id: Uuid::new_v4(),
                    timestamp: Utc::now(),
                    description: "Varying data latency".to_string(),
                    version: EXP_LOG_VERSION.to_string(),
                    data: ExpData::NoData,
                    no_results: true,
                };
            }
            no_opt_times.push(now.elapsed().as_millis());
            pred_no_opt_times.push(exec_stats.pred_only_time_ms);
            no_opt_num_preds_evaled.push(exec_stats.num_preds_evaled);
        }
        println!(
            "done with no-opt, num_preds: {}, depth {}",
            selectivities.len(),
            depth
        );

        drop_caches();
        let now = Instant::now();
        debug!("*** PARSING QUERY ***");
        let query = parser.parse(&query_str, &Default::default());
        if query.is_err() {
            panic!("Cannot parse query: {}", query_str);
        } else {
            let exec_params: ExecParams = Default::default();
            let mut exec_stats = ExecStats::new();
            opt_result = exec.run(query.unwrap(), &exec_params, &mut exec_stats);
            opt_times.push(now.elapsed().as_millis());
            pred_opt_times.push(exec_stats.pred_only_time_ms);
            opt_num_preds_evaled.push(exec_stats.num_preds_evaled);
        }
        println!(
            "done with opt, num_preds: {}, depth {}",
            selectivities.len(),
            depth
        );

        drop_caches();
        let now = Instant::now();
        debug!("*** PARSING QUERY ***");
        let query = parser.parse(&query_str, &Default::default());
        if query.is_err() {
            panic!("Cannot parse query: {}", query_str);
        } else {
            let mut exec_params: ExecParams = Default::default();
            exec_params.approx_opt_type = ApproxOptType::OnePredLookahead;
            let mut exec_stats = ExecStats::new();
            approx_result = exec.run(query.unwrap(), &exec_params, &mut exec_stats);
            approx_opt_times.push(now.elapsed().as_millis());
            pred_approx_opt_times.push(exec_stats.pred_only_time_ms);
            approx_opt_num_preds_evaled.push(exec_stats.num_preds_evaled);
        }
        println!(
            "done with approx, num_preds: {}, depth: {}",
            selectivities.len(),
            depth
        );

        if !no_tdacb {
            drop_caches();
            let now = Instant::now();
            debug!("*** PARSING QUERY ***");
            let query = parser.parse(&query_str, &Default::default());
            if query.is_err() {
                panic!("Cannot parse query: {}", query_str);
            } else {
                let mut exec_params: ExecParams = Default::default();
                exec_params.approx_opt_type = ApproxOptType::Tdacb;
                let mut exec_stats = ExecStats::new();
                tdacb_result = Some(exec.run(query.unwrap(), &exec_params, &mut exec_stats));
                tdacb_times.push(now.elapsed().as_millis());
                pred_tdacb_times.push(exec_stats.pred_only_time_ms);
                tdacb_num_preds_evaled.push(exec_stats.num_preds_evaled);
                tdacb_plan_times.push(exec_stats.plan_time_ms);
            }
            println!(
                "done with tdacb, num_preds: {}, depth: {}",
                selectivities.len(),
                depth
            );
        }

        //assert_eq!(opt_result, tdacb_result);
        assert_eq!(no_opt_result, opt_result);
        assert_eq!(opt_result, approx_result);
        if !no_tdacb {
            assert_eq!(approx_result, tdacb_result.unwrap());
        }
    }

    ExpItem {
        id: Uuid::new_v4(),
        timestamp: Utc::now(),
        description: "Varying data latency".to_string(),
        version: EXP_LOG_VERSION.to_string(),
        data: ExpData::ForestData(ForestData {
            opt_runtimes_ms: opt_times,
            no_opt_runtimes_ms: no_opt_times,
            approx_opt_runtimes_ms: approx_opt_times,
            tdacb_runtimes_ms: tdacb_times,
            pred_only_opt_runtimes_ms: pred_opt_times,
            pred_only_no_opt_runtimes_ms: pred_no_opt_times,
            pred_only_approx_opt_runtimes_ms: pred_approx_opt_times,
            pred_tdacb_runtimes_ms: pred_tdacb_times,
            opt_num_preds_evaled,
            no_opt_num_preds_evaled,
            tdacb_num_preds_evaled,
            approx_opt_num_preds_evaled,
            read_buf_size: READ_BUF_SIZE,
            db_path: opt.db_path.to_str().unwrap().to_string(),
            sleep_time: DEFAULT_SLEEP_TIME,
            pred_exprs,
            selectivities: select_vecs,
            costs: cost_vecs,
            tdacb_plan_times,
            depths,
            num_preds,
            uniform_cost,
        }),
        no_results: false,
    }
}

fn gen_pred_rand_children(
    level: u32,
    is_and: bool,
    rng: &mut impl Rng,
    attr_names: &mut Vec<String>,
    poss_preds: &HashMap<String, Vec<(String, u32, f64)>>,
    uniform_cost: bool,
) -> (String, HashMap<String, f64>, HashMap<String, u32>) {
    if attr_names.is_empty() {
        panic!("We ran out of attrs??");
    }
    if level == 0 {
        let attr_name = attr_names.remove(0);
        let vec: &Vec<(String, u32, f64)> = poss_preds.get(&attr_name).unwrap();
        //let (val, op_num, selec) = rng.choose(vec).unwrap();
        let (val, op_num, selec) = vec.choose(rng).unwrap();
        let sleep_time = if uniform_cost {
            1
        } else {
            rng.gen_range(1, 11)
        };
        let pred = format!(
            "forest_udf({}, {}, {}, {})",
            attr_name, val, op_num, sleep_time
        );
        (
            pred.clone(),
            vec![(pred.clone(), *selec)].into_iter().collect(),
            vec![(pred.clone(), sleep_time)].into_iter().collect(),
        )
    } else {
        let num_children = rng.gen_range(2, 6);
        let first_child = gen_pred_rand_children(
            level - 1,
            !is_and,
            rng,
            attr_names,
            poss_preds,
            uniform_cost,
        );
        let mut subexprs = (0..num_children - 1)
            .into_iter()
            .map(|i| {
                gen_pred_rand_children(
                    rng.gen_range(0, level),
                    !is_and,
                    rng,
                    attr_names,
                    poss_preds,
                    uniform_cost,
                )
            })
            .collect::<Vec<(String, HashMap<String, f64>, HashMap<String, u32>)>>();
        subexprs.insert(0, first_child);
        (
            subexprs
                .iter()
                .map(|(p, _, _)| format!("({})", p))
                .collect::<Vec<String>>()
                .join(if is_and { " and " } else { " or " }),
            subexprs.iter().fold(HashMap::new(), |acc, x| {
                acc.into_iter().chain(x.1.clone()).collect()
            }),
            subexprs.iter().fold(HashMap::new(), |acc, x| {
                acc.into_iter().chain(x.2.clone()).collect()
            }),
        )
    }
}

// Returns a pair of the query string and the corresponding selectivities
fn gen_pred_subexpr(
    level: u32,
    is_and: bool,
    rng: &mut impl Rng,
    idx: &mut usize,
    num_children: &Vec<usize>,
    attr_names: &mut Vec<String>,
    poss_preds: &HashMap<String, Vec<(String, u32, f64)>>,
) -> (String, HashMap<String, f64>) {
    if attr_names.is_empty() {
        panic!("We ran out of attrs??");
    }
    if level == 0 {
        let attr_name = attr_names.remove(0);
        let vec: &Vec<(String, u32, f64)> = poss_preds.get(&attr_name).unwrap();
        //let (val, op_num, selec) = rng.choose(vec).unwrap();
        let (val, op_num, selec) = vec.choose(rng).unwrap();
        let pred = format!("forest_udf({}, {}, {}, 0)", attr_name, val, op_num);
        (pred.clone(), vec![(pred, *selec)].into_iter().collect())
    } else {
        let nc = num_children[*idx];
        *idx += 1;
        let subexprs = (0..nc)
            .into_iter()
            .map(|i| {
                gen_pred_subexpr(
                    level - 1,
                    !is_and,
                    rng,
                    idx,
                    num_children,
                    attr_names,
                    poss_preds,
                )
            })
            .collect::<Vec<(String, HashMap<String, f64>)>>();
        (
            subexprs
                .iter()
                .map(|(p, _)| format!("({})", p))
                .collect::<Vec<String>>()
                .join(if is_and { " and " } else { " or " }),
            subexprs.into_iter().fold(HashMap::new(), |acc, x| {
                acc.into_iter().chain(x.1).collect()
            }),
        )
    }
}

// For return value, it's of the form {attr: [(val, op_num, selec)]
fn parse_preds_file(path: &PathBuf) -> HashMap<String, Vec<(String, u32, f64)>> {
    let mut preds = HashMap::new();
    let content = fs::read_to_string(path).unwrap();
    let lines = content.trim().split("\n");
    for line in lines {
        let tokens = line.split(",").collect::<Vec<&str>>();
        assert_eq!(tokens.len(), 4);
        let selec = tokens[0].parse::<f64>().unwrap();
        let attr = tokens[1].to_string();
        let val = tokens[2].to_string();
        let op_num = tokens[3].parse::<u32>().unwrap();
        let entry = preds.entry(attr);
        entry.or_insert(vec![]).push((val, op_num, selec));
    }
    preds
}

fn main() {
    let opt = Opt::from_args();

    let mut exp_log = vec![];

    let poss_preds = parse_preds_file(&opt.preds_path);

    let mut rng = rand::thread_rng();
    for depth in &[2, 3, 4] {
        for uniform_cost in &[true, false] {
            let mut query_idx = 0;
            while query_idx < opt.num_queries {
                let (pred_expr, selectivities, costs): (
                    String,
                    HashMap<String, f64>,
                    HashMap<String, u32>,
                ) = loop {
                    let mut attr_names = poss_preds.keys().cloned().collect();
                    let (pred_expr, selectivities, costs) = gen_pred_rand_children(
                        *depth,
                        rng.gen_range(0, 2) == 0,
                        &mut rng,
                        &mut attr_names,
                        &poss_preds,
                        *uniform_cost,
                    );
                    println!("num_preds: {}", selectivities.len());
                    //if *depth == 2 {
                    if 8 <= selectivities.len() && selectivities.len() <= 16 {
                        break (pred_expr, selectivities, costs);
                    }
                };
                println!(
                    "Starting query: {}, time: {}, expr: {}",
                    query_idx,
                    Local::now(),
                    &pred_expr
                );
                let exp_data = run_exp(
                    &opt,
                    &pred_expr,
                    selectivities,
                    *depth,
                    costs,
                    opt.no_tdacb,
                    //if opt.no_tdacb { true } else { *depth != 2 },
                    //false,
                    *uniform_cost,
                );
                if exp_data.no_results {
                    continue;
                }

                exp_log.push(exp_data);

                let stdout = io::stdout();
                let stdout = stdout.lock();
                let mut output = opt
                    .output_path
                    .as_ref()
                    .and_then(|path| {
                        File::create(path)
                            .ok()
                            .and_then(|file| Some(Box::new(file) as Box<dyn Write>))
                    })
                    .or_else(|| Some(Box::new(stdout)))
                    .expect("Could not get output");

                let s = serde_yaml::to_string(&exp_log);
                writeln!(output, "{}", s.expect("Trouble generating yaml"))
                    .expect("Trouble writing output");

                query_idx += 1;
            }
        }
    }

    let stdout = io::stdout();
    let stdout = stdout.lock();
    let mut output = opt
        .output_path
        .and_then(|path| {
            File::create(path)
                .ok()
                .and_then(|file| Some(Box::new(file) as Box<dyn Write>))
        })
        .or_else(|| Some(Box::new(stdout)))
        .expect("Could not get output");

    let s = serde_yaml::to_string(&exp_log);
    writeln!(output, "{}", s.expect("Trouble generating yaml")).expect("Trouble writing output");
}
