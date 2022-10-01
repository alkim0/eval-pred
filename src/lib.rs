mod bdc;
mod byp;
pub mod config;
mod db;
mod exec;
mod expr;
mod funcs;
mod parser;
pub mod query_utils;
mod table;
mod tdacb;
mod utils;

#[macro_use]
extern crate lazy_static;

pub use config as cham_config;
pub use db::DB;
pub use exec::{read_selectivities, ApproxOptType, ExecParams, ExecStats, Executor, RunContext};
pub use expr::{BinaryOperator, DBCol, DBResult, Expr};
use log::debug;
pub use parser::{JoinTableType, ParseParams, Parser, PredNode};
use std::fmt;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::Instant;
use structopt::StructOpt;
pub use table::{Id, Table};

#[derive(Debug, StructOpt)]
#[structopt(name = "cham", about = "Chameleon DB system.")]
pub struct Opt {
    #[structopt(short, long)]
    debug: bool,

    #[structopt(short = "a", default_value = "1")]
    start_index: u32,

    #[structopt(short, long)]
    num_queries: Option<u32>,

    #[structopt(short, long)]
    print_results: bool,

    #[structopt(long)]
    disable_or_opt: bool,

    #[structopt(long, default_value = "0")]
    extra_data_latency: u32,

    #[structopt(long, default_value = "32768")]
    read_buf_size: u32,

    #[structopt(short, long = "selec", parse(from_os_str))]
    selectivites_path: Option<PathBuf>,

    #[structopt(parse(from_os_str))]
    db_path: PathBuf,

    #[structopt(parse(from_os_str))]
    log_path: PathBuf,
}

// XXX This can be heavily optimized. Right now I just need it for debugging, so it's okay if it's
// slow.
pub fn print_results(results: &Vec<DBResult>) {
    assert!(!results.is_empty());
    let groups: Vec<Vec<DBCol>> = results[0].cols.keys().cloned().collect();
    for group in groups {
        let printables: Vec<Vec<Box<dyn fmt::Debug>>> = results
            .iter()
            .map(|result| {
                let col = &result.cols[&group];
                col.some(col.len())
            })
            .collect();
        let len = results[0].cols[&group].len();
        for i in 0..len {
            println!(
                "{}",
                printables
                    .iter()
                    .map(|x| format!("{:?}", x[i]))
                    .collect::<Vec<String>>()
                    .join(",")
            );
        }
    }
}

fn validate_args(opt: &Opt) {
    assert!(opt.start_index > 0, "start_index must be 1 or greater");
}

pub fn run(opt: Opt) {
    validate_args(&opt);

    let db = DB::new(&opt.db_path);
    let selectivities = opt
        .selectivites_path
        .as_ref()
        .and_then(|path| Some(read_selectivities(path)));
    let mut exec = Executor::new(&db, selectivities, None);
    let parser = Parser::new(&db);

    if opt.debug {
        fern::Dispatch::new()
            .level(log::LevelFilter::Debug)
            .chain(std::io::stderr())
            .apply()
            .expect("Logging failed");
    }

    let start = opt.start_index;
    let log_file = BufReader::new(File::open(opt.log_path).expect("Could not open given log path"));

    let now = Instant::now();

    let mut i: u32 = 0;
    let mut num_skipped: u32 = 0;
    for line in log_file.lines() {
        if i + 1 < start {
            i += 1;
            continue;
        }

        if opt.num_queries.is_some() && i >= opt.num_queries.unwrap() {
            break;
        }

        if line.is_err() {
            panic!("Reading line errored: {}", line.unwrap_err());
        }
        let line = line.unwrap();

        debug!("*** PARSING QUERY {} ***", i + 1);
        let query = line
            .splitn(3, ',')
            .nth(2)
            .expect("Line does not have 3 elements");
        let query = parser.parse(query, &Default::default());
        if query.is_err() {
            eprintln!("Skipped query: {}", i + 1);
            num_skipped += 1;
        } else {
            let mut exec_stats = ExecStats::new();
            let results = exec.run(query.unwrap(), &Default::default(), &mut exec_stats);
            assert!(!results.is_empty());
            if opt.print_results {
                print_results(&results);
            }
            eprintln!(
                "{}. Result is {} tuples",
                i + 1,
                results[0]
                    .cols
                    .iter()
                    .map(|(_, col)| col.len())
                    .sum::<usize>()
            );
        }
        i += 1;
    }

    println!("Total runtime: {} s", now.elapsed().as_secs());
    eprintln!("Total parsed: {}", i - num_skipped);
    eprintln!("Total skipped: {}", num_skipped);
}
