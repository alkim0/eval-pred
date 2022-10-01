use super::expr::{DBCol, DBResult};
use chrono::{Datelike, Timelike, Utc};
use num;
use spin_sleep;
use std::cmp::Ordering;
use std::collections::HashMap;

lazy_static! {
    //pub static ref BLAH: fn(Vec<DBResult>) -> DBResult = sum;
    pub static ref FUNC_MAP: HashMap<&'static str, fn(Vec<DBResult>) -> DBResult> =
        {
            let mut m = HashMap::new();
            m.insert("sum", sum as fn(Vec<DBResult>) ->DBResult);
            m.insert("max", max as fn(Vec<DBResult>) ->DBResult);
            m.insert("min", min as fn(Vec<DBResult>) ->DBResult);
            m.insert("avg", avg as fn(Vec<DBResult>) ->DBResult);
            m.insert("count", count as fn(Vec<DBResult>) ->DBResult);
            m.insert("coalesce", coalesce as fn(Vec<DBResult>) ->DBResult);
            m.insert("trunc", trunc as fn(Vec<DBResult>) ->DBResult);
            m.insert("round", round as fn(Vec<DBResult>) ->DBResult);
            m.insert("abs", abs as fn(Vec<DBResult>) ->DBResult);
            m.insert("substr", substr as fn(Vec<DBResult>) ->DBResult);
            m.insert("replace", replace as fn(Vec<DBResult>) ->DBResult);
            m.insert("json_path_lookup", json_path_lookup as fn(Vec<DBResult>) ->DBResult);
            m.insert("date_trunc", date_trunc as fn(Vec<DBResult>) ->DBResult);
            m.insert("date_part", date_part as fn(Vec<DBResult>) ->DBResult);
            m.insert("now", now as fn(Vec<DBResult>) ->DBResult);
            m.insert("current_timestamp", now as fn(Vec<DBResult>) ->DBResult);
            m.insert("dummy_udf", dummy_udf as fn(Vec<DBResult>) ->DBResult);
            m.insert("forest_udf", forest_udf as fn(Vec<DBResult>) -> DBResult);
            m
        };
}

//pub static function_map: HashSet<String> = vec!["sum"].iter().map(|f| f.to_string()).collect();

//pub static function_map: HashMap<String, &dyn Fn(Vec<DBResult>) -> DBResult> =
//    vec![("sum".to_string(), &sum)].iter().collect();
//

//fn assert_same_groups(args: &Vec<DBResult>) {
//    let groupings: Vec<HashSet<&Vec<DBVal>>> =
//        args.iter().map(|a| a.cols.keys().collect()).collect();
//    assert!(groupings.windows(2).all(|w| w[0] == w[1]));
//}

fn sum(mut args: Vec<DBResult>) -> DBResult {
    assert!(args.len() == 1);
    let mut arg = args.pop().unwrap();
    for (_, col) in arg.cols.iter_mut() {
        if col.len() == 0 {
            *col = DBCol::Int(vec![]);
        } else {
            *col = match col {
                DBCol::Int(vals) => DBCol::Int(vec![vals.iter().sum()]),
                DBCol::Long(vals) => DBCol::Long(vec![vals.iter().sum()]),
                DBCol::Float(vals) => DBCol::Float(vec![vals.iter().sum()]),
                DBCol::Double(vals) => DBCol::Double(vec![vals.iter().sum()]),
                _ => {
                    panic!("called sum on non-numerical type");
                }
            };
        }
    }
    arg
}

fn max(mut args: Vec<DBResult>) -> DBResult {
    assert!(args.len() == 1);
    let mut arg = args.pop().unwrap();
    for (_, col) in arg.cols.iter_mut() {
        if col.len() == 0 {
            *col = DBCol::Int(vec![]);
        } else {
            *col = match col {
                DBCol::Int(vals) => DBCol::Int(vec![*vals.iter().max().unwrap()]),
                DBCol::Long(vals) => DBCol::Long(vec![*vals.iter().max().unwrap()]),
                DBCol::Float(vals) => DBCol::Float(vec![*vals
                    .iter()
                    .max_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal))
                    .unwrap()]),
                DBCol::Double(vals) => DBCol::Double(vec![*vals
                    .iter()
                    .max_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal))
                    .unwrap()]),
                DBCol::Str(vals) => DBCol::Str(vec![vals.iter().max().unwrap().to_string()]),
                _ => {
                    panic!("called max on non-numerical type");
                }
            };
        }
    }
    arg
}

fn min(mut args: Vec<DBResult>) -> DBResult {
    assert!(args.len() == 1);
    let mut arg = args.pop().unwrap();
    for (_, col) in arg.cols.iter_mut() {
        if col.len() == 0 {
            *col = DBCol::Int(vec![]);
        } else {
            *col = match col {
                DBCol::Int(vals) => DBCol::Int(vec![*vals.iter().min().unwrap()]),
                DBCol::Long(vals) => DBCol::Long(vec![*vals.iter().min().unwrap()]),
                DBCol::Float(vals) => DBCol::Float(vec![*vals
                    .iter()
                    .min_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal))
                    .unwrap()]),
                DBCol::Double(vals) => DBCol::Double(vec![*vals
                    .iter()
                    .min_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal))
                    .unwrap()]),
                DBCol::Str(vals) => DBCol::Str(vec![vals.iter().min().unwrap().to_string()]),
                _ => {
                    panic!("called min on non-numerical type (e.g., {:?})", col.any());
                }
            };
        }
    }
    arg
}

fn avg(mut args: Vec<DBResult>) -> DBResult {
    assert!(args.len() == 1);
    let mut arg = args.pop().unwrap();
    for (_, col) in arg.cols.iter_mut() {
        if col.len() == 0 {
            *col = DBCol::Int(vec![0]);
        } else {
            *col = match col {
                DBCol::Int(vals) => {
                    let sum: i32 = vals.iter().sum();
                    DBCol::Int(vec![sum / (vals.len() as i32)])
                }
                DBCol::Long(vals) => {
                    let sum: i64 = vals.iter().sum();
                    DBCol::Long(vec![sum / (vals.len() as i64)])
                }
                DBCol::Float(vals) => {
                    let sum: f32 = vals.iter().sum();
                    DBCol::Float(vec![sum / (vals.len() as f32)])
                }
                DBCol::Double(vals) => {
                    let sum: f64 = vals.iter().sum();
                    DBCol::Double(vec![sum / (vals.len() as f64)])
                }
                _ => {
                    panic!("called sum on non-numerical type");
                }
            };
        }
    }
    arg
}

fn count(mut args: Vec<DBResult>) -> DBResult {
    assert!(args.len() == 1);
    let mut arg = args.pop().unwrap();
    for (_, col) in arg.cols.iter_mut() {
        *col = DBCol::Int(vec![col.len() as i32]);
    }
    arg
}

// FIXME This version of coalesce does nothing but return a copy of the first column.
fn coalesce(mut args: Vec<DBResult>) -> DBResult {
    assert!(args.len() >= 1);
    let mut arg = args.swap_remove(0);
    for (_, col) in arg.cols.iter_mut() {
        *col = col.clone();
    }
    arg
}

// FIXME This version of json_path_lookup does nothing but return a copy of the first column.
fn json_path_lookup(mut args: Vec<DBResult>) -> DBResult {
    assert!(args.len() >= 1);
    let mut arg = args.swap_remove(0);
    for (_, col) in arg.cols.iter_mut() {
        *col = col.clone();
    }
    arg
}

fn date_trunc(mut args: Vec<DBResult>) -> DBResult {
    assert!(args.len() == 2);
    let mut arg = args.swap_remove(1);
    for (group, col) in arg.cols.iter_mut() {
        let granularity;
        if let DBCol::Str(vals) = &args[0].cols[group] {
            assert!(!vals.is_empty());
            granularity = vals[0].clone();
        } else {
            panic!("granularity {:?} not supported", args[0].cols[group].any());
        }
        match &granularity[..] {
            "day" => {
                if let DBCol::DateTime(vals) = col {
                    vals.iter_mut().for_each(|x| *x = x.date().and_hms(0, 0, 0));
                } else {
                    panic!("Arg to date_trunc is not datettime format");
                }
            }
            _ => panic!("Don't support date_trunc for {}", granularity),
        }
    }
    arg
}

fn date_part(mut args: Vec<DBResult>) -> DBResult {
    assert!(args.len() == 2);
    let mut arg = args.swap_remove(1);
    for (group, ycol) in arg.cols.iter_mut() {
        let new_col;
        if let (DBCol::Str(xvals), DBCol::DateTime(yvals)) = (&args[0].cols[group], &ycol) {
            new_col = DBCol::Int(
                yvals
                    .iter()
                    .map(|y| match &xvals[0][..] {
                        "dow" => y.weekday().number_from_sunday() as i32,
                        "hour" => y.hour() as i32,
                        "epoch" => y.timestamp() as i32,
                        _ => {
                            panic!("Unsupported date_part arg {}", xvals[0]);
                        }
                    })
                    .collect(),
            );
        } else {
            panic!(
                "Unsupported arguments to date_part({:?}, {:?})",
                args[0].cols[group].any(),
                ycol.any()
            );
        }
        *ycol = new_col;
    }
    arg
}

fn trunc(mut args: Vec<DBResult>) -> DBResult {
    assert!(args.len() == 1);
    let mut arg = args.pop().unwrap();
    for (_, col) in arg.cols.iter_mut() {
        match col {
            DBCol::Int(_) | DBCol::Long(_) => {}
            DBCol::Float(vals) => {
                vals.iter_mut().for_each(|x| *x = x.trunc());
            }
            DBCol::Double(vals) => {
                vals.iter_mut().for_each(|x| *x = x.trunc());
            }
            _ => {
                panic!("called trunc on non-numerical type");
            }
        }
    }
    arg
}

// TODO This ignores round lengths for now
fn round(mut args: Vec<DBResult>) -> DBResult {
    assert!(args.len() == 1);
    for (_, col) in args[0].cols.iter_mut() {
        //let round_lengths;
        //if let DBCol::Int(vals) = args[1].cols[group] {
        //    round_lengths = vals;
        //} else {
        //    panic!("Round lengths are not int type (e.g., {:?})", args[1].cols[group].any());
        //}

        match col {
            DBCol::Int(_) | DBCol::Long(_) => {}
            DBCol::Float(vals) => {
                vals.iter_mut().for_each(|x| *x = x.round());
            }
            DBCol::Double(vals) => {
                vals.iter_mut().for_each(|x| *x = x.round());
            }
            _ => {
                panic!("called round on non-numerical type");
            }
        }
    }

    let mut arg = args.pop().unwrap();
    for (_, col) in arg.cols.iter_mut() {
        match col {
            DBCol::Int(_) | DBCol::Long(_) => {}
            DBCol::Float(vals) => {
                vals.iter_mut().for_each(|x| *x = x.trunc());
            }
            DBCol::Double(vals) => {
                vals.iter_mut().for_each(|x| *x = x.trunc());
            }
            _ => {
                panic!("called trunc on non-numerical type");
            }
        }
    }
    arg
}

fn abs(mut args: Vec<DBResult>) -> DBResult {
    assert!(args.len() == 1);
    let mut arg = args.pop().unwrap();
    for (_, col) in arg.cols.iter_mut() {
        match col {
            DBCol::Int(vals) => {
                vals.iter_mut().for_each(|x| *x = num::abs(*x));
            }
            DBCol::Long(vals) => {
                vals.iter_mut().for_each(|x| *x = num::abs(*x));
            }
            DBCol::Float(vals) => {
                vals.iter_mut().for_each(|x| *x = num::abs(*x));
            }
            DBCol::Double(vals) => {
                vals.iter_mut().for_each(|x| *x = num::abs(*x));
            }
            _ => {
                panic!("Called abs on non-numeric type (e.g., {:?})", col.any());
            }
        }
    }
    arg
}

fn substr(mut args: Vec<DBResult>) -> DBResult {
    assert!(args.len() == 3);
    let mut arg = args.swap_remove(0);
    for (group, col) in arg.cols.iter_mut() {
        let pos = args[0].cols[group]
            .iter_as_long()
            .next()
            .expect("No pos arg given for substr");
        let len = args[1].cols[group]
            .iter_as_long()
            .next()
            .expect("No len arg given for substr");
        match col {
            DBCol::Str(vals) => {
                vals.iter_mut()
                    .for_each(|x| *x = x.chars().skip(pos as usize).take(len as usize).collect());
            }
            _ => {
                panic!("called substr on non-string type");
            }
        }
    }
    arg
}

fn replace(mut args: Vec<DBResult>) -> DBResult {
    assert!(args.len() == 3);
    let mut arg = args.swap_remove(0);
    for (group, col) in arg.cols.iter_mut() {
        if let (DBCol::Str(vals), DBCol::Str(old_val), DBCol::Str(new_val)) =
            (col, &args[0].cols[group], &args[1].cols[group])
        {
            let old_val = old_val.iter().next().expect("No old val given");
            let new_val = new_val.iter().next().expect("No new val given");
            vals.iter_mut()
                .for_each(|x| *x = x.replace(old_val, new_val));
        } else {
            panic!(
                "Replacing with non-string values (e.g., {:?} {:?}",
                args[0].cols[group].any(),
                args[1].cols[group].any()
            );
        }
    }
    arg
}

fn now(args: Vec<DBResult>) -> DBResult {
    assert!(args.is_empty());
    DBResult {
        cols: vec![(vec![], DBCol::DateTime(vec![Utc::now()]))]
            .into_iter()
            .collect(),
        ref_table: None,
    }
}

fn dummy_udf(mut args: Vec<DBResult>) -> DBResult {
    assert!(args.len() == 3);
    let selectivity;
    let sleep_time;
    if let DBCol::Double(vals) = args[1].cols.iter().next().unwrap().1 {
        selectivity = vals[0];
    } else {
        panic!("Expecting double literal for selectivity");
    }
    if let DBCol::Long(vals) = args[2].cols.iter().next().unwrap().1 {
        sleep_time = vals[0];
    } else {
        panic!("Expecting long literal for sleep_time");
    }

    assert!(sleep_time >= 0);

    let mut arg = args.swap_remove(0);
    let num_vals = arg.len();
    let sleeper = spin_sleep::SpinSleeper::new(1_000);
    sleeper.sleep_ns(num_vals as u64 * sleep_time as u64);
    for (_, col) in arg.cols.iter_mut() {
        *col = if let DBCol::Double(vals) = col {
            DBCol::Bool(vals.iter().map(|x| *x < selectivity).collect())
        } else {
            panic!("Expecting doubles as vals, instead got {:?}", col.some(3));
        }
    }
    arg
}

// forest_udf(attr, val, op_num, extra_sleep_time)
// forest_udf(attr, val, 0, 0)
// If 3rd value is 0, then it's <, if 3rd value is 1, it's =
// 4th value is time of extra sleep
fn forest_udf(mut args: Vec<DBResult>) -> DBResult {
    assert!(args.len() == 4);
    let val;
    let op_num;
    let sleep_time;

    if let DBCol::Long(vals) = args[1].cols.iter().next().unwrap().1 {
        val = vals[0];
    } else {
        panic!("Expecting long literal for val");
    }

    if let DBCol::Long(vals) = args[2].cols.iter().next().unwrap().1 {
        op_num = vals[0];
    } else {
        panic!("Expecting int literal for op_num");
    }

    if let DBCol::Long(vals) = args[3].cols.iter().next().unwrap().1 {
        sleep_time = vals[0];
    } else {
        panic!("Expecting int literal for sleep_time");
    }

    assert!(sleep_time >= 0);

    let mut arg = args.swap_remove(0);
    let num_vals = arg.len();
    let sleeper = spin_sleep::SpinSleeper::new(1_000);
    sleeper.sleep_ns(num_vals as u64 * sleep_time as u64);
    for (_, col) in arg.cols.iter_mut() {
        *col = if let DBCol::Long(vals) = col {
            DBCol::Bool(
                vals.iter()
                    .map(|x| if op_num == 0 { *x < val } else { *x == val })
                    .collect(),
            )
        } else {
            panic!("Expecting doubles as longs, instead got {:?}", col.some(3));
        }
    }
    arg
}
