use super::config::{BLOCK_SIZE, READ_BUF_SIZE};
use super::exec::{ExecParams, ExecStats, RunContext};
use super::expr::{BinaryOperator, DBCol, DBResult, DataType, Expr};
use super::parser::PredNode;
use byteorder::{NativeEndian, ReadBytesExt};
use chrono::{Duration, TimeZone, Utc};
use log::debug;
use roaring::RoaringBitmap;
use snowflake::ProcessUniqueId;
use std::alloc::{self, Layout};
use std::cell::RefCell;
use std::cmp;
use std::collections::HashMap;
use std::ffi::CStr;
use std::fmt;
use std::fs::{self, File};
use std::hash::Hash;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::mem;
use std::ops::Deref;
use std::os::raw;
use std::path::Path;
use std::rc::{Rc, Weak};
use std::slice;

pub type Id = ProcessUniqueId;

// A column that refers to a specific column of a specific table.
// Here, `table` should always refer to a FileTable.
#[derive(Debug)]
pub struct FileCol {
    id: Id,
    name: String,
    data_type: DataType,
    pub table: Weak<FileTable>,
}

// `ReadContext` holds various mappings from which elements to read from the file to
// the order in which they should be returned.
// For performance, we split this up into 3 indexes of vecs and bitmaps rather than a hash map, and
// we have 2 types of indices:
// `file_index`: A bitmap which stores the indices of the elements we have to read from file (we
// call these file indices). Note this is different from `index` from the `exec` module. That index
// is with respect to the joined reference table, whereas this is with respect to the file itself.
// `output_index`: A Vec<usize> in which the ith element refers to an index in file_index
#[derive(Debug, Clone)]
pub struct ReadContext {
    pub file_index: RoaringBitmap,
    pub output_index: Vec<usize>,
}

impl FileCol {
    pub fn id(&self) -> Id {
        self.id
    }

    pub fn read(
        &self,
        run_context: &RunContext,
        exec_stats: &mut ExecStats,
        ref_table: &Rc<dyn Table>,
    ) -> DBResult {
        debug!(
            "[FileCol] Reading {}.{} under ref table {}",
            self.table.upgrade().unwrap().name(),
            self.name,
            ref_table.name()
        );
        // Get the vector of indices which we have to fetch
        let read_context = ref_table.get_read_context(self, run_context);
        let table = self.table.upgrade().unwrap();
        let mut result = table.read(self, read_context, run_context, exec_stats);
        result.ref_table = Some(ref_table.clone());
        debug!(
            "[FileCol] Read {} values for {}.{}; index size is {:?}",
            result.cols.values().map(|c| c.len()).sum::<usize>(),
            self.table.upgrade().unwrap().name(),
            self.name,
            run_context.index.as_ref().and_then(|i| Some(i.len()))
        );
        result
    }

    fn data_size(&self) -> usize {
        match self.data_type {
            DataType::Str => 128,
            DataType::Int => mem::size_of::<raw::c_int>(),
            DataType::Long => mem::size_of::<raw::c_long>(),
            DataType::Float => mem::size_of::<raw::c_float>(),
            DataType::Double => mem::size_of::<raw::c_double>(),
            DataType::Bool => 1,
            DataType::DateTime => mem::size_of::<raw::c_long>(),
            DataType::Duration => mem::size_of::<raw::c_long>(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Debug)]
pub struct FileTable {
    id: Id,
    name: String,
    path: Box<Path>,
    cols: RefCell<HashMap<String, Rc<FileCol>>>,
    data_num: RefCell<usize>,
}

#[derive(Debug, Clone)]
pub enum JoinType {
    Inner,
    LeftOuter,
    RightOuter,
}

#[derive(Debug)]
pub struct JoinTable {
    id: Id,
    name: String,
    left: Rc<dyn Table>,
    right: Rc<dyn Table>,
    cols: HashMap<String, Vec<Rc<FileCol>>>,
    join_type: JoinType,
    pub constraint: Expr,
    // Mapping from FileCold ID
    join_idx: RefCell<HashMap<String, Vec<usize>>>,
}

pub trait Table: fmt::Debug + fmt::Display {
    fn id(&self) -> Id;

    fn all_cols(&self) -> Vec<Rc<FileCol>>;

    fn find_col(&self, col_name: &str, table_name: Option<&str>) -> Option<Rc<FileCol>>;

    fn get_read_context(&self, col: &FileCol, run_context: &RunContext) -> ReadContext;

    fn len(&self) -> usize;

    fn name<'a>(&'a self) -> &'a str;

    fn contains_subtable(&self, table_name: &str) -> bool;

    fn get_all_join_exprs(&self) -> HashMap<String, Expr>;

    fn eval_join(
        &self,
        exec_params: &ExecParams,
        exec_stats: &mut ExecStats,
        pred: Option<&PredNode>,
    );

    fn get_map<'a>(&'a self) -> Box<dyn Deref<Target = HashMap<String, Vec<usize>>> + 'a>;
}

impl Table for FileTable {
    fn name(&self) -> &str {
        &self.name
    }

    fn id(&self) -> Id {
        self.id
    }

    fn contains_subtable(&self, table_name: &str) -> bool {
        table_name == self.name
    }

    fn get_read_context(&self, _col: &FileCol, run_context: &RunContext) -> ReadContext {
        if run_context.index.is_none() {
            let data_num = *self.data_num.borrow();
            ReadContext {
                file_index: (0..data_num as u32).collect(),
                output_index: (0..data_num).collect(),
            }
        } else {
            //let index: Vec<usize> = run_context
            //    .index
            //    .as_ref()
            //    .unwrap()
            //    .iter()
            //    .map(|x| x as usize)
            //    .collect();
            let index = run_context.index.as_ref().unwrap();
            ReadContext {
                file_index: run_context.index.clone().unwrap(),
                output_index: (0..index.len() as usize).collect(),
            }
        }
    }

    fn all_cols(&self) -> Vec<Rc<FileCol>> {
        self.cols.borrow().values().cloned().collect()
    }

    fn find_col(&self, col_name: &str, _table_name: Option<&str>) -> Option<Rc<FileCol>> {
        self.cols.borrow().get(col_name).cloned()
    }

    fn len(&self) -> usize {
        *self.data_num.borrow()
    }

    fn get_all_join_exprs(&self) -> HashMap<String, Expr> {
        HashMap::new()
    }

    fn eval_join(
        &self,
        _exec_params: &ExecParams,
        _exec_stats: &mut ExecStats,
        _pred: Option<&PredNode>,
    ) {
    }

    fn get_map<'a>(&'a self) -> Box<dyn Deref<Target = HashMap<String, Vec<usize>>> + 'a> {
        let mut map = HashMap::new();
        map.insert(
            self.name.to_string(),
            (0..*self.data_num.borrow()).collect(),
        );
        Box::new(Box::new(map))
    }
}

impl FileTable {
    pub fn new(path: Box<Path>) -> Rc<FileTable> {
        let schema_path = path.join("__schema__");
        assert!(
            schema_path.exists(),
            "schema file for {} does not exist",
            path.display()
        );
        let schema = fs::read_to_string(schema_path).unwrap();
        let mut lines = schema.split("\n");
        let attr_names: Vec<&str> = lines.next().unwrap().split(',').collect();
        let attr_types: Vec<&str> = lines.next().unwrap().split(',').collect();

        let table = Rc::new(FileTable {
            id: Id::new(),
            name: path
                .file_name()
                .unwrap()
                .to_os_string()
                .into_string()
                .unwrap(),
            path: path.clone(),
            cols: RefCell::new(HashMap::new()),
            data_num: RefCell::new(0),
        });

        for (name, data_type) in attr_names.iter().zip(attr_types) {
            let data_type = match data_type {
                "string" => DataType::Str,
                "int" => DataType::Int,
                "long" => DataType::Long,
                "float" => DataType::Float,
                "double" => DataType::Double,
                "boolean" => DataType::Bool,
                "timestamp" | "date" => DataType::DateTime,
                "interval" => DataType::Duration,
                _ => panic!("Unknown type: {}", data_type),
            };

            let col = Rc::new(FileCol {
                id: Id::new(),
                name: name.to_string(),
                data_type,
                table: Rc::downgrade(&table),
            });

            let col_file_size = fs::metadata(path.join(col.name.to_string()))
                .expect("Problem finding col")
                .len();
            let data_num = col_file_size as usize / col.data_size();
            if *table.data_num.borrow() != 0 {
                assert_eq!(
                    data_num,
                    *table.data_num.borrow(),
                    "{:?} {}",
                    path,
                    name.to_string()
                );
            } else {
                *table.data_num.borrow_mut() = data_num;
            }

            table.cols.borrow_mut().insert(name.to_string(), col);
        }

        table
    }

    fn cols(&self) -> &RefCell<HashMap<String, Rc<FileCol>>> {
        &self.cols
    }
    // TODO Change this to direct io in future
    // TODO Change this to have buffers
    // TODO Implement the sequential scan case
    fn read(
        &self,
        col: &FileCol,
        read_context: ReadContext,
        run_context: &RunContext,
        exec_stats: &mut ExecStats,
    ) -> DBResult {
        //thread::sleep(time::Duration::from_millis(
        //    run_context.exec_params.extra_data_retrieval_latency as u64,
        //));

        let mut file = File::open(self.path.join(&col.name)).expect("Could not find file");
        //let mut file = OpenOptions::new()
        //    .read(true)
        //    .custom_flags(libc::O_DIRECT | libc::O_SYNC)
        //    .open(self.path.join(&col.name))
        //    .expect(&format!(
        //        "Could not open file {:?}",
        //        self.path.join(&col.name)
        //    ));
        let file_size = file.metadata().expect("Error finding size of file").len() as usize;
        let data_size = col.data_size();
        let data_num = file_size / data_size;
        let _selectivity = read_context.file_index.len() as f32 / data_num as f32;
        let _col_name = col.name.clone();
        // XXX This allocates more than necessary space since it does not depend on selectivity.
        let mut col = match col.data_type {
            DataType::Int => DBCol::Int(Vec::with_capacity(data_num)),
            DataType::Long => DBCol::Long(Vec::with_capacity(data_num)),
            DataType::Float => DBCol::Float(Vec::with_capacity(data_num)),
            DataType::Double => DBCol::Double(Vec::with_capacity(data_num)),
            DataType::Str => DBCol::Str(Vec::with_capacity(data_num)),
            DataType::Bool => DBCol::Bool(Vec::with_capacity(data_num)),
            DataType::DateTime => DBCol::DateTime(Vec::with_capacity(data_num)),
            DataType::Duration => DBCol::Duration(Vec::with_capacity(data_num)),
        };
        let mut num_bufs_read = 0;
        // XXX Screw random I/O for a moment
        // TODO - FINISH Remember to implement the selectivity < SELECTIVITY_THRESHOLD case
        //if selectivity < 0. {
        //    let mut buf = Vec::with_capacity(data_size);
        //    buf.resize(data_size, 0);

        //    //if selectivity < SELECTIVITY_THRESHOLD {
        //    for idx in &read_context.file_index {
        //        file.seek(SeekFrom::Start(idx as u64 * data_size as u64))
        //            .expect("seeking failed");
        //        file.read_exact(&mut buf)
        //            .expect("trouble reading from file");
        //        let mut rdr = Cursor::new(&buf);
        //        match &mut col {
        //            DBCol::Int(vals) => {
        //                vals.push(rdr.read_i32::<NativeEndian>().unwrap());
        //            }
        //            DBCol::Long(vals) => {
        //                vals.push(rdr.read_i64::<NativeEndian>().unwrap());
        //            }
        //            DBCol::Float(vals) => {
        //                vals.push(rdr.read_f32::<NativeEndian>().unwrap());
        //            }
        //            DBCol::Double(vals) => {
        //                vals.push(rdr.read_f64::<NativeEndian>().unwrap());
        //            }
        //            DBCol::Str(vals) => {
        //                let end = buf.iter_mut().position(|c| *c == b'\0').unwrap_or_else(|| {
        //                    *buf.last_mut().unwrap() = b'\0';
        //                    buf.len()
        //                }) + 1;
        //                let val = CStr::from_bytes_with_nul(&buf[..end])
        //                    .expect("Error converting to string")
        //                    .to_str()
        //                    .expect("Error converting to utf8");

        //                vals.push(val.to_string());
        //            }
        //            DBCol::Bool(vals) => {
        //                vals.push(rdr.read_u8().unwrap() != 0);
        //            }
        //        }
        //    }
        //} else {
        //let mut buf = Vec::with_capacity(READ_BUF_SIZE);
        //buf.resize(READ_BUF_SIZE, 0);
        let mut buf = unsafe {
            slice::from_raw_parts_mut(
                alloc::alloc(
                    Layout::from_size_align(READ_BUF_SIZE, BLOCK_SIZE).expect(&format!(
                        "Error with alignment settings, size: {} align: {}",
                        READ_BUF_SIZE, BLOCK_SIZE
                    )),
                ),
                READ_BUF_SIZE,
            )
        };
        let data_per_buf = READ_BUF_SIZE / data_size;
        let mut buf_idx = 0;
        let bytes_read = file
            .read(&mut buf)
            //file.read_exact(&mut buf[..READ_BUF_SIZE])
            .expect(&format!(
                "trouble with first read, file_size: {}",
                file_size
            ));
        assert_eq!(bytes_read, cmp::min(READ_BUF_SIZE, file_size));
        num_bufs_read += 1;

        for idx in &read_context.file_index {
            let idx = idx as usize;
            // If the next element is on another page
            if idx >= data_per_buf * (buf_idx + 1) {
                // If the next element is on a page later than the next page
                if idx >= data_per_buf * (buf_idx + 2) {
                    buf_idx = idx / data_per_buf;
                    file.seek(SeekFrom::Start(buf_idx as u64 * READ_BUF_SIZE as u64))
                        .expect(&format!(
                            "seeking to {} failed",
                            buf_idx as u64 * READ_BUF_SIZE as u64
                        ));
                } else {
                    buf_idx += 1;
                }
                let expected_to_read =
                    cmp::min((buf_idx + 1) * READ_BUF_SIZE, file_size) - (buf_idx * READ_BUF_SIZE);
                let bytes_read = file.read(&mut buf).expect(&format!(
                    "trouble reading from file at index {} for {} bytes",
                    buf_idx * READ_BUF_SIZE,
                    expected_to_read
                ));
                assert_eq!(bytes_read, expected_to_read);
                num_bufs_read += 1;
            }

            let start = (idx % data_per_buf) * data_size;
            let mut rdr = Cursor::new(&buf[start..(start + data_size)]);
            match &mut col {
                DBCol::Int(vals) => {
                    vals.push(rdr.read_i32::<NativeEndian>().unwrap());
                }
                DBCol::Long(vals) => {
                    vals.push(rdr.read_i64::<NativeEndian>().unwrap());
                }
                DBCol::Float(vals) => {
                    vals.push(rdr.read_f32::<NativeEndian>().unwrap());
                }
                DBCol::Double(vals) => {
                    vals.push(rdr.read_f64::<NativeEndian>().unwrap());
                }
                DBCol::Str(vals) => {
                    let buf = &mut buf[start..(start + data_size)];
                    let end = buf.iter().position(|c| *c == b'\0').unwrap_or_else(|| {
                        *buf.last_mut().unwrap() = b'\0';
                        buf.len()
                    }) + 1;
                    let val = CStr::from_bytes_with_nul(&buf[..end])
                        .expect("Error converting to string")
                        .to_str()
                        .expect("Error converting to utf8");
                    //println!("{}", val);
                    vals.push(val.to_string());
                }
                DBCol::Bool(vals) => {
                    vals.push(rdr.read_u8().unwrap() != 0);
                }
                DBCol::DateTime(vals) => {
                    let ts = rdr.read_i64::<NativeEndian>().unwrap();
                    vals.push(Utc.timestamp(ts, 0));
                }
                DBCol::Duration(vals) => {
                    let dur = rdr.read_i64::<NativeEndian>().unwrap();
                    vals.push(Duration::seconds(dur));
                }
            }
        }
        //}

        unsafe {
            alloc::dealloc(
                buf.as_mut_ptr(),
                Layout::from_size_align(READ_BUF_SIZE, BLOCK_SIZE).expect(&format!(
                    "Error with alignment settings, size: {} align: {}",
                    READ_BUF_SIZE, BLOCK_SIZE
                )),
            );
        }

        assert_eq!(col.len(), read_context.file_index.len() as usize);
        exec_stats.num_bufs_read += num_bufs_read as u128;
        //eprintln!(
        //    "**** Read {}/{} bufs",
        //    num_bufs_read,
        //    file_size / (READ_BUF_SIZE) + (if file_size % READ_BUF_SIZE != 0 { 1 } else { 0 })
        //);

        fn make_output_col<T: Clone + fmt::Debug>(
            vals: Vec<T>,
            read_context: ReadContext,
        ) -> Vec<T> {
            read_context
                .output_index
                .iter()
                .map(|pidx| vals[*pidx].clone())
                .collect()
        }

        let output_col = match col {
            DBCol::Int(vals) => DBCol::Int(make_output_col(vals, read_context)),
            DBCol::Long(vals) => DBCol::Long(make_output_col(vals, read_context)),
            DBCol::Float(vals) => DBCol::Float(make_output_col(vals, read_context)),
            DBCol::Double(vals) => DBCol::Double(make_output_col(vals, read_context)),
            DBCol::Str(vals) => DBCol::Str(make_output_col(vals, read_context)),
            DBCol::Bool(vals) => DBCol::Bool(make_output_col(vals, read_context)),
            DBCol::DateTime(vals) => DBCol::DateTime(make_output_col(vals, read_context)),
            DBCol::Duration(vals) => DBCol::Duration(make_output_col(vals, read_context)),
        };

        let mut result = DBResult {
            cols: vec![(vec![], output_col)].into_iter().collect(),
            ref_table: None,
        };

        if !run_context.groups.is_empty() {
            result.group(&run_context.groups);
        }

        result
    }
}

impl Table for JoinTable {
    fn id(&self) -> Id {
        self.id
    }

    fn all_cols(&self) -> Vec<Rc<FileCol>> {
        self.cols.values().cloned().flatten().collect()
    }

    fn find_col(&self, col_name: &str, table_name: Option<&str>) -> Option<Rc<FileCol>> {
        table_name
            .and_then(|name| {
                if self.left.contains_subtable(name) {
                    self.left.find_col(col_name, Some(name))
                } else if self.right.contains_subtable(name) {
                    self.right.find_col(col_name, Some(name))
                } else {
                    None
                }
            })
            .or_else(|| {
                self.cols.get(col_name).and_then(|cols| {
                    assert!(
                        cols.len() >= 1,
                        "Found multiple instances of col ({}) in table ({})",
                        col_name,
                        self.name
                    );
                    Some(cols[0].clone())
                })
            })
    }

    fn get_read_context(&self, col: &FileCol, run_context: &RunContext) -> ReadContext {
        fn pack_index<'a>(idxs: impl Iterator<Item = &'a usize>) -> ReadContext
where {
            let mut file_index = RoaringBitmap::new();

            let mut output_index = vec![];
            for idx in idxs {
                file_index.insert(*idx as u32);
                output_index.push(*idx);
            }
            let rev_map = file_index
                .iter()
                .enumerate()
                .map(|(i, fidx)| (fidx, i))
                .collect::<HashMap<u32, usize>>();
            let output_index = output_index
                .iter()
                .map(|fidx| *rev_map.get(&(*fidx as u32)).unwrap())
                .collect::<Vec<usize>>();

            ReadContext {
                file_index,
                output_index,
            }
        }

        let join_idx = self.join_idx.borrow();
        assert!(!join_idx.is_empty());

        // The ordered vector of indices for the given col with respect to the file table.
        let table = col
            .table
            .upgrade()
            .expect(&format!("Trouble getting table for col {}", col.name));
        let col_idx = join_idx.get(table.name()).expect(&format!(
            "Trouble finding table ({}) in join_idx",
            table.name()
        ));
        if run_context.index.is_none() {
            pack_index(col_idx.iter())
        } else {
            pack_index(
                run_context
                    .index
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|i| &col_idx[i as usize]),
            )
        }
    }

    fn len(&self) -> usize {
        assert!(!self.join_idx.borrow().is_empty());
        self.join_idx.borrow().iter().next().unwrap().1.len()
    }

    fn name<'a>(&'a self) -> &'a str {
        &self.name
    }

    fn contains_subtable(&self, table_name: &str) -> bool {
        self.left.contains_subtable(table_name) || self.right.contains_subtable(table_name)
    }

    fn get_all_join_exprs(&self) -> HashMap<String, Expr> {
        let left_exprs = self.left.get_all_join_exprs();
        let right_exprs = self.left.get_all_join_exprs();
        let mut all_exprs = left_exprs
            .into_iter()
            .chain(right_exprs)
            .collect::<HashMap<String, Expr>>();
        all_exprs.insert(self.constraint.to_string(), self.constraint.clone());
        all_exprs
    }

    fn eval_join(
        &self,
        exec_params: &ExecParams,
        exec_stats: &mut ExecStats,
        pred: Option<&PredNode>,
    ) {
        debug!("[JoinTable] Evaluating {}", self.name);
        //let mut join_idx = self.join_idx.borrow_mut();
        if !self.join_idx.borrow().is_empty() {
            return;
        }

        self.left.eval_join(exec_params, exec_stats, pred);
        self.right.eval_join(exec_params, exec_stats, pred);

        let left_result;
        let right_result;

        if let Expr::BinaryOp {
            left,
            right,
            op: BinaryOperator::Eq,
        } = &self.constraint
        {
            left_result = left.eval(
                &RunContext {
                    index: None,
                    groups: vec![],
                    ref_table: None,
                    exec_params: exec_params.clone(),
                },
                exec_stats,
            );
            right_result = right.eval(
                &RunContext {
                    index: None,
                    groups: vec![],
                    ref_table: None,
                    exec_params: exec_params.clone(),
                },
                exec_stats,
            );
        } else {
            panic!(
                "We do not support non-equality join constraint ({:?})",
                self.constraint
            );
        }

        fn inner_join<'a, T: Eq + Hash + fmt::Debug + Clone>(
            left_vals: &Vec<T>,
            left_map: Box<dyn Deref<Target = HashMap<String, Vec<usize>>> + 'a>,
            right_vals: &Vec<T>,
            right_map: Box<dyn Deref<Target = HashMap<String, Vec<usize>>> + 'a>,
        ) -> HashMap<String, Vec<usize>> {
            let left_maps: Vec<(&String, &Vec<usize>)> = left_map.iter().collect();
            let right_maps: Vec<(&String, &Vec<usize>)> = right_map.iter().collect();

            let mut all_maps: Vec<(String, Vec<usize>)> =
                Vec::with_capacity(left_maps.len() + right_maps.len());
            for (name, _) in &left_maps {
                all_maps.push((name.to_string(), vec![]));
            }
            for (name, _) in &right_maps {
                all_maps.push((name.to_string(), vec![]));
            }

            let mut left_rev_map: HashMap<T, Vec<usize>> = HashMap::new();
            for (i, val) in left_vals.iter().enumerate() {
                left_rev_map.entry(val.clone()).or_default().push(i);
            }

            for (right_idx, val) in right_vals.iter().enumerate() {
                let left_idxs = left_rev_map.get(val);
                if left_idxs.is_some() {
                    for left_idx in left_idxs.unwrap() {
                        for (i, left_map) in left_maps.iter().enumerate() {
                            all_maps[i].1.push(left_map.1[*left_idx]);
                        }
                        for (i, right_map) in right_maps.iter().enumerate() {
                            all_maps[i + left_maps.len()].1.push(right_map.1[right_idx]);
                        }
                    }
                }
            }

            debug!(
                "[JoinTable] Joined and returning maps for [{}]",
                all_maps
                    .iter()
                    .map(|(name, _)| name.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            );
            all_maps.into_iter().collect()
        }

        #[allow(unused_variables)]
        fn left_outer_join<T: Eq + Hash + Clone>(
            left_vals: &Vec<T>,
            left_map: Box<dyn Deref<Target = HashMap<Id, Vec<usize>>>>,
            right_vals: &Vec<T>,
            right_map: Box<dyn Deref<Target = HashMap<Id, Vec<usize>>>>,
        ) -> HashMap<Id, Vec<usize>> {
            unimplemented!();
        }

        let left_vals = left_result
            .cols
            .get(&vec![])
            .expect("We're grouping on joins?");
        let left_table = left_result.ref_table.expect("join requires ref table");
        left_table.eval_join(exec_params, exec_stats, pred);
        let left_map = left_table.get_map();
        let right_vals = right_result
            .cols
            .get(&vec![])
            .expect("We're grouping on joins?");
        let right_table = right_result.ref_table.expect("join requires ref table");
        right_table.eval_join(exec_params, exec_stats, pred);
        let right_map = right_table.get_map();

        if let JoinType::Inner = self.join_type {
            // TODO We may have to cast types here if the key types don't match exactly.
            self.join_idx.replace(match (left_vals, right_vals) {
                (DBCol::Int(left_vals), DBCol::Int(right_vals)) => {
                    inner_join(left_vals, left_map, right_vals, right_map)
                }
                (DBCol::Long(left_vals), DBCol::Long(right_vals)) => {
                    inner_join(left_vals, left_map, right_vals, right_map)
                }
                (DBCol::Str(left_vals), DBCol::Str(right_vals)) => {
                    inner_join(left_vals, left_map, right_vals, right_map)
                }
                _ => {
                    panic!("unuspported join types");
                }
            });
        } else {
            // XXX All joins are inner joins! MUHAHAHA
            self.join_idx.replace(match (left_vals, right_vals) {
                (DBCol::Int(left_vals), DBCol::Int(right_vals)) => {
                    inner_join(left_vals, left_map, right_vals, right_map)
                }
                (DBCol::Long(left_vals), DBCol::Long(right_vals)) => {
                    inner_join(left_vals, left_map, right_vals, right_map)
                }
                (DBCol::Str(left_vals), DBCol::Str(right_vals)) => {
                    inner_join(left_vals, left_map, right_vals, right_map)
                }
                _ => {
                    panic!("unuspported join types");
                }
            });
        }

        debug!("[Join] Done evaluating {}", self.name);
    }

    fn get_map<'a>(&'a self) -> Box<dyn Deref<Target = HashMap<String, Vec<usize>>> + 'a> {
        Box::new(self.join_idx.borrow())
    }
}

impl JoinTable {
    pub fn new(
        join_type: JoinType,
        constraint: Expr,
        left: &Rc<dyn Table>,
        right: &Rc<dyn Table>,
    ) -> Rc<dyn Table> {
        let mut table = JoinTable {
            id: Id::new(),
            name: format!("join({}, {})", left.name(), right.name()),
            left: left.clone(),
            right: right.clone(),
            cols: HashMap::new(),
            join_type,
            constraint,
            join_idx: RefCell::new(HashMap::new()),
        };
        for col in left.all_cols().iter().chain(right.all_cols().iter()) {
            table
                .cols
                .entry(col.name.to_string())
                .or_default()
                .push(col.clone());
        }
        Rc::new(table)
    }

    fn cols(&self) -> &HashMap<String, Vec<Rc<FileCol>>> {
        &self.cols
    }
}

impl fmt::Display for JoinTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} JOIN {} ON {}",
            self.left, self.right, self.constraint
        )
    }
}

impl fmt::Display for FileTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

//#[cfg(test)]
//mod tests {
//    use super::*;
//
//    #[test]
//    fn create_table() {}
//}
