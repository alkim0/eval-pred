use super::table::{FileTable, Id, Table};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::rc::Rc;

pub struct DB {
    //table_map: HashMap<String, &'a Table>,
    tables: HashMap<Id, Rc<dyn Table>>,
    file_tables: HashMap<String, Rc<dyn Table>>,
    //col_map: HashMap<&'a FileCol, &'a Table>,
    pub path: PathBuf,
}

impl DB {
    pub fn new(path: &Path) -> DB {
        assert!(path.is_dir());

        let mut db = DB {
            //tables: vec![],
            //table_map: HashMap::new(),
            tables: HashMap::new(),
            file_tables: HashMap::new(),
            //col_map: HashMap::new(),
            path: path.to_path_buf(),
        };
        for table in fs::read_dir(path).unwrap() {
            let table = table.unwrap();
            let table_name = table.file_name().into_string().unwrap();
            if table_name == "__join_keys__" {
                continue;
            }
            let table_path = table.path();
            let table = FileTable::new(table_path.into_boxed_path());
            db.tables.insert(table.id(), table.clone());
            db.file_tables.insert(table_name, table);
        }
        db
    }

    pub fn tables(&self) -> &HashMap<Id, Rc<dyn Table>> {
        &self.tables
    }

    pub fn file_tables(&self) -> &HashMap<String, Rc<dyn Table>> {
        &self.file_tables
    }
}
