use super::byp::Byp;
use super::config::SMOOTHING_PARAMETER;
use super::db::DB;
use super::exec::{ApproxOptType, ExecParams, ExecStats, RunContext};
use super::expr::{BinaryOperator, DBCol, DBResult, Expr, ExprCreateError};
use super::table::{FileTable, Id, JoinTable, JoinType, Table};
use super::tdacb::Tdacb;
use crate::bdc::BDC;
use log::debug;
use roaring::RoaringBitmap;
use sqlparser::ast;
use sqlparser::dialect::PostgreSqlDialect;
use sqlparser::parser::{Parser as RawParser, ParserError as RawParserError};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::time::Instant;

#[derive(Clone)]
pub enum Node {
    PredNode(PredNode),
    SelectNode(SelectNode),
    GroupByNode(GroupByNode),
    TableNode(TableNode),
}

#[derive(Clone)]
pub struct SelectNode {
    items: Vec<SelectItem>,
}

#[derive(Clone)]
pub enum SelectItem {
    OrNode(Vec<SelectItem>),
    AndNode(Vec<SelectItem>),
    SelectAtom(Expr),
}

impl fmt::Display for SelectItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SelectItem::OrNode(items) => {
                let items: Vec<String> = items.iter().map(|i| format!("({})", i)).collect();
                write!(f, "{}", items.join(" OR "))
            }
            SelectItem::AndNode(items) => {
                let items: Vec<String> = items.iter().map(|i| format!("({})", i)).collect();
                write!(f, "{}", items.join(" AND "))
            }
            SelectItem::SelectAtom(expr) => write!(f, "{}", expr),
        }
    }
}

impl SelectNode {
    pub fn eval(&self, run_context: &RunContext, exec_stats: &mut ExecStats) -> Vec<DBResult> {
        let results = self
            .items
            .iter()
            .map(|item| item.eval(run_context, exec_stats))
            .collect::<Vec<DBResult>>();
        assert!(results.len() > 0);
        results
    }
}

impl SelectItem {
    fn new(expr: &ast::Expr, context: &ParseContext) -> Result<Self, ExprCreateError> {
        match expr {
            ast::Expr::BinaryOp {
                left,
                op: ast::BinaryOperator::And,
                right,
            } => {
                let left = SelectItem::new(left, context);
                let right = SelectItem::new(right, context);
                if left.is_ok() && right.is_ok() {
                    Ok(SelectItem::AndNode(vec![left.unwrap(), right.unwrap()]))
                } else if left.is_ok() {
                    left
                } else if right.is_ok() {
                    right
                } else {
                    Err(ExprCreateError::ColDoesNotExist)
                }
            }
            ast::Expr::BinaryOp {
                left,
                op: ast::BinaryOperator::Or,
                right,
            } => {
                let left = SelectItem::new(left, context);
                let right = SelectItem::new(right, context);
                if left.is_ok() && right.is_ok() {
                    Ok(SelectItem::OrNode(vec![left.unwrap(), right.unwrap()]))
                } else if left.is_ok() {
                    left
                } else if right.is_ok() {
                    right
                } else {
                    Err(ExprCreateError::ColDoesNotExist)
                }
            }
            ast::Expr::Nested(subexpr) => SelectItem::new(subexpr, context),
            //_ => Ok(SelectItem::SelectAtom(Expr::new(expr, context)?)),
            _ => Ok(SelectItem::SelectAtom(Expr::new(expr, context)?)),
        }
    }

    fn eval(&self, run_context: &RunContext, exec_stats: &mut ExecStats) -> DBResult {
        match self {
            SelectItem::OrNode(items) => {
                let mut result = HashMap::new();
                assert!(items.len() == 2);
                let left = items[0].eval(run_context, exec_stats);
                let right = items[1].eval(run_context, exec_stats);
                if let (Some(left_table), Some(right_table)) = (&left.ref_table, &right.ref_table) {
                    assert_eq!(left_table.id(), right_table.id());
                }
                assert_eq!(left.cols.len(), right.cols.len());
                for group in left.cols.keys() {
                    let left = left.cols.get(group).unwrap();
                    let right = right
                        .cols
                        .get(group)
                        .expect("left and right don't have same groups");
                    assert_eq!(left.len(), right.len());
                    if let (DBCol::Bool(left), DBCol::Bool(right)) = (left, right) {
                        result.insert(
                            group.clone(),
                            DBCol::Bool(
                                left.iter()
                                    .zip(right.iter())
                                    .map(|(x, y)| *x || *y)
                                    .collect(),
                            ),
                        );
                    } else {
                        panic!("Incompatible OR types");
                    }
                }
                DBResult {
                    cols: result,
                    ref_table: left
                        .ref_table
                        .or(right.ref_table)
                        .and_then(|table| Some(table.clone())),
                }
            }
            SelectItem::AndNode(items) => {
                let mut result = HashMap::new();
                assert!(items.len() == 2);
                let left = items[0].eval(run_context, exec_stats);
                let right = items[1].eval(run_context, exec_stats);
                if let (Some(left_table), Some(right_table)) = (&left.ref_table, &right.ref_table) {
                    assert_eq!(left_table.id(), right_table.id());
                }
                assert_eq!(left.cols.len(), right.cols.len());
                for group in left.cols.keys() {
                    let left = left.cols.get(group).unwrap();
                    let right = right
                        .cols
                        .get(group)
                        .expect("left and right don't have same groups");
                    assert_eq!(left.len(), right.len());
                    if let (DBCol::Bool(left), DBCol::Bool(right)) = (left, right) {
                        result.insert(
                            group.clone(),
                            DBCol::Bool(
                                left.iter()
                                    .zip(right.iter())
                                    .map(|(x, y)| *x && *y)
                                    .collect(),
                            ),
                        );
                    } else {
                        panic!("Incompatible OR types");
                    }
                }
                DBResult {
                    cols: result,
                    ref_table: left
                        .ref_table
                        .or(right.ref_table)
                        .and_then(|table| Some(table.clone())),
                }
            }
            SelectItem::SelectAtom(expr) => expr.eval(run_context, exec_stats),
        }
    }
}

#[derive(Debug, Clone)]
pub enum PredNode {
    OrNode(Vec<PredNode>),
    AndNode(Vec<PredNode>),
    PredAtomNode(PredAtomNode),
}

impl fmt::Display for PredNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PredNode::OrNode(items) => {
                let items: Vec<String> = items.iter().map(|i| format!("({})", i)).collect();
                write!(f, "{}", items.join(" OR "))
            }
            PredNode::AndNode(items) => {
                let items: Vec<String> = items.iter().map(|i| format!("({})", i)).collect();
                write!(f, "{}", items.join(" AND "))
            }
            PredNode::PredAtomNode(node) => write!(f, "{}", node.expr),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum PredGraphNodeType {
    And,
    Or,
    Atom,
}

struct PredGraphNode {
    parent: Option<Rc<RefCell<PredGraphNode>>>,
    children: Vec<Rc<RefCell<PredGraphNode>>>,
    atom_node: Option<Rc<PredAtomNode>>,
    node_type: PredGraphNodeType,
    id: Id,
    // If is_applied is true, the node is complete
    is_applied: bool,
    // If pos_selec is not None, it is positively determinable,
    pos_selec: Option<f64>,
    // If neg_selec is not None, it is negatively determinable,
    neg_selec: Option<f64>,
}

impl fmt::Debug for PredGraphNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.node_type {
            PredGraphNodeType::And | PredGraphNodeType::Or => write!(
                f,
                "({} {})",
                if let PredGraphNodeType::And = self.node_type {
                    "and"
                } else {
                    "or"
                },
                self.children
                    .iter()
                    .map(|c| format!("{:?}", c.borrow()))
                    .collect::<Vec<String>>()
                    .join(" ")
            ),
            PredGraphNodeType::Atom => write!(f, "{}", self.atom_node.as_ref().unwrap().expr),
        }
    }
}

pub struct PredGraph {
    root: Rc<RefCell<PredGraphNode>>,
    node_map: HashMap<Id, Rc<RefCell<PredGraphNode>>>,
    // Keeps track of positively determinable indices
    pos_map: HashMap<Id, RoaringBitmap>,
    // Keeps track of negatively determinable indices
    neg_map: HashMap<Id, RoaringBitmap>,
}

impl PredGraph {
    fn new(root: Rc<RefCell<PredGraphNode>>) -> PredGraph {
        let mut node_map = HashMap::new();
        let mut fringe = vec![root.clone()];
        while !fringe.is_empty() {
            let node = fringe.pop().unwrap();
            node_map.insert(node.borrow().id, node.clone());
            let node = node.borrow();
            if let PredGraphNodeType::And | PredGraphNodeType::Or = node.node_type {
                for child in &node.children {
                    fringe.push(child.clone());
                }
            }
        }
        PredGraph {
            root,
            node_map,
            pos_map: HashMap::new(),
            neg_map: HashMap::new(),
        }
    }

    fn copy(&self) -> PredGraph {
        fn copy_helper(
            root: &Rc<RefCell<PredGraphNode>>,
            parent: Option<Rc<RefCell<PredGraphNode>>>,
        ) -> Rc<RefCell<PredGraphNode>> {
            let root = root.borrow();
            let new_node = Rc::new(RefCell::new(PredGraphNode {
                parent,
                children: vec![],
                atom_node: None,
                node_type: root.node_type,
                id: root.id,
                is_applied: root.is_applied,
                pos_selec: root.pos_selec,
                neg_selec: root.neg_selec,
            }));

            match root.node_type {
                PredGraphNodeType::And | PredGraphNodeType::Or => {
                    new_node.borrow_mut().children = root
                        .children
                        .iter()
                        .map(|c| copy_helper(c, Some(new_node.clone())))
                        .collect();
                }
                PredGraphNodeType::Atom => {
                    assert!(root.atom_node.is_some());
                    new_node.borrow_mut().atom_node =
                        Some(root.atom_node.as_ref().unwrap().clone());
                }
            }
            new_node
        }
        PredGraph::new(copy_helper(&self.root, None))
    }

    fn pred_atoms(&self) -> Vec<Rc<RefCell<PredGraphNode>>> {
        let mut fringe = vec![self.root.clone()];
        let mut pred_atoms = vec![];
        while !fringe.is_empty() {
            let node = fringe.pop().unwrap();
            let borrowed = node.borrow();
            if let PredGraphNodeType::Atom = borrowed.node_type {
                pred_atoms.push(node.clone());
            } else {
                for child in &borrowed.children {
                    fringe.push(child.clone());
                }
            }
        }
        pred_atoms
    }

    // This only updates the pos_selec/neg_selec/is_applied of the graph nodes based on the
    // estimated selectivities
    fn fake_apply_pred_atom(&self, id: &Id) {
        let mut parent = {
            let mut node = self
                .node_map
                .get(&id)
                .expect(&format!("Could not find node: {}", id))
                .borrow_mut();
            if let PredGraphNodeType::Atom = node.node_type {
                // Ok
            } else {
                panic!("Found node was not pred atom");
            }

            node.is_applied = true;
            node.pos_selec = Some(node.atom_node.as_ref().unwrap().lookup_selectivity());
            node.neg_selec = Some(1. - node.pos_selec.unwrap());

            node.parent.as_ref().and_then(|p| Some(p.clone()))
        };

        while parent.is_some() {
            parent =
                {
                    let mut node = parent.as_ref().unwrap().borrow_mut();
                    match node.node_type {
                        PredGraphNodeType::And => {
                            if node.children.iter().all(|c| c.borrow().pos_selec.is_some()) {
                                node.pos_selec =
                                    Some(node.children.iter().fold(1., |selec, c| {
                                        selec * c.borrow().pos_selec.unwrap()
                                    }));
                            }
                            if node.children.iter().any(|c| c.borrow().neg_selec.is_some()) {
                                node.neg_selec = Some(
                                    node.children
                                        .iter()
                                        .filter_map(|c| c.borrow().neg_selec)
                                        .fold(0., |selec, ns| selec + ns - selec * ns),
                                );
                            }
                            node.is_applied = node.children.iter().all(|c| c.borrow().is_applied);
                        }
                        PredGraphNodeType::Or => {
                            if node.children.iter().any(|c| c.borrow().pos_selec.is_some()) {
                                node.pos_selec = Some(
                                    node.children
                                        .iter()
                                        .filter_map(|c| c.borrow().pos_selec)
                                        .fold(0., |selec, ps| selec + ps - selec * ps),
                                );
                            }
                            if node.children.iter().all(|c| c.borrow().neg_selec.is_some()) {
                                node.neg_selec =
                                    Some(node.children.iter().fold(1., |selec, c| {
                                        selec * c.borrow().neg_selec.unwrap()
                                    }));
                            }
                            node.is_applied = node.children.iter().all(|c| c.borrow().is_applied);
                        }
                        _ => {
                            panic!("wtf, why atom");
                        }
                    }

                    node.parent.as_ref().and_then(|p| Some(p.clone()))
                };
        }
    }

    fn total_cost(&self) -> f64 {
        fn total_cost_helper(root: &PredGraphNode) -> f64 {
            if root.is_applied {
                return 0.;
            }

            match root.node_type {
                PredGraphNodeType::Atom => root.atom_node.as_ref().unwrap().lookup_cost(),
                PredGraphNodeType::And => {
                    let mut selec = 1.;
                    for child in &root.children {
                        let child = child.borrow();
                        if child.is_applied {
                            selec *= child.pos_selec.expect("Why no pos selec when applied");
                        } else if child.neg_selec.is_some() {
                            selec *= 1. - child.neg_selec.unwrap();
                        }
                    }

                    let mut cost = 0.;
                    for child in &root.children {
                        let child = child.borrow();
                        cost += total_cost_helper(&child)
                            * if child.is_applied {
                                selec / child.pos_selec.unwrap()
                            } else if child.neg_selec.is_some() {
                                selec / (1. - child.neg_selec.unwrap())
                            } else {
                                selec
                            };
                    }
                    cost
                }
                PredGraphNodeType::Or => {
                    let mut selec = 1.;
                    for child in &root.children {
                        let child = child.borrow();
                        if child.pos_selec.is_some() {
                            selec *= 1. - child.pos_selec.unwrap();
                        }
                    }

                    let mut cost = 0.;
                    for child in &root.children {
                        let child = child.borrow();
                        cost += total_cost_helper(&child)
                            * if child.pos_selec.is_some() {
                                selec / (1. - child.pos_selec.unwrap())
                            } else {
                                selec
                            };
                    }
                    cost
                }
            }
        }
        total_cost_helper(&self.root.borrow())
    }

    pub fn get_root_idx(&self) -> Option<RoaringBitmap> {
        self.pos_map.get(&self.root.borrow().id).cloned()
    }

    fn one_lookahead(
        &mut self,
        index: &RoaringBitmap,
        exec_params: &ExecParams,
        exec_stats: &mut ExecStats,
    ) -> RoaringBitmap {
        let mut pred_atoms = self.pred_atoms();

        for _ in 0..pred_atoms.len() {
            let mut atom_costs = pred_atoms
                .into_iter()
                .map(|atom| {
                    let total_cost = {
                        let new_graph = self.copy();
                        new_graph.fake_apply_pred_atom(&atom.borrow().id);
                        new_graph.total_cost()
                    };
                    (atom, total_cost)
                })
                .collect::<Vec<(Rc<RefCell<PredGraphNode>>, f64)>>();
            atom_costs.sort_unstable_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(Ordering::Equal));
            let (first, rest) = atom_costs
                .split_first()
                .expect("Why is there nothing in the atom_costs");
            let id = first.0.borrow().id;
            self.apply_pred_atom(id, index, exec_params, exec_stats);
            pred_atoms = rest.into_iter().map(|x| x.0.clone()).collect();
        }

        self.get_root_idx()
            .expect("Why is root applied but not in pos_map")
    }

    // Returns ancestors from root to given pred node.
    fn get_lineage(&self, atom: &Rc<RefCell<PredGraphNode>>) -> Vec<Rc<RefCell<PredGraphNode>>> {
        if let PredGraphNodeType::Atom = atom.borrow().node_type {
            // Ok
        } else {
            panic!("Given node was not a pred atom");
        }

        let mut lineage = vec![];
        let mut node = atom.clone();
        loop {
            lineage.push(node.clone());
            let parent = node.borrow().parent.as_ref().and_then(|p| Some(p.clone()));
            if parent.is_some() {
                node = parent.unwrap();
            } else {
                break;
            }
        }
        lineage.reverse();
        lineage
    }

    fn calc_bestd_index(
        &self,
        lineage: &Vec<Rc<RefCell<PredGraphNode>>>,
        init_index: &RoaringBitmap,
    ) -> RoaringBitmap {
        let mut index = init_index.clone();
        for win in lineage.windows(2) {
            let parent = win[0].borrow();
            let child = win[1].borrow();
            let other_children = parent.children.iter().filter(|c| c.borrow().id != child.id);

            for other_child in other_children {
                let other_child = other_child.borrow();
                match parent.node_type {
                    PredGraphNodeType::And => {
                        if other_child.is_applied {
                            index &= self.pos_map.get(&other_child.id).expect(
                                "Did not update pos_map even though other_child is applied",
                            );
                        } else if other_child.neg_selec.is_some() {
                            index -= self.neg_map.get(&other_child.id).expect(
                                "Did not update neg_map even though other_child has neg_selec",
                            );
                        }
                    }
                    PredGraphNodeType::Or => {
                        if other_child.pos_selec.is_some() {
                            index -= self.pos_map.get(&other_child.id).expect(
                                "Did not update pos_map even though other_child has pos_selec",
                            );
                        }
                    }
                    _ => {
                        panic!("Wtf");
                    }
                }
            }
        }
        index
    }

    // This will actually go and evaluate the predicates
    pub fn apply_pred_atom(
        &mut self,
        id: Id,
        init_index: &RoaringBitmap,
        exec_params: &ExecParams,
        exec_stats: &mut ExecStats,
    ) -> PredGraph {
        let pred_atom = self
            .node_map
            .get(&id)
            .expect(&format!("Could not find node: {}", id))
            .clone();

        let mut lineage = self.get_lineage(&pred_atom);
        let index = self.calc_bestd_index(&lineage, init_index);
        let evaled_idx = pred_atom
            .borrow()
            .atom_node
            .as_ref()
            .expect("Why does atom not have atom_node")
            .eval(&index, exec_params, exec_stats);

        {
            let mut pred_atom = pred_atom.borrow_mut();
            pred_atom.is_applied = true;
            pred_atom.pos_selec = Some(evaled_idx.len() as f64 / init_index.len() as f64);
            pred_atom.neg_selec = Some(1. - pred_atom.pos_selec.unwrap());
            self.pos_map.insert(pred_atom.id, evaled_idx.clone());
            self.neg_map.insert(pred_atom.id, init_index - &evaled_idx);
        }

        lineage.reverse();
        for node in &lineage[1..] {
            let mut node = node.borrow_mut();
            match node.node_type {
                PredGraphNodeType::And => {
                    if node.children.iter().all(|c| c.borrow().pos_selec.is_some()) {
                        self.pos_map.insert(
                            node.id,
                            node.children.iter().fold(init_index.clone(), |i, c| {
                                i & self.pos_map.get(&c.borrow().id).expect(
                                    &format!("Why do you have no pos_map entry when you have pos_selec {} {:?} {:?}", c.borrow().id, c.borrow().node_type, c.borrow()),
                                )
                            }),
                        );
                    }

                    if node.children.iter().any(|c| c.borrow().neg_selec.is_some()) {
                        self.neg_map.insert(
                            node.id,
                            node.children
                                .iter()
                                .filter(|c| c.borrow().neg_selec.is_some())
                                .fold(RoaringBitmap::new(), |i, c| {
                                    i | self.neg_map.get(&c.borrow().id).expect(&format!(
                                        "Child has no entry in neg_map {:?}",
                                        c.borrow()
                                    ))
                                }),
                        );
                    }
                }
                PredGraphNodeType::Or => {
                    if node.children.iter().any(|c| c.borrow().pos_selec.is_some()) {
                        self.pos_map.insert(
                            node.id,
                            node.children
                                .iter()
                                .filter(|c| c.borrow().pos_selec.is_some())
                                .fold(RoaringBitmap::new(), |i, c| {
                                    i | self.pos_map.get(&c.borrow().id).expect(&format!(
                                        "Why do you have no pos_map entry when you have
                                            pos_selec {} {:?} {:?}",
                                        c.borrow().id,
                                        c.borrow().node_type,
                                        c.borrow()
                                    ))
                                }),
                        );
                    }

                    if node.children.iter().all(|c| c.borrow().neg_selec.is_some()) {
                        self.neg_map.insert(
                            node.id,
                            node.children.iter().fold(init_index.clone(), |i, c| {
                                i & self.neg_map.get(&c.borrow().id).expect(&format!(
                                    "Child has no entry in neg_map {:?}",
                                    c.borrow()
                                ))
                            }),
                        );
                    }
                }
                PredGraphNodeType::Atom => {
                    panic!("wtf");
                }
            }
            let pos_idx = self.pos_map.get(&node.id);
            if pos_idx.is_some() {
                node.pos_selec = Some(pos_idx.unwrap().len() as f64 / init_index.len() as f64);
            }
            let neg_idx = self.neg_map.get(&node.id);
            if neg_idx.is_some() {
                node.neg_selec = Some(neg_idx.unwrap().len() as f64 / init_index.len() as f64);
            }
            node.is_applied = node.children.iter().all(|c| c.borrow().is_applied);

            if let (None, None) = (pos_idx, neg_idx) {
                break;
            }
        }

        self.copy()
    }
}

impl PredNode {
    // TODO In future allow this to take a `Option<&RoaringBitmap>` and have None stand for no
    // index. This will probably speed things up.
    pub fn eval(
        &self,
        index: &RoaringBitmap,
        exec_params: &ExecParams,
        exec_stats: &mut ExecStats,
    ) -> RoaringBitmap {
        if let ApproxOptType::OnePredLookahead = exec_params.approx_opt_type {
            let mut graph = self.make_graph();
            return graph.one_lookahead(index, exec_params, exec_stats);
        } else if let ApproxOptType::Byp = exec_params.approx_opt_type {
            let byp = Byp::new(self);
            let plan = byp.find_plan(exec_stats);
            return plan.eval(index, exec_params, exec_stats);
        } else if let ApproxOptType::Tdacb = exec_params.approx_opt_type {
            let now = Instant::now();
            let tdacb = Tdacb::new(self);
            let plan = tdacb.find_plan(exec_stats);
            exec_stats.plan_time_ms = now.elapsed().as_millis();
            return tdacb.eval(&plan, index, exec_params, exec_stats);
        } else if let ApproxOptType::BDC | ApproxOptType::BDCWithBestD = exec_params.approx_opt_type
        {
            let now = Instant::now();
            let bdc = BDC::new(self);
            let plan = bdc.plan();
            exec_stats.plan_time_ms = now.elapsed().as_millis();
            if exec_params.check_plan_only {
                plan.check_ordering();
                return RoaringBitmap::new();
            }
            if let ApproxOptType::BDC = exec_params.approx_opt_type {
                return bdc.eval(&plan, None, index, exec_params, exec_stats);
            } else {
                return bdc.eval(
                    &plan,
                    Some(self.make_graph()),
                    index,
                    exec_params,
                    exec_stats,
                );
            }
        }

        match self {
            PredNode::OrNode(children) => {
                let mut children = children.iter().collect::<Vec<&PredNode>>();
                if !exec_params.disable_or_opt {
                    children.sort_unstable_by(|a, b| {
                        a.get_or_weight().partial_cmp(&b.get_or_weight()).unwrap()
                    });
                    //println!(
                    //    "or: {:?}",
                    //    children
                    //        .iter()
                    //        .map(|c| c.get_or_weight())
                    //        .collect::<Vec<f64>>()
                    //);
                    let running = index.clone();
                    let evaled = children.iter().fold(running, |running, c| {
                        let evaled = c.eval(&running, exec_params, exec_stats);
                        running - evaled
                    });
                    index - evaled
                } else {
                    let mut total = RoaringBitmap::new();
                    for child in children.iter() {
                        total |= child.eval(&index, exec_params, exec_stats);
                    }
                    total
                }
            }
            PredNode::AndNode(children) => {
                let mut children = children.iter().collect::<Vec<&PredNode>>();
                children.sort_unstable_by(|a, b| {
                    a.get_and_weight().partial_cmp(&b.get_and_weight()).unwrap()
                });
                //println!(
                //    "and: {:?}",
                //    children
                //        .iter()
                //        .map(|c| c.get_and_weight())
                //        .collect::<Vec<f64>>()
                //);
                let running = index.clone();
                children.iter().fold(running, |running, c| {
                    c.eval(&running, exec_params, exec_stats)
                })
            }
            PredNode::PredAtomNode(node) => {
                let ret = node.eval(index, exec_params, exec_stats);
                ret
            }
        }
    }

    /// Returns the likelihood that this predicate node resolves to true given a set of
    /// assignments.
    pub fn calc_likelihood(&self, assignments: &HashMap<&PredAtomNode, bool>) -> f64 {
        match self {
            PredNode::PredAtomNode(atom) => assignments
                .get(atom)
                .map_or_else(|| atom.lookup_selectivity(), |&a| if a { 1. } else { 0. }),
            PredNode::AndNode(children) => children
                .iter()
                .map(|c| c.calc_likelihood(assignments))
                .product(),
            PredNode::OrNode(children) => children
                .iter()
                .map(|c| c.calc_likelihood(assignments))
                .fold(0., |accum, item| accum + item - accum * item),
        }
    }

    pub fn set_selec_map(
        &mut self,
        selectivities: &Rc<HashMap<String, f64>>,
        costs: &Rc<HashMap<String, f64>>,
    ) {
        match self {
            PredNode::OrNode(children) | PredNode::AndNode(children) => {
                for child in children {
                    child.set_selec_map(selectivities, costs);
                }
            }
            PredNode::PredAtomNode(ref mut node) => {
                node.selec_map = Some(selectivities.clone());
                node.cost_map = Some(costs.clone());
            }
        }
    }

    // Return the estimated (selectivity, cost) pair for a node.
    pub fn get_selec_cost(&self) -> (f64, f64) {
        let mut total_selec;
        let mut total_cost = 0.0;
        match self {
            PredNode::OrNode(children) => {
                let mut children = children.iter().collect::<Vec<&PredNode>>();
                children.sort_unstable_by(|a, b| {
                    a.get_or_weight().partial_cmp(&b.get_or_weight()).unwrap()
                });
                total_selec = 0.;
                for child in children {
                    let (selec, cost) = child.get_selec_cost();
                    total_cost += (1.0 - total_selec) * cost;
                    total_selec = selec + total_selec * (1.0 - selec);
                }
            }
            PredNode::AndNode(children) => {
                let mut children = children.iter().collect::<Vec<&PredNode>>();
                children.sort_unstable_by(|a, b| {
                    a.get_and_weight().partial_cmp(&b.get_and_weight()).unwrap()
                });
                total_selec = 1.;
                for child in children {
                    let (selec, cost) = child.get_selec_cost();
                    total_cost += total_selec * cost;
                    total_selec = selec * total_selec;
                }
            }
            PredNode::PredAtomNode(node) => {
                total_selec = node.lookup_selectivity();
                total_cost = node.lookup_cost();
            }
        }
        (total_selec, total_cost)
    }

    pub fn get_all_atoms(&self) -> Vec<&PredAtomNode> {
        match self {
            PredNode::PredAtomNode(atom) => vec![atom],
            PredNode::OrNode(children) | PredNode::AndNode(children) => children
                .iter()
                .map(|c| c.get_all_atoms())
                .flatten()
                .collect(),
        }
    }

    fn get_or_weight(&self) -> f64 {
        let (selec, cost) = self.get_selec_cost();
        (cost + SMOOTHING_PARAMETER) / (selec + SMOOTHING_PARAMETER)
    }

    fn get_and_weight(&self) -> f64 {
        let (selec, cost) = self.get_selec_cost();
        (cost + SMOOTHING_PARAMETER) / (1.0 - selec + SMOOTHING_PARAMETER)
    }

    pub fn depth(&self) -> u32 {
        match self {
            PredNode::OrNode(children) | PredNode::AndNode(children) => {
                1 + children.iter().map(|c| c.depth()).max().unwrap()
            }
            PredNode::PredAtomNode(_) => 0,
        }
    }

    pub fn num_preds(&self) -> usize {
        match self {
            PredNode::OrNode(children) | PredNode::AndNode(children) => {
                children.iter().map(|c| c.num_preds()).sum()
            }
            PredNode::PredAtomNode(_) => 1,
        }
    }

    fn make_graph(&self) -> PredGraph {
        fn make_graph_helper(
            node_self: &PredNode,
            parent: Option<Rc<RefCell<PredGraphNode>>>,
        ) -> Rc<RefCell<PredGraphNode>> {
            let node_type = match node_self {
                PredNode::OrNode(_) => PredGraphNodeType::Or,
                PredNode::AndNode(_) => PredGraphNodeType::And,
                PredNode::PredAtomNode(_) => PredGraphNodeType::Atom,
            };
            let id = if let PredNode::PredAtomNode(atom) = node_self {
                atom.id
            } else {
                Id::new()
            };
            let node = Rc::new(RefCell::new(PredGraphNode {
                parent,
                children: vec![],
                atom_node: None,
                node_type,
                id,
                is_applied: false,
                pos_selec: None,
                neg_selec: None,
            }));

            match node_self {
                PredNode::OrNode(children) | PredNode::AndNode(children) => {
                    node.borrow_mut().children = children
                        .iter()
                        .map(|c| make_graph_helper(c, Some(node.clone())))
                        .collect();
                }
                PredNode::PredAtomNode(atom) => {
                    node.borrow_mut().atom_node = Some(Rc::new(atom.clone()));
                }
            };

            node
        }

        PredGraph::new(make_graph_helper(self, None))
    }

    // Get all file tables this predicate references
    pub fn get_ref_tables(&self) -> Vec<Rc<FileTable>> {
        let mut tables = match self {
            PredNode::OrNode(children) | PredNode::AndNode(children) => children
                .iter()
                .map(|child| child.get_ref_tables())
                .flatten()
                .collect(),
            PredNode::PredAtomNode(atom) => {
                let cols = atom.expr.get_all_cols();
                cols.iter()
                    .map(|col| col.table.upgrade().unwrap().clone())
                    .collect::<Vec<Rc<FileTable>>>()
            }
        };
        tables.sort_unstable_by_key(|t| t.id());
        tables.dedup_by_key(|t| t.id());
        tables
    }

    pub fn is_outwardly_conjunctive(&self) -> bool {
        match self {
            PredNode::PredAtomNode(_) => true,
            PredNode::AndNode(_) => true,
            PredNode::OrNode(_) => false,
        }
    }

    pub fn has_or_descendant(&self) -> bool {
        match self {
            PredNode::PredAtomNode(_) => false,
            PredNode::OrNode(_) => true,
            PredNode::AndNode(children) => children.iter().any(|c| c.has_or_descendant()),
        }
    }

    pub fn has_multi_table_clauses(&self) -> bool {
        match self {
            PredNode::PredAtomNode(_) => false,
            PredNode::AndNode(children) | PredNode::OrNode(children) => {
                children.iter().any(|c| c.get_ref_tables().len() > 1)
            }
        }
    }
}

#[derive(Clone)]
pub struct PredAtomNode {
    pub id: Id,
    pub expr: Expr,
    pub selec_map: Option<Rc<HashMap<String, f64>>>,
    pub cost_map: Option<Rc<HashMap<String, f64>>>,
}

impl PartialEq for PredAtomNode {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for PredAtomNode {}

impl Hash for PredAtomNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl fmt::Debug for PredAtomNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.expr)
    }
}

impl PredAtomNode {
    pub fn lookup_selectivity(&self) -> f64 {
        *self
            .selec_map
            .as_ref()
            .and_then(|map| map.get(&self.expr.to_string()))
            .unwrap_or(&1.0)
    }

    pub fn lookup_cost(&self) -> f64 {
        *self
            .cost_map
            .as_ref()
            .and_then(|map| map.get(&self.expr.to_string()))
            .unwrap_or(&1.0)
    }

    pub fn eval(
        &self,
        index: &RoaringBitmap,
        exec_params: &ExecParams,
        exec_stats: &mut ExecStats,
    ) -> RoaringBitmap {
        let result = self.expr.eval(
            &RunContext {
                index: Some(index.clone()),
                groups: vec![],
                ref_table: None,
                exec_params: exec_params.clone(),
            },
            exec_stats,
        );
        exec_stats.num_preds_evaled += index.len() as u128;
        assert!(result.cols.len() == 1);
        let (_, col) = result.cols.iter().next().unwrap();
        if let DBCol::Bool(vals) = col {
            let ret: RoaringBitmap = index
                .iter()
                .enumerate()
                .filter(|(i, _)| {
                    if result.ref_table.is_none() {
                        vals[0]
                    } else {
                        vals[*i]
                    }
                })
                .map(|(_, idx)| idx)
                .collect();
            debug!(
                "Evaluated {}, est sel: {} index size: {}, ret size: {}",
                self.expr,
                self.lookup_selectivity(),
                index.len(),
                ret.len()
            );
            ret
        } else {
            panic!("Pred atom returned non-bool type {:?}", col);
        }
    }
}

#[derive(Clone)]
pub struct GroupByNode {
    items: Vec<SelectItem>,
}

impl GroupByNode {
    pub fn eval(&self, run_context: &RunContext, exec_stats: &mut ExecStats) -> Vec<Vec<DBCol>> {
        let mut groups: Vec<Vec<DBCol>> = vec![];
        for item in &self.items {
            let mut result = item.eval(&run_context, exec_stats);
            assert!(result.cols.len() == 1);
            let (_, col) = result.cols.iter_mut().next().unwrap();
            if let DBCol::Float(_) | DBCol::Double(_) = col {
                // Don't want to deal with float/double-based groups
                continue;
            }
            debug!("what is index: {:?}", col.len());

            // XXX We are using this to see if this is a value-based result. It might not be that
            // only value-based results have no ref_table.
            if result.ref_table.is_none() {
                if groups.len() > col.len() {
                    col.repeat(groups.len());
                } else {
                    continue;
                }
            } else {
                groups.resize_with(col.len(), Default::default);
            }
            for i in 0..col.len() {
                match col {
                    DBCol::Int(vals) => {
                        groups[i].push(DBCol::Int(vec![vals[i]]));
                    }
                    DBCol::Long(vals) => {
                        groups[i].push(DBCol::Long(vec![vals[i]]));
                    }
                    DBCol::Str(vals) => {
                        groups[i].push(DBCol::Str(vec![vals[i].clone()]));
                    }
                    DBCol::Bool(vals) => {
                        groups[i].push(DBCol::Bool(vec![vals[i]]));
                    }
                    _ => {
                        panic!("Don't support group by type");
                    }
                }
            }
        }
        groups
    }
}

// This is just another pointer to the `ref_table` in `ParseContext`
#[derive(Clone)]
pub struct TableNode {
    pub table: Rc<dyn Table>,
    pub join_table_type: JoinTableType,
}

// `tables` is a vector of tables specific to this query (i.e., mostly join tables which are
// created for this query)
// `aliases` is a mapping of aliased table names to their respective tables
#[derive(Clone)]
pub struct Query<'a> {
    pub projection: SelectNode,
    pub filter: Option<PredNode>,
    pub group_by: GroupByNode,
    pub table: TableNode,
    pub context: ParseContext<'a>,
}

impl fmt::Display for Query<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SELECT {} FROM {} {}{};",
            self.projection
                .items
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<String>>()
                .join(", "),
            self.table.table,
            self.filter
                .as_ref()
                .and_then(|pred| Some(format!("WHERE {}", pred)))
                .unwrap_or("".to_string()),
            if self.group_by.items.is_empty() {
                "".to_string()
            } else {
                format!(
                    " GROUP BY {}",
                    self.group_by
                        .items
                        .iter()
                        .map(|i| i.to_string())
                        .collect::<Vec<String>>()
                        .join(", ")
                )
            }
        )
    }
}

// Here `other_table` is another possible reference table that can be required if an expression is
// a join constraint; the expressions need to be evaluated under different tables
#[derive(Clone)]
pub struct ParseContext<'a> {
    pub ref_table: Rc<dyn Table>,
    pub other_tables: Vec<Rc<dyn Table>>,
    pub db: &'a DB,
    pub aliases: HashMap<String, Rc<dyn Table>>,
    pub file_tables: &'a HashMap<String, Rc<dyn Table>>,
}

#[derive(Debug)]
pub enum ParseError {
    RawParserError(RawParserError),
    JoinConstraintError(ExprCreateError),
}

pub struct Parser<'a> {
    dialect: PostgreSqlDialect,
    db: &'a DB,
}

#[derive(Clone)]
pub enum JoinTableType {
    Normal,
}

pub struct ParseParams {
    pub join_table_type: JoinTableType,
}

impl Default for ParseParams {
    fn default() -> Self {
        Self {
            join_table_type: JoinTableType::Normal,
        }
    }
}

impl<'a> Parser<'a> {
    pub fn new(db: &'a DB) -> Parser<'a> {
        Parser {
            dialect: PostgreSqlDialect {},
            db,
        }
    }

    pub fn parse(&self, sql: &str, params: &ParseParams) -> Result<Query<'a>, ParseError> {
        let parsed = &RawParser::parse_sql(&self.dialect, sql.to_string())
            .map_err(|err| ParseError::RawParserError(err))?[0];
        let query;
        if let ast::Statement::Query(parsed) = parsed {
            if let ast::Query {
                body: ast::SetExpr::Select(select),
                ..
            } = parsed.as_ref()
            {
                query = select.as_ref()
            } else {
                panic!("Did not match expected select pattern");
            }
        } else {
            panic!("Did not match expected select pattern");
        }

        assert!(query.from.len() == 1, "query has more than one FROM table");
        let context = if let JoinTableType::Normal = params.join_table_type {
            self.parse_table(&query.from[0])?
        } else {
            panic!("no other type");
        };
        let table = TableNode {
            table: context.ref_table.clone(),
            join_table_type: params.join_table_type.clone(),
        };
        let filter = query
            .selection
            .as_ref()
            .and_then(|s| Some(self.parse_predicate(&s, &context)))
            .and_then(|n| n.and_then(|n| Ok(self.flatten_predicate(n))).ok());
        let (projection, valid_projs) = self.parse_projection(&query.projection, &context);
        let group_by = self.parse_group_by(&query.group_by, &context, &projection, valid_projs);

        Ok(Query {
            context,
            filter,
            projection,
            group_by,
            table,
        })
    }

    fn parse_projection(
        &self,
        items: &Vec<ast::SelectItem>,
        context: &ParseContext,
    ) -> (SelectNode, Vec<usize>) {
        debug!("*** PARSING PROJECTION ***");
        let mut valid_projections = Vec::new();
        let sel = SelectNode {
            items: items
                .iter()
                .enumerate()
                .filter_map(|(i, item)| {
                    let ret = match item {
                        ast::SelectItem::UnnamedExpr(expr) => SelectItem::new(expr, context),
                        ast::SelectItem::ExprWithAlias { expr, .. } => {
                            SelectItem::new(expr, context)
                        }
                        ast::SelectItem::QualifiedWildcard(ast::ObjectName(idents)) => {
                            SelectItem::new(&ast::Expr::QualifiedWildcard(idents.to_vec()), context)
                        }
                        ast::SelectItem::Wildcard => SelectItem::new(&ast::Expr::Wildcard, context),
                    }
                    .ok();
                    if ret.is_some() {
                        valid_projections.push(i);
                    }
                    ret
                })
                .collect(),
        };
        debug!(
            "[Projection] Items:\n{}",
            sel.items
                .iter()
                .map(|i| format!("{}{}", " ".repeat(8), i))
                .collect::<Vec<String>>()
                .join(",\n")
        );
        (sel, valid_projections)
    }

    // FIXME We skip over all the groups identified by aliases
    fn parse_group_by(
        &self,
        items: &Vec<ast::Expr>,
        context: &ParseContext,
        projection: &SelectNode,
        valid_projs: Vec<usize>,
    ) -> GroupByNode {
        debug!("*** PARSING GROUP BY ***");
        let node = GroupByNode {
            items: items
                .iter()
                .filter_map(|item| match item {
                    ast::Expr::Value(val) => {
                        let num;

                        if let ast::Value::Long(val) = val {
                            num = (*val - 1) as usize
                        } else {
                            panic!("Sighhh");
                        }

                        let r = valid_projs.binary_search(&num);
                        r.ok().and_then(|idx| Some(projection.items[idx].clone()))
                    }
                    _ => SelectItem::new(item, context).ok(),
                })
                .collect(),
        };
        debug!(
            "[GroupBy] Items:\n{}",
            node.items
                .iter()
                .map(|i| format!("{}{}", " ".repeat(8), i))
                .collect::<Vec<String>>()
                .join(",\n")
        );
        node
    }

    fn flatten_predicate(&self, pred_node: PredNode) -> PredNode {
        macro_rules! flatten_helper {
            ($children:expr, $pred_type:path) => {{
                let mut new_children = vec![];
                for child in $children {
                    let flattened = flatten(child);
                    match flattened {
                        $pred_type(mut grandchildren) => {
                            new_children.append(&mut grandchildren);
                        }
                        _ => {
                            new_children.push(flattened);
                        }
                    };
                }
                $pred_type(new_children)
            }};
        }
        fn flatten(pred_node: PredNode) -> PredNode {
            match pred_node {
                PredNode::AndNode(children) => flatten_helper!(children, PredNode::AndNode),
                PredNode::OrNode(children) => flatten_helper!(children, PredNode::OrNode),
                _ => pred_node,
            }
        }

        flatten(pred_node)
    }

    fn parse_predicate(
        &self,
        expr: &ast::Expr,
        context: &ParseContext,
    ) -> Result<PredNode, ExprCreateError> {
        // Unwrap until we hit non-ANDs/ORs, then call Expr::new
        match expr {
            ast::Expr::BinaryOp {
                left,
                op: ast::BinaryOperator::And,
                right,
            } => {
                let left = self.parse_predicate(left, context);
                let right = self.parse_predicate(right, context);
                if left.is_ok() && right.is_ok() {
                    Ok(PredNode::AndNode(vec![left.unwrap(), right.unwrap()]))
                } else if left.is_ok() {
                    left
                } else if right.is_ok() {
                    right
                } else {
                    Err(ExprCreateError::ColDoesNotExist)
                }
            }
            ast::Expr::BinaryOp {
                left,
                op: ast::BinaryOperator::Or,
                right,
            } => {
                let left = self.parse_predicate(left, context);
                let right = self.parse_predicate(right, context);
                if left.is_ok() && right.is_ok() {
                    Ok(PredNode::OrNode(vec![left.unwrap(), right.unwrap()]))
                } else if left.is_ok() {
                    left
                } else if right.is_ok() {
                    right
                } else {
                    Err(ExprCreateError::ColDoesNotExist)
                }
            }
            ast::Expr::Nested(expr) => self.parse_predicate(expr, context),
            _ => Ok(PredNode::PredAtomNode(PredAtomNode {
                id: Id::new(),
                expr: Expr::new(expr, context)?,
                selec_map: None,
                cost_map: None,
            })),
        }
    }

    // Parses the FROM table of the query and returns a ParseContext where the ref_table part is
    // the overall joined table in the query.
    fn parse_table(&self, table: &ast::TableWithJoins) -> Result<ParseContext<'a>, ParseError> {
        // This submethod also adds to aliases if an alias exists.
        fn unwrap_table<'a>(
            table: &ast::TableFactor,
            db: &'a DB,
            aliases: &mut HashMap<String, Rc<dyn Table>>,
        ) -> &'a Rc<dyn Table> {
            if let ast::TableFactor::Table { name, alias, .. } = table {
                assert!(name.0.len() == 1);
                let name = &name.0[0];
                let result = db.file_tables().get(name).unwrap();
                if alias.is_some() {
                    let alias = &alias.as_ref().unwrap().name;
                    aliases.insert(alias.to_string(), result.clone());
                }
                result
            } else {
                panic!("Relation is not of form table: {:?}", table);
            }
        }

        let mut aliases = HashMap::new();
        let mut running = unwrap_table(&table.relation, self.db, &mut aliases).clone();
        let mut context = ParseContext {
            ref_table: running.clone(),
            other_tables: vec![],
            db: self.db,
            aliases: aliases,
            file_tables: self.db.file_tables(),
        };
        for join in &table.joins {
            let to_join = unwrap_table(&join.relation, self.db, &mut context.aliases);
            context.ref_table = running.clone();
            context.other_tables = vec![to_join.clone()];
            running = match &join.join_operator {
                ast::JoinOperator::Inner(constraint) => {
                    let constraint = match constraint {
                        ast::JoinConstraint::On(constraint) => Expr::new(&constraint, &context)
                            .map_err(|err| ParseError::JoinConstraintError(err))?,
                        ast::JoinConstraint::Using(idents) => {
                            assert_eq!(idents.len(), 1);
                            let left =
                                Expr::new(&ast::Expr::Identifier(idents[0].clone()), &context)
                                    .map_err(|err| ParseError::JoinConstraintError(err))?;
                            context.ref_table = to_join.clone();
                            context.other_tables = vec![running.clone()];
                            let right =
                                Expr::new(&ast::Expr::Identifier(idents[0].clone()), &context)
                                    .map_err(|err| ParseError::JoinConstraintError(err))?;
                            Expr::BinaryOp {
                                left: Box::new(left),
                                right: Box::new(right),
                                op: BinaryOperator::Eq,
                            }
                        }
                        _ => {
                            panic!("Unknown join constraint type: {:?}", constraint);
                        }
                    };
                    JoinTable::new(JoinType::Inner, constraint, &running, &to_join)
                }
                ast::JoinOperator::LeftOuter(constraint) => {
                    let constraint = match constraint {
                        ast::JoinConstraint::On(constraint) => Expr::new(&constraint, &context)
                            .map_err(|err| ParseError::JoinConstraintError(err))?,
                        ast::JoinConstraint::Using(idents) => {
                            assert_eq!(idents.len(), 1);
                            let left =
                                Expr::new(&ast::Expr::Identifier(idents[0].clone()), &context)
                                    .map_err(|err| ParseError::JoinConstraintError(err))?;
                            context.ref_table = to_join.clone();
                            context.other_tables = vec![running.clone()];
                            let right =
                                Expr::new(&ast::Expr::Identifier(idents[0].clone()), &context)
                                    .map_err(|err| ParseError::JoinConstraintError(err))?;
                            Expr::BinaryOp {
                                left: Box::new(left),
                                right: Box::new(right),
                                op: BinaryOperator::Eq,
                            }
                        }
                        _ => {
                            panic!("Unknown join constraint type: {:?}", constraint);
                        }
                    };
                    JoinTable::new(JoinType::LeftOuter, constraint, &running, &to_join)
                }
                _ => {
                    panic!(
                        "We do not handle join type ({:?}) for now",
                        join.join_operator
                    );
                }
            };
        }
        Ok(ParseContext {
            ref_table: running,
            other_tables: vec![],
            db: self.db,
            aliases: context.aliases,
            file_tables: self.db.file_tables(),
        })
    }
}
