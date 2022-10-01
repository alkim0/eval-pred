use super::exec::{ExecParams, ExecStats};
use super::parser::{PredAtomNode, PredNode};
use super::table::Id;
use itertools::Itertools;
use roaring::RoaringBitmap;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::ops::Add;
use std::rc::Rc;

type PredId = Id;
type Cost = f64;

pub struct Byp {
    summands: Vec<BypSummand>,
}

#[derive(Clone)]
struct BypPlanNode {
    pred: BypPred,
    true_child: Option<Box<BypPlanNode>>,
    false_child: Option<Box<BypPlanNode>>,
}

impl BypPlanNode {
    fn from_preds(mut preds: Vec<BypPred>) -> Self {
        assert!(!preds.is_empty());
        preds.reverse();
        let mut node = BypPlanNode {
            pred: preds[0].clone(),
            true_child: None,
            false_child: None,
        };
        for pred in &preds[1..] {
            node = BypPlanNode {
                pred: pred.clone(),
                true_child: Some(Box::new(node)),
                false_child: None,
            };
        }
        node
    }
}

// A plan
#[derive(Clone)]
pub struct BypPlan {
    root: Option<BypPlanNode>,
}

// Each BypSummand
#[derive(Clone)]
struct BypSummand {
    preds: HashMap<PredId, BypPred>,
}

#[derive(Clone)]
struct BypPred {
    id: PredId,
    atom: Rc<PredAtomNode>,
}

impl PartialEq for BypSummand {
    fn eq(&self, other: &Self) -> bool {
        self.preds.keys().collect::<HashSet<&PredId>>()
            == other.preds.keys().collect::<HashSet<&PredId>>()
    }
}

impl Eq for BypSummand {}

impl Add for BypSummand {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        BypSummand {
            preds: self.preds.into_iter().chain(other.preds).collect(),
        }
    }
}

impl Hash for BypSummand {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for id in self.preds.keys() {
            id.hash(state);
        }
    }
}

impl BypSummand {
    fn new(preds: Vec<BypPred>) -> Self {
        let preds = preds.into_iter().map(|p| (p.id, p)).collect();
        BypSummand { preds }
    }
}

impl Byp {
    pub fn new(root: &PredNode) -> Self {
        let summands = Self::convert_to_dnf(root);
        Self { summands }
    }

    fn convert_to_dnf(root: &PredNode) -> Vec<BypSummand> {
        let mut pred_map = HashMap::new();
        let mut fringe = vec![root];
        while !fringe.is_empty() {
            let node = fringe.pop().unwrap();
            match node {
                PredNode::AndNode(children) | PredNode::OrNode(children) => {
                    for child in children {
                        fringe.push(child);
                    }
                }
                PredNode::PredAtomNode(atom) => {
                    pred_map.insert(PredId::new(), Rc::new(atom.clone()));
                }
            }
        }

        // Returns ORs of summands
        fn convert(node: &PredNode) -> Vec<BypSummand> {
            let summands = match node {
                PredNode::PredAtomNode(atom) => vec![BypSummand::new(vec![BypPred {
                    id: PredId::new(),
                    atom: Rc::new(atom.clone()),
                }])],

                PredNode::OrNode(children) => {
                    children.iter().map(|c| convert(c)).flatten().collect()
                }

                PredNode::AndNode(children) => children
                    .iter()
                    .map(|c| convert(c))
                    .multi_cartesian_product()
                    .map(|prod| {
                        prod.into_iter()
                            .fold(BypSummand::new(vec![]), |sum, i| sum + i)
                    })
                    .collect(),
            };

            summands
                .into_iter()
                .collect::<HashSet<BypSummand>>()
                .into_iter()
                .collect::<Vec<BypSummand>>()
        }

        convert(root)
    }

    pub fn find_plan(&self, exec_stats: &mut ExecStats) -> BypPlan {
        let plan = BypPlan { root: None };
        let result = plan.find_plan(self.summands.clone(), exec_stats);
        result.expect("Why no plan").0
    }
}

impl BypPlan {
    fn add_new(
        &mut self,
        true_ids: HashSet<PredId>,
        mut false_ids: HashSet<PredId>,
        new: BypPlanNode,
    ) {
        println!("00 {}", self.to_string());
        println!(
            "++ {:?}",
            true_ids
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<String>>()
                .join(" ")
        );
        println!(
            "-- {:?}",
            false_ids
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<String>>()
                .join(" ")
        );
        let mut node = self.root.as_mut().expect("Why find when no root");
        while true_ids.len() + false_ids.len() > 1 {
            if true_ids.contains(&node.pred.id) {
                false_ids.remove(&node.pred.id);
                node = node.true_child.as_mut().expect("Mismatch ids and child");
            }
            if false_ids.contains(&node.pred.id) {
                false_ids.remove(&node.pred.id);
                node = node.false_child.as_mut().expect("Mismatch ids and child");
            }
        }
        assert_eq!(false_ids.len(), 1);
        assert_eq!(node.pred.id, *false_ids.iter().next().unwrap());
        node.false_child = Some(Box::new(new));
    }

    fn get_cost(&self) -> Cost {
        fn cost_of_branch(node: &BypPlanNode) -> Cost {
            let selec = node.pred.atom.lookup_selectivity();
            1. + selec
                * (node
                    .true_child
                    .as_ref()
                    .and_then(|c| Some(cost_of_branch(c)))
                    .unwrap_or(0.)
                    + node
                        .false_child
                        .as_ref()
                        .and_then(|c| Some(cost_of_branch(c)))
                        .unwrap_or(0.))
        }

        self.root
            .as_ref()
            .and_then(|r| Some(cost_of_branch(r)))
            .unwrap_or(0.)
    }

    fn attach(&self, to_attach: &BypSummand) -> Vec<BypPlan> {
        if self.root.is_none() {
            to_attach
                .preds
                .values()
                .permutations(to_attach.preds.len())
                .map(|perm| BypPlan {
                    root: Some(BypPlanNode::from_preds(
                        perm.into_iter().map(|p| p.clone()).collect(),
                    )),
                })
                .collect()
        } else {
            // Tuple of (node, true pred set, false pred set)
            let mut fringe = vec![(self.root.as_ref().unwrap(), HashSet::new(), HashSet::new())];
            let mut branches: Vec<BypPlan> = vec![];
            while !fringe.is_empty() {
                //let top = fringe.pop().unwrap();
                //let node = top.0;
                //let true_ids = top.1.iter().cloned().collect::<HashSet<PredId>>();
                //let mut false_ids = top.2.iter().cloned().collect::<HashSet<PredId>>();
                let (node, true_ids, mut false_ids) = fringe.pop().unwrap();
                let attach_ids = to_attach.preds.keys().cloned().collect::<HashSet<PredId>>();

                if node.true_child.is_some() {
                    let mut new_true_ids = true_ids.clone();
                    new_true_ids.insert(node.pred.id);
                    fringe.push((
                        node.true_child.as_ref().unwrap(),
                        new_true_ids,
                        false_ids.clone(),
                    ));
                }

                if node.false_child.is_some() {
                    let mut new_false_ids = false_ids.clone();
                    new_false_ids.insert(node.pred.id);
                    fringe.push((
                        node.false_child.as_ref().unwrap(),
                        true_ids.clone(),
                        new_false_ids,
                    ));
                    continue;
                }

                false_ids.insert(node.pred.id);
                if false_ids.is_disjoint(&attach_ids) && !true_ids.is_superset(&attach_ids) {
                    let attach_ids = attach_ids.difference(&true_ids).collect::<Vec<&PredId>>();
                    branches.extend(
                        attach_ids
                            .iter()
                            .permutations(attach_ids.len())
                            .map(|perm| {
                                let mut branch = self.clone();
                                branch.add_new(
                                    true_ids.clone(),
                                    false_ids.clone(),
                                    BypPlanNode::from_preds(
                                        perm.into_iter()
                                            .map(|id| to_attach.preds.get(id).unwrap().clone())
                                            .collect(),
                                    ),
                                );
                                branch
                            })
                            .collect::<Vec<BypPlan>>(),
                    );
                }
            }

            branches
        }
    }

    fn find_plan(
        &self,
        remaining_summands: Vec<BypSummand>,
        exec_stats: &mut ExecStats,
    ) -> Option<(BypPlan, Cost)> {
        if remaining_summands.is_empty() {
            exec_stats.num_plans_considered += 1;
            return Some((self.clone(), self.get_cost()));
        }

        let mut best_plan_cost: Option<(BypPlan, Cost)> = None;
        for (i, summand) in remaining_summands.iter().enumerate() {
            let possible_plans = self.attach(summand);
            for plan in possible_plans {
                let result = plan.find_plan(
                    [&remaining_summands[..i], &remaining_summands[i + 1..]].concat(),
                    exec_stats,
                );
                if result.is_some()
                    && (best_plan_cost.is_none()
                        || result.as_ref().unwrap().1 < best_plan_cost.as_ref().unwrap().1)
                {
                    best_plan_cost = result;
                }
            }
        }

        best_plan_cost
    }

    pub fn eval(
        &self,
        index: &RoaringBitmap,
        exec_params: &ExecParams,
        exec_stats: &mut ExecStats,
    ) -> RoaringBitmap {
        let mut fringe = vec![(self.root.as_ref().unwrap(), index.clone())];
        let mut total = None;
        while !fringe.is_empty() {
            let (node, index) = fringe.pop().unwrap();
            let evaled = node.pred.atom.eval(&index, exec_params, exec_stats);

            if node.true_child.is_none() && node.false_child.is_none() {
                total = total
                    .and_then(|i| Some(i | evaled.clone()))
                    .or(Some(evaled));
            } else {
                if node.true_child.is_some() {
                    fringe.push((node.true_child.as_ref().unwrap(), evaled.clone()));
                }
                if node.false_child.is_some() {
                    fringe.push((node.true_child.as_ref().unwrap(), index - evaled));
                }
            }
        }
        total.expect("No bitmaps?")
    }

    fn to_string(&self) -> String {
        fn to_string_helper(node: &BypPlanNode) -> String {
            format!(
                "{}{}{}",
                //node.pred.atom.expr,
                node.pred.id,
                node.true_child
                    .as_ref()
                    .and_then(|c| Some(format!(" (+ {})", to_string_helper(c))))
                    .unwrap_or(" ".to_string()),
                node.false_child
                    .as_ref()
                    .and_then(|c| Some(format!(" (+ {})", to_string_helper(c))))
                    .unwrap_or(" ".to_string()),
            )
        }

        self.root
            .as_ref()
            .and_then(|n| Some(to_string_helper(n)))
            .unwrap_or("[]".to_string())
    }
}
