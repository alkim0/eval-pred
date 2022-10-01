use crate::exec::{ApproxOptType, ExecParams, ExecStats};
use crate::parser::{PredAtomNode, PredGraph, PredNode};
use approx;
use roaring::RoaringBitmap;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::iter::FromIterator;

type Assignments<'a> = HashMap<&'a PredAtomNode, bool>;

pub struct BDC<'a> {
    root: PredNode,
    all_preds: HashSet<&'a PredAtomNode>,
}

impl<'a> BDC<'a> {
    pub fn new(root: &'a PredNode) -> Self {
        Self {
            root: root.clone(),
            all_preds: HashSet::from_iter(root.get_all_atoms().into_iter()),
        }
    }

    pub fn plan(&self) -> BDCPlan {
        let pred_assignments = HashMap::new();
        let mut root = self.build_plan(&pred_assignments);
        //root.prune();
        BDCPlan { root }
    }

    fn build_plan(&self, pred_assignments: &Assignments<'a>) -> PlanNode {
        let mut assignments = pred_assignments.clone();
        let mut pred_ratios: Vec<(&PredAtomNode, f64)> = self
            .all_preds
            .difference(&pred_assignments.keys().cloned().collect())
            .map(|&p| (p, self.calc_ratio(p, &pred_assignments)))
            .collect();

        if pred_ratios.is_empty() {
            let likelihood = self.root.calc_likelihood(&pred_assignments);
            assert!(approx::abs_diff_eq!(likelihood, 1.) || approx::abs_diff_eq!(likelihood, 0.));
            return PlanNode::Val(self.root.calc_likelihood(&pred_assignments) > 0.);
        }

        pred_ratios.sort_unstable_by(|x, y| {
            (y.1, y.0.expr.to_string())
                .partial_cmp(&(x.1, x.0.expr.to_string()))
                .unwrap_or(Ordering::Equal)
        });
        if pred_ratios
            .iter()
            .all(|(_, ratio)| ratio.partial_cmp(&0.).unwrap_or(Ordering::Equal) == Ordering::Equal)
        {
            return PlanNode::Val(false);
        } else if pred_ratios
            .iter()
            .all(|(_, ratio)| ratio.partial_cmp(&1.).unwrap_or(Ordering::Equal) == Ordering::Equal)
        {
            return PlanNode::Val(true);
        }

        let atom = pred_ratios.first().unwrap().0;

        assignments.insert(atom, false);
        let neg = self.build_plan(&assignments);
        assignments.insert(atom, true);
        let pos = self.build_plan(&assignments);

        PlanNode::Branch(BranchNode {
            atom,
            pos: Some(Box::new(pos)),
            neg: Some(Box::new(neg)),
        })
    }

    fn calc_ratio(&self, atom: &PredAtomNode, pred_assignments: &Assignments) -> f64 {
        self.calc_achievement(atom, pred_assignments) / self.calc_cost(atom, pred_assignments)
    }

    fn calc_achievement(&self, atom: &PredAtomNode, pred_assignments: &Assignments) -> f64 {
        let mut assignments = pred_assignments.clone();

        // Likelihood that the predicate expression resolves to 1 if atom is set to 0
        assignments.insert(atom, false);
        let neg_likelihood = self.root.calc_likelihood(pred_assignments);

        // Likelihood that the predicate expression resolves to 1 if atom is set to 1
        assignments.insert(atom, true);
        let pos_likelihood = self.root.calc_likelihood(pred_assignments);

        (1. - neg_likelihood) * pos_likelihood + neg_likelihood * (1. - pos_likelihood)
            - neg_likelihood * (1. - neg_likelihood) * pos_likelihood * (1. - pos_likelihood)
    }

    fn calc_cost(&self, atom: &PredAtomNode, _pred_assignments: &Assignments) -> f64 {
        atom.lookup_cost()
    }

    pub fn eval(
        &self,
        plan: &BDCPlan,
        graph: Option<PredGraph>,
        index: &RoaringBitmap,
        exec_params: &ExecParams,
        exec_stats: &mut ExecStats,
    ) -> RoaringBitmap {
        if let Some(mut graph) = graph {
            let ordering = plan.get_ordering();
            for pred_atom in ordering {
                graph.apply_pred_atom(pred_atom.id, index, exec_params, exec_stats);
            }
            graph
                .get_root_idx()
                .expect("Why is root applied but not in pos_map")
        } else {
            self.eval_helper(&plan.root, index, exec_params, exec_stats)
        }
    }

    fn eval_helper(
        &self,
        node: &PlanNode,
        index: &RoaringBitmap,
        exec_params: &ExecParams,
        exec_stats: &mut ExecStats,
    ) -> RoaringBitmap {
        match node {
            PlanNode::Val(v) => {
                assert_eq!(*v, true);
                index.clone()
            }
            PlanNode::Branch(node) => {
                let evaled = node.atom.eval(index, exec_params, exec_stats);
                let pos_map = if let Some(pos) = &node.pos {
                    self.eval_helper(&pos, &evaled, exec_params, exec_stats)
                } else {
                    RoaringBitmap::new()
                };
                let neg_map = if let Some(neg) = &node.neg {
                    self.eval_helper(&neg, &(index - evaled), exec_params, exec_stats)
                } else {
                    RoaringBitmap::new()
                };
                pos_map | neg_map
            }
        }
    }
}

pub struct BDCPlan<'a> {
    root: PlanNode<'a>,
}

impl<'a> BDCPlan<'a> {
    fn get_ordering(&self) -> Vec<&'a PredAtomNode> {
        self.root.get_longest_ordering()
    }

    pub fn check_ordering(&self) {
        let orderings = self.root.get_all_orderings();
        let mut order_pairs = HashSet::new();

        for ordering in &orderings {
            for i in 0..ordering.len() {
                for j in (i + 1)..ordering.len() {
                    if order_pairs.contains(&(ordering[j], ordering[i])) {
                        panic!(
                            "Inconsistency in orderings {}",
                            orderings
                                .into_iter()
                                .map(|o| PredOrdering(o).to_string())
                                .collect::<Vec<String>>()
                                .join("\n  ")
                        );
                    }
                    order_pairs.insert((ordering[i], ordering[j]));
                }
            }
        }

        println!();
        println!(
            "  {}",
            orderings
                .into_iter()
                .map(|o| PredOrdering(o).to_string())
                .collect::<Vec<String>>()
                .join("\n  ")
        );
        println!("* {}", PredOrdering::from(PredOrderPairs(order_pairs)));
    }
}

enum PlanNode<'a> {
    Val(bool),
    Branch(BranchNode<'a>),
}

struct BranchNode<'a> {
    atom: &'a PredAtomNode,
    pos: Option<Box<PlanNode<'a>>>,
    neg: Option<Box<PlanNode<'a>>>,
}

impl<'a> PlanNode<'a> {
    fn prune(&mut self) -> bool {
        match self {
            PlanNode::Val(v) => *v,
            PlanNode::Branch(node) => {
                if let Some(child) = &mut node.pos {
                    if !child.prune() {
                        node.pos = None;
                    }
                }
                if let Some(child) = &mut node.neg {
                    if !child.prune() {
                        node.neg = None;
                    }
                }
                node.pos.is_some() || node.neg.is_some()
            }
        }
    }

    fn get_all_orderings(&self) -> Vec<Vec<&'a PredAtomNode>> {
        match self {
            PlanNode::Val(_) => vec![vec![]],
            PlanNode::Branch(node) => {
                let mut orderings = vec![];
                for branch in [&node.pos, &node.neg] {
                    if let Some(branch) = branch {
                        let branch_orderings = branch.get_all_orderings();
                        orderings.append(
                            &mut branch_orderings
                                .into_iter()
                                .map(|mut o| {
                                    o.insert(0, node.atom);
                                    o
                                })
                                .collect(),
                        );
                    }
                }
                orderings
            }
        }
    }

    fn get_longest_ordering(&self) -> Vec<&'a PredAtomNode> {
        match self {
            PlanNode::Val(_) => vec![],
            PlanNode::Branch(node) => {
                let pos_ordering = node
                    .pos
                    .as_ref()
                    .map_or(vec![], |branch| branch.get_longest_ordering());
                let neg_ordering = node
                    .neg
                    .as_ref()
                    .map_or(vec![], |branch| branch.get_longest_ordering());
                let mut ordering = if pos_ordering.len() >= neg_ordering.len() {
                    pos_ordering
                } else {
                    neg_ordering
                };
                ordering.insert(0, node.atom);
                ordering
            }
        }
    }
}

struct PredAtomNodeWrapper<'a>(&'a PredAtomNode);

impl<'a> fmt::Display for PredAtomNodeWrapper<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let expr = self.0.expr.to_string();
        let mut iter = expr.split(", ");
        //assert_eq!(iter.size(), 4);
        let first = &iter.next().unwrap()[18..];
        let second = iter.next().unwrap();
        write!(f, "({} >= {})", first, second)
    }
}

struct PredOrdering<'a>(Vec<&'a PredAtomNode>);

impl<'a> fmt::Display for PredOrdering<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}]",
            self.0
                .iter()
                .map(|p| {
                    let expr = p.expr.to_string();
                    let mut iter = expr.split(", ");
                    //assert_eq!(iter.size(), 4);
                    let first = &iter.next().unwrap()[18..];
                    let second = iter.next().unwrap();
                    format!("({} {})", first, second)
                })
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}

struct PredOrderPairs<'a>(HashSet<(&'a PredAtomNode, &'a PredAtomNode)>);

impl<'a> From<PredOrderPairs<'a>> for PredOrdering<'a> {
    fn from(item: PredOrderPairs<'a>) -> Self {
        let mut all_preds = Vec::from_iter(
            HashSet::<&PredAtomNode>::from_iter(
                item.0
                    .iter()
                    .map(|(left, right)| [*left, *right])
                    .flatten()
                    .collect::<Vec<&PredAtomNode>>()
                    .into_iter(),
            )
            .into_iter(),
        );

        for i in 0..all_preds.len() {
            for j in (i + 1)..all_preds.len() {
                assert!(
                    item.0.contains(&(all_preds[i], all_preds[j]))
                        || item.0.contains(&(all_preds[j], all_preds[i]))
                );

                if item.0.contains(&(all_preds[j], all_preds[i])) {
                    all_preds.swap(i, j);
                }
            }
        }

        PredOrdering(all_preds)
    }
}
