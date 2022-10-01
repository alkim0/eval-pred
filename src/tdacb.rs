use super::exec::{ExecParams, ExecStats};
use super::parser::{PredAtomNode, PredNode};
use log::debug;
use roaring::RoaringBitmap;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

type Cost = f64;

const MAX_BUDGET: Cost = 999999999999999999.;

const TAB: &str = "    ";

// Bit vector representation of predicate expressions
// We use u64's. The first bit is 1 if the expression is AND, second bit is 1 if the expression is
// OR. The following 32 bits are a bitmap where the ith bit is set to 1 if the ith AND/OR node is a
// child of this node. The next 20 bits are a bitmap of the predicate atoms, where the ith bit
// represents the presence of the ith predicate atom.

const NUM_PRED_ATOMS: usize = 20;

pub struct Tdacb {
    // From pred to idx
    pred_atoms: Vec<Rc<PredAtomNode>>,
    nonleaf: Vec<u64>,
    nonleaf_idx: usize,
    expr: u64,
    //expr: TdacbExpr,
    // plan, cost
    memo: RefCell<HashMap<(u64, u64), TdacbPlan>>,
    bounds: RefCell<HashMap<(u64, u64), Cost>>,
}

#[derive(Clone)]
pub struct TdacbPlan {
    // None for scan at beg
    //pred: Option<Rc<PredAtomNode>>,
    pred_idx: Option<usize>,
    pos: Option<Box<TdacbPlan>>,
    neg: Option<Box<TdacbPlan>>,
    cost: Cost,
    selectivity_til_now: f64, // Not including this node's selectivity
}

//impl fmt::Display for TdacbPlan {
//    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//        write!(
//            f,
//            "(pred: {}{}{})",
//            self.pred_idx
//                .as_ref()
//                .and_then(|p| p.to_string())
//                .unwrap_or("None".to_string()),
//            self.pos
//                .as_ref()
//                .and_then(|x| Some(format!(" (+: {})", x)))
//                .unwrap_or("".to_string()),
//            self.neg
//                .as_ref()
//                .and_then(|x| Some(format!(" (-: {})", x)))
//                .unwrap_or("".to_string()),
//        )
//    }
//}

impl Tdacb {
    pub fn new(root: &PredNode) -> Self {
        fn build_tdacb_expr(
            node: &PredNode,
            pred_atoms: &mut Vec<Rc<PredAtomNode>>,
            nonleaf: &mut Vec<u64>,
            atom_idx: &mut usize,
            nonleaf_idx: &mut usize,
        ) -> usize {
            match node {
                PredNode::OrNode(children) | PredNode::AndNode(children) => {
                    nonleaf.push(0);
                    let my_idx = *nonleaf_idx;
                    *nonleaf_idx += 1;
                    let mut node_val: u64 = 0;

                    if let PredNode::OrNode(_) = node {
                        node_val |= 1 << 62;
                    } else {
                        node_val |= 1 << 63;
                    }

                    children.iter().for_each(|c| {
                        if let PredNode::PredAtomNode(atom) = c {
                            pred_atoms.push(Rc::new(atom.clone()));
                            node_val |= 1 << *atom_idx;
                            *atom_idx += 1;
                        } else {
                            let nonleaf_idx =
                                build_tdacb_expr(c, pred_atoms, nonleaf, atom_idx, nonleaf_idx);
                            node_val |= 1 << (NUM_PRED_ATOMS + nonleaf_idx);
                        }
                    });

                    nonleaf[my_idx] = node_val;
                    my_idx
                }
                _ => {
                    panic!("no");
                }
            }
        }

        let mut pred_atoms = vec![];
        let mut nonleaf = vec![];
        let mut atom_idx = 0;
        let mut nonleaf_idx = 0;
        build_tdacb_expr(
            root,
            &mut pred_atoms,
            &mut nonleaf,
            &mut atom_idx,
            &mut nonleaf_idx,
        );
        // Here we assume root nonleaf node has index 0
        let expr = nonleaf[0];
        Tdacb {
            pred_atoms,
            nonleaf,
            expr,
            nonleaf_idx,
            memo: RefCell::new(HashMap::new()),
            bounds: RefCell::new(HashMap::new()),
        }
    }

    pub fn find_plan(&self, _exec_stats: &mut ExecStats) -> TdacbPlan {
        let true_asg: u64 = 0;
        let false_asg: u64 = 0;

        let plan = self.try_find(
            &TdacbPlan {
                pred_idx: None,
                pos: None,
                neg: None,
                cost: 0.,
                selectivity_til_now: 1.,
            },
            self.nonleaf[0],
            true_asg,
            false_asg,
            true,
            MAX_BUDGET,
            0,
        );

        // FIXME
        assert!(plan.is_some());
        //let mut asg = vec![];
        //for _ in 0..self.preds.len() {
        //    asg.push(None);
        //}
        let true_asg: u64 = 0;
        let false_asg: u64 = 0;
        let plan = self.prune_inconsistencies(&plan.unwrap(), true_asg, false_asg);
        assert!(plan.is_some());
        *plan.unwrap()
    }

    // Return value is result of applying assignment and remaining pred atoms
    fn apply_asg(&self, expr: u64, true_asg: u64, false_asg: u64) -> (Option<bool>, u64) {
        let mut rem: u64 = 0;
        let is_and = expr & 1 << 63 != 0;
        if expr & ((1 << NUM_PRED_ATOMS) - 1) != 0 {
            // This has pred atoms as children
            if is_and {
                if expr & false_asg != 0 {
                    debug!("{:b} {:b} {:b} result: false", expr, true_asg, false_asg);
                    return (Some(false), 0);
                }
                rem |= expr & !true_asg & ((1 << NUM_PRED_ATOMS) - 1);
            } else {
                if expr & true_asg != 0 {
                    debug!("{:b} {:b} {:b} result: true", expr, true_asg, false_asg);
                    return (Some(true), 0);
                }
                rem |= expr & !false_asg & ((1 << NUM_PRED_ATOMS) - 1);
            }
        }

        for i in 0..self.nonleaf.len() {
            debug!(
                "{} {:b} {:b} {:b}",
                i,
                expr,
                1 << (NUM_PRED_ATOMS + i),
                expr & 1 << (NUM_PRED_ATOMS + i)
            );
            if expr & 1 << (NUM_PRED_ATOMS + i) != 0 {
                // ith nonleaf node is child
                let (result, child_rem) = self.apply_asg(self.nonleaf[i], true_asg, false_asg);
                match result {
                    Some(false) if is_and => {
                        debug!("{:b} {:b} {:b} result: false", expr, true_asg, false_asg);
                        return (Some(false), 0);
                    }
                    Some(true) if !is_and => {
                        debug!("{:b} {:b} {:b} result: true", expr, true_asg, false_asg);
                        return (Some(true), 0);
                    }
                    _ => {
                        rem |= child_rem;
                    }
                }
            }
        }

        if rem == 0 {
            debug!(
                "{:b} {:b} {:b} result: {}",
                expr, true_asg, false_asg, is_and,
            );
            (Some(is_and), rem)
        } else {
            debug!("{:b} {:b} {:b} rem: {:b}", expr, true_asg, false_asg, rem);
            (None, rem)
        }
    }

    fn try_find(
        &self,
        orig_plan: &TdacbPlan,
        expr: u64,
        true_asg: u64,
        false_asg: u64,
        branch: bool,
        mut budget: Cost,
        depth: usize,
    ) -> Option<TdacbPlan> {
        {
            let inner_memo = self.memo.borrow();
            let entry = inner_memo.get(&(true_asg, false_asg));
            if let Some(orig_plan) = entry {
                if orig_plan.cost <= budget {
                    return Some(orig_plan.clone());
                }
            }

            let bounds = self.bounds.borrow();
            let entry = bounds.get(&(true_asg, false_asg));
            if let Some(bound) = entry {
                if bound >= &budget {
                    return None;
                }

                if bound > &0. {
                    budget = f64::max(budget, bound * &2.);
                }
            }
        }

        let (result, rem_preds) = self.apply_asg(expr, true_asg, false_asg);
        debug!(
            "{}result: {:?}, rem: {:b}",
            TAB.repeat(depth),
            result,
            rem_preds,
        );
        if result.is_some() {
            let bestplan = TdacbPlan {
                pred_idx: None,
                pos: None,
                neg: None,
                cost: 0.,
                selectivity_til_now: 0.,
            };
            let mut memo = self.memo.borrow_mut();
            memo.insert((true_asg, false_asg), bestplan.clone());
            return Some(bestplan);
        }

        // Pair of (plan, cost)
        let mut best: Option<TdacbPlan> = None;
        if budget >= 0. {
            for i in 0..NUM_PRED_ATOMS {
                if (rem_preds & (1 << i)) == 0 {
                    continue;
                }
                let plan = self.build_plan(i, orig_plan, branch);
                let mut plan_budget = best
                    .as_ref()
                    .and_then(|p| Some(f64::min(p.cost, budget)))
                    .unwrap_or(budget)
                    - plan.cost;
                let new_true_asg = true_asg | (1 << i);
                let pos_plan = self.try_find(
                    &plan,
                    expr,
                    new_true_asg,
                    false_asg,
                    true,
                    plan_budget,
                    depth + 1,
                );
                if pos_plan.is_some() {
                    plan_budget = plan_budget - pos_plan.as_ref().unwrap().cost;

                    let new_false_asg = false_asg | (1 << i);
                    let neg_plan = self.try_find(
                        &plan,
                        expr,
                        true_asg,
                        new_false_asg,
                        false,
                        plan_budget,
                        depth + 1,
                    );

                    debug!(
                        "{}Got pos_plan: {}",
                        TAB.repeat(depth),
                        pos_plan
                            .as_ref()
                            .and_then(|p| Some(p.to_string(self)))
                            .unwrap_or("".to_string())
                    );
                    debug!(
                        "{}Got neg_plan: {}",
                        TAB.repeat(depth),
                        neg_plan
                            .as_ref()
                            .and_then(|p| Some(p.to_string(self)))
                            .unwrap_or("".to_string())
                    );

                    if neg_plan.is_some() {
                        let cost = plan.cost
                            + pos_plan.as_ref().and_then(|p| Some(p.cost)).unwrap_or(0.)
                            + neg_plan.as_ref().and_then(|p| Some(p.cost)).unwrap_or(0.);

                        if best.is_none() || best.as_ref().unwrap().cost > cost {
                            best = Some(TdacbPlan {
                                pred_idx: Some(i),
                                pos: pos_plan.and_then(|p| Some(Box::new(p))),
                                neg: neg_plan.and_then(|p| Some(Box::new(p))),
                                cost,
                                selectivity_til_now: plan.selectivity_til_now,
                            });
                            //println!(
                            //    "{}Updating best: {}",
                            //    TAB.repeat(depth),
                            //    best.as_ref().unwrap()
                            //);
                        }
                    }
                }
            }
        }

        debug!(
            "{}best is {} result: {:?}",
            TAB.repeat(depth),
            best.as_ref()
                .and_then(|b| Some(b.to_string(self)))
                .unwrap_or("None".to_string()),
            result
        );

        if best.is_none()
            || best.as_ref().unwrap().pos.is_none()
            || best.as_ref().unwrap().neg.is_none()
        {
            let mut bounds = self.bounds.borrow_mut();
            bounds.insert((true_asg, false_asg), budget);
            None
        } else {
            let mut memo = self.memo.borrow_mut();
            let bestplan = best.unwrap();
            memo.insert((true_asg, false_asg), bestplan.clone());
            Some(bestplan)
        }
    }

    fn prune_inconsistencies(
        &self,
        plan: &TdacbPlan,
        true_asg: u64,
        false_asg: u64,
    ) -> Option<Box<TdacbPlan>> {
        let (result, _rem_preds) = self.apply_asg(self.expr, true_asg, false_asg);
        match result {
            Some(false) => None,
            Some(true) => Some(Box::new(plan.clone())),
            _ => {
                if plan.pred_idx.is_none() {
                    Some(Box::new(plan.clone()))
                } else {
                    Some(Box::new(TdacbPlan {
                        pred_idx: plan.pred_idx.clone(),
                        pos: {
                            let new_true_asg = true_asg | (1 << *plan.pred_idx.as_ref().unwrap());
                            self.prune_inconsistencies(
                                plan.pos.as_ref().unwrap(),
                                new_true_asg,
                                false_asg,
                            )
                        },
                        neg: {
                            let new_false_asg = false_asg | (1 << *plan.pred_idx.as_ref().unwrap());
                            self.prune_inconsistencies(
                                plan.neg.as_ref().unwrap(),
                                true_asg,
                                new_false_asg,
                            )
                        },
                        cost: plan.cost,
                        selectivity_til_now: plan.selectivity_til_now,
                    }))
                }
            }
        }
    }

    fn build_plan(&self, pred_idx: usize, partial: &TdacbPlan, branch: bool) -> TdacbPlan {
        let prev_selec = partial
            .pred_idx
            .as_ref()
            .and_then(|i| Some(self.pred_atoms[*i].lookup_selectivity()))
            .unwrap_or(1.);

        let selectivity_til_now = if branch {
            partial.selectivity_til_now * prev_selec
        } else {
            partial.selectivity_til_now * (1. - prev_selec)
        };

        let cost = self.pred_atoms[pred_idx].lookup_cost() * selectivity_til_now;

        let new_plan = TdacbPlan {
            pred_idx: Some(pred_idx),
            pos: None,
            neg: None,
            cost,
            selectivity_til_now,
        };

        new_plan
    }

    pub fn eval(
        &self,
        plan: &TdacbPlan,
        index: &RoaringBitmap,
        exec_params: &ExecParams,
        exec_stats: &mut ExecStats,
    ) -> RoaringBitmap {
        let evaled = plan
            .pred_idx
            .and_then(|i| Some(self.pred_atoms[i].eval(index, exec_params, exec_stats)));
        let pos_evaled = plan.pos.as_ref().and_then(|pos| {
            let index = evaled.as_ref().unwrap_or(index);
            Some(self.eval(pos, index, exec_params, exec_stats))
        });
        let neg_evaled = plan.neg.as_ref().and_then(|neg| {
            let index = evaled
                .as_ref()
                .and_then(|e| Some(index - e))
                .unwrap_or(RoaringBitmap::new());
            Some(self.eval(neg, &index, exec_params, exec_stats))
        });

        match (pos_evaled, neg_evaled) {
            (Some(pos), Some(neg)) => pos | neg,
            (Some(pos), None) => pos,
            (None, Some(neg)) => neg,
            (None, None) => index.clone(),
        }
    }
}

impl TdacbPlan {
    pub fn to_string(&self, tdacb: &Tdacb) -> String {
        format!(
            "(pred: {}{}{})",
            self.pred_idx
                .as_ref()
                .and_then(|i| Some(tdacb.pred_atoms[*i].expr.to_string()))
                .unwrap_or("None".to_string()),
            self.pos
                .as_ref()
                .and_then(|x| Some(format!(" (+: {})", x.to_string(tdacb))))
                .unwrap_or("".to_string()),
            self.neg
                .as_ref()
                .and_then(|x| Some(format!(" (-: {})", x.to_string(tdacb))))
                .unwrap_or("".to_string()),
        )
    }
}
