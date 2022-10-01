#!/usr/bin/env ppython


import csv
import os
import sys
import argparse

import numpy as np
import pandas as pd
import sklearn

import build_col

TABLE_NAME = "forest"

col_names = [
    "elevation",
    "aspect",
    "slope",
    "horiz_dist_hydro",
    "vert_dist_hydro",
    "horiz_dist_road",
    "hillshade_9am",
    "hillshade_noon",
    "hillshade_3pm",
    "horiz_dist_fire",
    "area",
    "cover",
]

quantile_percents = np.arange(0.1, 1, 0.1)

NUM_COL_REPEAT = 40
NUM_ROW_REPEAT = 100


def make_db(args, df):
    os.makedirs(args.db_path, exist_ok=True)
    os.makedirs(os.path.join(args.db_path, TABLE_NAME), exist_ok=True)

    for col_name in df.columns:
        a = df[col_name]
        assert a.dtype == np.int64
        with open(os.path.join(args.db_path, TABLE_NAME, col_name), "wb") as f:
            a.to_numpy().tofile(f)

    with open(os.path.join(args.db_path, TABLE_NAME, "__schema__"), "w") as f:
        print(",".join(df.columns), file=f)
        print(",".join(["long"] * len(df.columns)), file=f)
        print(",".join([""] * len(df.columns)), file=f)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("db_path")
    parser.add_argument("preds_path")
    return parser.parse_args()


def parse_nobin_csv(args, force_reload=False):
    refined_fname = "{}.nobin{}".format(*os.path.splitext(args.input))
    if not force_reload and os.path.exists(refined_fname):
        return pd.read_csv(refined_fname)

    df = pd.read_csv(args.input, header=None)
    area = df.apply(lambda x: np.array([1, 2, 3, 4])[x[10:14] == 1][0], axis=1)
    quant = df[list(range(10))]
    cover = df[len(df.columns) - 1]

    df = pd.DataFrame()
    for j in range(len(quant.columns)):
        df["{}".format(col_names[j])] = quant[j]
    df["area"] = area
    df["cover"] = cover

    df.to_csv(refined_fname, index=False)

    return df


def munge_df(args, df, force_reload=False):
    refined_fname = "{}.refined{}".format(*os.path.splitext(args.input))
    if not force_reload and os.path.exists(refined_fname):
        return pd.read_csv(refined_fname)

    total_df = pd.DataFrame()
    for i in range(NUM_COL_REPEAT):
        newdf = pd.DataFrame()
        for col in df.columns:
            newdf["{}_{}".format(col, i + 1)] = df[col]
        newdf = sklearn.utils.shuffle(newdf).reset_index(drop=True)

        total_df = pd.concat([total_df, newdf], axis=1)

        print(f"\rCreated column clone {i+1}/{NUM_COL_REPEAT}")

    total_df.to_csv(refined_fname, index=False)

    return total_df


def make_preds_file(args, df):
    with open(args.preds_path, "w") as f:
        quant_cols = [
            c
            for c in df.columns
            if not c.startswith("area") and not c.startswith("cover")
        ]
        quantiles = df[quant_cols].quantile(quantile_percents)
        for col in quant_cols:
            for percent in quantile_percents:
                print(
                    "{},{}.{},{},0".format(
                        percent, TABLE_NAME, col, int(quantiles[col][percent])
                    ),
                    file=f,
                )

        area_cols = [c for c in df.columns if c.startswith("area")]
        for area in (1, 2, 3, 4):
            selec = np.count_nonzero(df["area_1"] == area) / len(df.index)
            for col in area_cols:
                print("{},{}.{},{},1".format(selec, TABLE_NAME, col, area), file=f)

        cover_cols = [c for c in df.columns if c.startswith("cover")]
        for cover in range(1, 8):
            selec = np.count_nonzero(df["cover_1"] == cover) / len(df.index)
            for col in cover_cols:
                print("{},{}.{},{},1".format(selec, TABLE_NAME, col, cover), file=f)


def duplicate_df_vals(df):
    newdf = pd.DataFrame(np.tile(df.values, (NUM_ROW_REPEAT, 1)))
    newdf.columns = df.columns
    return newdf


def main():
    args = get_args()
    df = parse_nobin_csv(args)
    print("Read and refined csv file")
    df = munge_df(args, df, True)
    print("Duplicated columns")
    make_preds_file(args, df)
    print("Created preds file")
    df = duplicate_df_vals(df)
    print("Duplicated data")
    make_db(args, df)


if __name__ == "__main__":
    main()
