#!/usr/bin/env python

import json
from pathlib import Path

import yaml

import click


@click.command()
@click.argument("input-log", type=Path)
@click.argument("output-dir", type=Path)
def cli(input_log: Path, output_dir: Path) -> None:
    with open(input_log) as f:
        log = yaml.load(f, Loader=yaml.FullLoader)

    output = {}
    for item in log:
        item = item["data"]["ForestData"]
        uniform: bool = item["uniform_cost"]
        depth: int = item["depths"][0]

        output.setdefault((uniform, depth), []).append(
            {
                "depth": depth,
                "uniform_cost": uniform,
                "pred_exprs": item["pred_exprs"][0],
                "selectivities": item["selectivities"][0],
                "costs": item["costs"][0],
            }
        )

    output_dir.mkdir(exist_ok=True)
    for (uniform, depth) in output:
        path = (
            output_dir
            / f"queries-{'uniform' if uniform else 'varcost'}-depth{depth}.json"
        )
        with open(path, "w") as f:
            json.dump(output[(uniform, depth)], f, indent="  ")

    with open(output_dir / "meta.json", "w") as f:
        json.dump({"generated-from": input_log.name}, f)


if __name__ == "__main__":
    cli()
