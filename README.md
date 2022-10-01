# eval-pred
To create the dataset, download the data (`covtype.data.gz`) from https://archive.ics.uci.edu/ml/datasets/covertype
and unzip to the `input` directory. Then generate the test dataset with:
```
mkdir data
python3 scripts/gen_forest_data.py input/covtype.data data/forest data/preds
```

To run experiments, run:
```
cargo build --release
./target/release/run-forest-exp data/forest -n 500
```

Make sure you have access to a `drop_caches` command, which will drop cached files.
