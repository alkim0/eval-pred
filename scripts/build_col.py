#!/usr/bin/env python

import struct


BLOCK_SIZE = 8 * 1024 * 1024 * 1024


def build_col(path, col_type, vals=None, rng=None, num_records=None):
    """
    col_type must be one of "int", "float", or "string"
    Generates a tightly packed column based on given input. If vals is not
    None, then that list of values is used. If rng is not None, then it is
    assumed to be a function which can be called for random number generation.
    num_records must be defined if rng is not None. Othrwise if
    num_records > len(vals), the elements in are repeated in order until
    num_records is reached. If num_records < len(vals), then only the first
    num_records elements of vals is used.
    """
    if col_type == "int":
        fmt = "i"
    elif col_type == "long":
        fmt = "l"
    elif col_type == "float":
        fmt = "f"
    elif col_type == "double":
        fmt = "d"
    elif col_type == "string":
        fmt = "128s"
    else:
        raise Exception("Unknown col_type: {}".format(col_type))

    print("Building: {}".format(path))
    record_idx = 0

    def write_val(f, val):
        if col_type == "string":
            val = val.encode()
        f.write(struct.pack(fmt, val))

        nonlocal record_idx
        record_idx += 1

        if record_idx % int(1e6) == 0:
            print("{}M".format(record_idx / int(1e6)))

    if vals is not None:
        if num_records is None:
            num_records = len(vals)

        with open(path, "wb") as f:
            for i in range(num_records):
                val = vals[i % len(vals)]
                write_val(f, val)

    elif rng is not None:
        with open(path, "wb") as f:
            if num_records is None:
                for val in rng():
                    write_val(f, val)
            else:
                for i in range(num_records):
                    write_val(f, rng())

    return record_idx


def main():
    pass


if __name__ == "__main__":
    main()
