#!/usr/bin/env python3
import re
import csv
import math
import statistics
import argparse
from collections import defaultdict


def parse_statistics(path):
    records = []
    current = {}

    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    pattern = re.compile(rf"^(cputime|FPGA time|NQ):\s*({number})")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            m = pattern.match(line)
            if not m:
                continue

            key, val = m.group(1), m.group(2)
            if key == "NQ":
                current["NQ"] = int(float(val))
            elif key == "cputime":
                current["cputime"] = float(val)
            elif key == "FPGA time":
                current["FPGA_time"] = float(val)

            if {"NQ", "cputime", "FPGA_time"} <= current.keys():
                records.append(current)
                current = {}

    return records


def make_summary(records, sample_std=False):
    groups = defaultdict(lambda: {"cputime": [], "FPGA_time": []})
    for r in records:
        groups[r["NQ"]]["cputime"].append(r["cputime"])
        groups[r["NQ"]]["FPGA_time"].append(r["FPGA_time"])

    rows = []
    for nq in sorted(groups):
        cpu_vals = groups[nq]["cputime"]
        fpga_vals = groups[nq]["FPGA_time"]

        cpu_ave = statistics.mean(cpu_vals)
        fpga_ave = statistics.mean(fpga_vals)

        if sample_std:
            cpu_std = statistics.stdev(cpu_vals) if len(cpu_vals) > 1 else 0.0
            fpga_std = statistics.stdev(fpga_vals) if len(fpga_vals) > 1 else 0.0
        else:
            cpu_std = statistics.pstdev(cpu_vals) if len(cpu_vals) > 1 else 0.0
            fpga_std = statistics.pstdev(fpga_vals) if len(fpga_vals) > 1 else 0.0

        ratio = cpu_ave / fpga_ave if fpga_ave != 0 else math.nan
        rows.append([nq, cpu_ave, cpu_std, fpga_ave, fpga_std, ratio])

    return rows


def write_csv(rows, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "NQ",
            "cputime.ave",
            "cputime.std",
            "FPGA_time.ave",
            "FPGA_time.std",
            "cpu/FPGA",
        ])
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize cputime and FPGA time by NQ from statistics.txt."
    )
    parser.add_argument("input", help="Input statistics file, e.g. statistics.txt")
    parser.add_argument("output", help="Output CSV file, e.g. summary.csv")
    parser.add_argument(
        "--sample-std",
        action="store_true",
        help="Use sample standard deviation instead of population standard deviation.",
    )
    args = parser.parse_args()

    records = parse_statistics(args.input)
    rows = make_summary(records, sample_std=args.sample_std)
    write_csv(rows, args.output)


if __name__ == "__main__":
    main()
