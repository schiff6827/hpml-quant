"""Pair benchmark result JSONs (perf/quality/context_sweep) with their
matching profile CSV by parsing the run_name out of each filename.

Usage:
    python /opt/hpml_project/app/scripts/join_results.py
"""
import glob
import json
import os
import re


def run_name_from_filename(fname):
    base = os.path.basename(fname)
    base = re.sub(r'^run_', '', base)
    base = re.sub(r'_(perf|quality|context_sweep)_.*$', '', base)
    return base


def main():
    csvs = {
        run_name_from_filename(p): p
        for p in glob.glob('/opt/hpml_project/app/metrics/*.csv')
    }

    print(f'{"run_name":<60} | {"type":<7} | csv')
    print('-' * 100)
    for j in sorted(glob.glob('/opt/hpml_project/app/benchmarks/*.json')):
        rn = run_name_from_filename(j)
        matching_csv = csvs.get(rn)
        try:
            d = json.load(open(j))
        except Exception:
            continue
        bench = d.get('type', '?')
        print(f'{rn:<60} | {bench:<7} | {os.path.basename(matching_csv) if matching_csv else "—"}')


if __name__ == '__main__':
    main()
