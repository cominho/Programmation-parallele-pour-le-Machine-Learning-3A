import pandas as pd
import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table

def compare_results(
    csv1: str,
    csv2: str,
    key_cols: list,
    name1: str = None,
    name2: str = None
):
    """
    Compare two result CSVs and display a Rich table.
    - csv1, csv2: paths to the two timing-summary CSVs
    - key_cols: list of column names to join on (e.g. ['dim'])
    - name1, name2: labels for the two methods; if None, inferred from filenames
    Returns (merged_df, stats_dict).
    """
    p1, p2 = Path(csv1), Path(csv2)
    # infer labels if not provided
    if name1 is None:
        name1 = p1.stem.split('_')[-1]
    if name2 is None:
        name2 = p2.stem.split('_')[-1]

    # read data
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    # select and rename relevant columns
    mean_col = 'mean_cython'
    speedup_col = 'speedup_cython'
    df1r = df1[key_cols + [mean_col, speedup_col]].rename(columns={
        mean_col: f"mean_{name1}",
        speedup_col: f"speedup_{name1}"
    })
    df2r = df2[key_cols + [mean_col, speedup_col]].rename(columns={
        mean_col: f"mean_{name2}",
        speedup_col: f"speedup_{name2}"
    })

    # merge on key columns
    df = pd.merge(df1r, df2r, on=key_cols)

    # compute delta and best method
    df['delta'] = df[f"mean_{name2}"] - df[f"mean_{name1}"]
    df['best'] = df['delta'].apply(
        lambda d: name2 if d < 0 else (name1 if d > 0 else 'equal')
    )

    # display with Rich
    console = Console()
    table = Table(show_header=True, header_style="bold cyan")
    display_cols = key_cols + [
        f"mean_{name1}", f"mean_{name2}", "delta", "best"
    ]
    for col in display_cols:
        table.add_column(col, justify="center")
    for _, row in df.iterrows():
        table.add_row(*[str(row[c]) for c in display_cols])
    console.print(table)

    # compute aggregate stats
    total = len(df)
    wins1 = (df['best'] == name1).sum()
    wins2 = (df['best'] == name2).sum()
    ties  = (df['best'] == 'equal').sum()
    stats = {
        'total_cases':   total,
        f'{name1}_wins': wins1,
        f'{name2}_wins': wins2,
        'ties':          ties,
        f'pct_{name1}':  wins1 / total * 100,
        f'pct_{name2}':  wins2 / total * 100,
        'pct_ties':      ties / total * 100,
        'avg_delta':     df['delta'].mean()
    }

    # save summary next to first CSV
    output_dir   = p1.parent
    summary_path = output_dir / f"summary_{name1}_vs_{name2}.csv"
    pd.DataFrame([stats]).to_csv(summary_path, index=False)
    console.print(f"[INFO] Saved summary to {summary_path}")

    return df, stats

def compare_all_methods(csv_map: dict, key_cols: list):
    """
    Compare multiple methods in one go.
    Args:
        csv_map: dict mapping method names to CSV file paths.
        key_cols: list of columns to join on (e.g. ['dim']).
    Returns:
        df_all: pandas DataFrame with columns for each mean_<method>
                and a 'best_method' column.
    """
    dfs = []
    for name, path in csv_map.items():
        df = pd.read_csv(path)
        df_sub = df[key_cols + ['mean_cython']].rename(
            columns={'mean_cython': f"mean_{name}"}
        )
        dfs.append(df_sub)

    # merge all DataFrames on key columns
    df_all = dfs[0]
    for df_sub in dfs[1:]:
        df_all = pd.merge(df_all, df_sub, on=key_cols)

    # determine best method per row
    mean_cols = [f"mean_{name}" for name in csv_map]
    df_all['best_method'] = (
        df_all[mean_cols]
        .idxmin(axis=1)
        .str.replace('mean_', '', regex=False)
    )

    return df_all

def main():
    parser = argparse.ArgumentParser(
        description="Compare two or more timing-summary CSVs"
    )
    parser.add_argument('--csv1', required=True,
                        help="First CSV file (e.g. timings_summary_bandit.csv)")
    parser.add_argument('--csv2', required=False,
                        help="Second CSV file (e.g. timings_summary_random.csv)")
    parser.add_argument('--keys', nargs='+', default=['dim'],
                        help="Key columns to join on, e.g. dim")
    parser.add_argument('--name1', help="Label for first method (inferred if omitted)")
    parser.add_argument('--name2', help="Label for second method (inferred if omitted)")
    parser.add_argument('--all', nargs='+', metavar='NAME=PATH',
                        help="List of method=csv pairs for multi-method comparison")
    args = parser.parse_args()

    if args.all:
        # parse NAME=PATH entries into dict
        csv_map = {entry.split('=')[0]: entry.split('=')[1]
                   for entry in args.all}
        df_all = compare_all_methods(csv_map, key_cols=args.keys)
        print(df_all)
    else:
        compare_results(
            csv1=args.csv1,
            csv2=args.csv2,
            key_cols=args.keys,
            name1=args.name1,
            name2=args.name2
        )

if __name__ == '__main__':
    main()
