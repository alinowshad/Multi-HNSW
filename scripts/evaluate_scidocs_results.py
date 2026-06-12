#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUERIES = REPO_ROOT / "scidocs_evaluation" / "queries.jsonl"
DEFAULT_CORPUS = REPO_ROOT / "scidocs_evaluation" / "corpus.jsonl"
DEFAULT_QRELS = REPO_ROOT / "scidocs_evaluation" / "test.tsv"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SciDocs runs against local BEIR/SciDocs ground truth."
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES,
        help=f"Path to queries.jsonl (default: {DEFAULT_QUERIES})",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help=f"Path to corpus.jsonl (default: {DEFAULT_CORPUS})",
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=DEFAULT_QRELS,
        help=f"Path to SciDocs qrels TSV (default: {DEFAULT_QRELS})",
    )
    parser.add_argument(
        "--run",
        type=Path,
        action="append",
        default=[],
        help="Path to one run file. Supports BEIR-style 3-column TSVs and GEM-style 4-column TSVs.",
    )
    parser.add_argument(
        "--runs-glob",
        type=str,
        default=str(DEFAULT_RESULTS_DIR / "*_beir.tsv"),
        help="Optional glob pattern for run files.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        action="append",
        default=[],
        help="Optional C++ summary CSV with columns including tk,k,QPS. Can be passed multiple times.",
    )
    parser.add_argument(
        "--summary-glob",
        type=str,
        default=str(DEFAULT_RESULTS_DIR / "*.csv"),
        help="Optional glob pattern for summary CSVs used to attach QPS.",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[10, 100],
        help="Metrics cutoffs to report.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional path to save a summary CSV.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Optional GEM run log file to parse average query time and QPS.",
    )
    return parser.parse_args()


def load_jsonl_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL in {path} at line {line_no}: {exc}") from exc
            ids.append(str(payload["_id"]))
    return ids


def load_corpus_ids(path: Path) -> list[str]:
    ids: list[str] = []
    failed = False
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                failed = True
                break
            ids.append(str(payload["_id"]))

    if not failed:
        return ids

    text = path.read_text(encoding="utf-8")
    regex_ids = re.findall(r'"_id"\s*:\s*"([^"]+)"', text)
    print(
        f"Warning: {path} is not clean JSONL; using regex fallback and extracted {len(regex_ids)} document IDs.",
        file=sys.stderr,
    )
    return regex_ids


def build_positive_qrels(
    qrels_path: Path, query_to_idx: dict[str, int], doc_to_idx: dict[str, int]
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    stats = {
        "positive_rows": 0,
        "missing_query_ids": 0,
        "missing_doc_ids": 0,
    }

    with qrels_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            score = int(row["score"])
            if score <= 0:
                continue

            stats["positive_rows"] += 1
            query_id = str(row["query-id"])
            corpus_id = str(row["corpus-id"])

            if query_id not in query_to_idx:
                stats["missing_query_ids"] += 1
                continue
            if corpus_id not in doc_to_idx:
                stats["missing_doc_ids"] += 1
                continue

            mapped_qid = str(query_to_idx[query_id])
            mapped_did = str(doc_to_idx[corpus_id])
            qrels[mapped_qid][mapped_did] = score

    return dict(qrels), stats


def load_run(
    path: Path, allowed_query_ids: set[str] | None = None
) -> dict[str, list[tuple[int, str, float]]]:
    results: dict[str, list[tuple[int, str, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        implicit_rank_by_query: dict[str, int] = defaultdict(int)
        for row_no, row in enumerate(reader, start=1):
            if len(row) == 3:
                query_id, doc_id, score = row
                query_id = str(query_id)
                if allowed_query_ids is not None and query_id not in allowed_query_ids:
                    continue
                implicit_rank_by_query[query_id] += 1
                rank = implicit_rank_by_query[query_id]
            elif len(row) == 4:
                query_id, doc_id, score, rank = row
                query_id = str(query_id)
                if allowed_query_ids is not None and query_id not in allowed_query_ids:
                    continue
                rank = int(rank)
            else:
                raise RuntimeError(f"Unexpected TSV row shape in {path} at line {row_no}: {row}")
            results[query_id].append((rank, str(doc_id), float(score)))

    for query_id in results:
        results[query_id].sort(key=lambda item: item[0])
    return dict(results)


def discover_runs(args: argparse.Namespace) -> list[Path]:
    run_paths = list(args.run)
    if args.runs_glob:
        run_paths.extend(sorted(Path(match) for match in glob.glob(args.runs_glob)))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in run_paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(resolved)
    return deduped


def parse_beir_run_key_from_name(path: Path) -> tuple[int, int] | None:
    match = re.search(r"_k(\d+)_tk(\d+)_beir\.tsv$", path.name)
    if not match:
        return None
    return int(match.group(2)), int(match.group(1))


def parse_gem_run_key_from_name(path: Path) -> tuple[int, int] | None:
    match = re.search(r"_rerank(\d+)_ef(\d+)\.tsv$", path.name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def parse_run_key_from_name(path: Path) -> tuple[int, int] | None:
    return parse_beir_run_key_from_name(path) or parse_gem_run_key_from_name(path)


def load_qps_map_from_sidecars(run_paths: list[Path]) -> dict[tuple[Path, tuple[int, int]], float]:
    qps_map: dict[tuple[Path, tuple[int, int]], float] = {}
    for run_path in run_paths:
        run_key = parse_gem_run_key_from_name(run_path)
        if run_key is None:
            continue
        meta_path = Path(str(run_path) + ".meta.json")
        if not meta_path.exists():
            continue
        with meta_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        qps = payload.get("qps")
        if qps is not None:
            qps_map[(run_path.resolve(), run_key)] = float(qps)
    return qps_map


def load_qps_map_from_log(path: Path | None) -> dict[tuple[int, int], float]:
    if path is None:
        return {}

    text = path.read_text(encoding="utf-8")
    rerank_matches = list(re.finditer(r"rerankK:\s*(\d+)\s+ef:\s*(\d+)", text))
    time_matches = list(re.finditer(r"Average query time:\s*([0-9.]+)\s+seconds", text))
    pair_count = min(len(rerank_matches), len(time_matches))

    qps_map: dict[tuple[int, int], float] = {}
    for idx in range(pair_count):
        rerank = int(rerank_matches[idx].group(1))
        ef = int(rerank_matches[idx].group(2))
        avg_query_time = float(time_matches[idx].group(1))
        if avg_query_time > 0:
            qps_map[(rerank, ef)] = 1.0 / avg_query_time
    return qps_map


def discover_summary_csvs(args: argparse.Namespace) -> list[Path]:
    summary_paths = list(args.summary_csv)
    if args.summary_glob:
        summary_paths.extend(sorted(Path(match) for match in glob.glob(args.summary_glob)))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in summary_paths:
        resolved = path.resolve()
        if resolved not in seen and resolved.exists():
            seen.add(resolved)
            deduped.append(resolved)
    return deduped


def parse_summary_basename_prefix(name: str) -> str | None:
    match = re.match(r"(.+?)_(?:hnsw-quantized-complete|bruteforce)_k\.csv$", name)
    if match:
        return match.group(1)
    return None


def parse_run_basename_prefix(name: str) -> str | None:
    match = re.match(r"(.+?)_(?:maxsim_approx|full|bruteforce)_k\d+_tk\d+_beir\.tsv$", name)
    if match:
        return match.group(1)
    return None


def load_qps_map_from_summary_csvs(
    run_paths: list[Path], summary_paths: list[Path]
) -> dict[tuple[Path, tuple[int, int]], float]:
    summaries_by_prefix: dict[str, list[Path]] = defaultdict(list)
    for summary_path in summary_paths:
        prefix = parse_summary_basename_prefix(summary_path.name)
        if prefix is not None:
            summaries_by_prefix[prefix].append(summary_path)

    qps_map: dict[tuple[Path, tuple[int, int]], float] = {}
    for run_path in run_paths:
        run_key = parse_beir_run_key_from_name(run_path)
        run_prefix = parse_run_basename_prefix(run_path.name)
        if run_key is None or run_prefix is None:
            continue

        for summary_path in summaries_by_prefix.get(run_prefix, []):
            with summary_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    try:
                        tk = int(row["tk"])
                        k = int(row["k"])
                        qps = float(row["QPS"])
                    except (KeyError, ValueError):
                        continue
                    if (tk, k) == run_key:
                        qps_map[(run_path.resolve(), run_key)] = qps
                        break
            if (run_path.resolve(), run_key) in qps_map:
                break
    return qps_map


def to_beir_results(
    ranked_results: dict[str, list[tuple[int, str, float]]]
) -> dict[str, dict[str, float]]:
    beir_results: dict[str, dict[str, float]] = {}
    for query_id, rows in ranked_results.items():
        query_scores: dict[str, float] = {}
        for _, doc_id, score in rows:
            best = query_scores.get(doc_id)
            if best is None or score > best:
                query_scores[doc_id] = score
        if query_scores:
            beir_results[query_id] = query_scores
    return beir_results


def evaluate_run_official(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: list[int],
) -> dict[str, float]:
    try:
        from beir.retrieval.evaluation import EvaluateRetrieval
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The official BEIR evaluator is required but not installed. "
            "Install the `beir` package in this environment before running this script."
        ) from exc

    common_qids = sorted(set(qrels) & set(results), key=int)
    summary: dict[str, float] = {
        "queries_with_qrels": float(len(qrels)),
        "queries_with_results": float(len(results)),
        "common_queries": float(len(common_qids)),
    }

    if not common_qids:
        return summary

    qrels_common = {qid: qrels[qid] for qid in common_qids}
    results_common = {qid: results[qid] for qid in common_qids}

    evaluator = EvaluateRetrieval()
    ndcg, mean_ap, recall, precision = evaluator.evaluate(
        qrels_common,
        results_common,
        k_values=k_values,
    )
    mrr = EvaluateRetrieval.evaluate_custom(
        qrels_common,
        results_common,
        k_values=k_values,
        metric="mrr",
    )

    summary.update(ndcg)
    summary.update(mean_ap)
    summary.update(recall)
    summary.update(precision)
    summary.update(mrr)
    return summary


def write_summary_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    run_paths = discover_runs(args)
    if not run_paths:
        raise RuntimeError("No run files provided. Use --run or --runs-glob.")

    query_ids = load_jsonl_ids(args.queries)
    corpus_ids = load_corpus_ids(args.corpus)
    summary_paths = discover_summary_csvs(args)
    qps_map = load_qps_map_from_sidecars(run_paths)
    qps_map.update(load_qps_map_from_summary_csvs(run_paths, summary_paths))
    for run_path in run_paths:
        run_key = parse_gem_run_key_from_name(run_path)
        if run_key is None:
            continue
        qps = load_qps_map_from_log(args.log_file).get(run_key)
        if qps is not None:
            qps_map[(run_path.resolve(), run_key)] = qps
    query_to_idx = {query_id: idx for idx, query_id in enumerate(query_ids)}
    doc_to_idx = {doc_id: idx for idx, doc_id in enumerate(corpus_ids)}
    qrels, qrel_stats = build_positive_qrels(args.qrels, query_to_idx, doc_to_idx)

    print(f"query_ids\t{len(query_ids)}")
    print(f"doc_ids\t{len(corpus_ids)}")
    print(f"positive_qrels\t{qrel_stats['positive_rows']}")
    print(f"mapped_positive_qrels\t{sum(len(v) for v in qrels.values())}")
    print(f"queries_with_positive_qrels\t{len(qrels)}")
    print(f"missing_positive_qrel_query_ids\t{qrel_stats['missing_query_ids']}")
    print(f"missing_positive_qrel_doc_ids\t{qrel_stats['missing_doc_ids']}")
    print()

    allowed_query_ids = set(qrels)

    csv_rows: list[dict[str, str]] = []
    for run_path in run_paths:
        ranked_results = load_run(run_path, allowed_query_ids=allowed_query_ids)
        beir_results = to_beir_results(ranked_results)
        metrics = evaluate_run_official(qrels, beir_results, args.k_values)
        run_key = parse_run_key_from_name(run_path)
        qps = qps_map.get((run_path.resolve(), run_key)) if run_key is not None else None
        if qps is not None:
            metrics["QPS"] = qps

        print(run_path)
        for key, value in metrics.items():
            if key.startswith(("queries_", "common_queries")):
                print(f"{key}\t{int(value)}")
            else:
                print(f"{key}\t{value:.6f}")
        print()

        row: dict[str, str] = {"run": str(run_path)}
        for key, value in metrics.items():
            if key.startswith(("queries_", "common_queries")):
                row[key] = str(int(value))
            else:
                row[key] = f"{value:.6f}"
        csv_rows.append(row)

    if args.output_csv:
        write_summary_csv(args.output_csv, csv_rows)
        print(f"Saved summary CSV to {args.output_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
