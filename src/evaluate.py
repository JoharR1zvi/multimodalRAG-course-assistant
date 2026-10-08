# Measures how good the search is, on questions whose answers are known.
#
# The evaluation file (data/eval/retrieval_eval.json) lists questions. For each answerable
# question it says in which lecture, and between which seconds, the answer is spoken.
# We run the real search and ask: does one of the top results cover that stretch of the lecture?
#
#   HIT  = a result from the right lecture whose time range overlaps an answer range
#   rank = the position (1 = best) of the first hit in the results, or "miss" if there is none
#
# From the ranks we compute:
#   hit@k = the share of questions with a hit among the top k results
#   MRR   = the average of 1 / rank (a hit at rank 1 scores 1, rank 2 scores 0.5, a miss scores 0)
#
# Questions the lectures do not cover have no answer range. For those we only look at how
# high the best score is, and (with --answers) whether the answer step refuses to answer.
#
# Usage (from the project root):
#   python -m src.evaluate                     # search only, no Gemini calls
#   python -m src.evaluate --top 5             # look at the top 5 results only
#   python -m src.evaluate --answers           # also write answers with Gemini (about 14 calls)

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from src import config
from src.database.vector_store import open_client
from src.generation.llm_service import answer_question
from src.retrieval.retriever import format_timestamp, retrieve

# The evaluation questions. Private (course material), so the folder is git-ignored.
DEFAULT_EVAL_FILE = "data/eval/retrieval_eval.json"

# The hit@k numbers that are reported (only those not larger than --top are shown)
HIT_K_VALUES = [1, 3, 5, 10]

# How many results the answer step reads. Same default as `python -m src.ask`.
ANSWER_TOP_K = 5

# The stricter hit test: a chunk must cover at least this share of the answer range
STRICT_MIN_SHARE = 0.5

# The answer step already retries a failed Gemini call a few times, quickly. When Gemini stays
# overloaded for longer, one failed question must not end a run of 40 questions: the evaluation
# waits this many seconds after each failure and tries again, and records an error at the end.
ANSWER_RETRY_WAITS = [20, 40, 80]


# ---------------------------------------------------------------------------
# Pure helpers (no database, no model): these are the ones covered by tests
# ---------------------------------------------------------------------------

def ranges_overlap(chunk_start: float, chunk_end: float, range_start: float, range_end: float) -> bool:
    # True when the two stretches of time share at least some seconds.
    # Two stretches that only touch (one ends exactly where the other starts) do not overlap.
    return chunk_start < range_end and chunk_end > range_start


def overlap_seconds(chunk_start: float, chunk_end: float, range_start: float, range_end: float) -> float:
    # How many seconds the two stretches of time share (0 when they do not overlap)
    start = max(chunk_start, range_start)
    end = min(chunk_end, range_end)

    if end > start:
        return end - start

    return 0.0


def chunk_covers_location(chunk, location: dict, min_share: float) -> bool:
    # A stricter test than "any overlap": the chunk must be from the right lecture AND cover at
    # least min_share of one answer range (0.5 = at least half of the seconds of the answer).
    # A bigger chunk overlaps an answer range more easily, so "any overlap" favours big chunks;
    # this test does not.
    if chunk.lecture_id != location["lecture"]:
        return False

    for answer_range in location["ranges"]:
        range_length = answer_range[1] - answer_range[0]
        if range_length <= 0:
            continue

        shared = overlap_seconds(chunk.start_timestamp, chunk.end_timestamp, answer_range[0], answer_range[1])

        if shared / range_length >= min_share:
            return True

    return False


def find_first_strict_hit_rank(results: list, item: dict, min_share: float = STRICT_MIN_SHARE):
    # Like find_first_hit_rank, but a result only counts when its chunk covers at least min_share
    # of an answer range of the question (see chunk_covers_location)
    locations = get_locations(item)

    for position in range(len(results)):
        for location in locations:
            if chunk_covers_location(results[position].chunk, location, min_share):
                return position + 1

    return None


def get_locations(item: dict) -> list:
    # Every place in the lectures that counts as a correct find for this question, as a list of
    # {"lecture": ..., "ranges": [[start, end], ...]}:
    #   "lecture" + "answer_ranges" : the main place
    #   "extra_locations"           : other places that give the same answer (finding any one is right)
    #   "hops"                      : a question that needs several places; each hop is one place
    locations = []

    main_ranges = item.get("answer_ranges", [])
    if len(main_ranges) > 0:
        locations.append({"lecture": item["lecture"], "ranges": main_ranges})

    for extra in item.get("extra_locations", []):
        locations.append(extra)

    for hop in item.get("hops", []):
        locations.append(hop)

    return locations


def chunk_hits_location(chunk, location: dict) -> bool:
    # Is this chunk from the lecture of this place AND does it overlap one of the place's ranges?
    if chunk.lecture_id != location["lecture"]:
        return False

    for answer_range in location["ranges"]:
        range_start = answer_range[0]
        range_end = answer_range[1]

        if ranges_overlap(chunk.start_timestamp, chunk.end_timestamp, range_start, range_end):
            return True

    return False


def chunk_is_hit(chunk, item: dict) -> bool:
    # Is this chunk a correct place to find the answer to this question?
    # It is when it hits any of the question's places (see get_locations).
    for location in get_locations(item):
        if chunk_hits_location(chunk, location):
            return True

    return False


def find_first_hit_rank(results: list, item: dict):
    # Position (1 = best) of the first result that is a hit, or None if no result is.
    # results: list of SearchResult, best first.
    for position in range(len(results)):
        if chunk_is_hit(results[position].chunk, item):
            return position + 1

    return None


def count_hits_at_k(ranks: list, k: int) -> int:
    # How many questions have a hit at rank k or better (a None rank is a miss)
    count = 0

    for rank in ranks:
        if rank is not None and rank <= k:
            count = count + 1

    return count


def mean_reciprocal_rank(ranks: list) -> float:
    # The average of 1 / rank over all questions; a miss counts as 0
    if len(ranks) == 0:
        return 0.0

    total = 0.0

    for rank in ranks:
        if rank is not None:
            total = total + 1.0 / rank

    return total / len(ranks)


def compute_metrics(ranks: list, k_values: list) -> dict:
    # All the numbers for one group of questions
    metrics = {"count": len(ranks)}

    for k in k_values:
        hits = count_hits_at_k(ranks, k)
        metrics[f"hits@{k}"] = hits

        if len(ranks) == 0:
            metrics[f"hit@{k}"] = 0.0
        else:
            metrics[f"hit@{k}"] = hits / len(ranks)

    metrics["mrr"] = mean_reciprocal_rank(ranks)

    return metrics


def split_by_completeness(rows: list) -> tuple:
    # "full" = the lecturer answers the question completely. "partial" = only part of it is
    # answered in the lecture, so a good result there is a partial answer.
    full_rows = []
    partial_rows = []

    for row in rows:
        if row["completeness"] == "partial":
            partial_rows.append(row)
        else:
            full_rows.append(row)

    return full_rows, partial_rows


def count_hops_found(results: list, item: dict, k: int) -> int:
    # For a question that needs several places ("hops"): how many of the places have a hit
    # among the top k results
    found = 0

    for hop in item.get("hops", []):
        for result in results[:k]:
            if chunk_hits_location(result.chunk, hop):
                found = found + 1
                break

    return found


def compute_hop_metrics(rows: list, k_values: list) -> dict:
    # Numbers for the multi-hop questions (rows that have "hops_total" and "hops_found"):
    #   all_hops@k   = the share of questions whose places are ALL in the top k
    #   hop_recall@k = the share of all places that are in the top k
    metrics = {"count": len(rows)}

    for k in k_values:
        questions_complete = 0
        hops_found = 0
        hops_total = 0

        for row in rows:
            found_here = row["hops_found"][k]
            hops_found = hops_found + found_here
            hops_total = hops_total + row["hops_total"]

            if found_here == row["hops_total"]:
                questions_complete = questions_complete + 1

        if len(rows) == 0 or hops_total == 0:
            metrics[f"all_hops@{k}"] = 0.0
            metrics[f"hop_recall@{k}"] = 0.0
        else:
            metrics[f"all_hops@{k}"] = questions_complete / len(rows)
            metrics[f"hop_recall@{k}"] = hops_found / hops_total

    return metrics


def split_by_type(rows: list) -> dict:
    # Groups the rows by their "type" (for example detail, paraphrase, decoy), keeping the
    # order in which each type first appears. Rows without a type are left out.
    groups = {}

    for row in rows:
        question_type = row.get("type")
        if question_type is None:
            continue

        if question_type not in groups:
            groups[question_type] = []
        groups[question_type].append(row)

    return groups


def summarize_results(results: list, item: dict | None = None) -> list:
    # The search results as plain data for the saved file (and for printing)
    summary = []

    for position in range(len(results)):
        result = results[position]
        chunk = result.chunk

        entry = {
            "rank": position + 1,
            "chunk_id": chunk.chunk_id,
            "lecture_id": chunk.lecture_id,
            "start": chunk.start_timestamp,
            "end": chunk.end_timestamp,
            "score": result.score,
        }

        # Only questions with a known answer can say whether a result is a hit
        if item is not None:
            entry["hit"] = chunk_is_hit(chunk, item)

        summary.append(entry)

    return summary


def mean_top_chunk_seconds(rows: list) -> float:
    # The average length, in seconds, of the best-scoring result of each question. It tells how
    # big the chunks are, which matters when settings that change the chunk size are compared.
    lengths = []

    for row in rows:
        if len(row["results"]) > 0:
            top = row["results"][0]
            lengths.append(top["end"] - top["start"])

    if len(lengths) == 0:
        return 0.0

    return sum(lengths) / len(lengths)


def score_range(rows: list) -> tuple:
    # (lowest, highest) best-result score over some rows, or None when no row has a result
    scores = []

    for row in rows:
        if row["top_score"] is not None:
            scores.append(row["top_score"])

    if len(scores) == 0:
        return None

    return min(scores), max(scores)


def load_eval_set(path: str) -> list:
    # The list of evaluation items, exactly as written in the file
    with open(path, "r", encoding="utf-8") as file:
        items = json.load(file)

    return items


def settings_snapshot(top_k: int) -> dict:
    # The settings that decide what the search can find, saved with every result so two runs
    # can be compared later. They are read from the config; if you changed a setting without
    # rebuilding the chunks (python -m src.pipeline --all --force chunk), they do not describe
    # what is stored in the database.
    return {
        "retrieval_method": "dense" if len(config.RETRIEVAL_SIGNALS) == 1 else "dense fusion",
        "retrieval_signals": config.RETRIEVAL_SIGNALS,
        "top_k": top_k,
        "chunk_target_words": config.CHUNK_TARGET_WORDS,
        "chunk_max_words": config.CHUNK_MAX_WORDS,
        "chunk_overlap_segments": config.CHUNK_OVERLAP_SEGMENTS,
        "chunk_min_last_words": config.CHUNK_MIN_LAST_WORDS,
        "chunk_slide_aware": config.CHUNK_SLIDE_AWARE,
        "chunk_slide_aware_min_words": config.CHUNK_SLIDE_AWARE_MIN_WORDS,
        "chunk_slide_text_source": config.CHUNK_SLIDE_TEXT_SOURCE,
        "chunk_include_description": config.CHUNK_INCLUDE_DESCRIPTION,
        "qdrant_path": config.QDRANT_PATH,
        "embedding_provider": config.EMBEDDING_PROVIDER,
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_max_tokens": config.EMBEDDING_MAX_TOKENS,
    }


# ---------------------------------------------------------------------------
# Running the evaluation
# ---------------------------------------------------------------------------

def run_retrieval(items: list, top_k: int, client) -> tuple:
    # Searches every question once. Returns (answerable_rows, unanswerable_rows, all_results):
    #   the rows hold the numbers for each question
    #   all_results maps the question id to the SearchResult list (needed for --answers)
    # The search always fetches at least ANSWER_TOP_K results, so the answer step can use them.
    fetch_count = max(top_k, ANSWER_TOP_K)

    answerable_rows = []
    unanswerable_rows = []
    all_results = {}

    for item in items:
        fetched = retrieve(item["question"], top_k=fetch_count, client=client)
        all_results[item["id"]] = fetched

        # The ranks and metrics only look at the top_k results
        ranked = fetched[:top_k]

        top_score = None
        if len(ranked) > 0:
            top_score = ranked[0].score

        if item["answerable"]:
            row = {
                "id": item["id"],
                "type": item.get("type"),
                "completeness": item.get("answer_completeness", "full"),
                "lecture": item["lecture"],
                "rank": find_first_hit_rank(ranked, item),
                "strict_rank": find_first_strict_hit_rank(ranked, item),
                "top_score": top_score,
                "results": summarize_results(ranked, item),
            }

            # A question that needs several places: also count how many places are in the top k
            # (its "rank" above is the position of the first result that hits any of the places)
            if len(item.get("hops", [])) > 0:
                row["hops_total"] = len(item["hops"])
                row["hops_found"] = {}
                for k in HIT_K_VALUES:
                    if k <= top_k:
                        row["hops_found"][k] = count_hops_found(ranked, item, k)

            answerable_rows.append(row)
        else:
            row = {
                "id": item["id"],
                "type": item.get("type"),
                "top_score": top_score,
                "results": summarize_results(ranked),
            }
            unanswerable_rows.append(row)

    return answerable_rows, unanswerable_rows, all_results


def answer_with_patience(question: str, results: list):
    # Writes the answer, waiting and trying again when the Gemini call keeps failing.
    # Returns the FinalAnswer, or None when every try failed (the caller records an error).
    attempt = 0

    while True:
        try:
            return answer_question(question, results)
        except Exception as error:
            if attempt >= len(ANSWER_RETRY_WAITS):
                print(f"  giving up on this question: {error}")
                return None

            wait = ANSWER_RETRY_WAITS[attempt]
            print(f"  the answer step failed, waiting {wait}s before trying again...")
            time.sleep(wait)
            attempt = attempt + 1


def run_answers(items: list, all_results: dict, answerable_rows: list, unanswerable_rows: list) -> dict:
    # Writes an answer for every question with Gemini (one call each) and records what happened.
    # The answers are added to the rows. Returns the totals.
    rows_by_id = {}
    for row in answerable_rows + unanswerable_rows:
        rows_by_id[row["id"]] = row

    # Stay under the free quota of requests per minute
    pause_seconds = 60.0 / config.GEMINI_REQUESTS_PER_MINUTE

    for i, item in enumerate(items):
        if i > 0:
            time.sleep(pause_seconds)

        results = all_results[item["id"]][:ANSWER_TOP_K]
        print(f"  answering {item['id']} ({i + 1} of {len(items)})...")
        final = answer_with_patience(item["question"], results)

        row = rows_by_id[item["id"]]

        # Every try failed: record it as an error and go on with the next question
        if final is None:
            row["generated_answerable"] = None
            row["generated_coverage"] = None
            row["generated_answer"] = "(no answer: the Gemini call kept failing)"
            row["generated_warnings"] = []
            if item["answerable"]:
                row["cites_right_place"] = False
            continue

        row["generated_answerable"] = final.answerable
        row["generated_coverage"] = final.coverage
        row["generated_answer"] = final.text
        row["generated_warnings"] = final.warnings

        if item["answerable"]:
            # Does the answer cite at least one chunk from the right place in the lecture?
            cites_right_place = False
            for source in final.sources:
                if chunk_is_hit(source.chunk, item):
                    cites_right_place = True
            row["cites_right_place"] = cites_right_place

    # Totals (a question whose answer failed counts as neither refused nor answered)
    refused_correctly = 0
    answer_errors = 0
    for row in unanswerable_rows:
        if row["generated_answerable"] is False:
            refused_correctly = refused_correctly + 1
        if row["generated_answerable"] is None:
            answer_errors = answer_errors + 1

    wrongly_refused = 0
    cited_right_place = 0
    for row in answerable_rows:
        if row["generated_answerable"] is False:
            wrongly_refused = wrongly_refused + 1
        if row["generated_answerable"] is None:
            answer_errors = answer_errors + 1
        if row["cites_right_place"]:
            cited_right_place = cited_right_place + 1

    # Does the answer step say "partial" for the questions the lecture only partly answers, and
    # not for the ones it answers fully (hedging)? Answers that failed are left out.
    partial_total = 0
    partial_marked_partial = 0
    full_total = 0
    full_marked_partial = 0

    for row in answerable_rows:
        if row["generated_coverage"] is None:
            continue

        if row["completeness"] == "partial":
            partial_total = partial_total + 1
            if row["generated_coverage"] == "partial":
                partial_marked_partial = partial_marked_partial + 1
        else:
            full_total = full_total + 1
            if row["generated_coverage"] == "partial":
                full_marked_partial = full_marked_partial + 1

    return {
        "partial_questions_total": partial_total,
        "partial_questions_marked_partial": partial_marked_partial,
        "full_questions_total": full_total,
        "full_questions_marked_partial": full_marked_partial,
        "unanswerable_refused": refused_correctly,
        "unanswerable_total": len(unanswerable_rows),
        "answerable_wrongly_refused": wrongly_refused,
        "answerable_citing_right_place": cited_right_place,
        "answerable_total": len(answerable_rows),
        "answer_errors": answer_errors,
    }


# ---------------------------------------------------------------------------
# Printing and saving
# ---------------------------------------------------------------------------

def format_rank(rank, top_k: int) -> str:
    if rank is None:
        return f"miss (not in top {top_k})"
    return str(rank)


def print_answerable_table(rows: list, top_k: int) -> None:
    print("Questions with an answer in the lectures")
    print("(rank = position of the first result that covers the answer; top = the best-scoring result)\n")

    # The question type column is only shown when the questions have a type
    show_type = False
    for row in rows:
        if row.get("type") is not None:
            show_type = True

    for row in rows:
        top_text = "no results"
        if len(row["results"]) > 0:
            top = row["results"][0]
            start = format_timestamp(top["start"])
            end = format_timestamp(top["end"])
            top_text = f"{top['lecture_id']} {start} - {end}  score {top['score']:.3f}"

        type_text = ""
        if show_type:
            type_text = f"{row.get('type') or '':<21}"

        rank_text = format_rank(row["rank"], top_k)
        print(f"  {row['id']:<8} {type_text}{row['completeness']:<8} rank {rank_text:<22} top: {top_text}")

    print()


def print_metrics(title: str, metrics: dict, top_k: int) -> None:
    print(f"{title} ({metrics['count']} questions)")

    if metrics["count"] == 0:
        print("  none\n")
        return

    for k in HIT_K_VALUES:
        if k <= top_k:
            print(f"  hit@{k:<3} {metrics[f'hit@{k}']:.2f}   ({metrics[f'hits@{k}']} of {metrics['count']})")

    print(f"  MRR     {metrics['mrr']:.3f}")
    print()


def print_hop_metrics(metrics: dict, top_k: int) -> None:
    print(f"Questions that need several places ({metrics['count']} questions)")

    for k in HIT_K_VALUES:
        if k <= top_k:
            print(f"  all places in the top {k:<3} {metrics[f'all_hops@{k}']:.2f}     share of the places found {metrics[f'hop_recall@{k}']:.2f}")

    print()


def print_unanswerable(rows: list, answerable_rows: list) -> None:
    # Nothing to show when the question file has no question the lectures do not cover
    if len(rows) == 0:
        return

    print("Questions the lectures do not cover (the best score of a search result)")

    for row in rows:
        score_text = "no results"
        if row["top_score"] is not None:
            score_text = f"{row['top_score']:.3f}"
        print(f"  {row['id']:<8} top score {score_text}")

    answerable_range = score_range(answerable_rows)
    unanswerable_range = score_range(rows)

    if answerable_range is not None and unanswerable_range is not None:
        print(f"  best scores, questions with an answer:   {answerable_range[0]:.3f} to {answerable_range[1]:.3f}")
        print(f"  best scores, questions without an answer: {unanswerable_range[0]:.3f} to {unanswerable_range[1]:.3f}")

    print()


def print_answers(items: list, answerable_rows: list, unanswerable_rows: list, totals: dict) -> None:
    # The generated answer next to the reference answer, for a human to read
    rows_by_id = {}
    for row in answerable_rows + unanswerable_rows:
        rows_by_id[row["id"]] = row

    print("Generated answers (read them against the reference answers; there is no automatic judge yet)\n")

    for item in items:
        row = rows_by_id[item["id"]]
        print(f"--- {item['id']} ---")
        print(f"Question: {item['question']}")

        if item["answerable"]:
            print(f"Answer step said coverage = {row['generated_coverage']} (the lecture answers it: {item.get('answer_completeness', 'full')}), cites the right place = {row['cites_right_place']}")
        else:
            print(f"Answer step said coverage = {row['generated_coverage']}   (should be none)")

        print("Generated:")
        print(row["generated_answer"])

        if item["answerable"]:
            print("Reference:")
            print(item["reference_answer"])

        for warning in row["generated_warnings"]:
            print(f"WARNING: {warning}")

        print()

    print("Answer step totals")
    print(f"  questions without an answer that were refused: {totals['unanswerable_refused']} of {totals['unanswerable_total']}")
    print(f"  questions with an answer that were refused:    {totals['answerable_wrongly_refused']} of {totals['answerable_total']}")
    print(f"  answers that cite the right place:             {totals['answerable_citing_right_place']} of {totals['answerable_total']}")
    print(f"  partly answered questions marked partly covered: {totals['partial_questions_marked_partial']} of {totals['partial_questions_total']}")
    print(f"  fully answered questions marked partly covered:  {totals['full_questions_marked_partial']} of {totals['full_questions_total']}   (hedging)")
    if totals["answer_errors"] > 0:
        print(f"  questions whose answer failed (Gemini errors): {totals['answer_errors']}")
    print()


def save_results(output_folder: str, record: dict) -> Path:
    # results_<date>_<time>.json next to the evaluation file
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(output_folder) / f"results_{stamp}.json"

    # Two runs that finish in the same second would get the same name. Mode "x" refuses to open a
    # file that already exists, so in that case we add a counter (_2, _3, ...) instead of writing
    # into the other run's file (that mixed the text of two files once).
    counter = 1
    while True:
        try:
            file = open(path, "x", encoding="utf-8")
            break
        except FileExistsError:
            counter += 1
            path = Path(output_folder) / f"results_{stamp}_{counter}.json"

    with file:
        json.dump(record, file, indent=2, ensure_ascii=False)

    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the search (and optionally the answers) on the evaluation questions.")
    parser.add_argument("--top", type=int, default=10, help="how many results to look at per question (default 10)")
    parser.add_argument("--answers", action="store_true", help="also write answers with Gemini (about one call per question)")
    parser.add_argument("--eval-file", default=DEFAULT_EVAL_FILE, help="the evaluation questions (default data/eval/retrieval_eval.json)")
    parser.add_argument("--no-save", action="store_true", help="do not write a results file")
    args = parser.parse_args()

    items = load_eval_set(args.eval_file)

    # The database is opened once and shared by all searches (only one program may have it open)
    client = open_client()
    try:
        answerable_rows, unanswerable_rows, all_results = run_retrieval(items, args.top, client)
    finally:
        client.close()

    # No result for any question means nothing is indexed
    nothing_found = True
    for results in all_results.values():
        if len(results) > 0:
            nothing_found = False

    if nothing_found:
        print("Nothing found for any question. Are the lectures indexed? Run: python -m src.pipeline --all")
        return

    print(f"\nEvaluation: dense search, top {args.top}, {len(items)} questions\n")

    print_answerable_table(answerable_rows, args.top)

    # Metrics: all answerable questions, then the two kinds separately
    full_rows, partial_rows = split_by_completeness(answerable_rows)

    metric_groups = [
        ("All questions with an answer", answerable_rows),
        ("Fully answered in the lecture", full_rows),
        ("Only partly answered in the lecture (a good result is a partial answer)", partial_rows),
    ]
    metrics_record = {}

    for title, rows in metric_groups:
        ranks = []
        for row in rows:
            ranks.append(row["rank"])

        metrics = compute_metrics(ranks, HIT_K_VALUES)
        metrics_record[title] = metrics
        print_metrics(title, metrics, args.top)

    # The same numbers with the stricter hit test (the chunk must cover at least half of the
    # answer range), which does not favour big chunks
    strict_ranks = []
    for row in answerable_rows:
        strict_ranks.append(row["strict_rank"])

    strict_title = "Strict hits (the chunk covers at least half of the answer range)"
    strict_metrics = compute_metrics(strict_ranks, HIT_K_VALUES)
    metrics_record[strict_title] = strict_metrics
    print_metrics(strict_title, strict_metrics, args.top)

    average_length = mean_top_chunk_seconds(answerable_rows)
    metrics_record["Average length of the best result (seconds)"] = average_length
    print(f"Average length of the best result: {average_length:.0f} seconds\n")

    # The same numbers for each question type (only when the questions have a type).
    # A multi-hop question counts here with the position of its first hit on any of its places.
    rows_by_type = split_by_type(answerable_rows)

    for type_name in rows_by_type:
        ranks = []
        for row in rows_by_type[type_name]:
            ranks.append(row["rank"])

        metrics = compute_metrics(ranks, HIT_K_VALUES)
        metrics_record[f"Type: {type_name}"] = metrics
        print_metrics(f"Type: {type_name}", metrics, args.top)

    # Questions that need several places: are all of the places in the top k?
    hop_rows = []
    for row in answerable_rows:
        if "hops_total" in row:
            hop_rows.append(row)

    if len(hop_rows) > 0:
        usable_k_values = []
        for k in HIT_K_VALUES:
            if k <= args.top:
                usable_k_values.append(k)

        hop_metrics = compute_hop_metrics(hop_rows, usable_k_values)
        metrics_record["Questions that need several places"] = hop_metrics
        print_hop_metrics(hop_metrics, args.top)

    print_unanswerable(unanswerable_rows, answerable_rows)

    answer_totals = None
    if args.answers:
        print(f"Writing answers with Gemini ({len(items)} calls, paced under the free quota)...")
        answer_totals = run_answers(items, all_results, answerable_rows, unanswerable_rows)
        print()
        print_answers(items, answerable_rows, unanswerable_rows, answer_totals)

    if not args.no_save:
        record = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "settings": settings_snapshot(args.top),
            "metrics": metrics_record,
            "answerable_items": answerable_rows,
            "unanswerable_items": unanswerable_rows,
            "answer_totals": answer_totals,
        }
        path = save_results(str(Path(args.eval_file).parent), record)
        print(f"Results saved to {path}")


if __name__ == "__main__":
    main()
