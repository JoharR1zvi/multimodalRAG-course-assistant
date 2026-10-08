# Ask a question and get an answer written from the lectures, with sources.
#
# Usage (from the project root):
#   python -m src.ask "What is the difference between sensitivity and specificity?"
#   python -m src.ask "What is a confusion matrix?" --lecture lecture_01
#   python -m src.ask "..." --top 8          # give the model more excerpts to read (default 5)
#
# Steps: search (src/search.py) -> Gemini writes the answer from the found excerpts
# (src/generation/llm_service.py) -> the sources shown below the answer are built from the
# real metadata of the chunks, never from what the model says.

import argparse

from src.generation.llm_service import answer_question
from src.retrieval.retriever import format_timestamp, retrieve
from src.search import unique_titles


def format_source(number: int, result) -> str:
    # One line per source: [2] lecture_01 | 17:54 - 20:45 | slide titles
    chunk = result.chunk

    start = format_timestamp(chunk.start_timestamp)
    end = format_timestamp(chunk.end_timestamp)

    titles = unique_titles(chunk.slide_titles)
    if len(titles) == 0:
        titles_text = "(no slide titles)"
    else:
        titles_text = " | ".join(titles)

    lines = []
    lines.append(f"[{number}] {chunk.lecture_id} | {start} - {end} | {titles_text}")

    # The picture of the first slide of this stretch, so the student can look at it
    if len(chunk.image_paths) > 0:
        lines.append(f"     slide image: {chunk.image_paths[0]}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question about the indexed lectures.")
    parser.add_argument("question", help="your question, in quotes")
    parser.add_argument("--top", type=int, default=5, help="how many excerpts the model gets to read (default 5)")
    parser.add_argument("--lecture", default=None, help="only use this lecture, e.g. lecture_01")
    args = parser.parse_args()

    results = retrieve(args.question, top_k=args.top, lecture_id=args.lecture)
    final = answer_question(args.question, results)

    print(f"\nQuestion: {args.question}\n")

    if not final.answerable:
        print("NOT FOUND IN THE COURSE MATERIAL")
        print(final.text)
    else:
        print(final.text)

    if len(final.sources) > 0:
        print("\nSources:")
        for i, source in enumerate(final.sources):
            print(format_source(final.source_numbers[i], source))

    # If the question could not be answered, show where the search looked, so the student can check
    if not final.answerable and len(results) > 0:
        print("\nClosest passages the search found:")
        for i, result in enumerate(results[:3]):
            print(f"  score {result.score:.3f}  " + format_source(i + 1, result).split("\n")[0])

    for warning in final.warnings:
        print(f"\nWARNING: {warning}")

    print()


if __name__ == "__main__":
    main()
