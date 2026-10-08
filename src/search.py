# Ask a question, see which parts of the lectures match it best. No answer is written;
# this only shows the search results, so you can judge the search on its own.
#
# Usage (from the project root):
#   python -m src.search "What is a confusion matrix?"
#   python -m src.search "What is a confusion matrix?" --top 3
#   python -m src.search "What is a confusion matrix?" --lecture lecture_01

import argparse

from src.retrieval.retriever import format_timestamp, retrieve

# How many characters of the speech to show under each result
PREVIEW_CHARACTERS = 220


def unique_titles(titles: list) -> list:
    # The same slide title often appears several times in a row (one slide, several screen
    # states). Show each title once, in the order it first appears.
    seen = []
    for title in titles:
        if title not in seen:
            seen.append(title)
    return seen


def format_result(rank: int, result) -> str:
    # One search result as a few lines of text
    chunk = result.chunk

    start = format_timestamp(chunk.start_timestamp)
    end = format_timestamp(chunk.end_timestamp)

    titles = unique_titles(chunk.slide_titles)
    if len(titles) == 0:
        titles_text = "(no slide titles)"
    else:
        titles_text = " | ".join(titles)

    # Show the beginning of the speech, cut at a word boundary
    speech = chunk.text.strip().replace("\n", " ")
    if len(speech) > PREVIEW_CHARACTERS:
        speech = speech[:PREVIEW_CHARACTERS].rsplit(" ", 1)[0] + " ..."

    lines = []
    lines.append(f"{rank}. score {result.score:.3f} | {chunk.lecture_id} | {start} - {end}")
    lines.append(f"   Slides: {titles_text}")
    lines.append(f"   \"{speech}\"")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the indexed lectures with a question.")
    parser.add_argument("question", help="your question, in quotes")
    parser.add_argument("--top", type=int, default=5, help="how many results to show (default 5)")
    parser.add_argument("--lecture", default=None, help="only search inside this lecture, e.g. lecture_01")
    args = parser.parse_args()

    results = retrieve(args.question, top_k=args.top, lecture_id=args.lecture)

    if args.lecture is None:
        where = "all lectures"
    else:
        where = args.lecture

    print(f"\nQuestion: {args.question}")
    print(f"Searching {where}, top {args.top}\n")

    if len(results) == 0:
        print("Nothing found. Are the lectures indexed? Run: python -m src.pipeline --all")
        return

    for i, result in enumerate(results):
        print(format_result(i + 1, result))
        print()


if __name__ == "__main__":
    main()
