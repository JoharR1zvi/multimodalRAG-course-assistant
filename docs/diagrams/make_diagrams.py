# Draws the diagrams used in the README and in docs/architecture.md as plain SVG files.
#
# Run it from the project root:
#   python docs/diagrams/make_diagrams.py
#
# It writes four files next to this script:
#   overview.svg      the whole system in three parts (A, B, C)
#   phase1.svg        part A: from a lecture video to knowledge objects (pipeline stages 1 to 8)
#   phase2.svg        parts B and C: getting ready to search (stages 9 and 10) and answering a question
#   one-question.svg  part C followed step by step with one example question
#
# The SVGs use plain shapes and fixed colours (no scripts, no style sheets), so they look the
# same on GitHub in light and dark mode and in any image viewer. Change a text or a colour here
# and run the script again.

import html
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent

SANS = "Segoe UI, Helvetica, Arial, sans-serif"
MONO = "Consolas, Menlo, Courier New, monospace"

TEXT_COLOR = "#1F2A37"
MUTED_COLOR = "#4D5A6D"
ARROW_COLOR = "#7B8798"
PAGE_COLOR = "#FFFFFF"
PAGE_BORDER = "#D8DEE8"

# One colour family per part of the system: a light fill (for the lane behind the cards),
# a stroke (card outline and icon circle) and a dark shade (numbers and file names).
THEMES = {
    "phase1": {"fill": "#E8F0FE", "stroke": "#4A78C8", "dark": "#2A4F8C"},
    "index": {"fill": "#E2F4EE", "stroke": "#2E9C85", "dark": "#176654"},
    "answer": {"fill": "#FDEEDD", "stroke": "#DB8A33", "dark": "#94500B"},
    "planned": {"fill": "#F2F4F7", "stroke": "#8F99A8", "dark": "#586577"},
}

# Settings for white line icons drawn on top of a coloured circle
LINE = 'fill="none" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"'
SOLID = 'fill="#FFFFFF"'


def wrap_text(content, max_chars):
    # Splits a sentence into lines of at most max_chars characters, breaking only between words
    words = content.split(" ")
    lines = []
    current = ""

    for word in words:
        if current == "":
            candidate = word
        else:
            candidate = current + " " + word

        if len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = word

    if current != "":
        lines.append(current)

    return lines


def icon_shapes(kind):
    # The white drawing inside an icon circle. Every shape is drawn around the point (0, 0)
    # and fits in a square of about 22 x 22.
    if kind == "video":
        return (f'<rect x="-11" y="-7" width="15" height="14" rx="3" {LINE}/>'
                f'<path d="M4,-2 L11,-6 L11,6 L4,2 Z" {SOLID}/>')

    if kind == "sound":
        # five bars of different heights, like a sound wave
        bars = [(-11.5, 6), (-6.5, 14), (-1.5, 22), (3.5, 12), (8.5, 6)]
        shapes = ""
        for bar_x, bar_height in bars:
            shapes += f'<rect x="{bar_x}" y="{-bar_height / 2}" width="3" height="{bar_height}" rx="1.5" {SOLID}/>'
        return shapes

    if kind == "text":
        return (f'<rect x="-8" y="-11" width="16" height="22" rx="2" {LINE}/>'
                f'<path d="M-4,-5 H4 M-4,0 H4 M-4,5 H1" {LINE}/>')

    if kind == "frames":
        return (f'<rect x="-11" y="-8" width="19" height="14" rx="2" {LINE}/>'
                f'<rect x="-6" y="-3" width="19" height="14" rx="2" {LINE} fill="#4A78C8"/>')

    if kind == "ocr":
        return f'<text x="0" y="6" text-anchor="middle" font-family="{SANS}" font-size="16" font-weight="bold" {SOLID}>Aa</text>'

    if kind == "eye":
        return (f'<path d="M-11,0 Q0,-10 11,0 Q0,10 -11,0 Z" {LINE}/>'
                f'<circle cx="0" cy="0" r="3.5" {SOLID}/>')

    if kind == "clean":
        return f'<path d="M-10,-9 H10 L3,0 V9 L-3,6 V0 Z" {SOLID}/>'

    if kind == "link":
        return (f'<circle cx="-4.5" cy="0" r="7" {LINE}/>'
                f'<circle cx="4.5" cy="0" r="7" {LINE}/>')

    if kind == "records":
        return (f'<rect x="-10" y="-11" width="20" height="6" rx="2" {SOLID}/>'
                f'<rect x="-10" y="-3" width="20" height="6" rx="2" {SOLID}/>'
                f'<rect x="-10" y="5" width="20" height="6" rx="2" {SOLID}/>')

    if kind == "chunks":
        return (f'<rect x="-11" y="-5" width="6" height="10" rx="1.5" {SOLID}/>'
                f'<rect x="-3" y="-5" width="6" height="10" rx="1.5" {SOLID}/>'
                f'<rect x="5" y="-5" width="6" height="10" rx="1.5" {SOLID}/>')

    if kind == "numbers":
        dots = ""
        for dot_x in (-7, 0, 7):
            for dot_y in (-7, 0, 7):
                dots += f'<circle cx="{dot_x}" cy="{dot_y}" r="1.9" {SOLID}/>'
        return dots

    if kind == "database":
        return (f'<ellipse cx="0" cy="-7" rx="9" ry="3.5" {LINE}/>'
                f'<path d="M-9,-7 V7 A9,3.5 0 0 0 9,7 V-7" {LINE}/>'
                f'<path d="M-9,0 A9,3.5 0 0 0 9,0" {LINE}/>')

    if kind == "question":
        return f'<text x="0" y="8" text-anchor="middle" font-family="{SANS}" font-size="24" font-weight="bold" {SOLID}>?</text>'

    if kind == "search":
        return (f'<circle cx="-2" cy="-2" r="7" {LINE}/>'
                f'<path d="M3,3 L10,10" {LINE} stroke-width="3"/>')

    if kind == "write":
        # a four-pointed star
        return f'<path d="M0,-11 L2.5,-2.5 L11,0 L2.5,2.5 L0,11 L-2.5,2.5 L-11,0 L-2.5,-2.5 Z" {SOLID}/>'

    if kind == "check":
        return f'<path d="M-9,0 L-3,6 L9,-7" {LINE} stroke-width="3"/>'

    if kind == "answer":
        return (f'<rect x="-11" y="-9" width="22" height="15" rx="3" {LINE}/>'
                f'<path d="M-4,6 L-6,11 L1,6" {LINE}/>'
                f'<path d="M-6,-3 H6 M-6,1 H2" {LINE}/>')

    if kind == "target":
        return (f'<circle cx="0" cy="0" r="10" {LINE}/>'
                f'<circle cx="0" cy="0" r="5.5" {LINE}/>'
                f'<circle cx="0" cy="0" r="1.8" {SOLID}/>')

    if kind == "trend":
        return (f'<path d="M-10,7 L-3,0 L2,4 L10,-7" {LINE}/>'
                f'<path d="M4,-7 H10 V-1" {LINE}/>')

    raise ValueError(f"Unknown icon: {kind}")


class Drawing:
    def __init__(self, name, width, height, description):
        self.name = name
        self.width = width
        self.height = height
        self.description = description
        self.parts = []
        self.problems = []

    # ---------- basic shapes ----------

    def rect(self, x, y, w, h, fill, stroke=None, radius=12, dashed=False, stroke_width=1.5, opacity=1.0):
        attributes = f'x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}"'
        if opacity != 1.0:
            attributes += f' fill-opacity="{opacity}"'
        if stroke is not None:
            attributes += f' stroke="{stroke}" stroke-width="{stroke_width}"'
        if dashed:
            attributes += ' stroke-dasharray="7 5"'
        self.parts.append(f"<rect {attributes}/>")

    def text(self, x, y, content, size=13, weight="normal", fill=TEXT_COLOR, anchor="start", font=SANS, italic=False):
        attributes = f'x="{x}" y="{y}" font-family="{font}" font-size="{size}" font-weight="{weight}" fill="{fill}"'
        if anchor != "start":
            attributes += f' text-anchor="{anchor}"'
        if italic:
            attributes += ' font-style="italic"'
        # SVG squeezes runs of spaces into one. In code-style text the spacing matters, so every
        # space there becomes a non-breaking space, which is kept as it is.
        if font == MONO:
            content = content.replace(" ", " ")
        self.parts.append(f"<text {attributes}>{html.escape(content, quote=False)}</text>")

    def paragraph(self, x, y, content, max_chars, size=12.5, line_height=16, fill=MUTED_COLOR, anchor="start", max_lines=3):
        # Writes a sentence as several lines. Returns the y position after the last line.
        lines = wrap_text(content, max_chars)
        if len(lines) > max_lines:
            self.problems.append(f"{self.name}: more than {max_lines} lines: {content!r}")
        for line in lines:
            self.text(x, y, line, size=size, fill=fill, anchor=anchor)
            y += line_height
        return y

    def icon(self, kind, cx, cy, theme, radius=22):
        colors = THEMES[theme]
        scale = radius / 22
        self.parts.append(f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{colors["stroke"]}"/>')
        self.parts.append(f'<g transform="translate({cx},{cy}) scale({scale})">{icon_shapes(kind)}</g>')

    def badge(self, x, y, label, theme):
        # A small dark circle with a step number or letter in it, usually on a card's corner
        colors = THEMES[theme]
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="14" fill="{colors["dark"]}" stroke="#FFFFFF" stroke-width="2.5"/>')
        self.text(x, y + 4.5, str(label), size=12.5, weight="bold", fill="#FFFFFF", anchor="middle")

    def panel(self, x, y, w, h, theme, label, subtitle=None, dashed=False):
        # A tinted lane behind a group of cards, with its name in the top left corner
        colors = THEMES[theme]
        self.rect(x, y, w, h, fill=colors["fill"], radius=18, opacity=0.65)
        if dashed:
            self.rect(x, y, w, h, fill="none", stroke=colors["stroke"], radius=18, dashed=True, stroke_width=1.5)
        self.text(x + 18, y + 26, label, size=13.5, weight="bold", fill=colors["dark"])
        if subtitle is not None:
            self.text(x + 18, y + 44, subtitle, size=12, fill=MUTED_COLOR)

    # ---------- arrows ----------

    def arrow(self, points, dashed=False, label_lines=None, label_at=None, label_anchor="middle", size=1.0):
        # size makes the line and the arrowhead bigger (1.5 = half as big again)
        path = "M" + " L".join(f"{px},{py}" for px, py in points)
        dash = ' stroke-dasharray="6 5"' if dashed else ""
        self.parts.append(f'<path d="{path}" fill="none" stroke="{ARROW_COLOR}" stroke-width="{2.2 * size}" stroke-linecap="round" stroke-linejoin="round"{dash}/>')

        # The arrowhead is a small triangle at the last point, turned to match the last segment
        (x1, y1), (x2, y2) = points[-2], points[-1]
        length = 10 * size
        half = 5.5 * size
        if x1 == x2:
            direction = 1 if y2 > y1 else -1
            head = f"{x2},{y2} {x2 - half},{y2 - direction * length} {x2 + half},{y2 - direction * length}"
        else:
            direction = 1 if x2 > x1 else -1
            head = f"{x2},{y2} {x2 - direction * length},{y2 - half} {x2 - direction * length},{y2 + half}"
        self.parts.append(f'<polygon points="{head}" fill="{ARROW_COLOR}"/>')

        if label_lines is not None:
            label_x, label_y = label_at
            for line in label_lines:
                self.text(label_x, label_y, line, size=12, fill=MUTED_COLOR, anchor=label_anchor, italic=True)
                label_y += 14

    # ---------- cards ----------

    def card(self, x, y, w, h, theme, icon, title, description, tech=None, number=None, dashed=False):
        # A step of the pipeline: an icon, a title, one or two plain sentences, and a small line
        # naming the tool or the file.
        colors = THEMES[theme]
        self.rect(x, y, w, h, fill="#FFFFFF", stroke=colors["stroke"], radius=14, dashed=dashed, stroke_width=2)
        self.icon(icon, x + 38, y + h / 2, theme)

        text_x = x + 74
        max_chars = int((w - 74 - 14) / 6.2)

        if len(title) * 8.4 > (w - 74 - 14):
            self.problems.append(f"{self.name}: title may be too wide: {title!r}")
        self.text(text_x, y + 28, title, size=15, weight="bold")
        self.paragraph(text_x, y + 48, description, max_chars, size=12.5, line_height=16, max_lines=3)

        if tech is not None:
            if len(tech) * 6.7 > (w - 74 - 14):
                self.problems.append(f"{self.name}: tech line may be too wide: {tech!r}")
            self.text(text_x, y + h - 12, tech, size=11, fill=colors["dark"], font=MONO)

        if number is not None:
            self.badge(x + 4, y + 4, number, theme)

    def tall_card(self, x, y, w, h, theme, icon, title, description, tech):
        # A narrow card with everything centred, used for the database in the middle
        colors = THEMES[theme]
        self.rect(x, y, w, h, fill="#FFFFFF", stroke=colors["stroke"], radius=14, stroke_width=2)
        self.icon(icon, x + w / 2, y + 38, theme)
        self.text(x + w / 2, y + 86, title, size=15, weight="bold", anchor="middle")
        self.paragraph(x + w / 2, y + 106, description, int((w - 20) / 6.0), size=12, line_height=15, anchor="middle", max_lines=4)
        self.text(x + w / 2, y + h - 12, tech, size=11, fill=colors["dark"], font=MONO, anchor="middle")

    def wide_frame(self, x, y, w, h, theme, icon, title, number, fill="#FFFFFF"):
        # The outline of a wide step with an icon and a title. The caller writes the content.
        colors = THEMES[theme]
        self.rect(x, y, w, h, fill=fill, stroke=colors["stroke"], radius=14, stroke_width=2)
        self.icon(icon, x + 38, y + 38, theme)
        self.text(x + 74, y + 30, title, size=15.5, weight="bold")
        self.badge(x + 4, y + 4, number, theme)

    # ---------- output ----------

    def svg(self):
        body = "\n".join(self.parts)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}" role="img" aria-label="{html.escape(self.description)}">\n'
            f"<title>{html.escape(self.description)}</title>\n"
            f'<rect x="0.5" y="0.5" width="{self.width - 1}" height="{self.height - 1}" rx="16" fill="{PAGE_COLOR}" stroke="{PAGE_BORDER}"/>\n'
            f"{body}\n</svg>\n"
        )

    def save(self):
        path = OUTPUT_DIR / f"{self.name}.svg"
        path.write_text(self.svg(), encoding="utf-8")
        print(f"wrote {path.name} ({self.width} x {self.height})")


# =========================================================================================
# 1. The whole system in three parts: A, B, C
# =========================================================================================
def draw_overview():
    width = 984
    card_w = 280
    card_h = 250
    top = 24
    xs = [24, 24 + card_w + 48, 24 + 2 * (card_w + 48)]

    analogy_y = top + card_h + 42
    next_y = analogy_y + 26
    height = next_y + 56 + 24

    d = Drawing("overview", width, height,
                "The whole system in three parts: A understand the lecture, B get ready to search, C answer a question.")

    parts = [
        ("phase1", "A", "video", "Understand the lecture",
         "Listen to the speech, look at the slides, and pair the two up.",
         "a lecture video", "knowledge objects (1 per slide)"),
        ("index", "B", "database", "Get ready to search",
         "Cut the lecture into pieces (chunks) and file each one by what it is about, so it can be found by meaning.",
         "knowledge objects", "a searchable database"),
        ("answer", "C", "answer", "Answer a question",
         "Find the closest chunks and write an answer using only the lecture, with sources.",
         "your question", "answer with sources"),
    ]

    for i, (theme, letter, icon, title, description, takes_in, gives_out) in enumerate(parts):
        x = xs[i]
        colors = THEMES[theme]
        d.rect(x, top, card_w, card_h, fill="#FFFFFF", stroke=colors["stroke"], radius=16, stroke_width=2)
        d.badge(x + 6, top + 6, letter, theme)
        d.icon(icon, x + card_w / 2, top + 62, theme, radius=32)
        d.text(x + card_w / 2, top + 124, title, size=17.5, weight="bold", anchor="middle")
        d.paragraph(x + card_w / 2, top + 148, description, 38, size=13, line_height=18, anchor="middle", max_lines=3)

        # what goes in and what comes out
        d.rect(x + 14, top + card_h - 56, card_w - 28, 1.5, fill=PAGE_BORDER, radius=0)
        d.text(x + 18, top + card_h - 33, "IN", size=11, weight="bold", fill=colors["dark"])
        d.text(x + 48, top + card_h - 33, takes_in, size=12.5)
        d.text(x + 18, top + card_h - 13, "OUT", size=11, weight="bold", fill=colors["dark"])
        d.text(x + 48, top + card_h - 13, gives_out, size=12.5)

    middle = top + card_h / 2
    d.arrow([(xs[0] + card_w + 6, middle), (xs[1] - 6, middle)], size=1.5)
    d.arrow([(xs[1] + card_w + 6, middle), (xs[2] - 6, middle)], size=1.5)

    d.text(width / 2, analogy_y,
           "Like a library: first catalogue every book (A), then shelve them by topic (B), then ask the librarian a question (C).",
           size=13.5, fill=TEXT_COLOR, anchor="middle")

    # what comes next
    d.rect(24, next_y, width - 48, 56, fill="#FFFFFF", stroke=THEMES["planned"]["stroke"], radius=14, dashed=True, stroke_width=2)
    d.icon("target", 24 + 38, next_y + 28, "planned", radius=18)
    d.text(24 + 72, next_y + 25, "Next (planned)", size=13.5, weight="bold", fill=THEMES["planned"]["dark"])
    d.text(24 + 72, next_y + 44,
           "Test the search with real questions, and improve it only where the test shows it falls short.", size=12.5, fill=MUTED_COLOR)
    return d


# =========================================================================================
# 2. Part A: from a lecture video to knowledge objects (pipeline stages 1 to 8)
# =========================================================================================
def draw_phase1():
    d = Drawing("phase1", 780, 1104,
                "Part A: a lecture video is split into what was said and what was shown, and the two are matched slide by slide.")

    card_w = 300
    card_h = 112
    step = 134
    left_x = 40
    right_x = 440
    lane_top = 176

    # the title of this part, in the top left corner
    d.badge(40, 46, "A", "phase1")
    d.text(64, 52, "Understand", size=15, weight="bold", fill=THEMES["phase1"]["dark"])
    d.text(64, 72, "the lecture", size=15, weight="bold", fill=THEMES["phase1"]["dark"])
    d.text(64, 94, "pipeline stages 1 to 8", size=12, fill=MUTED_COLOR)

    # the video at the top
    d.card(240, 22, card_w, 100, "phase1", "video", "Lecture video",
           "A recording of one lecture, stored as a file on your computer.", "data/raw/<lecture>/")

    # two lanes
    first_y = lane_top + 40
    right_last_bottom = first_y + 3 * step + card_h
    d.panel(24, lane_top, 332, first_y + step + card_h + 18 - lane_top, "phase1", "WHAT WAS SAID")
    d.panel(424, lane_top, 332, right_last_bottom + 18 - lane_top, "phase1", "WHAT WAS SHOWN")

    # what was said: two steps
    said_y = [first_y, first_y + step]
    d.card(left_x, said_y[0], card_w, card_h, "phase1", "sound", "Extract the audio",
           "Pull the sound track out of the video.", "FFmpeg", number=1)
    d.card(left_x, said_y[1], card_w, card_h, "phase1", "text", "Write down the speech",
           "Turn the speech into text, with the start and end time of every sentence.", "Faster-Whisper on the GPU", number=2)

    # what was shown: four steps
    shown_y = []
    for i in range(4):
        shown_y.append(first_y + step * i)
    d.card(right_x, shown_y[0], card_w, card_h, "phase1", "frames", "Find the slide changes",
           "Save a picture each time the screen changes enough.", "OpenCV frame comparison", number=3)
    d.card(right_x, shown_y[1], card_w, card_h, "phase1", "ocr", "Read the slide text (OCR)",
           "Read the words on every picture.", "Tesseract", number=4)
    d.card(right_x, shown_y[2], card_w, card_h, "phase1", "eye", "Describe each slide",
           "Each picture goes to Gemini (a cloud service), which gives the title and describes diagrams.", "Gemini (cloud API)", number=5)
    d.card(right_x, shown_y[3], card_w, card_h, "phase1", "clean", "Clean up the text",
           "Remove words that repeat on every slide (menus) and characters OCR was unsure about.", "repeated-word filter", number=6)

    # the video splits into the two lanes
    d.arrow([(390, 122), (390, 150), (190, 150), (190, first_y - 6)])
    d.arrow([(390, 150), (590, 150), (590, first_y - 6)])
    d.arrow([(190, said_y[0] + card_h + 6), (190, said_y[1] - 6)])
    for i in range(3):
        d.arrow([(590, shown_y[i] + card_h + 6), (590, shown_y[i + 1] - 6)])

    # the two lanes meet
    join_y = right_last_bottom + 62
    d.card(140, join_y, 500, 112, "phase1", "link", "Match speech to slides",
           "Every piece of speech is filed under the slide that was on screen when it started.", "alignment.json", number=7)
    d.arrow([(190, said_y[1] + card_h + 6), (190, join_y - 34), (270, join_y - 34), (270, join_y - 6)])
    d.arrow([(590, shown_y[3] + card_h + 6), (590, join_y - 6)])

    # the result
    out_y = join_y + 112 + 34
    d.card(140, out_y, 500, 112, "phase1", "records", "Knowledge objects: one record per slide",
           "Each record holds the slide text, a description of any diagram, what was said while it was shown, and the exact times.",
           "knowledge_objects.json", number=8)
    d.arrow([(390, join_y + 112 + 6), (390, out_y - 6)])

    d.text(390, out_y + 112 + 38, "The small text under each step names the tool or the file behind it.", size=12, fill=MUTED_COLOR, anchor="middle")
    return d


# =========================================================================================
# 3. Parts B and C: getting ready to search, and answering a question
# =========================================================================================
def draw_phase2():
    card_w = 288
    card_h = 112
    step = 130
    left_x = 36
    gap = 90
    db_x = left_x + card_w + gap
    db_w = 150
    right_x = db_x + db_w + gap

    first_y = 108
    row_y = []
    for i in range(5):
        row_y.append(first_y + step * i)

    # the left side: part B on top, the planned work below it
    b_panel_y = 44
    b_panel_h = row_y[2] + card_h + 18 - b_panel_y
    planned_panel_y = b_panel_y + b_panel_h + 22
    planned_card_h = 96
    planned_first_y = planned_panel_y + 44
    planned_second_y = planned_first_y + planned_card_h + 16
    planned_panel_h = planned_second_y + planned_card_h + 18 - planned_panel_y

    c_panel_y = 44
    c_panel_h = row_y[4] + card_h + 18 - c_panel_y

    bottom = max(planned_panel_y + planned_panel_h, c_panel_y + c_panel_h)
    height = bottom + 52
    width = right_x + card_w + 36

    d = Drawing("phase2", width, height,
                "Parts B and C: lectures are cut into chunks and stored by meaning once; each question is matched to the closest chunks and answered with sources.")

    d.panel(20, b_panel_y, card_w + 32, b_panel_h, "index", "B   GET READY TO SEARCH", "once per lecture, stages 9 and 10")
    d.panel(right_x - 16, c_panel_y, card_w + 32, c_panel_h, "answer", "C   ANSWER A QUESTION", "every time someone asks")
    d.panel(20, planned_panel_y, card_w + 32, planned_panel_h, "planned", "NEXT (planned, not built yet)", dashed=True)

    # part B
    d.card(left_x, row_y[0], card_w, card_h, "phase1", "records", "Knowledge objects",
           "Everything part A produced for the lecture.", "knowledge_objects.json")
    d.card(left_x, row_y[1], card_w, card_h, "index", "chunks", "Cut into chunks",
           "About 350 words of speech each, together with the slides shown meanwhile.", "chunks.json", number=9)
    d.card(left_x, row_y[2], card_w, card_h, "index", "numbers", "Turn chunks into numbers",
           "Each chunk becomes 1,024 numbers that capture its meaning.", "bge-m3 on the GPU", number=10)
    for i in range(2):
        d.arrow([(left_x + card_w / 2, row_y[i] + card_h + 6), (left_x + card_w / 2, row_y[i + 1] - 6)])

    # the planned work
    d.card(left_x, planned_first_y, card_w, planned_card_h, "planned", "target", "A fresh question set",
           "A clean test, before claiming a final number.", "next", dashed=True)
    d.card(left_x, planned_second_y, card_w, planned_card_h, "planned", "trend", "Compare embeddings",
           "Try the Gemini model against bge-m3.", "later", dashed=True)
    d.arrow([(left_x + card_w / 2, planned_first_y + planned_card_h + 4), (left_x + card_w / 2, planned_second_y - 4)], dashed=True)

    # part C
    d.card(right_x, row_y[0], card_w, card_h, "answer", "question", "Your question",
           "Typed on the command line.", 'python -m src.ask "..."', number="C1")
    d.card(right_x, row_y[1], card_w, card_h, "answer", "numbers", "Turn it into numbers",
           "With the same model that handled the chunks.", "bge-m3", number="C2")
    d.card(right_x, row_y[2], card_w, card_h, "answer", "search", "Find the closest chunks",
           "The 5 best matches, by meaning and by exact words.", "python -m src.search", number="C3")
    d.card(right_x, row_y[3], card_w, card_h, "answer", "write", "Write the answer",
           "Gemini answers from those 5 chunks only, and cites them.", "Gemini (cloud API)", number="C4")
    d.card(right_x, row_y[4], card_w, card_h, "answer", "check", "Check and show sources",
           "Our code drops invented citations, then lists each source: lecture, time, slides.", "no model involved", number="C5")
    for i in range(4):
        d.arrow([(right_x + card_w / 2, row_y[i] + card_h + 6), (right_x + card_w / 2, row_y[i + 1] - 6)])

    # the database in the middle, level with the two steps that use it
    db_height = 200
    middle = row_y[2] + card_h / 2
    db_y = middle - db_height / 2
    d.tall_card(db_x, db_y, db_w, db_height, "index", "database", "Database",
                "Keeps each chunk's numbers, times and text, searchable by meaning.", "data/qdrant/")
    d.arrow([(left_x + card_w + 6, middle), (db_x - 6, middle)], label_lines=["stores"], label_at=((left_x + card_w + db_x) / 2, middle - 12), size=1.3)
    d.arrow([(right_x - 6, middle - 20), (db_x + db_w + 6, middle - 20)], label_lines=["question", "as numbers"], label_at=((db_x + db_w + right_x) / 2, middle - 56), size=1.3)
    d.arrow([(db_x + db_w + 6, middle + 20), (right_x - 6, middle + 20)], label_lines=["5 closest", "chunks"], label_at=((db_x + db_w + right_x) / 2, middle + 46), size=1.3)

    d.text(width / 2, height - 22, "The small text under each step names the tool or the file behind it.", size=12, fill=MUTED_COLOR, anchor="middle")
    return d


# =========================================================================================
# 4. Part C followed step by step with one example question
# =========================================================================================
def draw_one_question():
    x = 40
    w = 720
    inner_x = x + 74
    centre_x = x + 38

    # the heights of the five steps and the space between them
    heights = [92, 134, 244, 156, 184]
    gap = 30
    ys = []
    y = 24
    for height in heights:
        ys.append(y)
        y += height + gap
    total_height = y - gap + 24

    d = Drawing("one-question", 800, total_height,
                "Part C with one example question: it becomes numbers, the closest chunks are found, Gemini answers from them, and the sources are looked up.")

    # --- step C1: the question ---
    d.wide_frame(x, ys[0], w, heights[0], "answer", "question", "You ask", "C1")
    d.rect(inner_x, ys[0] + 44, 420, 34, fill=THEMES["answer"]["fill"], radius=17)
    d.text(inner_x + 18, ys[0] + 66, "What is a confusion matrix?", size=15)

    # --- step C2: it becomes numbers ---
    y2 = ys[1]
    d.wide_frame(x, y2, w, heights[1], "answer", "numbers", "It becomes a list of numbers", "C2")
    d.text(inner_x, y2 + 54, "The same model that turned the chunks into numbers does this for the question.", size=12.5, fill=MUTED_COLOR)
    d.text(inner_x, y2 + 84, "[ 0.12   -0.40   0.07   0.31   ...   -0.05 ]", size=13.5, font=MONO)
    d.text(inner_x, y2 + 108, "1,024 numbers in total: a numeric summary of what the question means", size=12, fill=MUTED_COLOR)
    # a small picture of the numbers as bars above and below a line
    bars = [10, -6, 4, 14, -12, 8, -3, 11, -9, 5, 13, -7, 2, -10, 9, 6, -4, 12, -8, 3]
    baseline = y2 + 96
    bar_x = x + 470
    for height in bars:
        if height >= 0:
            d.rect(bar_x, baseline - height * 1.5, 8, height * 1.5, fill=THEMES["answer"]["stroke"], radius=2)
        else:
            d.rect(bar_x, baseline, 8, -height * 1.5, fill=THEMES["answer"]["stroke"], radius=2, opacity=0.55)
        bar_x += 12
    d.rect(x + 466, baseline - 0.5, 252, 1, fill=MUTED_COLOR, radius=0)

    # --- step C3: the database returns the closest chunks ---
    y3 = ys[2]
    d.wide_frame(x, y3, w, heights[2], "index", "database", "The database returns the 5 closest chunks", "C3")
    d.paragraph(inner_x, y3 + 54, "Similar meaning gives similar numbers, so the closest numbers point to the most relevant parts of the lecture.",
                98, size=12.5, line_height=17, max_lines=2)
    d.text(inner_x + 36, y3 + 106, "closeness (1.0 = same meaning)", size=11, weight="bold", fill=THEMES["index"]["dark"])
    d.text(inner_x + 270, y3 + 106, "lecture", size=11, weight="bold", fill=THEMES["index"]["dark"])
    d.text(inner_x + 400, y3 + 106, "time range", size=11, weight="bold", fill=THEMES["index"]["dark"])
    rows = [("[1]", 0.613, "lecture_01", "12:31 - 15:06"), ("[2]", 0.599, "lecture_01", "14:58 - 18:03")]
    row_y = y3 + 130
    for label, score, lecture, times in rows:
        d.text(inner_x, row_y, label, size=12.5, weight="bold", font=MONO, fill=THEMES["index"]["dark"])
        # a bar as long as the score (1.0 would fill the whole bar), then the number itself
        d.rect(inner_x + 36, row_y - 12, 160, 14, fill="#EEF1F5", radius=7)
        d.rect(inner_x + 36, row_y - 12, 160 * score, 14, fill=THEMES["index"]["stroke"], radius=7)
        d.text(inner_x + 206, row_y, f"{score:.3f}", size=12.5, font=MONO)
        d.text(inner_x + 270, row_y, lecture, size=12.5, font=MONO)
        d.text(inner_x + 400, row_y, times, size=12.5, font=MONO)
        row_y += 28
    d.text(inner_x, row_y + 2, "[3] to [5]: three more chunks", size=12.5, fill=MUTED_COLOR)
    d.text(inner_x, row_y + 28, "Each hit comes with its lecture, times and slide titles, because they are stored next to its numbers.",
           size=12, fill=MUTED_COLOR)
    d.text(inner_x, row_y + 46, "Neighbouring chunks overlap a little, on purpose.", size=12, fill=MUTED_COLOR)

    # --- step C4: the answer ---
    y4 = ys[3]
    d.wide_frame(x, y4, w, heights[3], "answer", "write", "Gemini writes the answer from those 5 chunks only", "C4")
    d.rect(inner_x, y4 + 44, 620, 50, fill=THEMES["answer"]["fill"], radius=12)
    d.text(inner_x + 16, y4 + 66, "A confusion matrix counts the four ways a classifier can be right or wrong. [1][2]", size=13)
    d.text(inner_x + 16, y4 + 84, "(example wording)", size=11, fill=MUTED_COLOR, italic=True)
    d.text(inner_x, y4 + 118, "It may only cite the chunk numbers [1] to [5].", size=12.5, fill=MUTED_COLOR)
    d.text(inner_x, y4 + 138, "It never writes a lecture name or a time.", size=12.5, weight="bold", fill=THEMES["answer"]["dark"])

    # --- step C5: the sources, which is what you finally see ---
    y5 = ys[4]
    d.wide_frame(x, y5, w, heights[4], "answer", "check", "Our code checks the citations and builds the sources", "C5", fill=THEMES["answer"]["fill"])
    d.text(inner_x, y5 + 60, "[1]  lecture_01  12:31 - 15:06  Classification metrics | Confusion matrix", size=12, font=MONO)
    d.text(inner_x, y5 + 80, "[2]  lecture_01  14:58 - 18:03  Confusion matrix", size=12, font=MONO)
    d.paragraph(inner_x, y5 + 110, "Example values. Looked up in the stored chunks, not written by the model. Each source also points to the first slide picture of that stretch. "
                "A citation to a chunk that does not exist, such as [9], is removed and a warning is shown.",
                92, size=12.5, line_height=17, max_lines=3, fill=TEXT_COLOR)

    # arrows between the steps
    for i in range(4):
        d.arrow([(centre_x, ys[i] + heights[i] + 4), (centre_x, ys[i + 1] - 4)])
    return d


def main():
    drawings = [draw_overview(), draw_phase1(), draw_phase2(), draw_one_question()]
    problems = []

    for drawing in drawings:
        drawing.save()
        problems.extend(drawing.problems)

    if len(problems) == 0:
        print("no text problems found")
    else:
        print("check these texts:")
        for problem in problems:
            print("  ", problem)


if __name__ == "__main__":
    main()
