"""build_corpus.py (owner: P2) - the documented dataset with source provenance.
Downloads each official page once, extracts clean text, splits it into small chunks,
and saves every chunk WITH its source URL, section and date (that is what makes answers verifiable)."""
import datetime, json, pathlib, re
import csv
import requests
from normalize import normalize_text
from bs4 import BeautifulSoup
from pypdf import PdfReader

SOURCES = [  # (url or local file, title). Official sources only.
    ("https://dgip.gov.pk/passport/ordinary-passport.php", "DGIP - Requirements, fees, renewal, lost passport"),
    ("https://dgip.gov.pk/passport/process.php", "DGIP - Passport application process"),
    ("https://dgip.gov.pk/passport/types.php", "DGIP - Passport types"),
    ("https://dgip.gov.pk/updates/passport-fast-track.php", "DGIP - Fast Track service"),
    ("https://dgip.gov.pk/faq/faq.php", "DGIP - Frequently asked questions"),
    ("https://dgip.gov.pk/eServices/online-passport.php", "DGIP - Online passport"),
    ("https://dgip.gov.pk/eServices/epassport.php", "DGIP - e-Passport"),
    ("https://dgip.gov.pk/downloads/PassportRules-2021.pdf", "Passport Rules 2021"),
]
RAW, OUT = pathlib.Path("data/raw"), pathlib.Path("data/chunks.jsonl")


def html_blocks(html):
    """Yield (heading, text) in reading order. Table rows become 'Header: value | Header: value'."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "form"]):
        tag.decompose()                                    # menus and scripts are noise
    heading, header, prev, table = "", [], [], None
    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr"]):
        if el.name != "tr" and el.find_parent("tr"):
            continue                                       # table text is handled row by row
        if el.name[0] == "h":
            heading = el.get_text(" ", strip=True)
            continue
        if el.name == "tr":
            if el.find_parent("table") is not table:       # new table: forget old header
                table, header, prev = el.find_parent("table"), [], []
            cells = [c.get_text(" ", strip=True) for c in el.find_all(["td", "th"])]
            if el.find("th"):
                header = cells
                continue
            if header and len(cells) < len(header):        # merged (rowspan) cells: copy from row above
                cells = prev[: len(header) - len(cells)] + cells
            prev = cells
            text = " | ".join(f"{h}: {c}" if h else c for h, c in zip(header or [""] * len(cells), cells) if c) + "."
        else:
            if el.name == "li" and el.a and el.a.get_text(strip=True) == el.get_text(strip=True):
                continue                                   # a list item that is only a link = menu
            text = el.get_text(" ", strip=True)
        if len(text.split()) >= 3:
            yield heading, text


def pdf_blocks(path):
    for n, page in enumerate(PdfReader(path).pages, 1):
        text = re.sub(r"\s+", " ", page.extract_text() or "")
        for sentence in re.split(r"(?<=[.;:])\s+", text):
            if len(sentence.split()) >= 3:
                yield f"Page {n}", sentence


def chunk(blocks, max_words=120):
    """Group consecutive blocks with the same heading into chunks of about 120 words."""
    buf, head = [], None
    for h, text in blocks:
        size = sum(len(b.split()) for b in buf)
        if buf and (h != head or size + len(text.split()) > max_words):
            yield head, " ".join(buf)
            buf = []
        head = h
        buf.append(text)
    if buf:
        yield head, " ".join(buf)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    n, manifest = 0, []
    with open(OUT, "w", encoding="utf-8") as out:
        for d, (src, title) in enumerate(SOURCES, 1):
            if src.startswith("http"):
                local = RAW / re.sub(r"[^\w.]+", "_", src.split("//")[1])
                if not local.exists():                     # download once, then work offline
                    r = requests.get(src, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
                    r.raise_for_status()
                    local.write_bytes(r.content)
            else:
                local = pathlib.Path(src)
            date = datetime.date.fromtimestamp(local.stat().st_mtime).isoformat()
            is_pdf = local.suffix.lower() == ".pdf"
            blocks = pdf_blocks(local) if is_pdf else html_blocks(local.read_text("utf-8", errors="ignore"))
            doc_id = f"D{d:02d}"
            kept = 0
            for p, (head, text) in enumerate(chunk(blocks)):
                if len(text.split()) < 8:                       # heading-only chunks ruin search
                    continue
                rec = {"id": n, "doc_id": doc_id, "passage": f"{doc_id}-P{p:03d}", "title": title,
                       "section": normalize_text(head or title), "url": src, "retrieved": date,
                       "text": normalize_text(f"{head}. {text}" if head else text)}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                kept += 1
            manifest.append({"doc_id": doc_id, "title": title, "url": src, "retrieved": date, "chunks": kept})
            print(f"{doc_id} {title}: {kept} chunks")
    with open("data/sources.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["doc_id", "title", "url", "retrieved", "chunks"])
        w.writeheader(); w.writerows(manifest)
    print(n, "chunks saved to", OUT, "| source list saved to data/sources.csv")


if __name__ == "__main__":
    main()
