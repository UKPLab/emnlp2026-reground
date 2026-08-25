#!/usr/bin/env python3
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from lxml import etree
from tqdm import tqdm

GROBID_URL = "http://localhost:8070/api/processFulltextDocument"
TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def norm(txt: str) -> str:
    return " ".join(txt.split()).strip()


def call_grobid(pdf: Path, timeout=300, retries=5) -> str:
    for attempt in range(1, retries + 1):
        with pdf.open("rb") as f:
            r = requests.post(
                GROBID_URL,
                files={"input": (pdf.name, f, "application/pdf")},
                timeout=timeout,
            )

        if r.status_code == 200:
            return r.text

        if r.status_code in (429, 502, 503, 504):
            time.sleep(2 * attempt)
            continue

        r.raise_for_status()

    raise RuntimeError(f"GROBID failed after {retries} retries for {pdf}")


def extract_paragraphs(tei_xml: str, pdf: Path) -> dict:
    root = etree.fromstring(tei_xml.encode("utf-8"))
    paras = {}

    for p in root.xpath(".//tei:text/tei:body//tei:p", namespaces=TEI_NS):
        text = norm(" ".join(p.xpath(".//text()")))
        if not text:
            continue

        head = p.xpath(
            "string((ancestor::tei:div[tei:head][1]/tei:head)[1])",
            namespaces=TEI_NS,
        )
        head = norm(head)

        paras.setdefault(pdf.stem, {}).setdefault(head, []).append(text)

    return paras


def process_one(pdf: Path, force=False) -> tuple[str, str]:
    out = pdf.with_name("paragraphs.json")

    if out.exists() and not force:
        return (str(pdf), "skipped")

    tei = call_grobid(pdf)
    paras = extract_paragraphs(tei, pdf)

    out.write_text(
        json.dumps(paras, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return (str(pdf), "written")


def main(root: str, workers: int, force: bool):

    root = Path(root)
    pdfs = sorted(root.rglob("v1/paper_clean.pdf"))

    if not pdfs:
        raise SystemExit("No v1/paper_clean.pdf files found")

    print(f"Found {len(pdfs)} PDFs")

    written = skipped = failed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process_one, p, force) for p in pdfs]

        for f in tqdm(
            as_completed(futures), total=len(futures), desc="Processing PDFs"
        ):
            try:
                _, status = f.result()
                if status == "written":
                    written += 1
                else:
                    skipped += 1
            except Exception as e:
                failed += 1
                tqdm.write(f"[FAIL] {e}")

    print(f"Done. written={written}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    path_base = "/data/papers"
    path_base_full = Path(path_base)
    print(f"Processing folder: {path_base_full}")
    main(str(path_base_full), workers=6, force=True)
    print(f"Finished folder: {path_base_full}")
    print("-" * 40)