#!/usr/bin/env python3

import os
import re
from io import BytesIO
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
from pypdf import PdfReader, PdfWriter
from pypdf.errors import PdfReadError
from pypdf.generic import ContentStream, TextStringObject, NameObject

LINE_NUM_RE = re.compile(r"^\d{3,4}$")
AUTHOR_STRING = "AnonymousACLsubmission"


def clean_pdf(input_pdf: Path) -> Path:
    output_pdf = input_pdf.with_name("paper_clean.pdf")
    if output_pdf.exists():
        return output_pdf  # skip

    line_accu = 0

    def is_line_number(operands):
        nonlocal line_accu

        if not operands or not operands[0]:
            return False

        text = operands[0][0]
        if not isinstance(text, TextStringObject):
            return False

        s = str(text)
        if not LINE_NUM_RE.match(s):
            return False

        n = int(s)
        if n == 0 and line_accu == 0:
            line_accu = 0
            return True

        if n == line_accu + 1:
            line_accu = n
            return True

        return False

    def is_author_info(operands):
        if len(operands) != 1:
            return False
        return "".join(
            str(o) for o in operands[0] if isinstance(o, TextStringObject)
        ) == AUTHOR_STRING

    # robust reader
    try:
        reader = PdfReader(str(input_pdf))
    except PdfReadError as e:
        if "PDF starts with" not in str(e):
            raise
        with input_pdf.open("rb") as f:
            data = f.read(2048)
            start = data.find(b"%PDF-")
            if start == -1:
                raise
            f.seek(start)
            reader = PdfReader(BytesIO(f.read()))

    writer = PdfWriter()

    for page in reader.pages:
        contents = page.get("/Contents")
        if contents is None:
            writer.add_page(page)
            continue

        content = ContentStream(contents, reader)
        delete_indices = []

        for i, (operands, operator) in enumerate(content.operations):
            if operator == b"TJ":
                if is_line_number(operands) or is_author_info(operands):
                    delete_indices.append(i)

        for i in reversed(delete_indices):
            del content.operations[i]

        page[NameObject("/Contents")] = content
        writer.add_page(page)

    writer.add_metadata({"/Author": "Anonymous"})

    with output_pdf.open("wb") as f:
        writer.write(f)

    return output_pdf


def clean_many_pdfs(pdfs, max_workers=None):
    if max_workers is None:
        max_workers = os.cpu_count() or 4

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(clean_pdf, pdf): pdf for pdf in pdfs}

        with tqdm(total=len(futures), desc="Cleaning PDFs") as pbar:
            for future in as_completed(futures):
                pdf = futures[future]
                try:
                    future.result()
                except Exception as e:
                    tqdm.write(f"\n✗ Failed: {pdf}\n  {e}")
                finally:
                    pbar.update(1)


def main(data_path: Path = None):
    if not data_path.exists():
        raise FileNotFoundError(f"Path not found: {data_path}")

    pdfs = list(data_path.rglob("v1/paper.pdf"))
    if not pdfs:
        tqdm.write("No v1/paper.pdf files found.")
        return

    tqdm.write(f"Found {len(pdfs)} PDFs")
    clean_many_pdfs(pdfs, max_workers=6)


if __name__ == "__main__":
    main(Path("data/papers"))
