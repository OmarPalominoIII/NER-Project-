import fitz
from pathlib import Path

def extract_text_from_pdf(pdf_path: str | Path) -> str:
    doc = fitz.open(pdf_path)
    full_text = []

    for page in doc:
        full_text.append(page.get_text())
    doc.close()
    return "\n".join(full_text)

def extract_pdf_dataset(pdf_dir: str | Path, gold_samples: list[dict]) -> list[dict]:
    pdf_dir_path = Path(pdf_dir)
    pdf_samples = []

    for sample in gold_samples:
        doc_id = sample.get("id")
        pdf_path = pdf_dir_path / f"{doc_id}.pdf"

        if pdf_path.exists():
            text = extract_text_from_pdf(pdf_path)
        else:
            text = sample.get("text", "")

        pdf_record = {
            "id": doc_id,
            "text": text,
            "labels": sample.get("labels", [])
        }
        pdf_samples.append(pdf_record)
    return pdf_samples