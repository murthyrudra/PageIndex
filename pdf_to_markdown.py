import os
from pathlib import Path
from docling.document_converter import DocumentConverter


def convert_folder_to_markdown(input_folder, output_folder):
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    output_path.mkdir(parents=True, exist_ok=True)

    converter = DocumentConverter()

    pdf_files = list(input_path.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        return

    for pdf_file in pdf_files:
        try:
            print(f"Processing: {pdf_file.name}")

            result = converter.convert(str(pdf_file))

            markdown = result.document.export_to_markdown()

            output_file = output_path / (pdf_file.stem + ".md")

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(markdown)

            print(f"Saved: {output_file}")

        except Exception as e:
            print(f"❌ Failed on {pdf_file.name}: {e}")


if __name__ == "__main__":
    input_folder = "data/rag/"  # change this
    output_folder = "data_markdown/"  # change this

    convert_folder_to_markdown(input_folder, output_folder)
