# System dependencies

PDF vector-text recovery and final OCR auditing use the local `tesseract`
binary with English language data. Font discovery expects at least one of the
Chinese fonts listed in `phoena_translator/pdf/fonts.py`.

Install Python packages reproducibly with:

```sh
python -m pip install -r requirements.txt -c constraints.txt
```

The application does not use EbookLib or ReportLab; both were removed from the
runtime dependency set during the Stage 5 audit.
