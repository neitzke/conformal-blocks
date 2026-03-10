# conformal-blocks

Compiled PDF is automatically placed at

https://storage.googleapis.com/public-export-bucket/conformal-blocks/conformal-blocks.pdf

## Environment setup

This project requires a TeX distribution with `pdflatex` and `biber`.

On Debian/Ubuntu, install the required tools with:

```bash
sudo apt-get update
sudo apt-get install -y texlive-latex-extra texlive-fonts-extra biber latexmk
```

## Verify `pdflatex` works

Run:

```bash
pdflatex --version
pdflatex -interaction=nonstopmode -halt-on-error conformal-blocks.tex
```

If compilation succeeds, `conformal-blocks.pdf` will be generated in the repository root.
