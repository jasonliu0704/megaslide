#!/usr/bin/env bash
# Build an arXiv-ready source tarball for MegaSlide-DiT.
# Current paper uses inline thebibliography + PNG figures (no BibTeX required).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${ROOT}/arxiv_submission"
TAR="${ROOT}/megaslide_dit_arxiv_source.tar.gz"

rm -rf "${OUT}"
mkdir -p "${OUT}/figures"

cp "${ROOT}/megaslide_dit_paper.tex" "${OUT}/"
cp "${ROOT}/00README.txt" "${OUT}/"

# Paper currently includes .png via \includegraphics
for f in fig1_memory_scaling fig2_speedup_scaling fig3_quality_ablation \
         fig4_long_training fig5_utilization fig6_fwd_bwd_breakdown; do
  if [[ -f "${ROOT}/figures/${f}.png" ]]; then
    cp "${ROOT}/figures/${f}.png" "${OUT}/figures/"
  elif [[ -f "${ROOT}/figures/${f}.pdf" ]]; then
    cp "${ROOT}/figures/${f}.pdf" "${OUT}/figures/"
  else
    echo "ERROR: missing figure ${f}.{png,pdf}" >&2
    exit 1
  fi
done

# Verify local compile
if command -v tectonic >/dev/null 2>&1; then
  (cd "${OUT}" && tectonic megaslide_dit_paper.tex)
elif command -v pdflatex >/dev/null 2>&1; then
  (cd "${OUT}" && pdflatex -interaction=nonstopmode megaslide_dit_paper.tex \
    && pdflatex -interaction=nonstopmode megaslide_dit_paper.tex)
fi

# Strip build artifacts; keep .tex + figures
find "${OUT}" -type f \( -name '*.aux' -o -name '*.log' -o -name '*.out' \
  -o -name '*.blg' -o -name '*.pdf' \) ! -path '*/figures/*' -delete 2>/dev/null || true

tar -czf "${TAR}" -C "${OUT}" .
echo "Created ${TAR}"
ls -lh "${TAR}"
echo "Contents:"
tar -tzf "${TAR}"
