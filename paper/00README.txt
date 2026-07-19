00README for arXiv submission (MegaSlide-DiT)
============================================

Processor: pdflatex (or tectonic locally)
TeX Live: 2023 or 2025

Main file:
  megaslide_dit_paper.tex

Bibliography:
  Inline (\\begin{thebibliography} ... \\end{thebibliography}).
  No .bib / BibTeX run required.

Figures (PNG for pdflatex, as referenced by the .tex):
  figures/fig1_memory_scaling.png
  figures/fig2_speedup_scaling.png
  figures/fig3_quality_ablation.png
  figures/fig4_long_training.png
  figures/fig5_utilization.png   (optional if unused)
  figures/fig6_fwd_bwd_breakdown.png (optional if unused)

Do NOT upload:
  *.aux *.log *.out *.blg
  megaslide_dit_paper.md
  .playwright-cli/

Suggested arXiv metadata
------------------------
Title: MegaSlide-DiT: Training Video Diffusion Transformers Larger Than GPU Memory via CPU-Master Streaming
Authors: Jason Liu
Primary category: cs.LG
Cross-lists: cs.CV, cs.DC
Comments: Preprint. Code and H100 NVL measurement suite accompany the paper.
License: choose at submit time (CC-BY-4.0 recommended if code is Apache-2.0)

Build locally
-------------
  cd paper
  bash pack_arxiv.sh
  # or: tectonic megaslide_dit_paper.tex
