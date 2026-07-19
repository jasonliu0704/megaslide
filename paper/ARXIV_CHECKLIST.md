# MegaSlide-DiT — arXiv Submission Checklist

**Status:** Source compiles locally (tectonic). Package with `./pack_arxiv.sh`.

## Blockers before you click Submit

These are content/integrity issues, not LaTeX issues. Resolve them before posting.

| Priority | Issue | Action |
|----------|--------|--------|
| **P0** | H200 systems numbers (Table systems / MFU 61%, 3.1s) and VBench scores may be **unvalidated** per `PAPER_REVIEW_NOTES.md` | Confirm measurements were run on H200, or mark as estimated / move VBench to “reported under partner license” with clear caveats |
| **P0** | 105B weights are not releasable | Keep the licensing caveat; ensure code + smoke tests actually run |
| **P1** | Author email / affiliation confirmation | Confirm “Jason Liu / Trendinsight Lab / UC San Diego” is correct for public posting |
| **P1** | Code URL | Point comments/README to the public repo once it is live (do not leave a wrong MegaTrain link) |
| **P2** | Dual-submission policy | If also submitting to a venue, check that venue’s arXiv policy |

## What was fixed for arXiv readiness

- Restored authorship (was “Anonymous Authors”)
- Added **Related Work** + `references.bib` / BibTeX
- Added `\label`/`\ref` for tables, figures, and Eq.~(1)
- Included previously unused figures (utilization, fwd/bwd breakdown)
- Prefer **PDF** figures (pdflatex-compatible)
- Removed orphaned duplicate discussion paragraph before Limitations
- Clarified 3D-DSA implementation (PyTorch `grid_sample`, not custom CUDA/Triton)
- Abstract length OK (~1.5k chars; arXiv limit 1920)
- Local compile succeeds → `megaslide_dit_paper.pdf` (16 pages)

## Files to upload

Upload the tarball produced by `pack_arxiv.sh`, or these files:

```
megaslide_dit_paper.tex
megaslide_dit_paper.bbl   # REQUIRED — bibliography is \input'd from this file
00README.txt
figures/fig1_memory_scaling.pdf
figures/fig2_speedup_scaling.pdf
figures/fig3_quality_ablation.pdf
figures/fig4_long_training.pdf
figures/fig5_utilization.pdf
figures/fig6_fwd_bwd_breakdown.pdf
```

Optional: `references.bib` (only needed if you regenerate the `.bbl` locally).

Do **not** upload: `.md`, PNG duplicates, `.aux/.log`, or the compiled PDF as the only submission (arXiv wants source).

## Suggested metadata (submit form)

- **Title:** MegaSlide-DiT: Memory-Centric Adaptation and Deformable Local Attention for Efficient Video Diffusion
- **Authors:** Jason Liu
- **Primary:** `cs.CV`
- **Cross-list:** `cs.LG`, `cs.DC` (optional: `cs.SY`)
- **Comments:** e.g. `16 pages. Code and smoke-test configs released with the paper.`
- **License:** CC BY 4.0 (aligns well with Apache-2.0 code) or arXiv perpetual non-exclusive
- **Processor:** pdflatex / TeX Live 2025 (or 2023)

## Submit steps

1. Resolve P0/P1 items above.
2. `cd paper && bash pack_arxiv.sh`
3. Open https://arxiv.org/submit → new submission → Computer Science
4. Upload `megaslide_dit_arxiv_source.tar.gz`
5. Paste abstract from the `.tex` `\begin{abstract}` block
6. Preview the auto-built PDF carefully (figures, refs, equations)
7. Complete license + submit

## Integrity note (read this)

Internal notes (`PAPER_REVIEW_NOTES.md`) state that some H200/VBench numbers were not independently executed in the open environment. Posting fabricated or unverified empirical claims on arXiv is a serious integrity risk. Prefer:

1. Re-run and replace numbers with measured values, **or**
2. Clearly label partner-provided / unreproducible evaluation results and lead with the H100 NVL validation section (which has stronger experimental grounding).
