#!/usr/bin/env bash
# =============================================================================
# build.sh -- the ONLY supported way to build the paper.
#
# Builds BOTH PDFs from the single source main.tex, then proves that the only
# differences between them are the ones we intend.
#
# WHY THIS EXISTS
#   main.pdf and main_named.pdf once drifted apart: main_named.pdf was produced
#   by a manual `cp main.pdf main_named.pdf` and was never rebuilt after a
#   correction to Section 6, so it kept superseded numbers for several commits.
#   Nothing in the build caught it. This script makes that failure impossible:
#   both PDFs are always built from the same source in one invocation, and the
#   text is diffed afterwards.
#
# WHITELIST -- the only permitted differences between the two builds
#   1. Author block.  Anonymous build: "Anonymous Author(s) / Affiliation /
#      Address / email". Named build: the placeholder author lines and the
#      \thanks footnote attached to them.
#   2. Notice / footer string. The two builds carry different venue notices.
#   3. Line-number gutter. The anonymous build enables lineno for reviewers;
#      the named build does not. Rather than strip the gutter out of extracted
#      text (unsafe -- it also eats figure axis labels), the diff uses a third,
#      throwaway anonymous build compiled with \DIFFBUILD, which suppresses the
#      gutter at the source. The shipped main.pdf keeps its line numbers.
#   Anything else differing is a bug and fails this script.
#
# EXIT CODES  0 ok | 1 latex error | 2 text diff outside whitelist
#             3 assertion failure (page budget or venue notice)
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; RST=$'\033[0m'
say() { printf '%s\n' "$*"; }
die() { printf '%s%s%s\n' "$RED" "$*" "$RST" >&2; exit "${2:-1}"; }

AUX="*.aux *.log *.out *.bbl *.blg *.toc *.fls *.fdb_latexmk"
clean() { rm -f $AUX a.raw b.raw a.norm b.norm 2>/dev/null; }

set_anon() {  # $1 = true|false
  if [ "$1" = "true" ]; then
    sed -i '' 's/^\\newif\\ifanon      \\anonfalse$/\\newif\\ifanon      \\anontrue/' main.tex
  else
    sed -i '' 's/^\\newif\\ifanon      \\anontrue$/\\newif\\ifanon      \\anonfalse/' main.tex
  fi
}

latex_run() {  # $1 = label
  rm -f $AUX
  pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null 2>&1 \
    || { pdflatex -interaction=nonstopmode main.tex 2>&1 | grep -E '^!' | head -5; \
         die "[$1] pdflatex failed" 1; }
  bibtex main >/dev/null 2>&1
  pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
  pdflatex -interaction=nonstopmode main.tex >build_$1.log 2>&1
}

check_log() {  # $1 = label
  local l="build_$1.log" e u o
  e=$(grep -cE '^! ' "$l" || true)
  u=$(grep -cE 'LaTeX Warning: (Citation|Reference).*undefined' "$l" || true)
  o=$(grep -oE 'Overfull .hbox \([0-9.]+pt' "$l" | sed 's/[^0-9.]//g' | awk '$1>10' | wc -l | tr -d ' ')
  say "  [$1] errors=$e undefined=$u overfull>10pt=$o pages=$(pdfinfo main.pdf | awk '/^Pages/{print $2}')"
  [ "$e" -eq 0 ] || die "[$1] LaTeX errors" 1
  [ "$u" -eq 0 ] || die "[$1] undefined references or citations" 1
  [ "$o" -eq 0 ] || die "[$1] overfull boxes above 10pt" 1
}

# ---------------------------------------------------------------- build both
say "${YEL}== building anonymized ==${RST}"
clean; set_anon true;  latex_run anon;  check_log anon
cp main.pdf main_anon_tmp.pdf

# Text for the diff comes from a THIRD build: anonymous, but with line numbers
# suppressed via \DIFFBUILD. Stripping a lineno gutter out of pdftotext output
# by pattern is not safe -- it also eats figure axis tick labels, and a running
# counter desynchronises on the first page. Not generating the gutter at all is
# exact and needs no heuristic. This build is thrown away.
say "${YEL}== building anonymized (line-number-free, for diffing only) ==${RST}"
rm -f $AUX
pdflatex -interaction=nonstopmode "\def\DIFFBUILD{}\input{main.tex}" >/dev/null 2>&1
bibtex main >/dev/null 2>&1
pdflatex -interaction=nonstopmode "\def\DIFFBUILD{}\input{main.tex}" >/dev/null 2>&1
pdflatex -interaction=nonstopmode "\def\DIFFBUILD{}\input{main.tex}" >/dev/null 2>&1
pdftotext -layout main.pdf a.raw

say "${YEL}== building named ==${RST}"
set_anon false; latex_run named; check_log named
cp main.pdf main_named.pdf
pdftotext -layout main.pdf b.raw

# leave the tree in its canonical state: main.tex anonymous, main.pdf anonymous
set_anon true
mv main_anon_tmp.pdf main.pdf

# ------------------------------------------------------------ normalise text
# No gutter stripping is needed: the diff text comes from a line-number-free
# build. Only the whitelist is removed, then whitespace is squeezed.
WHITELIST_RE='Anonymous Author\(s\)|^ *Affiliation$|^ *Address$|^ *email$|AUTHOR LIST TO BE CONFIRMED|Affiliation to be confirmed|email@example\.invalid|Placeholder\. No real names|committed to this source file|Submitted to .*Do not distribute|Preprint|Conference on Neural Information Processing'

for f in a b; do
  grep -avE "$WHITELIST_RE" $f.raw \
    | sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//' | grep -av '^$' > $f.norm
done

say "${YEL}== diffing the two builds outside the whitelist ==${RST}"
if diff -q a.norm b.norm >/dev/null; then
  say "  ${GRN}identical${RST} ($(wc -l < a.norm | tr -d ' ') lines compared)"
else
  say "  ${RED}DIFFERENCES FOUND${RST} (anon '<' vs named '>'):"
  diff a.norm b.norm | head -40
  clean; die "the two builds differ outside the whitelist" 2
fi

# ------------------------------------------------------- main-text page budget
# The venue limit applies to the MAIN TEXT, not the total page count: references,
# appendices and the checklist do not count. Main text is therefore everything
# before the References heading, i.e. (page on which References starts) - 1.
say "${YEL}== main-text page budget ==${RST}"
MAIN_LIMIT=8
refpage=$(python3 - <<'PYEOF'
import subprocess, sys
n = int(subprocess.run(["pdfinfo", "main.pdf"], capture_output=True, text=True)
        .stdout.split("Pages:")[1].split()[0])
for pg in range(1, n + 1):
    t = subprocess.run(["pdftotext", "-f", str(pg), "-l", str(pg), "main.pdf", "-"],
                       capture_output=True, text=True).stdout
    # The bibliography page: has the heading and is short (body pages are dense).
    if "References" in t and len(t) < 5000:
        print(pg); sys.exit(0)
print(0)
PYEOF
)
if [ "$refpage" -eq 0 ]; then
  die "could not locate the References page to measure the main text" 3
fi
maintext=$((refpage - 1))
total=$(pdfinfo main.pdf | awk '/^Pages/{print $2}')
say "  main text: ${maintext} pages (References starts on page ${refpage}); total incl. appendices and checklist: ${total}"
if [ "$maintext" -gt "$MAIN_LIMIT" ]; then
  die "main text is ${maintext} pages, over the ${MAIN_LIMIT}-page limit" 3
fi
say "  ${GRN}within the ${MAIN_LIMIT}-page main-text limit${RST}"

# ------------------------------------------------------ footer/notice assertions
# The style's default submission notice claims a NeurIPS 2026 submission. That is
# false for this paper. main.tex overrides \@noticestring; these assertions make
# sure the override is actually in effect in the shipped PDFs.
say "${YEL}== asserting the venue notice ==${RST}"
DEFAULT_NOTICE='Conference on Neural Information Processing Systems'
VENUE=$(sed -nE 's/^\\newcommand\{\\venuename\}\{(.*)\}$/\1/p' main.tex | head -1)
VENUE_PLAIN=$(printf '%s' "$VENUE" | sed 's/---/—/g')
for pdf in main.pdf main_named.pdf; do
  txt=$(pdftotext "$pdf" - 2>/dev/null)
  if printf '%s' "$txt" | grep -qF "$DEFAULT_NOTICE"; then
    die "[$pdf] contains the style's default NeurIPS notice; the \\@noticestring override is not in effect" 3
  fi
  if ! printf '%s' "$txt" | grep -qF "$VENUE_PLAIN"; then
    die "[$pdf] does not contain the current \\venuename value ($VENUE_PLAIN)" 3
  fi
  say "  [$pdf] notice ok: no default NeurIPS string; venue present"
done

clean
say "${GRN}== OK ==${RST}  main.pdf (anonymized) and main_named.pdf are in sync."
