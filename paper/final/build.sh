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
# EXIT CODES  0 ok | 1 latex error | 2 text diff outside whitelist | 3 assertion
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

clean
say "${GRN}== OK ==${RST}  main.pdf (anonymized) and main_named.pdf are in sync."
