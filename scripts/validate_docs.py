"""Automated quality gate for DotShot Game v2 handbook, LaTeX report, BibTeX, assets, and Humanizer prose."""

import pathlib
import re
import sys
from typing import List, Set, Tuple


ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
PAPER_TEX = DOCS_DIR / "paper" / "dotshot_marl_report.tex"
BIB_PATH = DOCS_DIR / "references.bib"
README_PATH = ROOT / "README.md"

BANNED_CLICHES = [
    r"\bdelve\b",
    r"\btestament\b",
    r"\btapestry\b",
    r"\bseamlessly\b",
    r"\bseamless\b",
    r"\bpivotal\b",
    r"\bgame-changer\b",
    r"\bholistic\b",
    r"\bunderscore\b",
    r"\bfoster\b",
    r"\bshowcase\b",
    r"\brobust landscape\b",
    r"\bparadigm shift\b",
    r"\bleverage\b",
    r"\bit is important to note\b",
    r"\bin conclusion\b",
]

ITALIAN_LEAKAGE = [
    r"\binoltre\b",
    r"\bquindi\b",
    r"\bperché\b",
    r"\btuttavia\b",
    r"\binsieme\b",
    r"\bprogetto\b",
    r"\bcartella\b",
    r"\bagenti\b",
    r"\bdocumentazione\b",
    r"\bsimulatore\b",
]


def load_bib_keys(bib_path: pathlib.Path) -> Set[str]:
    if not bib_path.exists():
        raise FileNotFoundError(f"Missing bibliography file: {bib_path}")
    text = bib_path.read_text(encoding="utf-8")
    return set(re.findall(r"@\w+\s*\{\s*([a-zA-Z0-9_:-]+)\s*,", text))


def extract_md_citations(md_text: str) -> List[str]:
    cleaned = re.sub(r"```[\s\S]*?```", "", md_text)
    cleaned = re.sub(r"`[^`\n]*`", "", cleaned)
    found: List[str] = []
    for bracket in re.findall(r"\[([^\]]*@[^\]]+)\]", cleaned):
        for key in re.findall(r"@([a-zA-Z0-9_:-]+)", bracket):
            found.append(key)
    return found


def extract_tex_citations(tex_text: str) -> List[str]:
    found: List[str] = []
    for group in re.findall(r"\\cite[pt]?\*?(?:\[[^\]]*\])*\{([^}]+)\}", tex_text):
        for key in group.split(","):
            k = key.strip()
            if k:
                found.append(k)
    return found


def validate_latex_file(tex_path: pathlib.Path, bib_keys: Set[str]) -> Tuple[List[str], int]:
    errors: List[str] = []
    raw = tex_path.read_text(encoding="utf-8")
    rel_name = str(tex_path.relative_to(ROOT))

    # Strip LaTeX comments for syntax/prose checks
    no_comments = "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in raw.splitlines())

    keys = extract_tex_citations(no_comments)
    for k in keys:
        if k not in bib_keys:
            errors.append(f"[{rel_name}] Undefined LaTeX BibTeX citation: {k}")

    # Check \includegraphics targets
    for img_ref in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", no_comments):
        resolved = (tex_path.parent / img_ref.strip()).resolve()
        if not resolved.exists():
            errors.append(f"[{rel_name}] Missing \\includegraphics file: {img_ref}")

    # Check balanced \begin{...} / \end{...}
    begins = re.findall(r"\\begin\{([^}]+)\}", no_comments)
    ends = re.findall(r"\\end\{([^}]+)\}", no_comments)
    if len(begins) != len(ends):
        errors.append(f"[{rel_name}] Unbalanced \\begin ({len(begins)}) vs \\end ({len(ends)}) environments.")
    else:
        stack: List[str] = []
        for token in re.findall(r"\\(begin|end)\{([^}]+)\}", no_comments):
            kind, env = token
            if kind == "begin":
                stack.append(env)
            else:
                if not stack or stack[-1] != env:
                    errors.append(f"[{rel_name}] Mismatched \\end{{{env}}} environment.")
                    break
                stack.pop()

    lower_txt = no_comments.lower()
    for pat in BANNED_CLICHES + ITALIAN_LEAKAGE:
        if re.search(pat, lower_txt):
            errors.append(f"[{rel_name}] Prose/language lint hit: {pat}")

    return errors, len(keys)


def validate_docs() -> Tuple[List[str], int]:
    errors: List[str] = []
    bib_keys = load_bib_keys(BIB_PATH)
    md_files = sorted(DOCS_DIR.glob("*.md"))
    if README_PATH.exists():
        md_files.append(README_PATH)

    if not md_files:
        errors.append("No Markdown files found under docs/.")
        return errors, 0

    total_citations = 0
    for md_path in md_files:
        raw = md_path.read_text(encoding="utf-8")
        rel_name = str(md_path.relative_to(ROOT))

        keys = extract_md_citations(raw)
        total_citations += len(keys)
        for k in keys:
            if k not in bib_keys:
                errors.append(f"[{rel_name}] Undefined BibTeX citation key: @{k}")

        no_code = re.sub(r"```[\s\S]*?```", "", raw)
        for _label, target in re.findall(r"!?\[([^\]]*)\]\(([^)]+)\)", no_code):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            clean_target = target.split("#")[0].strip()
            if not clean_target:
                continue
            resolved = (md_path.parent / clean_target).resolve()
            if not resolved.exists():
                errors.append(f"[{rel_name}] Broken relative link/asset: {target}")

        if md_path.name == "_sidebar.md":
            continue
        lower_no_code = no_code.lower()
        for pat in BANNED_CLICHES + ITALIAN_LEAKAGE:
            if re.search(pat, lower_no_code):
                errors.append(f"[{rel_name}] Prose/language lint hit: {pat}")

    if PAPER_TEX.exists():
        tex_errors, tex_cites = validate_latex_file(PAPER_TEX, bib_keys)
        errors.extend(tex_errors)
        total_citations += tex_cites

    return errors, total_citations


def main() -> None:
    errors, total_cites = validate_docs()
    bib_count = len(load_bib_keys(BIB_PATH))
    if errors:
        print("DOCUMENTATION QUALITY GATE FAILED:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    tex_status = " + LaTeX report verified" if PAPER_TEX.exists() else ""
    print(
        f"DOCUMENTATION QUALITY GATE PASSED: "
        f"{bib_count} BibTeX entries, {total_cites} verified inline citations, 0 broken links, 0 prose violations{tex_status}."
    )


if __name__ == "__main__":
    main()
