#!/usr/bin/env python3
"""Extract result tables from LaTeX files for verification."""

from pathlib import Path
from typing import Dict, List, Optional


def extract_tables_from_tex(file_path: Path) -> List[Dict]:
    """Extract tables from a LaTeX file.

    Returns a list of dictionaries containing:
    - caption: The table caption
    - category: The category (lit, code, data, discovery, overall)
    - rows: List of data rows (excluding headers)
    - line_start: Starting line number in the file
    - line_end: Ending line number in the file
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    tables = []
    i = 0

    while i < len(lines):
        # Look for table environment
        if "\\begin{table}" in lines[i]:
            table_start = i + 1  # Line numbers are 1-indexed
            table_data = {
                "line_start": table_start,
                "caption": "",
                "category": None,
                "rows": [],
            }

            # Find the caption and extract category
            while i < len(lines) and "\\end{table}" not in lines[i]:
                if "\\caption{" in lines[i]:
                    # Extract caption (may span multiple lines)
                    caption_text = ""
                    j = i
                    brace_count = 0
                    in_caption = False

                    while j < len(lines):
                        line = lines[j]
                        if "\\caption{" in line:
                            in_caption = True
                            # Start after \caption{
                            line = line[line.index("\\caption{") + 9 :]
                            brace_count = 1

                        if in_caption:
                            for char in line:
                                if char == "{":
                                    brace_count += 1
                                elif char == "}":
                                    brace_count -= 1
                                    if brace_count == 0:
                                        break
                                caption_text += char

                            if brace_count == 0:
                                break
                        j += 1

                    table_data["caption"] = caption_text.strip()

                    # Determine category from caption or content
                    caption_lower = caption_text.lower()
                    if "overall" in caption_lower:
                        table_data["category"] = "overall"
                    elif "\\catlit" in caption_text or "literature" in caption_lower:
                        table_data["category"] = "lit"
                    elif (
                        "\\catcoding" in caption_text
                        or "code" in caption_lower
                        or "coding" in caption_lower
                    ):
                        table_data["category"] = "code"
                    elif (
                        "\\catdata" in caption_text or "data analysis" in caption_lower
                    ):
                        table_data["category"] = "data"
                    elif (
                        "\\catendtoend" in caption_text
                        or "end-to-end" in caption_lower
                        or "discovery" in caption_lower
                    ):
                        table_data["category"] = "discovery"

                    # Also check for specific dataset names
                    if (
                        "paperfindingbench" in caption_lower
                        or "paper finding" in caption_lower
                    ):
                        table_data["category"] = "lit_search"
                    elif "scholarqa" in caption_lower or "litqa" in caption_lower:
                        table_data["category"] = "lit_qa"
                    elif (
                        "arxivdigestables" in caption_lower
                        or "table" in caption_lower
                        and "literature" in caption_lower
                    ):
                        table_data["category"] = "lit_table"

                # Extract data rows (lines containing & and \\ but not \toprule, \midrule, \bottomrule, \cmidrule)
                if "&" in lines[i] and "\\\\" in lines[i]:
                    line = lines[i].strip()
                    # Skip header rows and rule lines
                    if not any(
                        rule in line
                        for rule in [
                            "\\toprule",
                            "\\midrule",
                            "\\bottomrule",
                            "\\cmidrule",
                            "\\addlinespace",
                        ]
                    ):
                        # Check if it's a header row (contains 'Score', 'Cost', 'Agent', 'Model', etc.)
                        if not any(
                            header in line
                            for header in [
                                "Score",
                                "Cost",
                                "Agent",
                                "Model",
                                " O ",
                                " T ",
                            ]
                        ):
                            # Check if line is commented out
                            if not line.startswith("%"):
                                table_data["rows"].append(line)

                i += 1

            # Found end of table
            if i < len(lines):
                table_data["line_end"] = i + 1  # Line numbers are 1-indexed
                if table_data["rows"]:  # Only add tables with data rows
                    tables.append(table_data)

        i += 1

    return tables


def parse_table_row(row: str) -> Dict:
    """Parse a LaTeX table row into its components.

    Returns a dictionary with fields like:
    - openness: The openness symbol
    - tooling: The tooling symbol
    - agent: The agent name/macro
    - model: The model name/macro
    - scores: List of score values
    - costs: List of cost values
    """
    # Remove trailing \\
    row = row.replace("\\\\", "").strip()

    # Split by &
    cells = [cell.strip() for cell in row.split("&")]

    result = {
        "openness": cells[0] if len(cells) > 0 else None,
        "tooling": cells[1] if len(cells) > 1 else None,
        "agent": cells[2] if len(cells) > 2 else None,
        "model": cells[3] if len(cells) > 3 else None,
        "values": [],
    }

    # Remaining cells are score/cost pairs
    if len(cells) > 4:
        result["values"] = cells[4:]

    return result


def identify_table_type(table: Dict) -> Optional[str]:
    """Identify the specific type of table based on caption and content.

    Returns one of:
    - 'overall': Overall results table
    - 'lit_search': Literature search (PaperFindingBench, LitQA2-Search)
    - 'lit_qa': Literature QA (ScholarQA, LitQA2)
    - 'lit_table': Literature tables (ArxivDIGESTables)
    - 'code': Coding tasks (SUPER, CORE-Bench, DS-1000)
    - 'data': Data analysis (DiscoveryBench)
    - 'discovery': End-to-end discovery (E2E-Bench)
    """
    caption = table.get("caption", "").lower()

    # Check for specific dataset mentions
    if (
        "paperfindingbench" in caption
        or "paper finding" in caption
        or "literature search" in caption
    ):
        return "lit_search"
    elif "scholarqa" in caption or "literature qa" in caption:
        return "lit_qa"
    elif "arxivdigestables" in caption or (
        "table" in caption and "synthesis" in caption
    ):
        return "lit_table"
    elif (
        "super" in caption
        or "core-bench" in caption
        or "ds-1000" in caption
        or "coding" in caption
    ):
        return "code"
    elif "discoverybench" in caption or "data analysis" in caption:
        return "data"
    elif "e2e-bench" in caption or "end-to-end discovery" in caption:
        return "discovery"
    elif "overall" in caption:
        return "overall"

    # Fall back to category if set
    return table.get("category")


def main():
    """Extract and display tables from paper tex files."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract tables from LaTeX files")
    parser.add_argument(
        "tex_file", type=str, help="Path to the LaTeX file to extract tables from"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed output including all rows"
    )
    args = parser.parse_args()

    tex_path = Path(args.tex_file)
    if not tex_path.exists():
        print(f"Error: File {tex_path} does not exist")
        return

    tables = extract_tables_from_tex(tex_path)

    print(f"Found {len(tables)} tables in {tex_path.name}\n")

    for i, table in enumerate(tables, 1):
        table_type = identify_table_type(table)
        print(f"Table {i}:")
        print(f"  Type: {table_type or 'unknown'}")
        print(f"  Lines: {table['line_start']}-{table['line_end']}")
        print(
            f"  Caption: {table['caption'][:100]}{'...' if len(table['caption']) > 100 else ''}"
        )
        print(f"  Data rows: {len(table['rows'])}")

        if args.verbose and table["rows"]:
            print("  Sample rows:")
            for j, row in enumerate(table["rows"][:3], 1):
                parsed = parse_table_row(row)
                agent = parsed.get("agent", "?")
                model = parsed.get("model", "?")
                print(f"    {j}. {agent} with {model}")
                if len(table["rows"]) > 3 and j == 3:
                    print(f"    ... and {len(table['rows']) - 3} more rows")

        print()


if __name__ == "__main__":
    main()
