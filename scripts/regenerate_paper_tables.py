#!/usr/bin/env python3
"""Regenerate paper tables while preserving structure and agent ordering."""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from extract_paper_tables import parse_table_row

COMMENT_PREFIX = "%#gen:"


def extract_table_with_structure(lines: List[str], start_idx: int) -> Dict:
    """Extract a table preserving all its structure.

    Returns a dictionary with:
    - entries: List of all lines with their types
    - has_ci: Whether table has confidence intervals
    - category: Table category
    """
    result = {
        "entries": [],  # List of {"type": "raw"/"data_row", "original_line": ..., "agent": ..., "model": ...}
        "has_ci": False,
        "category": None,
        "start_line": start_idx,
        "end_line": None,
    }

    # First, find caption to determine category
    i = start_idx
    caption_text = ""
    while i < len(lines) and "\\end{table}" not in lines[i]:
        if "\\caption{" in lines[i]:
            # Extract caption (may span multiple lines)
            j = i
            brace_count = 0
            in_caption = False

            while j < len(lines):
                line = lines[j]
                if "\\caption{" in line:
                    in_caption = True
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

            # Determine category from caption
            caption_lower = caption_text.lower()
            if "overall" in caption_lower:
                result["category"] = "overall"
            elif (
                "paper finding" in caption_lower or "literature search" in caption_lower
            ):
                result["category"] = "lit_search"
            elif (
                "scholarqa" in caption_lower
                or "literature qa" in caption_lower
                or "litqa" in caption_lower
            ):
                result["category"] = "lit_qa"
            elif "arxivdigestables" in caption_lower or (
                "table" in caption_lower and "synthesis" in caption_lower
            ):
                result["category"] = "lit_table"
            elif (
                "super" in caption_lower
                or "core-bench" in caption_lower
                or "ds-1000" in caption_lower
                or "coding" in caption_lower
            ):
                result["category"] = "code"
            elif "discoverybench" in caption_lower or "data analysis" in caption_lower:
                result["category"] = "data"
            elif "\\catendtoend" in caption_lower or "end-to-end" in caption_lower:
                result["category"] = "discovery"
            elif "\\catlit" in caption_text and "\\evaltables" in caption_text:
                result["category"] = "lit_table"
            elif "\\catlit" in caption_text:
                # Need more specific detection
                if "search" in caption_text.lower():
                    result["category"] = "lit_search"
                else:
                    result["category"] = "lit"
        i += 1

    # Reset to start and process table
    i = start_idx
    while i < len(lines) and "\\end{table}" not in lines[i]:
        line = lines[i]

        # Check if this is a data row by looking for agent/model macros
        if (
            ("\\agent" in line or "\\model" in line or "\\missing" in line)
            and "&" in line
            and "\\\\" in line
            and not line.strip().startswith("%")
        ):
            # This is a data row
            parsed = parse_table_row(line.strip())
            entry = {
                "type": "data_row",
                "original_line": line,
                "agent": parsed["agent"],
                "model": parsed["model"],
            }
            result["entries"].append(entry)

            # Check for confidence intervals
            if " +- " in line:
                result["has_ci"] = True
        else:
            # Not a data row - just keep it as raw
            entry = {"type": "raw", "original_line": line}
            result["entries"].append(entry)

        i += 1

    # Add the \end{table} line
    if i < len(lines):
        result["entries"].append({"type": "raw", "original_line": lines[i]})
        result["end_line"] = i

    return result


def generate_table_rows_from_jsonl(
    category: str, agents_models: List[Tuple[str, str]], has_ci: bool
) -> Dict[Tuple[str, str], str]:
    """Generate table rows for specific agent/model combinations.

    Returns a dictionary mapping (agent_macro, model_macro) to LaTeX row string.
    """
    # Map category to table type for the generation script
    type_mapping = {
        "overall": "overall",
        "lit_search": "lit_search",
        "lit_qa": "lit_qa",
        "lit_table": "lit_table",
        "code": "code",
        "data": "data",
        "discovery": "discovery",
        "lit": "lit_search",  # Default for generic lit
    }

    if category not in type_mapping:
        return {}

    script_arg = type_mapping[category]

    # Run the generation script
    cmd = [
        "python",
        "scripts/generate_latex_from_jsonl.py",
        # "--no-submission-id",
        "--table",
        script_arg,
    ]

    if has_ci:
        cmd.append("--include-ci")
    else:
        cmd.append("--no-ci")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split("\n")

        # Parse generated rows into a dictionary
        rows_dict = {}
        for line in lines:
            line = line.strip()
            if line and not line.startswith("%") and not line.startswith("==="):
                if "&" in line and "\\\\" in line:
                    # Parse the row to get agent and model
                    parsed = parse_table_row(line)
                    if parsed["agent"] and parsed["model"]:
                        key = (parsed["agent"], parsed["model"])
                        rows_dict[key] = (
                            line + "\n"
                        )  # Keep newline for proper formatting

        return rows_dict
    except subprocess.CalledProcessError as e:
        print(f"Error generating table rows: {e}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        return {}


def parse_values_from_row(row: str, include_metadata_columns: bool = False) -> List:
    """Parse values from a LaTeX table row.

    Returns list of tuples (value, precision) where precision is detected from decimal places.
    If include_metadata_columns is True, also includes openness and tooling columns.
    """
    values = []
    # Split by & to get columns
    parts = row.split("&")

    # Determine which columns to include
    parts = parts if include_metadata_columns else parts[3:]

    # Process requested columns
    for i, part in enumerate(parts):
        # Clean up the part - handle \\ terminator and comments
        part = part.partition("\\\\")[0].strip()

        # For openness/tooling columns (first 2), just keep as string
        if include_metadata_columns and i < 2:
            values.append((part, None))
        # Check for special non-numeric values
        elif part in ["{--}", "{\\red{?}}", "\\missing", "{\red{?}}"]:
            values.append((part, None))  # Keep as string to detect differences
        else:
            # Try to extract number with optional +- CI
            import re

            # Match number with optional +- and CI
            match = re.match(
                r"(\\textbf\{|\\bf(series)? |\\B )?(?P<num1>[\d.]+)(?:\s*\+-\s*(?P<num2>[\d.]+)[*}]?)?",
                part,
            )
            if not match:
                # If we can't parse it as a number, keep as string
                values.append((part, None))
                continue

            for number_str in [match.group("num1"), match.group("num2")]:
                if number_str is None:
                    continue
                number_val = float(number_str)

                # Calculate decimal places from the string representation
                if "." in number_str:
                    precision = len(number_str.split(".")[1])
                else:
                    precision = 0
                values.append((number_val, precision))
    return values


def values_match(
    old_values: List,
    new_values: List,
    decimal_places: int = 1,
    include_metadata: bool = False,
) -> Tuple[bool, List[str]]:
    """Check if two sets of values match when rounded to the same decimal places.

    Handles mixed types: floats for numbers, strings for special values.
    If decimal_places is -1, auto-detect precision from old_values (expects tuples).
    Returns (True if all match, list of differences).
    """
    differences = []

    if len(old_values) != len(new_values):
        differences.append(
            f"Different number of values: {len(old_values)} vs {len(new_values)}"
        )
        return False, differences

    # Column names for better reporting
    # Note: Confidence intervals create separate values, so Score becomes Score and Score CI
    column_names = [
        "Score",
        "Score CI",
        "Cost",
        "Cost CI",
    ]
    if include_metadata:
        column_names = ["Openness", "Tooling", "Agent", "Model"] + column_names

    for i, (old_val, new_val) in enumerate(zip(old_values, new_values)):
        col_name = column_names[i] if i < len(column_names) else f"Column {i+1}"

        # Extract precision info if using tuples
        old_precision = None
        if isinstance(old_val, tuple):
            old_val, old_precision = old_val
        if isinstance(new_val, tuple):
            new_val, new_precision = new_val

        # Handle string values (special LaTeX markers)
        if isinstance(old_val, str) or isinstance(new_val, str):
            if old_val != new_val:
                differences.append(f"{col_name}: {old_val} vs {new_val}")
        # Handle numeric values
        elif isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
            # Determine precision to use
            if decimal_places == -1:
                use_precision = min(old_precision, new_precision)
            else:
                use_precision = decimal_places

            # Round both values to the determined precision and compare
            old_rounded = round(old_val, use_precision)
            new_rounded = round(new_val, use_precision)
            if old_rounded != new_rounded and abs(new_val - old_val) > 0.002:
                # Format with the same decimal places used for comparison
                format_str = f"{{:.{use_precision}f}}"
                old_str = format_str.format(old_val)
                new_str = format_str.format(new_val)
                diff_str = format_str.format(new_val - old_val)
                differences.append(
                    f"{col_name}: {old_str} vs {new_str} (diff: +{diff_str})"
                    if new_val > old_val
                    else f"{col_name}: {old_str} vs {new_str} (diff: {diff_str})"
                )
        # Handle type mismatches
        elif type(old_val) != type(new_val):
            differences.append(
                f"{col_name}: {old_val} ({type(old_val).__name__}) vs {new_val} ({type(new_val).__name__})"
            )

    return len(differences) == 0, differences


def is_known_confirmed_change(entry: dict, new_row: str, diffs: list[str]) -> bool:
    if entry["agent"] == "\\agentAsta":
        return True

    if entry["agent"] == "\\agentFutureHouseFalcon":
        return True

    if entry["agent"] == "\\agentFutureHouseCrow":
        return True

    if (
        entry["agent"] == "\\agentScholarQANoTables"
        and entry["model"] == "\\modelGeminiTwoPointFiveFlashUnpinned"
    ):
        return True

    if (
        ("YouCom" in entry["agent"] or "Perplexity" in entry["agent"])
        and "Openness: \\opennessSymbolClosed vs \\opennessSymbolClosedWithApi" in diffs
    ):
        return True

    if (
        entry["agent"] == "\\agentScholarQANoTables"
        and "Cost CI: 0.039 vs 0.030 (diff: -0.009)" in diffs
    ):
        return True

    if (
        entry["agent"] == "\\agentOpenScholar"
        and "Score CI: 2.5 vs 2.6 (diff: +0.1)" in diffs
    ):
        return True

    if (
        entry["agent"] == "\\agentSTORM"
        and "Model: \\modelGPTFourOShort, \\modelGPTThreeFiveTurboShort vs \\modelGPTThreeFiveTurboShort, \\modelGPTFourOShort"
        in diffs
    ):
        return True

    if entry["agent"] == "\\agentAstaTableAgent":
        if diffs in [
            ["Cost: 0.16 vs 0.17 (diff: +0.01)"],
            ["Cost CI: 0.02 vs 0.01 (diff: -0.01)"],
            ["Model: \\modelGPTFiveShort vs \\modelGPTFiveUnpinned"],
            ["Model: \\modelGPTFiveMiniShort vs \\modelGPTFiveMiniUnpinned"],
        ]:
            return True

    if (
        entry["agent"] == "\\agentDataVoyager"
        and diffs
        and diffs[0] == "Tooling: \\toolingSymbolStandard vs \\toolingSymbolEquivalent"
    ):
        m = re.match(r"Model: (.*) vs (.*), \\modelGPTFourOUnpinned", diffs[1])
        if m and m.group(1) == m.group(2):
            return True

    if (
        entry["agent"] == "\\agentFaker"
        and entry["model"] == "\\modelGPTFourPointOneUnpinned"
    ):
        return True

    if entry["agent"] == "\\agentAutoAsta" and entry["model"] in (
        "\\modelClaudeSonnetFourShort",
        "\\modelGPTFourPointOneUnpinned",
    ):
        return True

    if (
        entry["agent"] == "\\agentCodeScientist"
        and entry["model"] == "\\modelClaudeSonnetThreeSevenShort"
    ):
        return True

    return False


def regenerate_table(
    table_structure: Dict,
    check_values: bool = False,
    decimal_places: int = 1,
    ignore_new_entries: bool = False,
    check_metadata: bool = True,
) -> List[str]:
    """Regenerate a table with updated data rows while preserving structure.

    Returns the complete regenerated table as a list of lines.
    """
    # Extract agent/model pairs from data rows
    agents_models = []
    for entry in table_structure["entries"]:
        if entry["type"] == "data_row":
            agents_models.append((entry["agent"], entry["model"]))

    # Generate all possible rows for this table type
    all_rows = generate_table_rows_from_jsonl(
        table_structure["category"], agents_models, table_structure["has_ci"]
    )

    if not all_rows:
        print(
            f"Warning: Could not generate rows for category {table_structure['category']}",
            file=sys.stderr,
        )
        return None

    # Build the new table by going through each entry
    new_lines = []
    missing_combos = []
    new_entries_added = set()
    values_different = []

    for entry in table_structure["entries"]:
        if entry["type"] == "raw":
            # Skip previously generated comments to avoid duplication
            if entry["original_line"].strip().startswith(COMMENT_PREFIX):
                continue
            # Keep other raw lines as-is
            new_lines.append(entry["original_line"])
        elif entry["type"] == "data_row":
            # Replace data rows with regenerated ones
            key = (entry["agent"], entry["model"])
            if key in all_rows:
                new_row = all_rows[key]

                # If check_values is enabled, compare the actual values
                if check_values:
                    # Check both metadata and data columns
                    old_all_values = parse_values_from_row(
                        entry["original_line"], include_metadata_columns=check_metadata
                    )
                    new_all_values = parse_values_from_row(
                        new_row, include_metadata_columns=check_metadata
                    )
                    match, diffs = values_match(
                        old_all_values,
                        new_all_values,
                        decimal_places,
                        include_metadata=check_metadata,
                    )

                    if is_known_confirmed_change(entry, new_row, diffs):
                        new_lines.append(new_row)
                        continue

                    if not match:
                        values_different.append((entry["agent"], entry["model"], diffs))
                        new_lines.append(
                            f"{COMMENT_PREFIX} VALUES DIFFER: {entry['agent']} with {entry['model']}\n"
                        )
                        for diff in diffs:
                            new_lines.append(f"{COMMENT_PREFIX}   {diff}\n")
                        # Add new row as comment for user to review
                        new_lines.append(f"{COMMENT_PREFIX} {new_row}")
                        # Keep original row below
                        new_lines.append(entry["original_line"])
                    else:
                        new_lines.append(new_row)
                        # new_lines.append(entry["original_line"] + (new_row[new_row.find("\\\\") + 2 :] if COMMENT_PREFIX not in entry['original_line'] else ""))
                        # new_lines.append(entry["original_line"])
                new_entries_added.add(key)
            else:
                # Try to find a close match
                found = False
                for (gen_agent, gen_model), row in all_rows.items():
                    # Check agent match (exact or substring)
                    agent_match = (
                        entry["agent"] == gen_agent
                        or entry["agent"] in gen_agent
                        or gen_agent in entry["agent"]
                    )

                    # Check model match - handle special cases
                    model_match = False

                    # Handle special cases like \missing or models in parentheses
                    entry_model_clean = entry["model"].strip()
                    if entry_model_clean.startswith("(") and entry_model_clean.endswith(
                        ")"
                    ):
                        # Remove parentheses for matching
                        entry_model_clean = entry_model_clean[1:-1]

                    # For missing model placeholder
                    if entry_model_clean == "\\missing" and gen_model == "\\missing":
                        model_match = True
                    elif entry_model_clean == "\\missing" or gen_model == "\\missing":
                        # Other \missing cases won't match
                        continue
                    elif "," in entry_model_clean and "," in gen_model:
                        # Both have multiple models - check if any models overlap
                        entry_models = [m.strip() for m in entry_model_clean.split(",")]
                        gen_models = [m.strip() for m in gen_model.split(",")]
                        n_matched_models = 0

                        # Check if there's substantial overlap (at least one model in common)
                        for em in entry_models:
                            em_base = (
                                em.replace("Short", "")
                                .replace("Unpinned", "")
                                .replace("$^\\dagger$", "")
                                .strip()
                            )
                            for gm in gen_models:
                                gm_base = (
                                    gm.replace("Short", "")
                                    .replace("Unpinned", "")
                                    .replace("$^\\dagger$", "")
                                    .strip()
                                )
                                if (
                                    em_base == gm_base
                                    or em_base in gm_base
                                    or gm_base in em_base
                                ):
                                    n_matched_models += 1
                                    break
                        if n_matched_models == len(gen_models):
                            model_match = True
                    elif entry_model_clean == gen_model:
                        model_match = True
                    elif (
                        entry_model_clean in gen_model or gen_model in entry_model_clean
                    ):
                        model_match = True
                    else:
                        # Handle "Short" suffix differences (e.g., modelGeminiTwoPointFiveFlashUnpinned vs modelGeminiTwoPointFiveFlashShort)
                        entry_model_base = (
                            entry_model_clean.replace("Short", "")
                            .replace("Unpinned", "")
                            .replace("$^\\dagger$", "")
                            .strip()
                        )
                        gen_model_base = (
                            gen_model.replace("Short", "")
                            .replace("Unpinned", "")
                            .replace("$^\\dagger$", "")
                            .strip()
                        )
                        if entry_model_base == gen_model_base:
                            model_match = True

                    if agent_match and model_match:
                        new_row = row

                        # If check_values is enabled, compare the actual values
                        if check_values:
                            old_all_values = parse_values_from_row(
                                entry["original_line"],
                                include_metadata_columns=check_metadata,
                            )
                            new_all_values = parse_values_from_row(
                                new_row, include_metadata_columns=check_metadata
                            )
                            match, diffs = values_match(
                                old_all_values,
                                new_all_values,
                                decimal_places,
                                include_metadata=check_metadata,
                            )

                            if is_known_confirmed_change(entry, new_row, diffs):
                                new_lines.append(new_row)
                            elif not match:
                                values_different.append((gen_agent, gen_model, diffs))
                                new_lines.append(
                                    f"{COMMENT_PREFIX} VALUES DIFFER: {gen_agent} with {gen_model}\n"
                                )
                                for diff in diffs:
                                    new_lines.append(f"{COMMENT_PREFIX}   {diff}\n")
                                # Add new row as comment for user to review
                                new_lines.append(f"{COMMENT_PREFIX} {new_row}")
                                # Keep original row below
                                new_lines.append(entry["original_line"])
                            else:
                                new_lines.append(new_row)
                                # new_lines.append(entry["original_line"] + (new_row[new_row.find("\\\\") + 2 :] if COMMENT_PREFIX not in entry['original_line'] else ""))
                                # new_lines.append(entry["original_line"])
                        new_entries_added.add((gen_agent, gen_model))
                        found = True
                        break

                if not found:
                    # Keep the original row but comment about failure to regenerate
                    new_lines.append(
                        f"{COMMENT_PREFIX} Could not regenerate: {entry['original_line'].strip()}\n"
                    )
                    new_lines.append(entry["original_line"])
                    missing_combos.append((entry["agent"], entry["model"]))

    # Check for completely new agent/model combinations that weren't in the original
    all_new_entries = []
    for key, row in all_rows.items():
        if key not in new_entries_added:
            # This is a new entry not in the original table
            agent, model = key
            # Check if it's really new (not just a variant)
            is_variant = False
            for orig_agent, orig_model in agents_models:
                if (agent in orig_agent or orig_agent in agent) and (
                    model in orig_model or orig_model in model
                ):
                    is_variant = True
                    break

            if not is_variant:
                all_new_entries.append((key, row))

    # Add new entries at the end, commented
    if all_new_entries and not ignore_new_entries:
        # Find where to insert (before \bottomrule and \end{table})
        insert_idx = len(new_lines)
        for i in range(len(new_lines) - 1, -1, -1):
            if "\\bottomrule" in new_lines[i] or "\\end{table" in new_lines[i]:
                insert_idx = i
                break

        # Insert the new entries
        new_entries_lines = [f"{COMMENT_PREFIX} New entries not in original table:\n"]
        for (agent, model), row in all_new_entries:
            new_entries_lines.append(f"{COMMENT_PREFIX} {row}")

        new_lines = new_lines[:insert_idx] + new_entries_lines + new_lines[insert_idx:]

    if missing_combos:
        print(
            f"Warning: Could not regenerate {len(missing_combos)} row(s)",
            file=sys.stderr,
        )
        for agent, model in missing_combos[:5]:
            print(f"  - {agent} with {model}", file=sys.stderr)

    if values_different:
        print(
            f"\nFound {len(values_different)} rows with actual value differences:",
            file=sys.stderr,
        )
        for agent, model, diffs in values_different[:3]:
            print(f"\n  {agent} with {model}:", file=sys.stderr)
            for diff in diffs:
                print(f"    - {diff}", file=sys.stderr)

    return new_lines


def main():
    """Main function to regenerate tables in LaTeX files."""
    parser = argparse.ArgumentParser(
        description="Regenerate paper tables with updated data"
    )
    parser.add_argument("tex_file", type=str, help="Path to the LaTeX file to update")
    parser.add_argument("--output", type=str, help="Output file (default: {input}.new)")
    parser.add_argument(
        "--table", type=int, help="Only regenerate specific table number"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write output, just show what would be done",
    )
    parser.add_argument(
        "--ignore-new-entries",
        action="store_true",
        default=False,
        help="Don't add comments indicating rows that are missing from the paper tables",
    )
    parser.add_argument(
        "--no-check-values",
        dest="check_values",
        action="store_false",
        default=True,
        help="Disable value checking",
    )
    parser.add_argument(
        "--decimal-places",
        type=int,
        default=-1,
        help="Number of decimal places to use when checking values; use -1 (the default) for auto-detection based on original values",
    )
    args = parser.parse_args()

    tex_path = Path(args.tex_file)
    if not tex_path.exists():
        print(f"Error: File {tex_path} does not exist", file=sys.stderr)
        return 1

    # Read the file
    with open(tex_path, "r") as f:
        lines = f.readlines()

    # Find and process tables
    new_lines = []
    i = 0
    table_num = 0
    tables_regenerated = []

    while i < len(lines):
        if "\\begin{table}" in lines[i]:
            table_num += 1

            # Extract table with structure
            table_structure = extract_table_with_structure(lines, i)

            # Check if we should process this table
            if args.table and args.table != table_num:
                # Keep original
                new_lines.extend(lines[i : table_structure["end_line"] + 1])
            else:
                print(
                    f"Processing table {table_num} ({table_structure['category']})",
                    file=sys.stderr,
                )

                # Regenerate the table
                regenerated = regenerate_table(
                    table_structure,
                    check_values=args.check_values,
                    decimal_places=args.decimal_places,
                    ignore_new_entries=args.ignore_new_entries,
                    check_metadata=True,
                )

                if regenerated:
                    new_lines.extend(regenerated)
                    tables_regenerated.append(table_num)
                else:
                    # Keep original if regeneration failed
                    print(f"  Keeping original table {table_num}", file=sys.stderr)
                    new_lines.extend(lines[i : table_structure["end_line"] + 1])

            # Skip to after this table
            i = table_structure["end_line"] + 1
        else:
            new_lines.append(lines[i])
            i += 1

    # Write output (default to .new file)
    output_path = Path(args.output) if args.output else tex_path.with_suffix(".tex.new")

    if not args.dry_run:
        with open(output_path, "w") as f:
            f.writelines(new_lines)
        print(f"Updated {output_path}", file=sys.stderr)

        if tables_regenerated:
            print(
                f"Regenerated tables: {', '.join(map(str, tables_regenerated))}",
                file=sys.stderr,
            )

            # Show diff between original and new file
            subprocess.run(
                [
                    "git",
                    "diff",
                    "--no-index",
                    "--word-diff-regex=[\\d.]+|\\*|[^*,&+ -]+|\\+-|&",
                    "--color=always",
                    str(tex_path),
                    str(output_path),
                ]
            )
    else:
        print("Dry run - no files modified", file=sys.stderr)
        if tables_regenerated:
            print(
                f"Would regenerate tables: {', '.join(map(str, tables_regenerated))}",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
