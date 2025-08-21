#!/usr/bin/env python3
"""Verify paper tables against generated tables from JSONL data."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from extract_paper_tables import extract_tables_from_tex, parse_table_row, identify_table_type


def normalize_value(value: str) -> str:
    """Normalize a table value for comparison.
    
    Handles:
    - Removing extra whitespace
    - Normalizing red question marks
    - Normalizing dashes
    """
    value = value.strip()
    
    # Normalize different ways of representing missing values
    if value in ['{\\red{?}}', '\\red{?}', '?']:
        return 'MISSING_UNEXPECTED'
    elif value in ['{--}', '--', '—']:
        return 'MISSING_EXPECTED'
    
    # Normalize numerical values (remove trailing zeros after decimal)
    try:
        # Check if it's a number with +- for CI
        if ' +- ' in value:
            parts = value.split(' +- ')
            val = float(parts[0])
            ci = float(parts[1])
            # Format consistently
            if val > 10:  # Likely a score percentage
                return f"{val:.1f} +- {ci:.1f}"
            else:  # Likely a cost
                return f"{val:.2f} +- {ci:.2f}"
        else:
            # Single number
            val = float(value)
            if val > 10:  # Likely a score percentage
                return f"{val:.1f}"
            else:  # Likely a cost
                return f"{val:.2f}"
    except (ValueError, IndexError):
        # Not a number, return as-is
        return value


def generate_expected_table(table_type: str, include_ci: bool = False) -> List[str]:
    """Generate expected table rows using generate_latex_from_jsonl.py.
    
    Returns list of LaTeX table rows.
    """
    # Map table types to script arguments
    type_mapping = {
        'overall': 'overall',
        'lit_search': 'lit_search',
        'lit_qa': 'lit_qa',
        'lit_table': 'lit_table',
        'code': 'code',
        'data': 'data',
        'discovery': 'discovery',
    }
    
    if table_type not in type_mapping:
        return []
    
    script_arg = type_mapping[table_type]
    
    # Run the generation script
    cmd = [
        'python', 
        'scripts/generate_latex_from_jsonl.py',
        '--table', script_arg
    ]
    
    if not include_ci:
        cmd.append('--no-ci')
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        
        # Filter out comment lines and headers
        data_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('%') and not line.startswith('==='):
                # This should be a data row
                if '&' in line and '\\\\' in line:
                    data_lines.append(line)
        
        return data_lines
    except subprocess.CalledProcessError as e:
        print(f"Error generating expected table: {e}")
        print(f"stderr: {e.stderr}")
        return []


def compare_rows(paper_row: str, expected_row: str) -> Tuple[bool, List[str]]:
    """Compare a paper table row with expected row.
    
    Returns (matches, list_of_differences)
    """
    paper_parsed = parse_table_row(paper_row)
    expected_parsed = parse_table_row(expected_row)
    
    differences = []
    
    # Compare key fields
    if paper_parsed['openness'] != expected_parsed['openness']:
        differences.append(f"Openness: '{paper_parsed['openness']}' vs '{expected_parsed['openness']}'")
    
    if paper_parsed['tooling'] != expected_parsed['tooling']:
        differences.append(f"Tooling: '{paper_parsed['tooling']}' vs '{expected_parsed['tooling']}'")
    
    if paper_parsed['agent'] != expected_parsed['agent']:
        differences.append(f"Agent: '{paper_parsed['agent']}' vs '{expected_parsed['agent']}'")
    
    if paper_parsed['model'] != expected_parsed['model']:
        differences.append(f"Model: '{paper_parsed['model']}' vs '{expected_parsed['model']}'")
    
    # Compare values (scores and costs)
    paper_values = [normalize_value(v) for v in paper_parsed['values']]
    expected_values = [normalize_value(v) for v in expected_parsed['values']]
    
    if len(paper_values) != len(expected_values):
        differences.append(f"Value count: {len(paper_values)} vs {len(expected_values)}")
    else:
        for i, (pv, ev) in enumerate(zip(paper_values, expected_values)):
            if pv != ev:
                # Try to be more specific about which column
                col_type = "Score" if i % 2 == 0 else "Cost"
                col_num = i // 2 + 1
                differences.append(f"Column {i+5} ({col_type} {col_num}): '{pv}' vs '{ev}'")
    
    return len(differences) == 0, differences


def find_matching_row(paper_row: str, expected_rows: List[str]) -> Tuple[Optional[int], List[str]]:
    """Find the best matching row from expected rows.
    
    Returns (index_of_match, differences) or (None, differences) if no good match.
    """
    paper_parsed = parse_table_row(paper_row)
    
    best_match = None
    best_differences = []
    min_differences = float('inf')
    
    for i, expected_row in enumerate(expected_rows):
        matches, differences = compare_rows(paper_row, expected_row)
        
        if matches:
            return i, []
        
        # Check if this is a potential match based on agent/model
        expected_parsed = parse_table_row(expected_row)
        if (paper_parsed['agent'] == expected_parsed['agent'] and 
            paper_parsed['model'] == expected_parsed['model']):
            # This is likely the intended match
            if len(differences) < min_differences:
                min_differences = len(differences)
                best_match = i
                best_differences = differences
    
    return best_match, best_differences


def verify_table(table: Dict, expected_rows: List[str], verbose: bool = False) -> Tuple[int, int, List[str]]:
    """Verify a table against expected rows.
    
    Returns (num_matches, num_mismatches, list_of_issues)
    """
    issues = []
    matches = 0
    mismatches = 0
    
    # Track which expected rows have been matched
    matched_expected = set()
    
    for i, paper_row in enumerate(table['rows']):
        # Skip commented rows
        if paper_row.strip().startswith('%'):
            continue
        
        match_idx, differences = find_matching_row(paper_row, expected_rows)
        
        if match_idx is not None:
            if differences:
                mismatches += 1
                parsed = parse_table_row(paper_row)
                issues.append(f"Row {i+1} ({parsed['agent']} with {parsed['model']}):")
                for diff in differences:
                    issues.append(f"  - {diff}")
            else:
                matches += 1
            matched_expected.add(match_idx)
        else:
            mismatches += 1
            parsed = parse_table_row(paper_row)
            issues.append(f"Row {i+1} ({parsed['agent']} with {parsed['model']}): No matching row in expected data")
    
    # Check for expected rows that weren't in the paper
    for i, expected_row in enumerate(expected_rows):
        if i not in matched_expected:
            parsed = parse_table_row(expected_row)
            issues.append(f"Missing from paper: {parsed['agent']} with {parsed['model']}")
    
    return matches, mismatches, issues


def main():
    """Main verification function."""
    parser = argparse.ArgumentParser(description="Verify paper tables against JSONL data")
    parser.add_argument(
        "tex_file",
        type=str,
        help="Path to the LaTeX file to verify"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output for all issues"
    )
    parser.add_argument(
        "--table",
        type=int,
        help="Only verify specific table number"
    )
    args = parser.parse_args()
    
    tex_path = Path(args.tex_file)
    if not tex_path.exists():
        print(f"Error: File {tex_path} does not exist")
        return 1
    
    # Extract tables from paper
    tables = extract_tables_from_tex(tex_path)
    print(f"Found {len(tables)} tables in {tex_path.name}\n")
    
    total_matches = 0
    total_mismatches = 0
    tables_with_issues = []
    
    for i, table in enumerate(tables, 1):
        # Skip if specific table requested and this isn't it
        if args.table and args.table != i:
            continue
        
        table_type = identify_table_type(table)
        
        print(f"Table {i} ({table_type or 'unknown'}):")
        print(f"  Lines {table['line_start']}-{table['line_end']}")
        
        if not table_type or table_type == 'unknown':
            print("  ⚠️  Could not identify table type, skipping verification")
            print()
            continue
        
        # Skip literature table type that's too generic
        if table_type == 'lit' and 'search' not in table['caption'].lower() and 'qa' not in table['caption'].lower():
            # Try to be more specific
            caption_lower = table['caption'].lower()
            if 'digest' in caption_lower or 'table' in caption_lower:
                table_type = 'lit_table'
            else:
                print(f"  ⚠️  Generic literature table, cannot determine specific type")
                print()
                continue
        
        # Determine if table has confidence intervals
        has_ci = any(' +- ' in row for row in table['rows'])
        
        # Generate expected rows
        expected_rows = generate_expected_table(table_type, include_ci=has_ci)
        
        if not expected_rows:
            print(f"  ⚠️  Could not generate expected data")
            print()
            continue
        
        # Verify the table
        matches, mismatches, issues = verify_table(table, expected_rows, verbose=args.verbose)
        
        total_matches += matches
        total_mismatches += mismatches
        
        print(f"  ✓ {matches} rows match")
        if mismatches > 0:
            print(f"  ✗ {mismatches} rows have issues")
            tables_with_issues.append(i)
            
            if args.verbose or mismatches <= 5:
                for issue in issues[:10]:  # Show first 10 issues
                    print(f"    {issue}")
                if len(issues) > 10:
                    print(f"    ... and {len(issues) - 10} more issues")
        
        print()
    
    # Summary
    print("=" * 60)
    print(f"SUMMARY: {total_matches} matches, {total_mismatches} issues")
    
    if tables_with_issues:
        print(f"Tables with issues: {', '.join(map(str, tables_with_issues))}")
        return 1
    else:
        print("✅ All tables verified successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())