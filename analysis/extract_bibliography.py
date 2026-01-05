import requests
import time
import re

# Configuration
OUTPUT_FILE = r"G:\My Drive\LIAA\doctorado\papers\lista_de_papers_sorted.txt"
INPUT_FILE = r"G:\My Drive\LIAA\doctorado\papers\lista_de_papers.txt"

def extract_doi(line):
    """
    Attempts to find a DOI in the line using regex.
    """
    # Standard regex for finding DOIs (10.xxxx/xxxxx)
    doi_pattern = r'(10\.\d{4,9}/[-._;()/:A-Z0-9]+)'
    match = re.search(doi_pattern, line, re.IGNORECASE)
    if match:
        # Clean trailing punctuation often found in lists
        return match.group(1).rstrip(';,.)') 
    return None

def extract_arxiv_year(line):
    """
    Extracts year from ArXiv IDs (e.g., 2203.16502 -> 2022).
    """
    # Matches arXiv patterns like: arXiv:2203.1234 or /2506.1234
    arxiv_pattern = r'arXiv[:/.](\d{2})(\d{2})\.'
    match = re.search(arxiv_pattern, line, re.IGNORECASE)
    if match:
        year_short = int(match.group(1))
        # Assumption: 90-99 -> 1990-1999, 00-89 -> 2000-2089
        if year_short > 50: 
            return 1900 + year_short
        else:
            return 2000 + year_short
    return None

def get_year_from_universal_doi(doi):
    """
    Queries doi.org directly with Content Negotiation.
    This works for CrossRef, DataCite (Zenodo/Dryad), and mEDRA.
    """
    url = f"https://doi.org/{doi}"
    headers = {'Accept': 'application/vnd.citationstyles.csl+json'}
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            # Navigate standard CSL JSON date structure
            date_parts = None
            if 'issued' in data and 'date-parts' in data['issued']:
                date_parts = data['issued']['date-parts']
            elif 'published-online' in data and 'date-parts' in data['published-online']:
                date_parts = data['published-online']['date-parts']
                
            if date_parts and len(date_parts) > 0:
                return int(date_parts[0][0]) # First part is the year
    except Exception:
        pass
    return None

def get_year_from_string_fallback(line):
    """
    Fallback: looks for a 4-digit year (1990-2029) in the text string.
    """
    year_pattern = r'(199\d|20[0-2]\d)'
    matches = re.findall(year_pattern, line)
    if matches:
        return int(matches[-1]) 
    return 9999

def process_papers(filename):
    print(f"Reading {filename}...")
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    paper_data = []
    
    print(f"Processing {len(lines)} papers...")
    
    for i, line in enumerate(lines):
        year = None
        source = "Text Scan"
        
        # 1. Try ArXiv Pattern First (Very reliable if present)
        if not year:
            year = extract_arxiv_year(line)
            if year: source = "ArXiv ID"

        # 2. Try DOI Lookup (Universal)
        if not year:
            doi = extract_doi(line)
            if doi:
                year = get_year_from_universal_doi(doi)
                if year: source = "DOI API"
                time.sleep(0.2) # Polite delay
        
        # 3. Fallback to Regex in text
        if not year:
            year = get_year_from_string_fallback(line)
            if year != 9999: source = "Regex Fallback"

        print(f"[{i+1}/{len(lines)}] Year: {year if year != 9999 else 'Unk'} | Source: {source} | Paper: {line[:30]}...")
        
        paper_data.append({
            'original_line': line,
            'year': year
        })

    # Sort
    sorted_papers = sorted(paper_data, key=lambda x: (x['year'], x['original_line']))
    return sorted_papers

def save_results(papers, output_filename):
    with open(output_filename, 'w', encoding='utf-8') as f:
        current_year = 0
        for paper in papers:
            year = paper['year']
            if year != current_year and year != 9999:
                f.write(f"\n### {year} ###\n")
                current_year = year
            elif year == 9999 and current_year != 9999:
                 f.write(f"\n### Date Unknown (Manual Check Required) ###\n")
                 current_year = 9999
            f.write(f"{paper['original_line']}\n")

if __name__ == "__main__":
    sorted_data = process_papers(INPUT_FILE)
    save_results(sorted_data, OUTPUT_FILE)
    print("Done.")