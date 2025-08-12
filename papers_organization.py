from tqdm import tqdm
from pathlib import Path
import requests

# Lista de DOIs
dois_path = Path(r"G:\Mi unidad\LIAA\doctorado\papers\lista_de_papers_doi.txt")

dois = []
if dois_path.exists():
    with open(dois_path, "r", encoding="utf-8") as file:
        dois = [line.strip() for line in file if line.strip()]
        dois = [doi[doi.index("10."):] for doi in dois if "10." in doi]
headers = {"Accept": "application/vnd.citationstyles.csl+json"}
titles = []


titles = []
for doi in tqdm(dois, total=len(dois), desc="Consultando DOIs"):
    url = f"https://doi.org/{doi}"
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()

        # Title
        raw_title = data.get("title", "")
        if isinstance(raw_title, list):
            title = raw_title[0] if raw_title else ""
        else:
            title = raw_title

        # Authors
        authors = []
        for author in data.get("author", []):
            given = author.get("given", "")
            family = author.get("family", "")
            authors.append(f"{family}, {given[0]}." if given else family)

        # Year
        year = ""
        issued = data.get("issued", {}).get("date-parts", [[""]])
        if issued and issued[0]:
            year = str(issued[0][0])

        # Journal
        journal = data.get("container-title", "")
        if isinstance(journal, list):
            journal = journal[0] if journal else ""

        # Volume, Issue, Pages
        volume = data.get("volume", "")
        issue = data.get("issue", "")
        pages = data.get("page", "")

        # URL
        url_out = data.get("URL", url)

        # Open Access
        license_info = data.get("license", [])
        if license_info:
            license_url = license_info[0].get("URL", "")
            oa_status = f"Open Access ({license_url})"
        else:
            oa_status = "No Open Access (o sin datos)"

        titles.append({
            "doi": doi,
            "title": title,
            "authors": authors,
            "year": year,
            "journal": journal,
            "volume": volume,
            "issue": issue,
            "pages": pages,
            "url": url_out,
            "open_access": oa_status
        })

    except Exception as e:
        titles.append({
            "doi": doi,
            "error": str(e)
        })

# Sort them by year and title
titles.sort(key=lambda x: (x.get("year", ""), x.get("title", "").lower()))

# Make print in APA style
for title in titles:
    if "error" not in title:
        text = f"{title['authors']} ({title['year']}). {title['title']}. {title['journal']}, {title['volume']}({title['issue']}), {title['pages']}. {title['url']}\n"
        text = text.replace("[", "").replace("]", "")  # Remove brackets if any
        text = text.replace("'", "")  # Remove single quotes
        if title["open_access"]:
            text = "[OA] " + text
        else:
            text = "[NOA] " + text
        print(f"\n\n{text}")
    else:
        print(f"\n\nError al procesar DOI {title['doi']}: {title['error']}")

# Save titles to a file
with open(r"G:\Mi unidad\LIAA\doctorado\papers\bibliografia.txt", "w", encoding="utf-8") as f:
    for title in titles:
        if "error" not in title:
            text = f"{title['authors']} ({title['year']}). {title['title']}. {title['journal']}, {title['volume']}({title['issue']}), {title['pages']}. {title['url']}"
            text += "\n" if title != titles[-1] else ""
            text = text.replace("[", "").replace("]", "")  # Remove brackets if any
            text = text.replace("'", "")  # Remove single quotes
            if title["open_access"].startswith("Open Access"):
                text = "[OA] " + text
            else:
                text = "[NOA] " + text
            f.write(text)
        else:
            f.write(f"Error al procesar DOI {title['doi']}: {title['error']}\n")

print("\nArchivo 'bibliografia.txt' generado con", len(titles), "entradas.")
titles_without_errors = [title for title in titles if "error" not in title]
print("\nHay {} open access y {} no liberados.".format(sum(1 for title in titles_without_errors if title["open_access"].startswith("Open Access")), sum(1 for title in titles_without_errors if not title["open_access"].startswith("Open Access"))))
print("\nErrores en DOIs:")
for title in titles:
    if "error" in title:
        print(f" - {title['doi']}: {title['error']}")