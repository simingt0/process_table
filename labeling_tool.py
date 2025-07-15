import table_utils
import process_paper_utils as utils
import os
from bs4 import BeautifulSoup
from collections import defaultdict
import html
import pandas as pd

PMC = "PMC11131026"

#fetching xml file
os.makedirs("papers", exist_ok=True)
if os.path.isfile(f"papers/{PMC}.xml"):
    with open(f"papers/{PMC}.xml", 'r', encoding='utf-8') as f:
        paper_xml = f.read()
else:
    paper_xml = utils.fetch_xml(PMC)
    with open(f"papers/{PMC}.xml", "w", encoding="utf-8") as f:
        f.write(paper_xml)

#getting tables
soup = BeautifulSoup(paper_xml, "lxml-xml")

#tables
table_passages = []
for p in soup.find_all("passage"):
    sect = p.find("infon", {"key": "section_type"})
    if sect and sect.text.strip() == "TABLE":
        table_passages.append(p)
tables = defaultdict(lambda: {"caption": None, "df": None, "footnotes": []})
for p in table_passages:
    table_id = p.find("infon", {"key": "id"}).text.strip()
    passage_type = p.find("infon", {"key": "type"}).text.strip()
    if passage_type == "table":
        table_xml = p.find("infon", {"key": "xml"}).text
        if "<table" not in table_xml:
            continue
        tables[table_id]["df"] = table_utils.markdown_to_dataframe(table_utils.single_html_table_to_markdown(html.unescape(table_xml)))
for tid, info in tables.items():
    out = pd.DataFrame(index=info["df"].index)
    for col in info["df"].columns:
        s = info["df"][col].astype(str)
        parts = s.str.partition('+-')
        
        out[f"{col}_mean"] = parts[0]
        out[f"{col}_SD"] = parts[2]
    info["df_split"] = out


with pd.ExcelWriter("tool.xlsx", engine="openpyxl") as writer:
    for tid, info in tables.items():
        info["df_split"].to_excel(writer, sheet_name=tid, index=False)