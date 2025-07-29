import table_utils
import process_paper_utils as utils
from collections import defaultdict
from bs4 import BeautifulSoup
import os
from datetime import datetime
import pandas as pd
import html
from openpyxl import load_workbook

PMC_LIST = ["PMC5514907", "PMC7767363", "PMC6449948", "PMC9322224", "PMC5373957", "PMC7716188", "PMC8536644", "PMC5043003", "PMC11131026"]

def main():
    os.makedirs("fetch_tables", exist_ok=True)
    os.makedirs("fetch_tables/responses", exist_ok=True)
    for PMC in PMC_LIST:
        print(f"--Processing {PMC}--")
        print("importing paper")
        #importing paper
        os.makedirs("papers", exist_ok=True)
        if os.path.isfile(f"papers/{PMC}.xml"):
            with open(f"papers/{PMC}.xml", 'r', encoding='utf-8') as f:
                paper_xml = f.read()
        else:
            paper_xml = utils.fetch_xml(PMC).replace("|", "/")
            with open(f"papers/{PMC}.xml", "w", encoding="utf-8") as f:
                f.write(paper_xml)
        
        print("parsing xml")
        #parsing w/ bs4
        soup = BeautifulSoup(paper_xml, "lxml-xml")

        #methods
        methods_passages = [p for p in soup.find_all("passage") if p.find("infon", {"key": "section_type"}) and 
                p.find("infon", {"key": "section_type"}).text.strip().upper() == "METHODS"]
        methods_text_list = [p.find("text").get_text() for p in methods_passages if p.find("text")]
        methods = "\n".join(methods_text_list)


        print("extracting tables")
        #tables
        table_passages = [p for p in soup.find_all("passage") if (sect := p.find("infon", {"key": "section_type"})) and sect.text.strip() == "TABLE"]
        tables = defaultdict(lambda: {"label": "o", "caption": None, "markdown": None, "footnotes": []})
        for p in table_passages:
            table_id = p.find("infon", {"key": "id"}).text.strip()
            passage_type = p.find("infon", {"key": "type"}).text.strip()
            if passage_type == "table_caption":
                tables[table_id]["caption"] = p.find("text").text.strip()
            elif passage_type == "table":
                table_xml = p.find("infon", {"key": "xml"}).text
                if "<table" not in table_xml:
                    continue
                table = table_utils.single_html_table_to_markdown(html.unescape(table_xml))
                tables[table_id]["markdown"] = table_utils.remove_headers(table)
            elif passage_type in ("table_foot", "table_footnote"):
                tables[table_id]["footnotes"].append(p.find("text").text.strip())

        print("LLM Request")
        id_tables_prompts = utils.id_tables_prompt(methods, tables)
        answers = []
        for id_tables_prompt in id_tables_prompts:
            id_tables_output = utils.llama_request(id_tables_prompt, 100, 0)
            time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            with open(f"fetch_tables/responses/{time}.txt", "w", encoding="utf-8") as file:
                file.write(f"{PMC}\n---\n{id_tables_prompt}\n---\n{id_tables_output}")
            answers.append(id_tables_output)
        id_tables_combined_output = "\n".join([answer.strip() for answer in answers])

        print("processing labels")
        for label in id_tables_combined_output.strip().split("\n"):
            if "|" not in label: continue
            label = label.strip()
            content = label.strip("<>")
            tid, table_label = content.split("|")
            tables[tid.strip()]["label"] = table_label.strip().lower()

        print("writing to xlsx")
        to_write = [(tid,info) for tid,info in tables.items() if info["label"] == "b"]
        if to_write:
            output_file_path = f"fetch_tables/{PMC}.xlsx"
            with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
                for tid, info in tables.items():
                    if info["label"] != "b": continue
                    table_utils.markdown_to_dataframe(info["markdown"]).to_excel(writer, sheet_name=tid, index=False)

if __name__ == "__main__":
    main()