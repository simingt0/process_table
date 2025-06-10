import requests
from bs4 import BeautifulSoup
import csv
import time
from datetime import datetime
import pandas as pd
import os
from dotenv import load_dotenv

def get_mesh_terms(pmid : str) -> list[str]:
    """
    Given a PMID, returns the MeSH terms for the respective research paper in a list
    """
    #requests paper from PMID
    if pmid == "0":
        return []
    load_dotenv()
    api_key = os.getenv("PUBMED_API_KEY")
    pm_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml&api_key={api_key}"
    )
    pm_response = requests.get(pm_url)
    pm_response.raise_for_status()

    #extracts MeSH terms
    soup2 = BeautifulSoup(pm_response.text, "xml")
    mesh_terms = [tag.get_text() for tag in soup2.select("MeshHeadingList > MeshHeading > DescriptorName")]
    return mesh_terms

def format_time(seconds):
    d_hours, d_rem = divmod(seconds, 3600)
    d_minutes, d_seconds = divmod(d_rem, 60)
    return f"{int(d_hours):02d}:{int(d_minutes):02d}:{int(d_seconds):02d}"

if __name__ == "__main__":
    read_directory = "mesh_term_extractor"
    write_directory = "data.csv"
    rows = []
    fieldnames = ["PMID", "PMCID", "outer folder", "tar file", "MeSH terms"]
    
    program_start_time = datetime.now()
    for outer_folder in os.listdir(read_directory):
        now = datetime.now()
        delta = now - program_start_time
        seconds_passed = delta.total_seconds()
        outer_folder_path = os.path.join(read_directory, outer_folder)
        if os.path.isdir(outer_folder_path):
            print("------------------------------------------")
            print(f"In outer folder '{outer_folder}'")
            print(f"\ttime elapsed: {format_time(seconds_passed)}")
            print("------------------------------------------")
            xml_folder = os.path.join(outer_folder_path, "xml")
            for filelist in os.listdir(xml_folder):
                if filelist.lower().endswith(".csv"):
                    filelist_path = os.path.join(xml_folder, filelist)
                    with open(filelist_path, newline="", encoding="utf-8") as read_file:      
                        reader = csv.reader(read_file)
                        header = next(reader)
                        i = 0
                        filelist_start_time = datetime.now()
                        print(f"*** Starting filelist '{filelist}'")
                        print(f"*** time elapsed: {format_time(seconds_passed)}")
                        for row in reader:
                            i += 1
                            term_list = get_mesh_terms(row[4])
                            rows.append({"PMID" : row[4], "PMCID" : row[2], "outer folder" : outer_folder, "tar file" : filelist[:-4], "MeSH terms" : term_list})
                            time.sleep(0.01)
                            if i%10 == 0:
                                now = datetime.now()
                                df = pd.read_csv(filelist_path)
                                delta = now - filelist_start_time
                                seconds_passed = int(delta.total_seconds())
                                projected_seconds = int(delta.total_seconds()*len(df)/i)
                                print(f"({i}/{len(df)}) PMIDs processed")
                                print(f"{format_time(seconds_passed)}/{format_time(projected_seconds)}")
    print("Sorting...")
    rows.sort(key=lambda row: row["PMID"])
    print("Writing...")
    with open(write_directory, mode="w", newline="", encoding="utf-8") as write_file:
        writer = csv.DictWriter(write_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("Done!")