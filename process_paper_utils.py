import table_utils
import requests
from collections import defaultdict
import copy
import os
from datetime import datetime
import pandas as pd

def fetch_xml(pmc_id: str, encoding: str = "ascii", db: str = "pmcoa") -> str:
    """
    Uses API call to import PMC dataset paper in xml format and default ASCII encoding
    """
    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/{db}.cgi/BioC_xml/{pmc_id}/{encoding}"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.text

def chatgpt_request(prompt: str, max_tokens: int = 500, temperature = 0):
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    version = os.getenv("OPENAI_API_VERSION")
    model = os.getenv("OPENAI_MODEL")
    api_key = os.getenv("OPENAI_API_KEY")

    url = f"{endpoint}openai/deployments/{model}/chat/completions?api-version={version}"
    print(url)

    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }

    table_payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    #saving as {PMC}\n---\n{prompt}\n---\n{response}
    table_response_json = requests.post(url, headers=headers, json=table_payload)
    print(f"ChatGPT Request Status: {table_response_json.status_code}")
    return table_response_json.json()['choices'][0]['message']['content']

def id_tables_prompt(abstract: str, tables) -> str:
    """
    Returns one-shot prompt for identifying table type using abstract and list of tables, labeling each as treatment group table (G), biomarker table (B), or other (O). Do not call with empty table list.
    """
    #types of table: treatment group table, results table, other
    #maybe TODO: gene expression table?
    #TODO: include qualitative data
    prompt = "# Instructions\n" \
    "You will be given the abstract of a research paper which includes at least one animal toxicity study then a list of all the tables included in that paper, each one labeled with its title. For each table identify it as one of three categories: treatment group table (G), biomarker table (B), or other (O). Treatment group tables should contain information on the specific treatment groups, such as the medications, the dosages, the sample size, etc. Tables that only give information on the chemicals used (such as the sourcing) should not be labeled G and should instead be labeled O. Even if the table meets these criteria, only label the table G if it describes the groups of specifically an animal toxicity study within the paper. Otherwise, label it O. Biomarker tables may contain treatment group dosage information, but the main difference between is that biomarker tables should contain biomarker data observed from the treatment groups. This can frequency data (such as survival or number occurences of a condition) or metric data (such as ALT levels or compound concentration), and the table must describe the results of specifically an animal toxicity study within the paper to be labeled B. Otherwise, label it O. Any other irrelevant papers should be labeled O, such as results of in vitro assays or gene expression tables. For each table on a new line, include the label in the following format in the order that the tables are given: <table name|label>. Do not include any extra text in your answer or extra whitespace inside the labels. Here is an example input and output:\n" \
    "# Example Input\n" \
    "<abstract>Abstract\nBackground: Compound ABC is a novel small molecule with potential therapeutic applications in metabolic disorders. To evaluate its safety profile, we conducted two complementary toxicity studies: an in vitro cytotoxicity assay and an in vivo rodent toxicity assessment.\nMethods:\nIn vitro assay: Human hepatocellular carcinoma (HepG2) and rat cardiomyoblast (H9c2) cell lines were exposed to ABC at concentrations of 1-200 µM for 24, 48, and 72 hours. Cell viability was measured by MTT assay, and apoptosis was assessed by flow cytometry using Annexin V/PI staining and caspase-3 activation.\nIn vivo study: Male and female Sprague-Dawley rats (n = 10 per sex per group) received daily oral doses of ABC at 0 (vehicle control), 10, 50, or 200 mg/kg for 28 days. Clinical observations, body weight, and food consumption were recorded. At study termination, hematology, serum biochemistry (ALT, AST, BUN, creatinine), and organ histopathology (liver, kidney, heart) were evaluated.\nResults:\nIn vitro, ABC induced a concentration- and time-dependent decrease in cell viability, with IC₅₀ values of 45 µM (HepG2) and 120 µM (H9c2) at 48 h. Apoptotic indices increased significantly at ≥50 µM, accompanied by a 3-fold rise in caspase-3 activity. In vivo, no mortality or adverse clinical signs were observed at ≤50 mg/kg. Rats dosed at 200 mg/kg exhibited transient body weight suppression (5% lower vs. control), mild elevations in ALT and creatinine (1.4- and 1.3-fold, respectively), and minimal hepatocellular vacuolation on histology. No significant changes were seen in hematological parameters or cardiac histopathology.\nConclusions: ABC demonstrates moderate cytotoxicity in vitro, primarily via apoptosis, and is well tolerated in rats at doses up to 50 mg/kg/day. The no-observed-adverse-effect level (NOAEL) in this 28-day study is 50 mg/kg. These findings support further preclinical development of ABC, with emphasis on mechanistic studies of hepatic metabolism and long-term safety.</abstract>\n" \
    "<example table 1>Caption: Overview of dosing regimens for both in vitro and in vivo studies.\n| Study | Model | Group | Dose / Concentration | Frequency / Exposure | Route | Replicates (n) |\n| --- | --- | --- | --- | --- | --- | --- |\n| In vitro | HepG2 & H9c2 cell lines | Control | 0 µM | 48 h single exposure | N/A | 3 per timepoint |\n| In vitro | HepG2 & H9c2 cell lines | Low | 10 µM | 48 h single exposure | N/A | 3 per timepoint |\n| In vitro | HepG2 & H9c2 cell lines | Medium | 50 µM | 48 h single exposure | N/A | 3 per timepoint |\n| In vitro | HepG2 & H9c2 cell lines | High | 100 µM | 48 h single exposure | N/A | 3 per timepoint |\n| In vivo | Sprague-Dawley rat | Control | 0 mg/kg/day | Once daily 28 days | Oral gavage | 10 rats/sex/group |\n| In vivo | Sprague-Dawley rat | Low | 10 mg/kg/day | Once daily 28 days | Oral gavage | 10 rats/sex/group |\n| In vivo | Sprague-Dawley rat | Medium | 50 mg/kg/day | Once daily 28 days | Oral gavage | 10 rats/sex/group |\n| In vivo | Sprague-Dawley rat | High | 200 mg/kg/day | Once daily 28 days | Oral gavage | 10 rats/sex/group |\nFootnotes:\n1 In vitro replicates refer to independent wells measured per concentration and timepoint.\n2 In vivo 'n' indicates number of animals per sex in each treatment group.\n</example table 1>\n" \
    "<example table 2>Caption: Summary of cell-based toxicity endpoints following 48 h exposure to ABC.\n| Group | Viability (% of control) | Apoptotic Cells (%) | Caspase-3 Activity (fold of control) |\n| --- | --- | --- | --- |\n| Control (0 µM) | 100 +- 5 | 5 +- 1 | 1.0 +- 0.1 |\n| Low (10 µM) | 95 +- 4 |  8 +- 2 | 1.2 +- 0.1 |\n| Medium (50 µM) | 60 +- 6 | 25 +- 3 | 3.0 +- 0.2 |\n| High (100 µM) | 30 +- 5 | 50 +- 4 | 5.5 +- 0.4 |\nFootnotes:\n3 Viability measured by MTT assay, expressed as percent of untreated control.\n4 Apoptotic fraction determined by Annexin V/PI staining via flow cytometry.\n5 Caspase-3 activity normalized to control, measured enzymatically.\n</example table 2>\n" \
    "<example table 3>Caption: Key serum biomarker changes in Sprague-Dawley rats after 28 days of ABC dosing.\n| Group | ALT (U/L) | AST (U/L) | Creatinine (mg/dL) | BUN (mg/dL) |\n| --- | --- | --- | --- | --- |\n| Control (0 mg/kg) | 35 +- 5 | 42 +- 6 | 0.6 +- 0.1 | 14 +- 2 |\n| Low (10 mg/kg) | 38 +- 6 | 45 +- 7 | 0.7 +- 0.1 | 15 +- 2 |\n| Medium (50 mg/kg) | 45 +- 7 | 52 +- 8 | 0.8 +- 0.1 | 18 +- 3 |\n| High (200 mg/kg) | 50 +- 8 | 60 +- 10 | 0.9 +- 0.1 | 20 +- 3 |\nFootnotes:\n6 Alanine aminotransferase (ALT) and aspartate aminotransferase (AST) reflect hepatic function.\n7 BUN: blood urea nitrogen, indicates renal function.\n</example table 3>\n" \
    "# Example output\n" \
    "<example table 1|G>\n" \
    "<example table 2|O>\n" \
    "<example table 3|B>\n"
    prompt += f"# Input\n<abstract>{abstract}</abstract>\n"

    for tid, info in tables.items():
        if not info["markdown"]: continue
        prompt += f"<{tid}>"
        if info["caption"]:
            prompt += f"Caption: {info["caption"]}\n"
        prompt += info["markdown"] + "\n"
        if len(info["footnotes"]) > 0:
            prompt += "Footnotes:\n"
            for fn in info["footnotes"]:
                prompt += fn + "\n"
        prompt += f"</{tid}>\n"

    return prompt

def gft_prompt(abstract, methods, results):
    prompt = f"# Instructions\nYou will be given the abstract, methods, and results of a paper with an animal toxicity study. Find each treatment group found in the paper that is related to an animal toxicity study. List each group and any parameters found in the following format on its own line, delimited by |: <group|species|n|terminal time|treatment1|dose1|units1|treatment2|dose2|units2>. group is the name given to the group (make up a short but descriptively unique one if the paper does not give one that is different from the others), species is the animal species (not the strain), and n is the sample size. terminal time is the time at which the group is euthanized/the metrics are measured. If the terminal time is used to distinguish between biomarker metrics rather than treatment groups, instead leave it blank. There are two slots for treatment medications, and in each, treatment is the medication name, dose is the numerical dosage amount, and units is the respective units. If there is only one treatment applied, leave treatment2, dose2, and units2 blank. If there is no treatment applied (ie. control group), leave all six parameters blank. Do not omit these from the list, just put the delimiters back to back with no whitespace like so: <control|mouse|10|30 days||||||>. If you are not sure of any of these, leave the space blank, and make sure to verify the source of your information before adding it to the answer. Do not include any text besides your answer. An example of a treatment group could look like this: <IBU100|rat|20||ibuprofen|100|mug/kg|||>.\n# Abstract\n{abstract}\n# Methods\n{methods}\n# Results\n{results}\n"
    return prompt

def formatting_prompt(abstract, methods, table):
    """
    Returns prompt for formatting
    """
    #TODO: one shot
    formatting_prompt = f"# Instructions\nYou will be given the abstract of a paper and a table representing animal toxicity data from a study in that paper. Your task is to identify the structure of the table in context of the study.\n 1. Using the text and the table, identify what the treatment groups are and what the measured metrics are.\n2. Observe how the table is oriented: do the groups go vertically, where each column corresponds to a treatment group and each row corresponds to a metric, or do they go horizontally, the inverse? Write <vertical> if vertical and <horizontal> if horizontal.\n3. If the groups go vertically, are there any columns on the very left that should be combined? If so, add that to the end of the answer like so: <vertical|n>, where n is the number of columns that make up the row header, such that the data starts at column n+1. If the groups go horizontally, skip this step and answer as so: <horizontal>.\n# Abstract\n{abstract}\n# Methods\n{methods}\n# Table\n{table}" \
    "Do not include any extra text or whitespace besides your answer."
    return formatting_prompt

def id_columns_prompt(table):
    """
    Returns prompt for identifying columns in a table
    """
    prompt = f"# Instructions\nYou will be given a table describing the results of an animal toxicity study. Your task is to identify label each of the columns with any of 8 types. Remember that each column can have multiple labels or none at all and that some types can be used to label multiple columns. Each label should be indicated as a list of information, each one occupying its own line with the columns being labelled in order from left to right. The first two elements should be the column header and the labels, followed by any additional information as instructed below. Format the lists within triangle brackets with terms separated by |'s like so: <intestinal lesions|count|any other information requested>. If there are multiple labels, separate them into multiple lists with the same first element but different labels. If no labels are applicable to a column, do not add any labels and continue to the next column. Do not add any other text besides your labels and do not add any extra whitespace within the labels. Below are the labels with instructions on what additional information to retrieve for each. Make sure to only select from these types, omitting any labels if not applicable, and always label with the specific word associated with each type, as notated in the titles. If you are unsure if a label is applicable or not, just omit it; not every column has to have a label.\n" \
    "# Labels\n" \
    "1. Group Name (group)\nIf the column contains information on the group name, it should be labeled as group. If the column is a group name column, do not label it with any other types even if it references other types of information such as the treatment medication or dosage. No additional information should be added.\n" \
    "2. Treatment Dosage (dose)\nIf the column contains information on the dosage amount of a medication, label it with dose. If the dosage information is directly linked to a column that gives information on what medication is used (it should contain multiple medications, indicating which one was used in each treatment group), indicate the column title in additional information, prefacing with 'link:'. Note that this linked column could be the same column as the one being labeled if that column lists different medications used and their respective dosages. If the column is instead related to the dosage of a singular medication/treatment, then return the name of that as additional information, prefacing with 'name:'. Also, if there is data on the units, add the units as additional information, prefacing the list entry with 'units:', with each different form of units as its own element in the list. Note that in the list of additional information, while there can only be one item for 'link' or 'name', there could be multiple 'units' listed. Remember that this information can come from the column title or other text in the table, just be sure to verify the information is specifically about this column before answering. If there is no units information, do not add any additional information. A dosage label linked to a column may look like this: <medication|dose|link:Treatment Medication|units:mg/kg|units:mug/kg>. One that is about a specific treatment may look like this: <Treatment.Tamoxifen|dose|name:Tamoxifen|units:mug/kg>\n" \
    "3. Sample Size (size)\nIf the column contains information about the groups' sample sizes, label it as size. Note that some counted biomarkers could contain information about this when representing the frequency (ie. one way of representing this is [count]/[sample size]). Do not add any additional information.\n" \
    "4. Animal Model (animal)\nIf the column contains information on what animal model was used for the groups, label it as animal. However, only label it if the information is the species (mouse, rat, zebrafish, etc.), not the strain (CD-1, Sprague-Dawley, etc.). If the species is the same across all groups, include that as additional information. If else, add no additional information. There should only be one column labeled this." \
    "5. Terminal Time (time)\nIf the column contains information about the time at which the measurements were taken or when the treatment group was euthanized, label it as time. Be careful as these columns could also be group columns, where the treatment groups differ in when they were euthanized. They can also be biomarker columns, where the metrics differ based on what time they were measured. Do not add any additional information." \
    "6. Biomarker (biomarker)\nIf the column contains biomarker data, label it as biomarker. As the first piece of additional information, write the name of the biomarker, excluding any irrelevant information. The name should be specific enough that it is unique (ie. 'Lung ITO concentration' is a good name for a column titled 'Lung.ITO concentration (mg/kg)'). This may be the same as the name of the column. In the second piece of information, categorize the biomarker as one of 4 types: mean, variation, frequency, or severity (label them exactly as written here). Mean data would be any data that expresses the mean of some metric (ie. ALT levels or WBC count). Variation would be data on the variation of this kind of data, such as the SD or SE. For both mean and variation data, write the units as the third piece of additional information. Frequency data is data on how often a certain condition occurs within the group population specifically (ie. survival rate, occurence of lesions) (note that even if a biomarker describes the frequency of a condition, if that condition's subject is not the group, it should be labeled as mean ie. offspring survival rate). For frequency data, choose out of one of the following units as the third piece of additional information (label exactly as written): count, percent, decimal. Count is the number of incidences, percent is the percentage representation of the frequency, and decimal is the decimal representation (between 0 and 1). Severity data describes the severity of some condition and MUST be in reference to some frequency biomarker. Do not add any more additional information for severity biomarkers. Some columns may include multiple biomarkers, in which case multiple labels should be written for them, even if one of the biomarkers is only present in some of the cells. An example of how a biomarker label would look like is this: <ALT (U/L)|biomarker|ALT|mean|U/L>\n" \
    "# Table\n"
    prompt += table
    return prompt

def link_prompt(methods, table_groups, text_groups):
    prompt = f"# Instructions\nYou will be given the methods section to a paper with an animal toxicity study as well as two tables describing the treatment groups of the paper. The first graph was derived from a graph, and the second graph was derived from the text. Your goal is to cross-reference information from both tables to identify the treatment groups in table 1. For each group in table 1, pair it with one of the groups in table 2 if you think they are referring to the same treatment group based on similarity in the metrics and group names. For each pairing, answer in the following format with each entry having its own row: <group1|group2>, where group1 is the group name from table 1 and group 2 is the group name from table 2 (here is an example: <control|Control Group>). Note that two groups could be paired even if they differ by one or two arguments. Also, the same group from table 2 can be paired to multiple groups in table 1. If none of the groups in table 2 seem like a good fit for a group in table 1, label the second group as 'NO MATCH' like so: <example group|NO MATCH>. Do not add any additional whitespace or text in your answer.\n# Methods\n{methods}\n# Table 1\n{table_groups}\n# Table 2\n{text_groups}"
    return prompt

def format_tables(pmc, abstract, methods, tables):
    """
    Prompts LLM for formatting instructions then performs them
    - Inverts
    - Combines mean and SD if separated in groups
    - Expands mean and SD into biomarkers
    - TODO: Special characters
    """
    formatted_tables = copy.deepcopy(tables)
    for tid, info in tables.items():
        if info["label"].lower() != "b": continue
        if not info["markdown"]: continue
        formatted_markdown = table_utils.remove_headers(info["markdown"])
        formatted_markdown = table_utils.combine_m_and_SD(formatted_markdown)
        info_combined = ""
        if info["caption"]:
            info_combined += f"Caption: {info["caption"]}\n"
        info_combined += formatted_markdown + "\n"
        if len(info["footnotes"]) > 0:
            info_combined += "Footnotes:\n"
            for fn in info["footnotes"]:
                info_combined += fn + "\n"
        prompt = formatting_prompt(abstract, methods, info_combined)
        answer = chatgpt_request(prompt, 10, 0)
        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(f"responses/table_formatting/{time}.txt", "w", encoding="utf-8") as file:
            file.write(f"{pmc}\n---\n{tid}\n---\n{prompt}\n---\n{answer}")
        if "vertical" in answer:
            num_cols = int(answer.split("|",1)[1].rstrip(">").strip())
            formatted_markdown = table_utils.combine_markdown_group_cols(formatted_markdown, num_cols)
            formatted_tables[tid]["markdown"] = table_utils.transpose_markdown_table(copy.deepcopy(formatted_markdown))
        formatted_tables[tid]["markdown"] = table_utils.separate_m_and_SD(formatted_tables[tid]["markdown"])
    return formatted_tables

def groups_from_text(pmc, abstract, methods, results):
    prompt = gft_prompt(abstract, methods, results)
    answer = chatgpt_request(prompt)
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f"responses/groups_from_text/{time}.txt", "w", encoding="utf-8") as file:
            file.write(f"{pmc}\n---\n{prompt}\n---\n{answer}")
    treatment_groups = [group.strip().strip("<>").split("|") for 
    group in answer.split("\n") if group]
    print(answer)
    print(treatment_groups)
    return treatment_groups

def link_groups(pmc, tid, methods, table_groups, text_groups):
    prompt = link_prompt(methods, table_utils.dataframe_to_markdown(table_groups), table_utils.dataframe_to_markdown(text_groups))
    answer = chatgpt_request(prompt)
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f"responses/link_groups/{time}.txt", "w", encoding="utf-8") as file:
            file.write(f"{pmc}\n---\n{tid}\n---\n{prompt}\n---\n{answer}")
    pairings = [pair.strip().strip("<>").split("|", 1) for pair in answer.split("\n") if pair]
    for group1_raw, group2_raw in pairings:
        group1 = group1_raw.strip().lower()
        group2 = group2_raw.strip().lower()
        for col in table_groups.columns:
            if pd.isna(table_groups.at[group1, col]):
                table_groups.at[group1, col] = text_groups.at[group2, col]

