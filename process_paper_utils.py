import table_utils
import requests
from collections import defaultdict
import copy
import os
from datetime import datetime
import pandas as pd
from llama_api_client import LlamaAPIClient
from google import genai
from dotenv import load_dotenv

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

def llama_request(prompt, max_tokens = 4096, temperature = 0):
    load_dotenv()
    api_key = os.getenv('LLAMA_API_KEY')
    model = os.getenv("LLAMA_MODEL")
    client = LlamaAPIClient(api_key=api_key)

    response = client.chat.completions.create(
        messages=[
        {
            "role": "user",
            "content": prompt
        },
        ],
        model=model,
        stream=False,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        top_p=0.9,
        repetition_penalty=1,
    )
    return response.completion_message.content.text

# def llama_request(prompt, blank1, blank2):
    
#     client = genai.Client()
    
#     response = client.models.generate_content(
#         model="gemini-2.5-flash", contents=prompt
#     )
#     return response.text

def id_tables_prompt(methods: str, tables) -> str:
    """
    Returns one-shot prompt for identifying table type using methods and list of tables, labeling each as treatment group table (G), biomarker table (B), or other (O). Do not call with empty table list.
    """
    #types of table: treatment group table, results table, other
    #maybe TODO: gene expression table?
    #TODO: include qualitative data
    
    TABLES_PER_REQUEST = 4

    count = 0
    prompts = []
    prompt = "# Instructions\n" \
    "You will be given the methods of a research paper which includes at least one animal toxicity study then a list of all the tables included in that paper, each one labeled with its title. Your goal is to identify each table as either a biomarker table (B), or other (O). Biomarker tables may contain treatment group dosage information, but the main difference between is that biomarker tables should contain biomarker data observed from the treatment groups. This can frequency data (such as survival or number occurences of a condition) or metric data (such as ALT levels or compound concentration), and the table must describe the results of specifically an animal toxicity study within the paper to be labeled B. Otherwise, label it O. Any tables from other irrelevant studies should be labeled O, such as results of in vitro assays or gene expression tables. Additionally, tables where the data is qualitative observations should also be labeled O, as well as any tables with individual subject data instead of summary statistics (do not confuse this with frequency metrics, which should be labeled B). For each table on a new line, include the label in the following format in the order that the tables are given: <table name|label>. Do not include any extra text in your answer or extra whitespace inside the labels.\n" \
    "# Example output\n" \
    "<example table 1|O>\n" \
    "<example table 2|O>\n" \
    "<example table 3|B>\n"
    prompt += f"# Input\n<methods>{methods}</methods>\n"
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
        count += 1
        if count == TABLES_PER_REQUEST:
            prompts.append(prompt)
            count = 0
            prompt = "# Instructions\n" \
            "You will be given the methods of a research paper which includes at least one animal toxicity study then a list of all the tables included in that paper, each one labeled with its title. Your goal is to identify each table as either a biomarker table (B), or other (O). Biomarker tables may contain treatment group dosage information, but the main difference between is that biomarker tables should contain biomarker data observed from the treatment groups. This can frequency data (such as survival or number occurences of a condition) or metric data (such as ALT levels or compound concentration), and the table must describe the results of specifically an animal toxicity study within the paper to be labeled B. Otherwise, label it O. Any tables from other irrelevant studies should be labeled O, such as results of in vitro assays or gene expression tables. Additionally, tables where the data is qualitative observations should also be labeled O, as should any tables that give individual data instead of summary statistics (do not confuse this with frequency metrics, which should be labeled B). For each table on a new line, include the label in the following format in the order that the tables are given: <table name|label>. Do not include any extra text in your answer or extra whitespace inside the labels.\n" \
            "# Example output\n" \
            "<example table 1|O>\n" \
            "<example table 2|O>\n" \
            "<example table 3|B>\n"
            prompt += f"# Input\n<methods>{methods}</methods>\n"
    if count != 0:
        prompts.append(prompt)

    return prompts

def gft_prompt(abstract, methods, results):
    prompt = f"# Instructions\nYou will be given the abstract, methods, and results of a paper with an animal toxicity study. Find each treatment group found in the paper that is related to an animal toxicity study. List each group and any parameters found in the following format on its own line, delimited by |: <group|species|n|treatment1|dose1|units1|treatment2|dose2|units2>. group is the name given to the group (make up a short but descriptively unique one if the paper does not give one that is different from the others), species is the animal species (not the strain), and n is the sample size. If the terminal time is used to distinguish between biomarker metrics rather than treatment groups, instead leave it blank. There are two slots for treatment medications, and in each, treatment is the medication name, dose is the numerical dosage amount, and units is the respective units. If there is only one treatment applied, leave treatment2, dose2, and units2 blank. If there is no treatment applied (ie. control group), leave all six parameters blank. Do not omit these from the list, just put the delimiters back to back with no whitespace like so: <control|mouse|10||||||>. If you are not sure of any of these, leave the space blank, and make sure to verify the source of your information before adding it to the answer. Do not include any text besides your answer. An example of a treatment group could look like this: <IBU100|rat|20|ibuprofen|100|mug/kg|||>. Make sure that each label is a list of length 9 even if some of the elements are blank.\n# Abstract\n{abstract}\n# Methods\n{methods}\n# Results\n{results}\n"
    return prompt

def formatting_prompt1(table):
    prompt = f"# Instructions\nYou will be given a table. Count how many left-most row header columns there are such that after this column/these columns numerical measurements start. Reply with this number and do not add any other text.\n# Table\n{table}"
    return prompt

def formatting_prompt2(text_groups, methods, first_col, first_row):
    """
    Returns prompt for formatting
    """
    formatting_prompt = f"# Instructions\nYou will be given the methods section to an animal toxicity study as well as some treatment groups extracted from the text. You will also be given two lists, list1 and list2, derived from a table in the text. Your task is to, using the text and the treatment groups provided to you, decide whether list1 or list2 consists of the names of the treatment groups in the study. Note that a term could be a treatment group even if its name contains 'mean' or 'sd'. Give your answer as <list1> or <list2> and do not give any other text.\n# Methods\n{methods}\n# Treatment Groups\n{text_groups}\n# list1\n{first_col}\n# list2\n{first_row}"
    return formatting_prompt

def id_columns_prompt(table):
    """
    Returns prompt for identifying columns in a table
    """
    # prompt = f"# Instructions\nYou will be given a table describing the results of an animal toxicity study. Your task is to identify label each of the columns with any of 5 types. Remember that each column can have multiple labels or none at all and that some types can be used to label multiple columns. Each label should be indicated as a list of information, each one occupying its own line with the columns being labelled in order from left to right. The first two elements should be the column header and the labels, followed by any additional information as instructed below. Format the lists within triangle brackets with terms separated by |'s like so: <intestinal lesions|count|any other information requested>. If there are multiple labels, separate them into multiple lists with the same first element but different labels. If no labels are applicable to a column, do not add any labels and continue to the next column. Do not add any other text besides your labels and do not add any extra whitespace within the labels. Below are the labels with instructions on what additional information to retrieve for each. Make sure to only select from these types, omitting any labels if not applicable, and always label with the specific word associated with each type, as notated in the titles. If you are unsure if a label is applicable or not, just omit it; not every column has to have a label. Some examples of columns you should omit include columns about terminal time, columns with data about drug efficacy, and columns that display data on p-values.\n" \
    prompt = f"# Instructions\nYou will be given a table describing the results of an animal toxicity study. Your task is to identify label each of the columns with any of 5 types. Remember that each column should only be labeled with one type though each type of label can be used multiple times. Each label should be indicated as a list of information, each one occupying its own line with the columns being labelled in order from left to right. The first two elements should be the column header and the labels, followed by any additional information as instructed below. **The column header must be entered exactly how it is in the table with the same capitalization and whitespace.** Format the lists within triangle brackets with terms separated by |'s like so: <intestinal lesions|count|any other information requested>. If no labels are applicable to a column, do not add any labels and continue to the next column. Do not add any other text besides your labels and do not add any extra whitespace within the labels. Below are the labels with instructions on what additional information to retrieve for each. Make sure to only select from these types, omitting any labels if not applicable, and always label with the specific word associated with each type, as notated in the titles. If you are unsure if a label is applicable or not, just omit it; not every column has to have a label. Some examples of columns you should omit include columns about terminal time, columns with data about drug efficacy, and columns that display data on p-values.\n" \
    "# Labels\n" \
    "1. Group Name (group)\nIf the column contains information on the group name, it should be labeled as group. If the column is a group name column, do not label it with any other types even if it references other types of information such as the treatment medication or dosage. No additional information should be added.\n" \
    "2. Treatment Dosage (dose)\nIf the column contains information on the dosage amount of a medication, label it with dose. If the dosage information is directly linked to a column that gives information on what medication is used (it should contain multiple medications, indicating which one was used in each treatment group), indicate the column title in additional information, prefacing with 'link:'. Note that this linked column could be the same column as the one being labeled if that column lists different medications used and their respective dosages. If the column is instead related to the dosage of a singular medication/treatment, then return the name of that as additional information, prefacing with 'name:'. Also, if there is data on the units, add the units as additional information, prefacing the list entry with 'units:', with each different form of units as its own element in the list. Note that in the list of additional information, while there can only be one item for 'link' or 'name', there could be multiple 'units' listed. Remember that this information can come from the column title or other text in the table, just be sure to verify the information is specifically about this column before answering. If there is no units information, do not add any additional information. A dosage label linked to a column may look like this: <medication|dose|link:Treatment Medication|units:mg/kg|units:mug/kg>. One that is about a specific treatment may look like this: <Treatment.Tamoxifen|dose|name:Tamoxifen|units:mug/kg>\n" \
    "3. Sample Size (size)\nIf the column contains information about the groups' sample sizes, label it as size. Note that some counted biomarkers could contain information about this when representing the frequency (ie. one way of representing this is [count]/[sample size]). Do not add any additional information.\n" \
    "4. Animal Model (animal)\nIf the column contains information on what animal model was used for the groups, label it as animal. However, only label it if the information is the species (mouse, rat, zebrafish, etc.), not the strain (CD-1, Sprague-Dawley, etc.). If the species is the same across all groups, include that as additional information. If else, add no additional information. There should only be one column labeled this." \
    "5. Biomarker (biomarker)\nIf the column contains biomarker data, label it as biomarker. Note that p-value columns are not biomarkers and should not be labeled. As the first piece of additional information, write the name of the biomarker, excluding any irrelevant information. The name should be specific enough that it is unique (ie. 'Lung ITO concentration' is a good name for a column titled 'Lung.ITO concentration (mg/kg)'). Note that this should be the name of the biomarker metric measured (ie. cell count, condition frequency, chemical level), so DO NOT include what summary statistic is being measured (ie. mean, SD, SE) in this title (the mean and SD column labels should have the same name). This may be the same as the name of the column. In the second piece of information, categorize the biomarker as one of 4 types: mean, variation, frequency, or severity (label them exactly as written here). Mean data would be any data that expresses the mean of some metric (ie. ALT levels or WBC count). Variation would be data on the variation of this kind of data, such as the SD or SE. For both mean and variation data, write the units as the third piece of additional information. Frequency data is data on how often a certain condition occurs within the group population specifically (ie. survival rate, occurence of lesions) (note that even if a biomarker describes the frequency of a condition, if that condition's subject is not the group, it should be labeled as mean ie. offspring survival rate). For frequency data, choose out of one of the following units as the third piece of additional information (label exactly as written): count, percent, decimal. Count is the number of incidences, percent is the percentage representation of the frequency, and decimal is the decimal representation (between 0 and 1). Severity data describes the severity of some condition and MUST be in reference to some frequency biomarker. Do not add any more additional information for severity biomarkers. Some columns may include multiple biomarkers, in which case multiple labels should be written for them, even if one of the biomarkers is only present in some of the cells. An example of how a biomarker label would look like is this: <ALT (U/L)|biomarker|ALT|mean|U/L>\n" \
    "# Table\n"
    prompt += table
    return prompt

def link_prompt(methods, table_groups, text_groups):
    prompt = f"# Instructions\nYou will be given the methods section to a paper with an animal toxicity study as well as two tables describing the treatment groups of the paper. The first graph was derived from a graph, and the second graph was derived from the text. Your goal is to cross-reference information from both tables to identify the treatment groups in table 1. For each group in table 1, pair it with one of the groups in table 2 if you think they are referring to the same treatment group based on similarity in the metrics and group names. For each pairing, answer in the following format with each entry having its own row: <group1|group2>, where group1 is the group name from table 1 and group 2 is the group name from table 2 (here is an example: <control|Control Group>). Note that two groups could be paired even if they differ by one or two arguments. Also, the same group from table 2 can be paired to multiple groups in table 1. Make these matches in order of the rows in table 1, making sure that you include every row, even if there is no match. If none of the groups in table 2 seem like a good fit for a group in table 1, label the second group as 'NO MATCH' like so: <example group|NO MATCH>. Do not add any additional whitespace or text in your answer.\n# Methods\n{methods}\n# Table 1\n{table_groups}\n# Table 2\n{text_groups}"
    return prompt

def label_batch_prompt(methods, group_list, batch):
    prompt = f"# Instructions\nYou will be given the methods section, a list of groups and a few paragraphs from the results section of a paper with an animal toxicity study. Your task is to determine whether each paragraph contains qualifying toxicity data. For each paragraph, check if the paragraph fits the following conditions:\n1. The paragraph describes a toxicity study specifically on in vivo animal subjects.\n2. The paragraph contains data on a treatment group, specifying what group it is describing. You can use the provided list of groups as reference, but be sure that the groups abide by condition 1.\n3. The data included is either a summary statistic of a toxicity biomarker (ie. mean +- SD, 'hepatotoxicity seen in 12/20 of group') or a toxicity description within a specific group (ie. 'significantly higher levels of lesions in treatment group'). The data must be specifically about toxicity as a result of the treatment; ignore any efficacy results.\nFor each paragraph, provide each answer as a pair, delineated by the '|' symbol, with each pair on a different line. Answer in order that the paragraphs are given. The first term of each pair should be the first 5 words of the paragraph, including any punctuation or symbols in between them. The second term of each pair should be 'yes' if the paragraph abides by the conditions, or 'no' if it does not. For example:\nBecause the mice (CD-1) were|yes\nDo not include any other text or whitespace besides your answer.\n# Methods\n{methods}\n# Group List\n{"\n".join(group["group"] for group in group_list)}\n# Paragraphs\n<paragraph>\n{"\n</paragraph>\n<paragraph>\n".join([paragraph["text"] for paragraph in batch])}</paragraph>"
    return prompt

def data_from_text_prompt(methods, group_list_df, paragraph):
    prompt = f"# Instructions\nYou will be given the methods section of an animal toxicity paper, a list of groups derived from the text, and a paragraph containing animal toxicity data. Your goal is to extract each piece of data from the text in a standardized manner. Each piece of data should specify a treatment group, be the results of specifically an in vivo animal toxicity study, and be describing a summary statistic (ie. mean +- SD, 'hepatotoxicity seen in 12/20 of group') or observation of a toxicity biomarker (ie. 'significantly higher levels of lesions in treatment group'). Your answer should be a list dilineated by '|' for each piece of data, each on a new line. For each piece of data, complete the following steps to complete its list:\n1. From the provided treatment groups, add the group name of the group that is being observed in this datum or the group the most closely resembles it.\n2. Add the name of the biomarker observed.\n3. Determine which type the observation is out of three options: numerical, frequency, and descriptive (output your choice as written here). Numerical is a discrete summary statistic, usually a mean and SD. Frequency is data describing how common a condition is within the treatment group population, usually a percentage or count. Descriptive data contains no numbers and simple states an observation of the treatment group specifically inferring a toxicity caused by a treatment.\n4. Add the value of the data, omitting any units. If it is a mean +- SD, only include the mean, and write the observation verbatim if it is descriptive data. If the data is frequency and the sample size is included as well (ie. '4 out of the 10 mice'), write it in a fraction like so: count/sample_size (ie. 4/10). Remember that all numerical and frequency data should be numeric and description data should be text.\n5. Add the observation value's units. If it is frequency data, choose out of the following: percentage, decimal, count. Percentage is a value between 0 and 100, decimal is a value between 0 and 1, and count is the number of subjects in the group with the condition. If the data is descriptive, put N/A here.\n6. If the data is numerical, put the SD here. If there is no SD or if the data is frequency or descriptive, put N/A here.\nThe end format looks like this:\ngroup|biomarker|type|value|units|SD\nDo not include any other text or whitespace besides your answer. If a piece of information requested is not available in the paragraph, put N/A in its respective spot.\n# Methods\n{methods}\n# Group List\n{table_utils.dataframe_to_markdown(group_list_df)}\n# Paragraph\n{paragraph}"
    return prompt

def data_from_text(pmc, methods, group_list_df, paragraph):
    prompt = data_from_text_prompt(methods, group_list_df, paragraph)
    answer = llama_request(prompt)
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f"fulltext_extraction/data_from_text/{time}.txt", "w", encoding="utf-8") as file:
        file.write(f"{pmc}\n---\n{prompt}\n---\n{answer}")

    data_dict = {"num": [], "freq": [], "desc": []}
    response_data = [datum.split("|") for datum in answer.split("\n") if "|" in datum]
    for response_datum in response_data:
        datum = {
            "Animal Model": group_list_df.at[response_datum[0].strip().lower(), "animal_model"],
            "Sample Size": group_list_df.at[response_datum[0].strip().lower(), "sample_size"],
            "Treatment 1": group_list_df.at[response_datum[0].strip().lower(), "treatment1"],
            "Dose 1": group_list_df.at[response_datum[0].strip().lower(), "dose1"],
            "Units 1": group_list_df.at[response_datum[0].strip().lower(), "units1"],
            "Treatment 2": group_list_df.at[response_datum[0].strip().lower(), "treatment2"],
            "Dose 2": group_list_df.at[response_datum[0].strip().lower(), "dose2"],
            "Units 2": group_list_df.at[response_datum[0].strip().lower(), "units2"],
            "Biomarker": response_datum[1],
        }
        if "num" in response_datum[2].lower():
            datum["Value"] = response_datum[3]
            datum["Units"] = response_datum[4]
            datum["Variation"] = response_datum[5]
            data_dict["num"].append(datum)
        elif "freq" in response_datum[2].lower():
            if not response_datum[3].isnumeric() or "decimal" in response_datum[4].lower():
                datum["Value"] = response_datum[3]
            elif "percent" in response_datum[4].lower():
                datum["Value"] = int(response_datum[3])/100
            elif "count" in response_datum[4].lower():
                if "/" in response_datum[3]:
                    numerator, denominator = response_datum[3].split("/")
                    datum["Value"] = int(numerator)/int(denominator)
                elif datum["Sample Size"].isnumeric():
                    datum["Value"] = int(response_datum[3])/int(datum["Sample Size"])
                else:
                    print("ERROR: Count but no sample size")
                    datum["Value"] = None
            data_dict["freq"].append(datum)
        elif "desc" in response_datum[2].lower():
            datum["Description"] = response_datum[3]
            data_dict["desc"].append(datum)
    return data_dict


def label_batch(pmc, methods, group_list, batch):
    prompt = label_batch_prompt(methods, group_list, batch)
    answer = llama_request(prompt)

    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f"fulltext_extraction/label_batch/{time}.txt", "w", encoding="utf-8") as file:
        file.write(f"{pmc}\n---\n{prompt}\n---\n{answer}")

    label_list = [label.strip().lower() for label in answer.split("\n") if label.strip() != ""]
    if len(label_list) != len(batch):
        print("ERROR: # paragraphs != # labels")
        for paragraph in batch:
            paragraph["label"] = False
        return
    for i in range(len(batch)):
        first_words, val = label_list[i].split("|", 1)
        if first_words.strip() not in batch[i]["text"].lower(): print("ERROR: first_words not in paragraph")
        if val.strip() == "yes": batch[i]["label"] = True
        elif val.strip() == "no": batch[i]["label"] = False
        else:
            print("ERROR: Invalid label value")

def format_tables(pmc, text_groups, methods, tables):
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
        print(f"START:\n{info["markdown"]}")
        formatted_markdown = table_utils.remove_headers(info["markdown"])
        print(f"AFTER REMOVE HEADER:\n{formatted_markdown}")
        formatted_markdown = table_utils.combine_m_and_SD(formatted_markdown)
        print(f"AFTER COMBINE:\n{formatted_markdown}")

        prompt1 = formatting_prompt1(formatted_markdown)
        answer1 = llama_request(prompt1, 5, 0)
        formatted_markdown = table_utils.combine_markdown_group_cols(formatted_markdown, int(answer1))
        # info_combined = ""
        # if info["caption"]:
        #     info_combined += f"Caption: {info["caption"]}\n"
        # info_combined += formatted_markdown + "\n"
        # if len(info["footnotes"]) > 0:
        #     info_combined += "Footnotes:\n"
        #     for fn in info["footnotes"]:
        #         info_combined += fn + "\n"
        
        formatted_df = table_utils.markdown_to_dataframe(formatted_markdown)
        prompt2 = formatting_prompt2(text_groups, methods, formatted_df.iloc[:, 0].tolist(), formatted_df.columns.tolist())
        answer2 = llama_request(prompt2, 100, 0)
        extracted_answer = answer2.split('<', 1)[1].split('>', 1)[0]
        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(f"responses/table_formatting/{time}.txt", "w", encoding="utf-8") as file:
            file.write(f"{pmc}\n---\n{tid}\n---\n{prompt1}\n---\n{answer1}\n---\n{prompt2}\n---\n{answer2}")
        if "list2" in extracted_answer.lower():
            formatted_markdown = table_utils.transpose_markdown_table(formatted_markdown)
        formatted_tables[tid]["markdown"] = table_utils.separate_m_and_SD(copy.deepcopy(formatted_markdown))
        print(f"tid : {tid}")
        print(formatted_tables[tid]["markdown"])
    return formatted_tables

def groups_from_text(pmc, abstract, methods, results):
    prompt = gft_prompt(abstract, methods, results)
    answer = llama_request(prompt)
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
    answer = llama_request(prompt)
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f"responses/link_groups/{time}.txt", "w", encoding="utf-8") as file:
            file.write(f"{pmc}\n---\n{tid}\n---\n{prompt}\n---\n{answer}")
    pairings = [pair.strip().strip("<>").split("|", 1) for pair in answer.split("\n") if pair]
    for group1_raw, group2_raw in pairings:
        group1 = group1_raw.strip().lower()
        group2 = group2_raw.strip().lower()
        if group2 in ["no match", "nomatch"]:
            continue
        for col in table_groups.columns:
            # print(f"Type: {type(table_groups.at[group1, col])}")
            # print(table_groups.at[group1, col])
            print(f"g1: {group1}, col: {col}")
            if pd.isna(table_groups.at[group1, col]):
                table_groups.at[group1, col] = text_groups.at[group2, col]