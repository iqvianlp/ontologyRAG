import logging
from pathlib import Path
from pyoxigraph import Store
from typing import Union
import pandas as pd

from scripts.llm.clients import BaseLLMClient
from scripts.utils.file_interface import InputDoc
from scripts.utils.model_loader import ModelLoader
from scripts.utils.query import read_prompts
from scripts.utils.general import init_logger

logger = init_logger(__name__, logging.INFO)
THIS_DIR = Path(__file__).parent
print(THIS_DIR)


class Rdf:
    """
    A class represents a rdf structure for further processing.
    """
    def __init__(self, subject, predicate, object) -> None:
        self.subject = subject
        self.predicate = predicate
        self.object = object


def llm_is_ready(llm: BaseLLMClient) -> bool:
    """
    Checks LLM connection status.
    """
    return True if llm.is_alive() else False
    

def is_valid_sparql(graph_store: Store, query: str) -> bool:
    """
    Checks if the SPARQL query is structured in the correct format.
    """
    try:
        graph_store.query(query)
        return True
    except:
        return False


def is_valid_json_output(answer: str, key: str) -> bool:
    """
    Checks if the mapping result generated by LLMs is in the correct json format.
    """
    try:
        return True if {key}.issubset(eval(answer).keys()) else False
    except:
        return False


def call_llm(llm: BaseLLMClient, prompt_file: Union[Path, str], input_question: InputDoc, llm_type: str,
             example_file: Union[Path, str] = None, max_output_tokens: int = 1000) -> str:
    """
    Formats prompt message and calls LLM.
    """
    queries = read_prompts(prompt_file, example_file)
    for query in queries:
        message = query.format_message(input_question)
        if llm_type == 'azure':
            answer = llm.invoke(str(message))
        else:
            answer = llm.invoke(str(message), params={"max_new_tokens": max_output_tokens})
        return answer
        

def run_nl2sparql(input_question: InputDoc, graph_store: Store, llm_client: BaseLLMClient, llm_type: str) -> str:
    logger.debug(f"-----Input question-----\n {input_question}")

    prompt_file = Path(THIS_DIR, "prompt_templates/nl2sparql/prompts_sparql.xlsx")
    example_file = Path(THIS_DIR, "prompt_templates/nl2sparql/examples_sparql.xlsx")
    
    answer_sparql = ""
    while not is_valid_sparql(graph_store, answer_sparql):
        answer_sparql = call_llm(llm=llm_client,
                                 prompt_file=prompt_file,
                                 input_question=input_question,
                                 example_file=example_file,
                                 llm_type=llm_type)

        # Post-processes output from llama-3
        answer_sparql = answer_sparql.strip("'")
        sys_word_loc = answer_sparql.find("PREFIX")
        if sys_word_loc != -1:
            answer_sparql = answer_sparql[sys_word_loc:]
        sys_word_loc = answer_sparql.find("assistant")
        if sys_word_loc != -1:
            answer_sparql = answer_sparql[:sys_word_loc]
        answer_sparql = answer_sparql.strip("`")

    logger.debug(f"-----Converted SPARQL query-----\n{answer_sparql}")
    return answer_sparql


def pair_disease_codes(rdfs: list) -> dict:
    """
    Extract disease code pairs from the give rdfs.
    """
    pairing = {}
    for instance in rdfs: 
        if instance.predicate == ":mapsFrom" or instance.predicate == ":mapsTo":
            if instance.subject not in pairing.keys():
                pairing[instance.subject] = {}
                pairing[instance.subject]["mapsTo"] = []
            if instance.predicate == ":mapsFrom":
                pairing[instance.subject]["mapsFromObject"] = instance.object
                for instance_2 in rdfs:
                    if instance.object == instance_2.subject:
                        if instance_2.predicate == ":mappedValue":
                            pairing[instance.subject]["mapsFromCode"] = instance_2.object
                        elif instance_2.predicate == "rdfs:label":
                            pairing[instance.subject]["mapsFromLabel"] = instance_2.object                
            elif instance.predicate == ":mapsTo":
                pairing_mapTo = {"mapsToObject": instance.object, "mapsToCode": []}
                for instance_2 in rdfs:
                    if instance.object == instance_2.subject:
                        if instance_2.predicate == ":mappedValue" and instance_2.object not in pairing_mapTo["mapsToCode"]:
                            pairing_mapTo["mapsToCode"].append(instance_2.object)
                        elif instance_2.predicate == "rdfs:label":
                            pairing_mapTo["mapsToLabel"] = instance_2.object
                if pairing_mapTo not in pairing[instance.subject]["mapsTo"]:
                    pairing[instance.subject]["mapsTo"].append(pairing_mapTo)
    
    pairing_deduplicated = {}
    for pair in pairing.values():
        del pair["mapsFromObject"]
        for mapsTo in pair["mapsTo"]: 
            del mapsTo["mapsToObject"]
            mapsTo["mapsToCode"] = ", ".join(mapsTo["mapsToCode"])
            mapsTo = {key: value for key, value in sorted(mapsTo.items())}

            if pair["mapsFromCode"] not in pairing_deduplicated.keys():
                pairing_deduplicated[pair["mapsFromCode"]] = pair
            elif mapsTo not in pairing_deduplicated[pair["mapsFromCode"]]["mapsTo"]:
                pairing_deduplicated[pair["mapsFromCode"]]["mapsTo"].append(mapsTo)

    return pairing_deduplicated


def retrieve_subgraph(graph_store: Store, sparql_query: str) -> dict:
    prefix = {"rdf:": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
              "rdfs:": "http://www.w3.org/2000/01/rdf-schema#",
              "common:": "http://iqvia.com/ontologies/common/",
              "meta:": "http://iqvia.com/ontologies/metadata/",
              "dc:": "http://purl.org/dc/elements/1.1/",
              ":": "http://iqvia.com/ontologies/mappings/"}

    # Queries ontology graph data with the generated SPARQL query
    query_results = graph_store.query(sparql_query)
    logger.debug(f"-----Retrieved query results-----\n{query_results}")

    rdfs = []
    is_labeled = False        
    
    # Processes query results and structures RDFs
    for query_result in query_results:
        subject = query_result.subject.value
        object = query_result.object.value
        predicate = query_result.predicate.value

        # Replaces prefix
        object_is_text = True
        for k, v in prefix.items():
            if v in object:
                object_is_text = False
            subject = subject.replace(v, k)
            object = object.replace(v, k)
            predicate = predicate.replace(v, k)

        # If rdfs:label exists, enter the reasoning step
        if predicate == "rdfs:label":
            is_labeled = True

        # Create a list of RDF triples
        rdfs.append(Rdf(subject, predicate, object))
    logger.debug(f"-----Extracted RDF triples from query results-----\n{rdfs}")

    # Extract disease code pairs
    pairing = pair_disease_codes(rdfs)
    logger.debug(f"-----Extracted code pairs from RDF triples-----\n{pairing}")

    return pairing


def remove_lower_mapping_level(mapping_results: list) -> list:
    """
    Removes lower mapping levels for the same mapping code.
    """
    streamlined_results = {}
    for result in mapping_results:
        code = result["mapsToCode"]
        if code not in streamlined_results.keys():
            streamlined_results[code] = result
        elif result["mapsToLabel"] > streamlined_results[code]["mapsToLabel"]:
            streamlined_results[code] = result
    return list(streamlined_results.values())


def sort_mapping_results(mapping_results: list) -> list:
    """
    Sorts mapping results according to mapping level.
    """
    new_mapping_results = {}
    for result in mapping_results:
        mapping_level = result["mappingLevel"]
        new_mapping_results[mapping_level] = result
    new_mapping_results = {k: v for k, v in sorted(new_mapping_results.items())}
    return list(new_mapping_results.values())


def prepare_mapping_results_for_summary(mapping_results: list) -> list:
    """
    Preprocesses and structures mapping results for LLM to perform summarization tasks.
    """
    streamlined_results = remove_lower_mapping_level(mapping_results)
    sorted_results = sort_mapping_results(streamlined_results)
    return sorted_results


def predict_map_level_and_reason(pairing: dict, llm_client: BaseLLMClient, llm_type: str, model_name: str, prompt_mode: str) -> dict:
    model_to_prompt_files = {
        "zero-shot": {"prompt": Path(THIS_DIR, "prompt_templates/reasoning_and_summarisation/mapping_level/prompts_mapping_zeroShot.xlsx"),
                      "example": Path(THIS_DIR, "prompt_templates/reasoning_and_summarisation/mapping_level/examples_mapping_zeroShot.xlsx")},
        "few-shot": {"prompt": Path(THIS_DIR, "prompt_templates/reasoning_and_summarisation/mapping_level/prompts_mapping_fewShot.xlsx"),
                     "example": Path(THIS_DIR, "prompt_templates/reasoning_and_summarisation/mapping_level/examples_mapping_fewShot.xlsx")},
        "few-shot-enhanced": {"prompt": Path(THIS_DIR, "prompt_templates/reasoning_and_summarisation/mapping_level/prompts_mapping_fewShotEnhanced.xlsx"),
                              "example": Path(THIS_DIR, "prompt_templates/reasoning_and_summarisation/mapping_level/examples_mapping_fewShotEnhanced.xlsx")},
        "cot": {"prompt": Path(THIS_DIR, "prompt_templates/reasoning_and_summarisation/mapping_level/prompts_mapping_fewShotCoT.xlsx"),
                "example": Path(THIS_DIR, "prompt_templates/reasoning_and_summarisation/mapping_level/examples_mapping_fewShotCoT.xlsx")},
    }
    prompt_file_mapping_level = model_to_prompt_files[prompt_mode]["prompt"]
    prompt_file_reasoning = Path(THIS_DIR, "prompt_templates/reasoning_and_summarisation/reasoning/prompts_reasoning.xlsx")
    example_file_mapping_level = model_to_prompt_files[prompt_mode]["example"]
    example_file_reasoning = Path(THIS_DIR, "prompt_templates/reasoning_and_summarisation/reasoning/examples_reasoning.xlsx")
    
    for pair in pairing.values():
        for mapsTo in pair["mapsTo"]:

            # Calls LLM to evaluate mapping levels
            if "mapsToLabel" in mapsTo.keys():
                input_text = str({"original_label": pair["mapsFromLabel"], "mapped_label": mapsTo["mapsToLabel"]})
                input_pair = InputDoc(doc_type='text', doc_text=input_text)
                answer_mapping = ""
                
                # Prompts llm to predict mapping level. The while loop will be executed and continue until the response is valid.
                while not is_valid_json_output(answer_mapping, "mapping_level"):
                    answer_mapping = call_llm(llm=llm_client,
                                              prompt_file=prompt_file_mapping_level,
                                              input_question=input_pair,
                                              example_file=example_file_mapping_level,
                                              llm_type=llm_type)
                
                    json_start = answer_mapping.find('{')
                    json_end = answer_mapping.find('}')
                    answer_mapping = answer_mapping[json_start:json_end+1]

                # Calls LLM to perform reasoning
                mapping_level = eval(answer_mapping[json_start:json_end+1])["mapping_level"]
                input_pair.doc_text = {"original_label": pair["mapsFromLabel"], "mapped_label": mapsTo["mapsToLabel"], "mapping_level": mapping_level}
                answer_reasoning = ""
                while not is_valid_json_output(answer_reasoning, "reasoning"):
                    answer_reasoning = call_llm(llm=llm_client,
                                                prompt_file=prompt_file_reasoning,
                                                input_question=input_pair,
                                                example_file=example_file_reasoning,
                                                llm_type=llm_type)
                    json_start = answer_reasoning.find('{')
                    json_end = answer_reasoning.find('}')

                    # Post-processes summarization output from llama-3
                    if model_name == "llama-3":
                        sys_word_loc = answer_reasoning.find("assistant")
                        if sys_word_loc != -1:
                            reasoning_dict = {"reasoning": answer_reasoning[:sys_word_loc]}
                            answer_reasoning = str(reasoning_dict)
                    elif json_start != -1 and json_end != -1:
                        answer_reasoning = answer_reasoning[json_start:json_end+1]

                mapsTo["mappingLevel"] = mapping_level
                mapsTo["reasoning"] = eval(answer_reasoning)["reasoning"]
            else: 
                mapsTo["mapsToLabel"] = "Not found in the graph."
                mapsTo["mappingLevel"] = "N/A"
                mapsTo["reasoning"] = "N/A"

    # Processes final results            
    for code, pair in pairing.items():
        logger.debug(f"Code: {code}")
        logger.debug(f"Label: {pair['mapsFromLabel']}")
        logger.debug(pd.DataFrame(pair["mapsTo"]).head())
        pair["mapsTo"] = prepare_mapping_results_for_summary(pair["mapsTo"])
    
    return pairing


def summarise(input_question: InputDoc, pairing: dict, llm_client: BaseLLMClient, llm_type: str) -> str:
    prompt_file = Path(THIS_DIR, "prompt_templates/reasoning_and_summarisation/summarisation/prompts_summary.xlsx")
    example_file = Path(THIS_DIR, "prompt_templates/reasoning_and_summarisation/summarisation/examples_summary.xlsx")
    for pair in pairing.values():
        input_question.doc_text += str(pair)

    # Processes prompts and calls LLM for summary
    answer_summary = call_llm(llm=llm_client,
                              prompt_file=prompt_file,
                              input_question=input_question,
                              example_file=example_file,
                              llm_type=llm_type)
    
    # Post-processes summarization output from llama-3
    sys_word_loc = answer_summary.find("assistant")
    if sys_word_loc != -1:
        answer_summary = answer_summary[:sys_word_loc]
    return answer_summary


def run_reasoning_n_summarisation(input_question: InputDoc, pairing: dict, llm_client: BaseLLMClient, llm_type: str,
                                  model_name: str, prompt_mode: str):
    pairing_with_reasoning = predict_map_level_and_reason(pairing, llm_client, llm_type, model_name, prompt_mode)
    return summarise(input_question, pairing_with_reasoning, llm_client, llm_type)


def run_retrieval(input_question: Union[str, list], graph_data: Union[str, Path], model_name: str, prompt_mode: str,
                  temperature: str):
    logger.info(f"Received input question: {input_question}")
    input_question = InputDoc(doc_type="text", doc_text=input_question)
    graph_store = Store.read_only(str(graph_data))
    model_loader = ModelLoader(model_name=model_name, temperature=float(temperature))
    llm_client = model_loader.get_client()
    llm_type = model_loader.get_deployment_type()
    
    if llm_is_ready(llm_client):
        
        logger.info(f"Model server is reachable. Model name: {model_name}; prompt mode: {prompt_mode}; temperature: {temperature}")

        # NL2SPARQL module
        logger.info("Start running NL2SPARQL, converting input question to SPARQL query ...")
        sparql_query = run_nl2sparql(input_question, graph_store, llm_client, llm_type)

        # Subgraph retrieval
        logger.info("Retrieving sub-graphs with generated SPARQL query ...")
        pairing = retrieve_subgraph(graph_store, sparql_query)

        # Reasoning & Summarisation module
        logger.info("Assessing retrieved results and generating summary ...")
        summary = run_reasoning_n_summarisation(input_question, pairing, llm_client, llm_type, model_name, prompt_mode)
        
        logger.info("Congratulations! All finished.")

        return summary
    
    else:
        logger.debug("Model server is unreachable. Stop process.")
