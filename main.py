import asyncio
import os

import pandas as pd
import tiktoken
import chainlit as cl
import nest_asyncio
from typing import Optional, Dict

# import asyncio

from graphrag.query.indexer_adapters import read_indexer_entities, read_indexer_reports
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.global_search.community_context import (
    GlobalCommunityContext,
)
from graphrag.query.structured_search.global_search.search import GlobalSearch
from interpreter import interpreter

from unittest import result

interpreter.os = True
interpreter.llm.supports_vision = True

interpreter.llm.model = "gpt-4o"

api_key = os.environ["GRAPHRAG_API_KEY"]
llm_model = os.environ["GRAPHRAG_LLM_MODEL"]

llm = ChatOpenAI(
    api_key=api_key,
    model=llm_model,
    api_type=OpenaiApiType.OpenAI,  # OpenaiApiType.OpenAI or OpenaiApiType.AzureOpenAI
    max_retries=20,
    # api_base="https://api.openai.com/v1/chat/completions"
)

token_encoder = tiktoken.get_encoding("cl100k_base")

# INPUT_DIR = "./input"
# COMMUNITY_REPORT_TABLE = "create_final_community_reports"
# ENTITY_TABLE = "create_final_nodes"
# ENTITY_EMBEDDING_TABLE = "create_final_entities"

# community level in the Leiden community hierarchy from which we will load the community reports
# higher value means we use reports from more fine-grained communities (at the cost of higher computation cost)
COMMUNITY_LEVEL = 2

# %%
entity_df = pd.read_parquet("./output/20240706-145118/artifacts/create_final_nodes.parquet")
report_df = pd.read_parquet("./output/20240706-145118/artifacts/create_final_community_reports.parquet")
entity_embedding_df = pd.read_parquet("output/20240706-145118/artifacts/create_final_entities.parquet")

reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
print(f"Report records: {len(report_df)}")
report_df.head()

# #### Build global context based on community reports
context_builder = GlobalCommunityContext(
    community_reports=reports,
    entities=entities,  # default to None if you don't want to use community weights for ranking
    token_encoder=token_encoder,
)

# #### Perform global search

context_builder_params = {
    "use_community_summary": False,  # False means using full community reports. True means using community short
    # summaries.
    "shuffle_data": True,
    "include_community_rank": True,
    "min_community_rank": 0,
    "community_rank_name": "rank",
    "include_community_weight": True,
    "community_weight_name": "occurrence weight",
    "normalize_community_weight": True,
    "max_tokens": 3_000,  # change this based on the token limit you have on your model (if you are using a model
    # with 8k limit, a good setting could be 5000)
    "context_name": "Reports",
}

map_llm_params = {
    "max_tokens": 1000,
    "temperature": 0.0,
    "response_format": {"type": "json_object"},
}

reduce_llm_params = {
    "max_tokens": 2000,  # change this based on the token limit you have on your model (if you are using a model with
    # 8k limit, a good setting could be 1000-1500)
    "temperature": 0.0,
}


async def agent_message():
    search_engine = GlobalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        max_data_tokens=12_000,  # change this based on the token limit you have on your model (if you are using a model
        # with 8k limit, a good setting could be 5000)
        map_llm_params=map_llm_params,
        reduce_llm_params=reduce_llm_params,
        allow_general_knowledge=False,  # set this to True will add instruction to encourage the LLM to incorporate
        # general knowledge in the response, which may increase hallucinations, but could be useful in some use cases.
        json_mode=True,  # set this to False if your LLM model does not support JSON mode.
        context_builder_params=context_builder_params,
        concurrent_coroutines=10,
        response_type="multiple-page report",
        # free form text describing the response type and format, can be anything,
        # e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
    )
    agent_task = f"""
        ROLE:
        As a seasoned agronomist/researcher , you have extensive knowledge of crop physiology, soil science, 
        and agricultural best practices. You excel at translating complex data into practical 
        advice for farmers to improve their crop yields and sustainability.
        Interpret the data analysis results and provide agronomic recommendations. 
        ____
        TASK:
        Focus on translating the identified trends into actionable advice for farmers. 
        Consider optimal planting times, irrigation strategies, fertilizer application, 
        and crop selection based on the historical data patterns. 
        Your recommendations should be practical, sustainable, and aimed at maximizing crop yield with with data reference.
        Make sure to provide a concise and clear, helpful and concise summary of your analysis and recommendations.
        ___
        QUESTION:
        Provide the data requirements necessary for analyzing tomato production using both historical and forecast soil 
        and weather data. The goal is to generate comprehensive insights and advisories for a complete crop calendar 
        with precise recommendations for key farming activities. The data requirements should cover:

        - Soil temperature ranges for different growth stages
        - Optimal weather conditions for each growth stage
        - Soil moisture levels and irrigation needs
        - Fertilizer application timing and rates
        - Pest and disease monitoring and control measures
        - Expected harvest dates based on growing conditions
        These data points should help in creating a detailed crop calendar and actionable insights to enhance 
        tomato production, mitigate risks, and improve overall crop yield and sustainability.
        """
    search_result = await search_engine.asearch(agent_task)

    interpreter.llm.supports_functions = True
    interpreter.llm.context_window = 110000
    interpreter.llm.max_tokens = 4096
    interpreter.auto_run = True
    interpreter.loop = True
    interpreter.system_message = ("""
                                   ROLE:
                                   Agricultural Data Analyst
                                   ___
                                   GOAL:
                                   Analyze historical weather and soil data to identify patterns and trends relevant to crop production.
                                   ___
                                   BACKSTORY:
                                   You are an expert in agricultural data analysis with a deep understanding of how weather and soil conditions affect crop growth. Your insights help farmers make data-driven decisions to optimize their crop production.
                                   """)

    interpreter.custom_instructions = """
                                    TASK DESCRIPTION:
                                    Analyze the historical weather and soil data for crop production. Focus on identifying significant patterns and correlations between variables. Pay special attention to how temperature, precipitation, humidity, wind speed, soil temperature, and soil moisture interact and affect potential crop growth. Your final report should clearly articulate the key trends, anomalies, and potential impact on different crop types.
                                    ___
                                    EXPECTED OUTPUT:
                                    A comprehensive 3-paragraph report on the key agricultural data trends and their implications.
    
                                    """

    message = f"""
    ################################
    # Agronomic Recommendations for Enhanced Tomato Production:
    {search_result.response}
    ################################
    Use this data https://raw.githubusercontent.com/Musbell/gis_data/main/data.csv to provide data
    using the historical data based on "Agronomic Recommendations for Enhanced Tomato Production".
    Include precised date and time with relevant data in reference to the requirements for decision making.
    """
    interpreter.chat(message)


asyncio.run(agent_message())
