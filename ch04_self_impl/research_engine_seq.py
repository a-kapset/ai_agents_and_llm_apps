from web_searching import web_search
from web_scraping import web_scrape
from llm_models import get_llm
from utilities import to_obj
from prompts import (
    ASSISTANT_SELECTION_PROMPT_TEMPLATE,
    WEB_SEARCH_PROMPT_TEMPLATE,
    SUMMARY_PROMPT_TEMPLATE,
    RESEARCH_REPORT_PROMPT_TEMPLATE
)

NUM_SEARCH_QUERIES = 2
NUM_SEARCH_RESULTS_PER_QUERY = 3
RESULT_TEXT_MAX_CHARACTERS = 10000

question = 'What can I see and do in the Spanish town of Astorga?'

# Instantiating the LLM client
llm = get_llm()


#  Generating the web searches and collecting the results

assistant_selection_prompt = ASSISTANT_SELECTION_PROMPT_TEMPLATE.format(user_question=question)
assistant_instructions = llm.invoke(assistant_selection_prompt)
assistant_instructions_dict = to_obj(assistant_instructions.content)

web_search_prompt = WEB_SEARCH_PROMPT_TEMPLATE.format(
    assistant_instructions = assistant_instructions_dict['assistant_instructions'],
    num_search_queries = NUM_SEARCH_QUERIES,
    user_question = question
)

web_search_queries = llm.invoke(web_search_prompt)
web_search_queries_list = to_obj(web_search_queries.content.replace('\n',''))

searches_and_results_urls = [
    {
        'result_urls': web_search(web_query=wq['search_query'], num_results=NUM_SEARCH_RESULTS_PER_QUERY),
        'search_query': wq['search_query']
    }
    for wq in web_search_queries_list
]

web_search_query_and_result_url_list = []

for qr in searches_and_results_urls:
    web_search_query_and_result_url_list.extend(
        [
            {
                'search_query': qr['search_query'],
                'result_url': r
            }
            for r in qr['result_urls']
        ]
    )

    
# Scraping the web results

result_text_list = [
    {
        'result_text': web_scrape(url=res['result_url'])[:RESULT_TEXT_MAX_CHARACTERS],
        'result_url': res['result_url'],
        'search_query': res['search_query']
    }
    for res in web_search_query_and_result_url_list
]


# Summarizing the web results

result_text_summary_list = []

for rt in result_text_list:
    summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(
        search_result_text=rt['result_text'],
        search_query=rt['search_query']
    )

    text_summary = llm.invoke(summary_prompt)

    result_text_summary_list.append(
        {
            'text_summary': text_summary.content,
            'result_url': rt['result_url'],
            'search_query': rt['search_query']
        }
    )
    

# Generating the research report

stringified_summary_list = [
    f'Source URL: {sr["result_url"]}\nSummary: {sr["text_summary"]}' for sr in result_text_summary_list
]

appended_result_summaries = '\n'.join(stringified_summary_list)

research_report_prompt = RESEARCH_REPORT_PROMPT_TEMPLATE.format(
    research_summary=appended_result_summaries,
    user_question=question
)

research_report = llm.invoke(research_report_prompt)