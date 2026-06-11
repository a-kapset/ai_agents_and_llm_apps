from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, List

class ResearchState(TypedDict):
    user_question: str
    assistant_info: Optional[dict]
    search_queries: Optional[List[dict]]
    search_results: Optional[List[dict]]
    research_summary: Optional[str]
    final_report: Optional[str]
    

graph = StateGraph(ResearchState)

def generate_search_queries(state: dict) -> dict:
    """Generate search queries based on user question."""
    question = state['user_question']
    queries = llm_generate_queries(question)
    
    return {'search_queries': queries}

graph.add_node('generate_queries', generate_search_queries)
graph.add_edge('generate_queries', 'perform_searches')

def should_refine_queries(state: dict) -> dict:
    if len(state['search_results']) < 2:
        return 'refine_queries'
    else:
        return 'summarize_results'

graph.add_conditional_edges('perform_searches', should_refine_queries)

graph.set_entry_point('parse_question')

graph.add_edge('write_final_report', END)