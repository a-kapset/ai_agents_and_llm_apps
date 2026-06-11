import os
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from typing import TypedDict, Dict, Any, List, Annotated, Tuple
from models import ResearchState
from agents.assistant_selector import select_assistant
from agents.web_researcher import generate_search_queries, perform_web_searches, summarize_search_results, evaluate_search_relevance
from agents.report_writer import write_research_report

def create_research_graph() -> StateGraph:
    """
    Create the LangGraph research graph that coordinates the agents.
    """
    
    graph = StateGraph(ResearchState)
    
    graph.add_node('select_assistant', select_assistant)
    graph.add_node('generate_search_queries', generate_search_queries)
    graph.add_node('perform_web_searches', perform_web_searches)
    graph.add_node('summarize_search_results', summarize_search_results)
    graph.add_node('evaluate_search_relevance', evaluate_search_relevance)
    graph.add_node('write_research_report', write_research_report)
    
    # Define the conditional routing function for relevance evaluation
    def route_based_on_relevance(state: Dict[str, Any]) -> str:
        """
        Route to either generate new search queries or continue to report writing
        based on the relevance evaluation.
        """
        
        iteration_count = state.get('iteration_count', 0)
        new_iteration_count = iteration_count + 1
        state['iteration_count'] = new_iteration_count
        
        if new_iteration_count >= 3:
            print(f"Reached maximum iterations ({new_iteration_count}). Proceeding to write report with current results.")
            return "write_research_report"
        
        if state.get("should_regenerate_queries", False):
            print(f"Iteration {new_iteration_count}: Regenerating search queries.")
            return "generate_search_queries"
        else:
            print(f"Iteration {new_iteration_count}: Search results are relevant. Proceeding to write report.")
            return "write_research_report"                    
    
    graph.add_edge("select_assistant", "generate_search_queries")
    graph.add_edge("generate_search_queries", "perform_web_searches")
    graph.add_edge("perform_web_searches", "summarize_search_results")
    graph.add_edge("summarize_search_results", "evaluate_search_relevance")
    
    graph.add_conditional_edges(
        'evaluate_search_relevance',
        route_based_on_relevance,
        {
            "generate_search_queries": "generate_search_queries",
            "write_research_report": "write_research_report"            
        }
    )
    
    graph.add_edge('write_research_report', END)
    
    graph.set_entry_point('select_assistant')
    
    return graph


def run_research(question: str) -> str:
    """
    Run the research graph with a user question.
    
    Args:
        question: The user's research question
        
    Returns:
        The final research report
    """
    
    # Create the graph
    research_graph = create_research_graph()
    
    # Compile the graph
    app = research_graph.compile()
    
    # Initialize the state
    initial_state = {
        "user_question": question,
        "assistant_info": None,
        "search_queries": None,
        "search_results": None,
        "search_summaries": None,
        "research_summary": None,
        "final_report": None,
        "used_fallback_search": False,
        "relevance_evaluation": None,
        "should_regenerate_queries": None,
        "iteration_count": 0
    }
    
    # Run the graph
    result = app.invoke(initial_state)
    
    return result['final_report']


# For testing purposes
if __name__ == "__main__":
    # Example usage
    question = "What can you tell me about Astorga's roman spas"
    report = run_research(question)
    print(report)