from research.web_provider import WebResearchProvider
from agent.analyzer import analyze


def run_pipeline(query: str, limit: int = 10) -> dict:
    provider = WebResearchProvider()
    products = provider.search(query, limit=limit)

    if not products:
        raise ValueError(f"No search results returned for query: '{query}'")

    return analyze([p.to_dict() for p in products])
