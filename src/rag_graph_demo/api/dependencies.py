from fastapi import Request

from rag_graph_demo.services.container import AppContainer


def get_container(request: Request) -> AppContainer:
    return request.app.state.container

