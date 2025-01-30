import os
import supervisely as sly
from dotenv import load_dotenv

from src.cas import run_server, get_client, get_vectors
from fastapi import Request

load_dotenv(os.path.expanduser("~/supervisely.env"))
load_dotenv("local.env")


run_server()
client = get_client()
app = sly.Application()
server = app.get_server()


@server.post("/get_vectors")
async def get_vectors_endpoint(request: Request):
    context = request.state.context
    queries = context["queries"]
    return await get_vectors(client, queries)
