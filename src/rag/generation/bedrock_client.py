import os

import boto3
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Model IDs read from env (see .env.example) rather than hardcoded, since
# exact Bedrock model IDs vary by region/account and change over time.
MODEL_IDS = {
    "haiku": os.environ.get("BEDROCK_HAIKU_MODEL_ID"),
    "gpt_oss": os.environ.get("BEDROCK_GPT_OSS_MODEL_ID"),
}

_client = None


def get_bedrock_client():
    """Lazily create and cache a boto3 bedrock-runtime client (same
    lazy-singleton pattern as get_model() in dense.py / _get_reranker() in
    rerank.py)."""


def generate_answer(query: str, context: str, model_id: str) -> str:
    """The generation-axis equivalent of retrieve_fn: call the given
    Bedrock model_id via the Converse API with (query, context) and return
    the generated answer text. The Converse API gives one unified
    request/response shape across vendors (Anthropic, OpenAI gpt-oss),
    so -- unlike invoke_model -- no per-vendor request/response branching
    is needed here. Pluggable by model_id so the eval harness can run the
    same gold set through haiku and gpt-oss and compare."""
