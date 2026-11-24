import os
import time

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Model IDs read from env (see .env.example) rather than hardcoded, since
# exact Bedrock model IDs vary by region/account and change over time.
MODEL_IDS = {
    "haiku": os.environ.get("BEDROCK_HAIKU_MODEL_ID"),
    "gpt_oss": os.environ.get("BEDROCK_GPT_OSS_MODEL_ID"),
}

# This account's on-demand Bedrock quota throttles hard enough that
# boto3's built-in adaptive retry (even at 10 attempts) isn't enough to
# ride it out during a full eval run -- a full gold-set run makes far
# more Converse calls per minute than casual use, so every call is
# deliberately paced this far apart up front rather than bursting and
# hoping retries absorb the overflow.
MIN_SECONDS_BETWEEN_CALLS = 12
_last_call_time = 0.0

_client = None


def get_bedrock_client():
    """Lazily create and cache a boto3 bedrock-runtime client (same
    lazy-singleton pattern as get_model() in dense.py / _get_reranker() in
    rerank.py)."""
    global _client
    if _client is None:
        _client = boto3.client(
            "bedrock-runtime",
            region_name=AWS_REGION,
            config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),
        )
    return _client


def converse_paced(model_id: str, prompt: str) -> str:
    """Shared Converse call used by both generate_answer and the judge
    metrics in generation_metrics.py, so every Bedrock call anywhere in
    an eval run -- not just generate_answer's -- respects the same
    account-wide pacing and retries on a lingering ThrottlingException
    with a long sleep rather than failing the whole run."""
    global _last_call_time
    client = get_bedrock_client()

    elapsed = time.monotonic() - _last_call_time
    if elapsed < MIN_SECONDS_BETWEEN_CALLS:
        time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)

    while True:
        try:
            response = client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
            )
            break
        except ClientError as e:
            if e.response["Error"]["Code"] != "ThrottlingException":
                raise
            time.sleep(MIN_SECONDS_BETWEEN_CALLS * 2)
        finally:
            _last_call_time = time.monotonic()

    for block in response["output"]["message"]["content"]:
        if "text" in block:
            return block["text"]
    raise ValueError(f"No text content block in Converse response: {response}")


def generate_answer(query: str, context: str, model_id: str) -> str:
    """The generation-axis equivalent of retrieve_fn: call the given
    Bedrock model_id via the Converse API with (query, context) and return
    the generated answer text. The Converse API gives one unified
    request/response shape across vendors (Anthropic, OpenAI gpt-oss),
    so -- unlike invoke_model -- no per-vendor request/response branching
    is needed here. Pluggable by model_id so the eval harness can run the
    same gold set through haiku and gpt-oss and compare."""
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    return converse_paced(model_id, prompt)