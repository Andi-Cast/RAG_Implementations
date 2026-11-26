import os
import time

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

from rag.observability.tracing import get_langfuse, mask_text
from rag.security.pii_redaction import redact_pii

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

MODEL_IDS = {
    "haiku": os.environ.get("BEDROCK_HAIKU_MODEL_ID"),
    "gpt_oss": os.environ.get("BEDROCK_GPT_OSS_MODEL_ID"),
}

MIN_SECONDS_BETWEEN_CALLS = 12
_last_call_time = 0.0

_client = None


def get_bedrock_client():
    global _client
    if _client is None:
        _client = boto3.client(
            "bedrock-runtime",
            region_name=AWS_REGION,
            config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),
        )
    return _client


def converse_paced(model_id: str, prompt: str, name: str, temperature: float | None = None, trace_redact_fn=None) -> str:
    global _last_call_time
    client = get_bedrock_client()

    elapsed = time.monotonic() - _last_call_time
    if elapsed < MIN_SECONDS_BETWEEN_CALLS:
        time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)

    converse_kwargs = {
        "modelId": model_id,
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
    }
    if temperature is not None:
        converse_kwargs["inferenceConfig"] = {"temperature": temperature}

    langfuse = get_langfuse()
    with langfuse.start_as_current_observation(
        as_type="generation",
        name=name,
        model=model_id,
        input=[{"role": "user", "content": mask_text(prompt, 200)}],
    ) as gen:
        while True:
            try:
                response = client.converse(**converse_kwargs)
                break
            except ClientError as e:
                if e.response["Error"]["Code"] != "ThrottlingException":
                    raise
                time.sleep(MIN_SECONDS_BETWEEN_CALLS * 2)
            finally:
                _last_call_time = time.monotonic()

        answer = None
        for block in response["output"]["message"]["content"]:
            if "text" in block:
                answer = block["text"]
                break
        if answer is None:
            raise ValueError(f"No text content block in Converse response: {response}")

        usage = response.get("usage", {})
        traced_answer = trace_redact_fn(answer) if trace_redact_fn else answer
        gen.update(
            output={"role": "assistant", "content": mask_text(traced_answer, 200)},
            usage_details={
                "input": usage.get("inputTokens"),
                "output": usage.get("outputTokens"),
                "total": usage.get("totalTokens"),
            },
        )

    return answer


def generate_answer(query: str, context: str, model_id: str) -> str:
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    return converse_paced(model_id, prompt, name="generate-answer", trace_redact_fn=redact_pii)