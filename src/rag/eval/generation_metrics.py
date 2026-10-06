from rag.generation.bedrock_client import get_bedrock_client

import re

SCORE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)")

def _ask_judge(prompt: str, judge_model_id: str) -> str:
    """Send a judge prompt to judge_model_id via the Converse API and
    return the raw text response. Shared by faithfulness/answer_relevance
    so they don't each duplicate the Converse call and content-block
    parsing (same block-scanning logic as generate_answer)."""
    client = get_bedrock_client()
    response = client.converse(
        modelId=judge_model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
    )
    for message in response["output"]["message"]["content"]:
        if "text" in message:
            return message["text"]
    raise ValueError(f"No text content block in judge Converse response: {response}")



def _parse_score(judge_response: str) -> float:
    """Extract a 0-1 numeric score from the judge model's raw text
    response (e.g. "0.8" or "Score: 0.8 - mostly supported"), clamped
    into the valid [0, 1] range."""
    match = SCORE_PATTERN.search(judge_response)
    if not match:
        raise ValueError(f"Could not parse numeric score from judge response: {judge_response}")
    score = float(match.group(1))
    return max(0.0, min(1.0, score))

def faithfulness(context: str, answer: str, judge_model_id: str) -> float:
    """LLM-as-judge metric: ask judge_model_id whether the claims in
    `answer` are actually supported by `context`, and return a 0-1 score
    (1.0 = fully grounded in the retrieved context, 0.0 = unsupported/
    hallucinated). Naive/hand-rolled, like retrieval_metrics.py -- no
    RAGAS/DeepEval dependency, so the judge prompt and scoring logic are
    fully inspectable rather than a black box."""
    prompt = (
        "You are evaluating whether an answer is faithful to its source context.\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:\n{answer}\n\n"
        "On a scale from 0.0 to 1.0, how well is every claim in the answer "
        "supported by the context? Respond with only a decimal number "
        "between 0.0 and 1.0."
    )
    judge_response = _ask_judge(prompt, judge_model_id)
    return _parse_score(judge_response)


def answer_correctness(reference_answer: str, answer: str, judge_model_id: str) -> float:
    """LLM-as-judge metric: ask judge_model_id whether `answer` conveys
    the same actual information as `reference_answer` (the gold set's
    hand-written correct answer), and return a 0-1 correctness score.
    Unlike faithfulness/answer_relevance, this is NOT reference-free --
    it directly checks the generated answer against ground truth, which
    is what catches a retrieval failure that faithfulness/relevance alone
    would miss (a model can be fully faithful to the wrong retrieved
    context and still be flatly incorrect)."""
    prompt = (
        "You are evaluating whether an answer conveys the same information "
        "as a reference answer.\n\n"
        f"Reference Answer:\n{reference_answer}\n\n"
        f"Answer:\n{answer}\n\n"
        "On a scale from 0.0 to 1.0, how well does the answer convey the "
        "same information as the reference answer? Respond with only a "
        "decimal number between 0.0 and 1.0."
    )
    judge_response = _ask_judge(prompt, judge_model_id)
    return _parse_score(judge_response)


def answer_relevance(query: str, answer: str, judge_model_id: str) -> float:
    """LLM-as-judge metric: ask judge_model_id how directly `answer`
    addresses `query`, and return a 0-1 relevance score. Distinct from
    faithfulness -- an answer can be fully grounded in the retrieved
    context while still being off-topic or incomplete relative to what
    was actually asked."""
    prompt = (
        "You are evaluating whether an answer directly addresses a question, "
        "regardless of whether the answer's claims are factually correct.\n\n"
        f"Question:\n{query}\n\n"
        f"Answer:\n{answer}\n\n"
        "On a scale from 0.0 to 1.0, how directly and completely does the "
        "answer address the question? Respond with only a decimal number "
        "between 0.0 and 1.0."
    )
    judge_response = _ask_judge(prompt, judge_model_id)
    return _parse_score(judge_response)
