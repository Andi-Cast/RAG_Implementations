from rag.generation.bedrock_client import converse_paced

import re

SCORE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)")

def _ask_judge(prompt: str, judge_model_id: str, name: str) -> str:
    return converse_paced(judge_model_id, prompt, name=name, temperature=0.0)



def _parse_score(judge_response: str) -> float:
    match = SCORE_PATTERN.search(judge_response)
    if not match:
        raise ValueError(f"Could not parse numeric score from judge response: {judge_response}")
    score = float(match.group(1))
    return max(0.0, min(1.0, score))

def faithfulness(context: str, answer: str, judge_model_id: str) -> float:
    prompt = (
        "You are evaluating whether an answer is faithful to its source context.\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:\n{answer}\n\n"
        "On a scale from 0.0 to 1.0, how well is every claim in the answer "
        "supported by the context? Respond with only a decimal number "
        "between 0.0 and 1.0."
    )
    judge_response = _ask_judge(prompt, judge_model_id, name="score-faithfulness")
    return _parse_score(judge_response)


def answer_correctness(reference_answer: str, answer: str, judge_model_id: str) -> float:
    prompt = (
        "You are evaluating whether an answer conveys the same information "
        "as a reference answer.\n\n"
        f"Reference Answer:\n{reference_answer}\n\n"
        f"Answer:\n{answer}\n\n"
        "On a scale from 0.0 to 1.0, how well does the answer convey the "
        "same information as the reference answer? Respond with only a "
        "decimal number between 0.0 and 1.0."
    )
    judge_response = _ask_judge(prompt, judge_model_id, name="score-answer-correctness")
    return _parse_score(judge_response)


def answer_relevance(query: str, answer: str, judge_model_id: str) -> float:
    prompt = (
        "You are evaluating whether an answer directly addresses a question, "
        "regardless of whether the answer's claims are factually correct.\n\n"
        f"Question:\n{query}\n\n"
        f"Answer:\n{answer}\n\n"
        "On a scale from 0.0 to 1.0, how directly and completely does the "
        "answer address the question? Respond with only a decimal number "
        "between 0.0 and 1.0."
    )
    judge_response = _ask_judge(prompt, judge_model_id, name="score-answer-relevance")
    return _parse_score(judge_response)
