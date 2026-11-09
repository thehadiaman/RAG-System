import time

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.deps import get_bedrock
from app.llm.bedrock import BedrockClient, BedrockError
from app.llm.messages import ChatMessage, ImageInput, solid_png

router = APIRouter(tags=["aws"])

CHECK_QUESTION = "What colour is this square? Answer with one word."
RED_SQUARE = ImageInput(solid_png(64, 64, (220, 30, 30)), "png")


class CheckResult(BaseModel):
    ok: bool  # the call worked
    configured: bool
    saw_red: bool | None = None  # the model described the test image correctly
    reply: str | None = None
    seconds: float | None = None
    error: str | None = None


class AwsCheckResponse(BaseModel):
    region: str
    nova_lite: CheckResult
    qwen_vl: CheckResult


async def _check(problem: str | None, ask) -> CheckResult:
    if problem:
        return CheckResult(ok=False, configured=False, error=problem)

    started = time.perf_counter()
    try:
        reply = await ask([ChatMessage("user", CHECK_QUESTION, images=(RED_SQUARE,))], max_tokens=10, temperature=0)
    except BedrockError as exc:
        return CheckResult(ok=False, configured=True, error=str(exc), seconds=_since(started))
    reply = reply.strip()
    return CheckResult(
        ok=True,
        configured=True,
        saw_red="red" in reply.lower(),
        reply=reply[:200],
        seconds=_since(started),
    )


def _since(started: float) -> float:
    return round(time.perf_counter() - started, 2)


@router.get("/aws/check")
async def check_aws(bedrock: BedrockClient = Depends(get_bedrock)) -> AwsCheckResponse:
    """Sends each configured AWS model a tiny question with a tiny image.

    Costs a fraction of a cent. The first Qwen call can take minutes if the
    imported model is asleep.
    """
    return AwsCheckResponse(
        region=bedrock.region,
        nova_lite=await _check(bedrock.nova_problem(), bedrock.nova),
        qwen_vl=await _check(bedrock.qwen_problem(), bedrock.qwen),
    )
