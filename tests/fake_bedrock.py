"""Stand-in for BedrockClient in API tests. Defaults to "not configured", like a fresh install."""

from app.llm.bedrock import BedrockError
from app.llm.messages import ChatMessage

NOT_SET = "AWS keys are not set (AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env)"


class FakeBedrock:
    def __init__(
        self,
        configured: bool = False,
        nova_reply: str = "Red",
        qwen_reply: str = "Red.",
        error: str | None = None,
    ):
        self.region = "us-east-1"
        self.configured = configured
        self.nova_reply = nova_reply
        self.qwen_reply = qwen_reply
        self.error = error
        self.calls: list[tuple[str, list[ChatMessage], dict]] = []

    def nova_problem(self) -> str | None:
        return None if self.configured else NOT_SET

    def qwen_problem(self) -> str | None:
        return None if self.configured else NOT_SET

    async def nova(self, messages: list[ChatMessage], **kwargs) -> str:
        return self._answer("nova", messages, kwargs, self.nova_reply)

    async def qwen(self, messages: list[ChatMessage], **kwargs) -> str:
        return self._answer("qwen", messages, kwargs, self.qwen_reply)

    def _answer(self, model: str, messages, kwargs, reply: str) -> str:
        self.calls.append((model, messages, kwargs))
        if self.error:
            raise BedrockError(self.error)
        return reply
