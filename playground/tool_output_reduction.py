import logging
import time

from pyrate_limiter import Limiter, Rate, BucketFullException
from strands.experimental.hooks import BeforeToolInvocationEvent
from strands.hooks import HookProvider, HookRegistry, AfterToolCallEvent
from strands.types.tools import ToolResultContent

MAX_CHARS = 100_000
suppressed: ToolResultContent = {
    "text": "[Output suppressed due to rate limiting]"
}

class OutputLimitHook(HookProvider):

    def __init__(self):
        self.rate_limit = Limiter(Rate(10, 60))

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(
            event_type=AfterToolCallEvent,
            callback=self.after_call
        )
        registry.add_callback(
            event_type=BeforeToolInvocationEvent,
            callback=self.before_call
        )

    def before_call(self, event: BeforeToolInvocationEvent) -> None:
        logging.info(event.invocation_state)

    def after_call(self, event: AfterToolCallEvent) -> None:
        #time.sleep(3)
        return
        result = event.result
        content = result.get("content", [])

        if not content:
            return

        for block in content:
            if "text" not in block:
                continue

            text = block["text"]
            if len(text) > MAX_CHARS:
                logging.info(
                    f"Truncating browse output "
                    f"({len(text)} → {MAX_CHARS})"
                )
                block["text"] = text[:MAX_CHARS] + "\n\n[Truncated]"
