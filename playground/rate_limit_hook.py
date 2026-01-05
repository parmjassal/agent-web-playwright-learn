import logging

import time
from pyrate_limiter import Limiter, Rate
from strands.hooks import HookProvider, HookRegistry, BeforeModelCallEvent


class RateLimitHook(HookProvider):

    def __init__(self):
        self.rate_limit = Limiter(Rate(1000, 60))

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(event_type=BeforeModelCallEvent, callback=self.before_call)

    def before_call(self, event: BeforeModelCallEvent) -> None:
        logging.info(f"Validating with limiter for making request {event}")
        while True:
            allowed = self.rate_limit.try_acquire("model", 1)
            if allowed:
                logging.info(f"Validated to make request {allowed}")
                return
            time.sleep(6)