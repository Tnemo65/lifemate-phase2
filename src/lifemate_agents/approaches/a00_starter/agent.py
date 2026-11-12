"""Starter text Agent. Replace the no-action policy with your tool policy."""
import json
import os
import urllib.request

from lifemate import llm, span


class Agent:
    def __init__(self, config):
        self.config = config

    def setup(self):
        self.model = self.config.get("llm", {}).get("model", "")

    def run(self, sample, env):
        with span("prepare request", input=sample.transcript) as step:
            messages = [{"role": "user", "content": sample.transcript}]
            step["output"] = {"installed_apps": len(sample.apps)}
        with span("choose policy", input={"model": self.model}) as step:
            step["output"] = "model" if self.model else "no action"
        if not self.model:
            return {"decision": "Starter policy makes no tool calls"}

        # Optional real inference, enabled only by a user-configured model endpoint.
        settings = self.config["llm"]
        payload = {"model": self.model, "messages": messages, "temperature": 0, "seed": 0,
                   "max_tokens": settings.get("max_tokens", 128)}
        headers = {"Content-Type": "application/json"}
        if token := os.environ.get("OPENAI_API_KEY"):
            headers["Authorization"] = "Bearer " + token
        request = urllib.request.Request(settings["base_url"].rstrip("/") + "/chat/completions",
                                         data=json.dumps(payload).encode(), headers=headers, method="POST")
        with llm(self.model, prompt=messages) as call:
            with urllib.request.urlopen(request, timeout=settings.get("timeout", 25)) as response:
                result = json.load(response)
            call["output"] = result["choices"][0]["message"]
            usage = result.get("usage") or {}
            call["token_counts"] = {"input": usage.get("prompt_tokens"), "output": usage.get("completion_tokens")}
        return {"model_output": result["choices"][0]["message"]}
