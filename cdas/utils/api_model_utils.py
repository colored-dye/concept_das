import asyncio
from openai import AsyncClient


class RemoteAPIModel(object):
    def __init__(self, model: str, client: AsyncClient, temperature: float = 1.0):
        self.model = model
        self.client = client
        self.temperature = temperature

    async def chat_completion(self, client: AsyncClient, prompt):
        # check if the prompt is cached
        raw_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=self.temperature,
            max_completion_tokens=500,
        )
        raw_completion = raw_completion.to_dict()

        # query sometimes returns None;
        # relaunch needed
        try:
            content = raw_completion["choices"][0]["message"]["content"]
        except Exception as e:
            print("Content is None!")
            raise ValueError(e)
        completion = self.normalize(content)

        usage = raw_completion["usage"]
        return (completion, usage)

    async def chat_completions(self, prompts, batch_size=32):
        """handling batched async calls with internal batching mechanism"""
        # Ensure api_names is a list of appropriate length
        # Process in batches
        all_completions = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]

            # batched calls
            async_responses = [
                self.chat_completion(self.client, prompt) for prompt in batch_prompts
            ]
            raw_completions = await asyncio.gather(*async_responses)
            # post handling for current batch
            for j, (completion, usage) in enumerate(raw_completions):
                all_completions.append(completion)

        return all_completions

    def normalize(self, text):
        return text.strip()