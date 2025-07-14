"""
title: Civitai Image Generation Pipe
author: Your Name (inspired by examples)
description: >-
  Generates images using the Civitai.com API and displays them in the chat.
  This pipe allows for specifying a model, negative prompts, and additional networks like LoRAs.
version: 1.0.0
license: MIT
requirements: civitai-py, httpx, requests, pyyaml
environment_variables:
  - CIVITAI_API_TOKEN
"""

# TODO: add support for non streaming responses

import time
import copy
import yaml
import ast
import os
import json
import base64
import asyncio
import httpx
import traceback
import requests

import yaml.dumper
import civitai
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Callable, Awaitable, AsyncGenerator
import re

DEFAULT_MODEL = "urn:air:sdxl:checkpoint:civitai:827184@1761560"
DEFAULT_NEGATIVEPROMPT = "easynegative, badhandv4, (worst quality, low quality, normal quality), bad-artist, blurry, ugly, ((bad anatomy)),((bad hands)),((bad proportions)),((duplicate limbs)),((fused limbs)),((interlocking fingers)),((poorly drawn face)), signature, watermark, artist logo, patreon logo"
# DEFAULT_NEGATIVEPROMPT = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft"
GENERATION_PARAMS_DEFAULTS = {
    # "prompt": prompt,
    "negativePrompt": "",
    "scheduler": "EulerA",
    "steps": 25,
    "cfgScale": 4,
    "width": 100000,
    "height": 100000,
    "seed": None,
    "clipSkip": 2,
}


def urn_str2dict(urn: str) -> dict:
    """

    Args:
        urn: A URN string that may contain a strength modifier (e.g. "urn:air:sdxl:lora:civitai:1115064@1253021*0.5")

    Returns:
        dict: A dictionary containing the URN, type and strength information

    Example:
        >>> urn_str2dict("urn:air:sdxl:lora:civitai:1115064@1253021*0.5")
        {'urn:air:sdxl:lora:civitai:1115064@1253021': {'type': 'Lora', 'strength': 0.5}}

        $ urn_str2dict("urn:air:sdxl:lora:civitai:1115064@1253021*0.5")
        >>> {"urn:air:sdxl:lora:civitai:1115064@1253021": {"type": "Lora", "strength": 0.5}}

        $ urn_str2dict("urn:air:sdxl:embedding:civitai:1115064@1253021!easynegative")
        >>> {"urn:air:sdxl:embedding:civitai:1115064@1253021": {"type": "TextualInversion", "strength": 1.0, "triggerWord": "easynegative"}}

        $ urn_str2dict("urn:air:sdxl:embedding:civitai:1115064@1253021*1.7!easynegative")
        >>> {"urn:air:sdxl:embedding:civitai:1115064@1253021": {"type": "TextualInversion", "strength": 1.7, "triggerWord": "easynegative"}}

        # in the cas of a url, we need to convert it to a urn using the civitai_url_to_urn function
        $ urn_str2dict("https://civitai.com/models/341353?modelVersionId=382152")
        >>> {"urn:air:sdxl:lora:civitai:341353@382152": {"type": "Lora", "strength": 1.0}}

    """
    additionalNetworks_types_map = {
        "TextualInversion": "embedding",
        "Lora": "lora",
    }

    if not urn:
        return {}

    if urn.startswith("https://civitai.com/models/"):
        urn = civitai_url_to_urn(urn)

    trigger_word = None
    if "!" in urn:
        urn, trigger_word = urn.strip().split("!", 1)
        trigger_word = trigger_word.strip()

    # Split URN and strength if present
    strength = 1.0
    if "*" in urn:
        urn_part, strength = urn.split("*")
        strength = float(strength.strip().strip(","))
    else:
        urn_part = urn

    # Determine type from URN
    network_type = urn_part.split(":")[3]  # Get 'lora' or 'embedding' from URN
    type_name = next(
        (k for k, v in additionalNetworks_types_map.items() if v == network_type),
        "TextualInversion",  # Default to TextualInversion if not found
    )

    result = {
        urn_part: {
            "type": type_name,
            "strength": strength,
        }
    }
    if trigger_word:
        result[urn_part]["triggerWord"] = trigger_word
    return result


def parse_seed(seed: str) -> int:
    """
    Parses a seed string into an integer.
    """
    if seed in ["", "None", "null", "undefined"]:
        return None
    return int(seed)


# TODO:
def civitai_url_to_urn(url: str) -> str:
    model_id = url.split("/models/")[1].split("/")[0]
    model_version_id = (
        url.split("modelVersionId=")[1].split("&")[0]
        if "modelVersionId=" in url
        else None
    )
    response = requests.get(f"https://civitai.com/api/v1/models/{model_id}")
    response.raise_for_status()
    model_data = response.json()
    model_type = model_data["type"]
    model_version_id = model_data["modelVersionId"]
    urn = f"urn:air:{model_type}:civitai:{model_id}@{model_version_id}"
    return urn


def civitai_urn_to_url(urn: str) -> str:
    """
    Converts a Civitai URN to a web URL.

    Args:
        urn: Civitai URN in format like "urn:air:sdxl:lora:civitai:212532@239420"

    Returns:
        str: The corresponding Civitai web URL

    Example:
        >>> civitai_urn_to_url("urn:air:sdxl:lora:civitai:212532@239420")
        "https://civitai.com/models/212532?modelVersionId=239420"
    """
    if not urn.startswith("urn:air:"):
        raise ValueError(f"Invalid URN format: {urn}")

    # Split URN parts: urn:air:sdxl:lora:civitai:212532@239420
    parts = urn.split(":")
    if len(parts) < 6 or parts[4] != "civitai":
        raise ValueError(f"Invalid Civitai URN format: {urn}")

    # Extract model ID and version ID from the last part
    model_part = parts[5]  # "212532@239420"
    if "@" not in model_part:
        raise ValueError(f"Invalid model part in URN: {model_part}")

    model_id, version_id = model_part.split("@", 1)

    return f"https://civitai.com/models/{model_id}?modelVersionId={version_id}"


def parse_prompt_components(user_input):
    other_metadata_possible_keys = [
        "Steps:",
        "CFG scale:",
        "Sampler:",
        "Seed:",
        "Size:",
        "Model:",
        "width:",
        "height:",
        "draft:",
        "baseModel:",
        "disablePoi:",
        "aspectRatio:",
        "Created Date:",
        "experimental:",
        "Clip skip:",
        "process:",
        "workflow:",
        "quantity:",
    ]

    def split_off_prompt(text):
        triggers = [
            "Negative prompt:",
            "Additional networks:",
        ] + other_metadata_possible_keys
        pattern = re.compile("|".join(map(re.escape, triggers)))
        match = pattern.search(text)
        if not match:
            return text, ""
        return text[: match.start()].rstrip(", \n"), text[match.start() :]

    def grab_section(label, text, all_triggers):
        pattern_str = "|".join(re.escape(t) for t in all_triggers if t != label)
        regex = re.compile(
            rf"{re.escape(label)}\s*(?P<content>.*?)(?=(?:{pattern_str})|$)",
            flags=re.DOTALL,
        )
        m = regex.search(text)
        if not m:
            return "", text
        content = m.group("content")
        start, end = m.span()
        new_text = text[:start] + text[end:]
        return content.strip(), new_text.strip()

    user_input = user_input.strip()
    all_triggers = [
        "Negative prompt:",
        "Additional networks:",
    ] + other_metadata_possible_keys

    prompt, remainder = split_off_prompt(user_input)
    negative_prompt, remainder = grab_section(
        "Negative prompt:", remainder, all_triggers
    )
    additional_networks, remainder = grab_section(
        "Additional networks:", remainder, all_triggers
    )
    other_metadata = remainder.strip()

    return {
        "prompt": prompt,
        "negativePrompt": negative_prompt,
        "additionalNetworks": additional_networks,
        "other_metadata": other_metadata,
    }


def parse_generation_data(input_string: str) -> Dict[str, Any]:
    """
    Parse a Civitai prompt block into its logical components using regex.

    Args:
        input_string (str): The input prompt string containing prompt, negative prompt and metadata

    Returns:
        Dict[str, Any]: Dictionary containing:
            - prompt (str): The main prompt text
            - negativePrompt (str): The negative prompt text (empty if not found)
            - additionalNetworks (Dict[str, Any]): Dictionary of additional networks (empty if not found)
            - other_metadata (Dict[str, Any]): Dictionary of additional metadata parameters (empty if not found)
    """

    def convert_sentence_to_camel_case(sentence: str) -> str:
        words = sentence.lower().strip().split(" ")
        return words[0] + "".join(word.capitalize() for word in words[1:])

    component_strings = parse_prompt_components(input_string)

    # Parse additional networks into list then convert to dict
    additional_networks_list = []
    if component_strings["additionalNetworks"]:
        additional_networks_list = [
            item.strip()
            for item in re.split(r",\s*", component_strings["additionalNetworks"])
            if item.strip()
        ]

    additionalNetworks = {}
    for urn in additional_networks_list:
        urn_dict = urn_str2dict(urn.strip())
        additionalNetworks.update(urn_dict)

    # Merge metadata lines and break into individual key-value pairs

    # Clean up metadata text by removing Created Date which contains colons
    metadata_text = re.sub(
        r"Created Date:[^,]*,?\s*", "", component_strings["other_metadata"]
    ).strip()
    metadata = {}
    for part in re.split(r",\s*", metadata_text):
        if ":" not in part:
            continue

        key, value = part.split(":", 1)
        key = convert_sentence_to_camel_case(key.strip())
        value = value.strip().replace(" ", "")

        # Convert values to appropriate types
        # TODO: try removing the conversion and see if the api still works
        if value.replace(".", "").replace("-", "").isdigit():
            metadata[key] = float(value) if "." in value else int(value)
        elif value.lower() in ("true", "false"):
            metadata[key] = value.lower() == "true"
        else:
            metadata[key] = value

    return {
        "prompt": component_strings["prompt"],
        "negativePrompt": component_strings["negativePrompt"],
        "additionalNetworks": additionalNetworks,
        "other_metadata": metadata,
    }


def parse_inputstring_to_generation_input(input_string: str) -> dict:
    generation_data = parse_generation_data(input_string)
    print(f"generation_data={json.dumps(generation_data, indent=4)}")

    # generation_data = parse_generation_data(prompt)
    METADATA_WHITELIST = [
        "prompt",
        "negativePrompt",
        "width",
        "height",
        "scheduler",
        "steps",
        "cfgScale",
        "seed",
        "clipSkip",
    ]
    params = copy.deepcopy(GENERATION_PARAMS_DEFAULTS)
    for k, v in generation_data["other_metadata"].items():
        if not not k in METADATA_WHITELIST:
            params[k] = v
    params["prompt"] = generation_data["prompt"]
    params["negativePrompt"] = generation_data["negativePrompt"]

    generation_input = dict(
        model=generation_data["other_metadata"].get("model", DEFAULT_MODEL),
        params=params,
        additionalNetworks=generation_data["additionalNetworks"],
    )
    return generation_input


try:
    os.system("pip uninstall -y civitai")
except Exception as e:
    print("")

# Set the Civitai API token from environment variables
# Ensure you have CIVITAI_API_TOKEN set in your .env file for OpenWebUI
if "CIVITAI_API_TOKEN" in os.environ:
    civitai.api_key = os.environ["CIVITAI_API_TOKEN"]


class Pipe:
    """
    A pipe for generating images using the Civitai API.
    """

    class Valves(BaseModel):
        """
        Configuration settings for the Civitai pipe, managed through the OpenWebUI settings interface.
        """

        CIVITAI_API_TOKEN: str = Field(
            default="",
            description="Your Civitai API Key. It's recommended to set this via environment variables.",
        )
        USE_B64: bool = Field(
            default=True,
            description="If enabled, encodes the image to Base64 and embeds it directly in the chat. If disabled, uses the direct image URL. Note that the image may disappear later from the chat if you disable this.",
        )

    def __init__(self):
        self.id = "civitai_image_generator"
        self.name = "Civitai Image Generator"
        self.valves = self.Valves()
        self.emitter = None

        # If the API key is set in valves, prioritize it. Otherwise, it should rely on the module-level setup.
        if self.valves.CIVITAI_API_TOKEN:
            civitai.api_key = self.valves.CIVITAI_API_TOKEN

    def _get_last_user_message(self, messages: List[Dict[str, Any]]) -> str:
        """Extracts the text content from the last user message."""
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    for part in content:
                        if part.get("type") == "text":
                            return part.get("text", "")
        return ""

    async def _emit_status(self, description: str, done: bool = False):
        """Emits a status update to the OpenWebUI frontend."""
        if self.emitter:
            await self.emitter(
                {
                    "type": "status",
                    "data": {"description": description, "done": done},
                }
            )

    async def pipe(
        self,
        body: dict,
        __event_emitter__: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        The main execution method for the pipe. It receives the request body,
        generates an image via Civitai, and streams the result back.
        """
        self.emitter = __event_emitter__

        try:
            # 1. Validate API Key
            civitai.api_key = self.valves.CIVITAI_API_TOKEN or os.getenv(
                "CIVITAI_API_TOKEN"
            )
            if not civitai.api_key:
                yield "Error: CIVITAI_API_TOKEN is not configured. Please set it in the pipe settings or as an environment variable."
                await self._emit_status("Error: API token missing.", done=True)
                return

            # 2. Extract Prompt
            user_prompt = self._get_last_user_message(body.get("messages", []))
            if not user_prompt:
                yield "Error: Could not find a user prompt in the chat history."
                await self._emit_status("Error: No prompt found.", done=True)
                return

            yield f'🎨 Starting image generation for prompt: "{user_prompt}"\n\n'
            generation_input = parse_inputstring_to_generation_input(user_prompt)
            print(f"generation_input={json.dumps(generation_input, indent=4)}")

            yield f"""```yaml
# generation input
{yaml.dump(generation_input)}
```
"""

            # 4. Submit Job to Civitai
            await self._emit_status("Submitting job to Civitai...")
            yield "generation_input=" + json.dumps(generation_input, indent=4)
            with open(__file__ + "-generation_input.json", "w") as f:
                json.dump(generation_input, f, indent=4)
            response = await civitai.image.create(generation_input, wait=False)
            job_token = response["token"]
            job_id = response["jobs"][0]["jobId"]
            await self._emit_status(
                f"Job submitted (ID: {job_id}). Waiting for result..."
            )

            job_status = await civitai.jobs.get(token=job_token, job_id=job_id)
            image_info = job_status["jobs"][0].get("result", [{}])[0]
            # Polling timeout and retry logic
            POLLING_TIMEOUT = 120  # seconds
            POLLING_INTERVAL = 2  # seconds
            start_time = time.time()

            while not image_info.get("available"):
                elapsed_time = time.time() - start_time
                if elapsed_time >= POLLING_TIMEOUT:
                    yield f"Error: Job timed out after {POLLING_TIMEOUT} seconds."
                    await self._emit_status("Error: Job timed out.", done=True)
                    return

                await asyncio.sleep(POLLING_INTERVAL)
                job_status = await civitai.jobs.get(token=job_token, job_id=job_id)
                image_info = job_status["jobs"][0].get("result", [{}])[0]

                remaining_time = POLLING_TIMEOUT - elapsed_time
                await self._emit_status(
                    f"Job still processing... (timeout in {remaining_time:.0f}s)"
                )

            # Image is now available
            image_url = image_info.get("blobUrl")
            if not image_url:
                yield "Error: Job completed but no image URL was found."
                await self._emit_status("Error: Missing image URL.", done=True)
                return

            # 6. Download and Display Image
            if self.valves.USE_B64:
                await self._emit_status(
                    f"Image generated! Downloading and encoding... ![Generated Image]({image_url})"
                )
                async with httpx.AsyncClient() as client:
                    img_response = await client.get(image_url, timeout=120.0)
                    img_response.raise_for_status()

                content_type = img_response.headers.get("Content-Type", "image/jpeg")
                image_base64 = base64.b64encode(img_response.content).decode("utf-8")
                image_url = f"data:{content_type};base64,{image_base64}"
            else:
                await self._emit_status("Image generated! Using direct URL.")
            yield f"![Generated Image]({image_url})"
            await self._emit_status("Image generation complete!", done=True)
            return  # Success, exit the loop

        except Exception as e:
            error_message = f"An unexpected error occurred: {str(e)}"
            stack_trace = traceback.format_exc()
            print(f"Error in Civitai Pipe: {error_message}\n{stack_trace}")
            yield f"{error_message}\n\n```\n{stack_trace}\n```"
            await self._emit_status(f"Error: {str(e)}", done=True)


async def main():
    xyz = """
score_9,score_8_up,score_7_up,
1 tanned curvy italian girl, hot girl, round face, thin nose, long hair, hair in pigtails , brown hair, (oversized huge breasts), sexy face, narrow face, makeup,
Cute pajamas top , pajamas top
pulled over midriff , lifting shirt, pierced belly button , close up view, on knees, knees spread wide, all fours, ass up, round ass,
In bedroom, frat house BREAK on bed,
By night, pajamas down, anus, pussy, (puffy labia:1.4), beautiful woman, intimate photo, realistic image, dim light, cozy atmosphere, perfect lighting, perfect shadows, shiny skin, perfect reflections, lots of shadows
Negative prompt: easynegative, badhandv4, (worst quality, low quality, normal quality), bad-artist, blurry, ugly, ((bad anatomy)),((bad hands)),((bad proportions)),((duplicate limbs)),((fused limbs)),((interlocking fingers)),((poorly drawn face)), signature, watermark, artist logo, patreon logo
Additional networks: urn:air:sd1:embedding:civitai:7808@9208*1.0!easynegative, urn:air:sdxl:lora:civitai:1115064@1253021*3.3, urn:air:sdxl:lora:civitai:212532@239420*1.0, urn:air:sdxl:lora:civitai:341353@382152*0.8, urn:air:sdxl:lora:civitai:888231@398847*0.45, urn:air:sdxl:lora:civitai:300005@436219*0.25
Steps: 25, CFG scale: 4, Sampler: Euler a, Seed: 1594260453, process: txt2img, workflow: txt2img, Size: 1024x1024, draft: false, width: 1024, height: 1024, quantity: 1, baseModel: Illustrious, disablePoi: true, aspectRatio: 1:1, Created Date: 2025-07-05T1255:24.7399976Z, experimental: false, Clip skip: 2, Model: urn:air:sdxl:checkpoint:civitai:257749@290640
"""
    pipe = Pipe()
    generator = pipe.pipe(dict(messages=[{"role": "user", "content": xyz}]))
    with open(__file__ + "-output.md", "w") as f:
        async for chunk in generator:
            f.write(chunk)

    generation_input = parse_inputstring_to_generation_input(xyz)
    with open(__file__ + "-generation_input.json", "w") as f:
        json.dump(generation_input, f, indent=4)

    import deepdiff

    GT = {
        "model": "urn:air:sdxl:checkpoint:civitai:257749@290640",
        "params": {
            "negativePrompt": "easynegative, badhandv4, (worst quality, low quality, normal quality), bad-artist, blurry, ugly, ((bad anatomy)),((bad hands)),((bad proportions)),((duplicate limbs)),((fused limbs)),((interlocking fingers)),((poorly drawn face)), signature, watermark, artist logo, patreon logo",
            "scheduler": "EulerA",
            "steps": 25,
            "cfgScale": 4,
            "width": 1024,
            "height": 1024,
            "seed": 1594260453,
            "clipSkip": 2,
            "prompt": "score_9,score_8_up,score_7_up,\n1 tanned curvy italian girl, hot girl, round face, thin nose, long hair, hair in pigtails , brown hair, (oversized huge breasts), sexy face, narrow face, makeup,\nCute pajamas top , pajamas top\npulled over midriff , lifting shirt, pierced belly button , close up view, on knees, knees spread wide, all fours, ass up, round ass,\nIn bedroom, frat house BREAK on bed,\nBy night, pajamas down, anus, pussy, (puffy labia:1.4), beautiful woman, intimate photo, realistic image, dim light, cozy atmosphere, perfect lighting, perfect shadows, shiny skin, perfect reflections, lots of shadows",
        },
        "additionalNetworks": {
            "urn:air:sd1:embedding:civitai:7808@9208": {
                "type": "TextualInversion",
                "strength": 1.0,
                "triggerWord": "easynegative",
            },
            "urn:air:sdxl:lora:civitai:1115064@1253021": {
                "type": "Lora",
                "strength": 3.3,
            },
            "urn:air:sdxl:lora:civitai:212532@239420": {
                "type": "Lora",
                "strength": 1.0,
            },
            "urn:air:sdxl:lora:civitai:341353@382152": {
                "type": "Lora",
                "strength": 0.8,
            },
            "urn:air:sdxl:lora:civitai:888231@398847": {
                "type": "Lora",
                "strength": 0.45,
            },
            "urn:air:sdxl:lora:civitai:300005@436219": {
                "type": "Lora",
                "strength": 0.25,
            },
        },
    }
    diff = deepdiff.DeepDiff(generation_input, GT)
    assert (
        not diff
    ), f"generation_input: {json.dumps(generation_input, indent=4)}\nGT: {json.dumps(GT, indent=4)}\ndiff: {json.dumps(diff, indent=4)}"

    # ================ test cases ================

    prompt = """
    score_9,score_8_up,score_7_up,
    1 tanned curvy italian girl, hot girl, round face, thin nose, long hair, hair in pigtails , brown hair, (oversized huge breasts), sexy face, narrow face, makeup,
    Cute pajamas top , pajamas top
    pulled over midriff , lifting shirt, pierced belly button , close up view, on knees, knees spread wide, all fours, ass up, round ass,
    In bedroom, frat house BREAK on bed, smiling with puffy pussy, crying
    By night, pajamas down, anus, pussy, (puffy labia:1.4), beautiful woman, intimate photo, realistic image, dim light, cozy atmosphere, perfect lighting, perfect shadows, shiny skin, perfect reflections, lots of shadows
    """.strip()
    negativePrompt = """
    Negative prompt: easynegative, badhandv4, (worst quality, low quality, normal quality), bad-artist, blurry, ugly, ((bad anatomy)),((bad hands)),((bad proportions)),((duplicate limbs)),((fused limbs)),((interlocking fingers)),((poorly drawn face)), signature, watermark, artist logo, patreon logo
    """.strip()
    additionalNetworks = """
    Additional networks: urn:air:sd1:embedding:civitai:7808@9208*1.0!easynegative, urn:air:sdxl:lora:civitai:1115064@1253021*3.3, urn:air:sdxl:lora:civitai:212532@239420*1.0, urn:air:sdxl:lora:civitai:341353@382152*0.8, urn:air:sdxl:lora:civitai:888231@398847*0.45, urn:air:sdxl:lora:civitai:300005@436219*0.25
    """.strip()
    other_metadata = """
    Steps: 25, CFG scale: 4, Sampler: Euler a, Seed: 1594260453, process: txt2img, workflow: txt2img, Size: 1024x1024, draft: false, width: 1024, height: 1024, quantity: 1, baseModel: Illustrious, disablePoi: true, aspectRatio: 1:1, Created Date: 2025-07-05T1255:24.7399976Z, experimental: false, Clip skip: 2, Model: urn:air:sdxl:checkpoint:civitai:257749@290640
    """.strip()
    o = {
        "prompt": prompt,
        "negativePrompt": negativePrompt,
        "additionalNetworks": additionalNetworks,
        "other_metadata": other_metadata,
    }

    from itertools import combinations

    # Generate all possible combinations and permutations
    all_keys = ["negativePrompt", "additionalNetworks", "other_metadata"]
    combination_keys = []

    # Generate all combinations of different lengths (1 to 4 elements)
    for r in range(1, len(all_keys) + 1):
        for combo in combinations(all_keys, r):
            # For each combination, generate all permutations
            from itertools import permutations

            for perm in permutations(combo):
                combination_keys.append(["prompt"] + list(perm))
    print(f"combination_keys={json.dumps(combination_keys, indent=4)}")

    for input_names in combination_keys:
        inputs = [o[name] for name in input_names]
        generation_data_str = "\n".join(inputs)
        generation_data = parse_generation_data(generation_data_str)
        for key in input_names:
            assert key in generation_data, f"{key} not in {generation_data.keys()}"

        pipe = Pipe()
        generator = pipe.pipe(
            dict(messages=[{"role": "user", "content": generation_data_str}])
        )
        async for chunk in generator:
            print(f"{chunk}", end="")


if __name__ == "__main__":
    asyncio.run(main())
