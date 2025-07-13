"""
title: Civitai Image Generation Pipe
author: Your Name (inspired by examples)
description: >-
  Generates images using the Civitai.com API and displays them in the chat.
  This pipe allows for specifying a model, negative prompts, and additional networks like LoRAs.
version: 1.0.0
license: MIT
requirements: civitai-py, httpx, requests
environment_variables:
  - CIVITAI_API_TOKEN
"""

import ast
import os
import json
import base64
import asyncio
import httpx
import traceback
import civitai
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Callable, Awaitable, AsyncGenerator

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
        MODEL_URN: str = Field(
            default="urn:air:sdxl:checkpoint:civitai:257749@290640",
            description="The URN of the base model to use for generation (e.g., Pony Diffusion V6 XL).",
        )
        NEGATIVE_PROMPT: str = Field(
            default="easynegative, badhandv4, (worst quality, low quality, normal quality), bad-artist, blurry, ugly, ((bad anatomy)),((bad hands)),((bad proportions)),((duplicate limbs)),((fused limbs)),((interlocking fingers)),((poorly drawn face)), signature, watermark, artist logo, patreon logo",
            description="Default negative prompt to improve image quality.",
        )
        ADDITIONAL_NETWORKS_JSON: str = Field(
            default="""
            {
                # easynegative https://civitai.com/models/7808/easynegative?modelVersionId=9208
                "urn:air:sd1:embedding:civitai:7808@9208": {
                    "type": "TextualInversion",
                    "strength": 1.0,
                    "triggerWord": "easynegative"
                },
                # Pony Realism Slider https://civitai.com/models/1115064?modelVersionId=1253021
                "urn:air:sdxl:lora:civitai:1115064@1253021": {
                    "type": "Lora",
                    "strength": 3.3,
                },
                # # Makeup master for ponyXL https://civitai.com/models/389775/makeup-master-for-ponyxl?modelVersionId=994789
                # "urn:air:sdxl:lora:civitai:389775@994789": {
                #     "type": "Lora",
                #     "strength": 1.0,
                # },
                # # detail tweaker xl https://civitai.com/models/122359?modelVersionId=135867
                # "urn:air:sdxl:lora:civitai:122359@135867": {
                #     "type": "Lora",
                #     "strength": 1.0,
                # },
                # all disney princesses from ralph breaks the internet https://civitai.com/models/212532?modelVersionId=239420
                "urn:air:sdxl:lora:civitai:212532@239420": {
                    "type": "Lora",
                    "strength": 1.0,
                },
                # ExpressiveH (Hentai LoRa Style) エロアニメ https://civitai.com/models/341353?modelVersionId=382152
                "urn:air:sdxl:lora:civitai:341353@382152": {
                    "type": "Lora",
                    "strength": 0.8,
                },
                # Vixon's Pony Styles - gothic neon https://civitai.com/models/888231?modelVersionId=398847
                "urn:air:sdxl:lora:civitai:888231@398847": {
                    "type": "Lora",
                    "strength": 0.45,
                },
                # Incase Style [PonyXL] https://civitai.com/models/300005?modelVersionId=436219
                "urn:air:sdxl:lora:civitai:300005@436219": {
                    "type": "Lora",
                    "strength": 0.25,
                },
            }
            """,
            description="A JSON string defining additional networks like LoRAs or Textual Inversions.",
        )
        POLLING_INTERVAL_SECONDS: int = Field(
            default=5,
            description="How often to check if the image generation job is complete (in seconds).",
        )
        JOB_TIMEOUT_SECONDS: int = Field(
            default=300,  # 5 minutes
            description="Maximum time to wait for the image generation job to complete.",
        )
        USE_B64: bool = Field(
            default=True,
            description="If enabled, encodes the image to Base64 and embeds it directly in the chat. If disabled, uses the direct image URL.",
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
            api_key = self.valves.CIVITAI_API_TOKEN or os.getenv("CIVITAI_API_TOKEN")
            if not api_key:
                yield "Error: CIVITAI_API_TOKEN is not configured. Please set it in the pipe settings or as an environment variable."
                await self._emit_status("Error: API token missing.", done=True)
                return
            civitai.api_key = api_key  # Ensure the SDK is using the correct key

            # 2. Extract Prompt
            prompt = self._get_last_user_message(body.get("messages", []))
            if not prompt:
                yield "Error: Could not find a user prompt in the chat history."
                await self._emit_status("Error: No prompt found.", done=True)
                return

            yield f'🎨 Starting image generation for prompt: "{prompt[:80]}..."\n\n'

            # 3. Prepare Civitai Payload
            try:
                additional_networks = ast.literal_eval(
                    self.valves.ADDITIONAL_NETWORKS_JSON.strip()
                )
            except json.JSONDecodeError:
                yield "Error: Invalid JSON format for 'Additional Networks'."
                await self._emit_status("Error: Invalid JSON.", done=True)
                return

            generation_input = {
                "model": self.valves.MODEL_URN,
                "params": {
                    "prompt": prompt,
                    "negativePrompt": self.valves.NEGATIVE_PROMPT,
                    "scheduler": "EulerA",
                    "steps": 25,
                    "cfgScale": 4,
                    "width": 1024,
                    "height": 1024,
                    "seed": 1594260453,
                    "clipSkip": 2,
                },
                "additionalNetworks": additional_networks,
            }

            # 4. Submit Job to Civitai
            await self._emit_status("Submitting job to Civitai...")
            response = await civitai.image.create(generation_input, wait=False)
            job_token = response["token"]
            job_id = response["jobs"][0]["jobId"]
            await self._emit_status(
                f"Job submitted (ID: {job_id}). Waiting for result..."
            )

            # 5. Poll for Job Completion
            elapsed_time = 0
            # while elapsed_time < self.valves.JOB_TIMEOUT_SECONDS:
            job_status = await civitai.jobs.get(token=job_token, job_id=job_id)
            image_info = job_status["jobs"][0].get("result", [{}])[0]

            if image_info.get("available"):
                image_url = image_info.get("blobUrl")
                if not image_url:
                    yield "Error: Job completed but no image URL was found."
                    await self._emit_status("Error: Missing image URL.", done=True)
                    return

                # 6. Download and Display Image
                if self.valves.USE_B64:
                    await self._emit_status(
                        "Image generated! Downloading and encoding..."
                    )
                    async with httpx.AsyncClient() as client:
                        img_response = await client.get(image_url, timeout=120.0)
                        img_response.raise_for_status()

                    content_type = img_response.headers.get(
                        "Content-Type", "image/jpeg"
                    )
                    image_base64 = base64.b64encode(img_response.content).decode(
                        "utf-8"
                    )
                    image_markdown = (
                        f"![Generated Image](data:{content_type};base64,{image_base64})"
                    )
                else:
                    await self._emit_status("Image generated! Using direct URL.")
                    image_markdown = f"![Generated Image]({image_url})"

                yield image_markdown
                await self._emit_status("Image generation complete!", done=True)
                return  # Success, exit the loop

            # # Wait before polling again
            # await asyncio.sleep(self.valves.POLLING_INTERVAL_SECONDS)
            # elapsed_time += self.valves.POLLING_INTERVAL_SECONDS
            # await self._emit_status(
            #     f"Polling for result... ({elapsed_time}s elapsed)"
            # )

        except Exception as e:
            error_message = f"An unexpected error occurred: {str(e)}"
            stack_trace = traceback.format_exc()
            print(f"Error in Civitai Pipe: {error_message}\n{stack_trace}")
            yield f"{error_message}\n\n```\n{stack_trace}\n```"
            await self._emit_status(f"Error: {str(e)}", done=True)
