import os
import requests
from openai import OpenAI
from datetime import datetime
import hashlib

from nexora.logger import logger
from nexora.tools.basetool import BaseTool

class ImageTool(BaseTool):
    name = "image"

    client = None

    def generate(self, prompt: str, model_name: str = "dall-e-3", size: str = "1024x1024", quality: str = "standard") -> str:
        """
        openai.function: Generate an image using OpenAI DALL-E 3 based on the given prompt. Returns the absolute path of the saved image.
        
        :param str prompt: Prompt for the image.
        :param str model_name: Model name to use for generation, default is "dall-e-3"
        :param str size: Desired image size, options are "1024x1024", "1024x1792", "1792x1024"
        :param str quality: Desired image quality, options are "standard", "hd"
        """

        if not self.client:
            api_key = self.app.config.settings["providers"]["openai"]["api_key"]
            self.client = OpenAI(api_key=api_key)

        try:
            response = self.client.images.generate(
                model=model_name,
                prompt=prompt,
                size=size,
                quality=quality,
                n=1,
            )
            image_url = response.data[0].url
            image_data = requests.get(image_url).content

            # Ensure the directory exists
            image_dir = os.path.expanduser("~/.config/nexora/images")
            os.makedirs(image_dir, exist_ok=True)

            # Create a truncated and hashed version of the prompt for the filename
            prompt_slug = hashlib.md5(prompt.encode('utf-8')).hexdigest()[:6]
            truncated_prompt = prompt[:20].replace(' ', '_')
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            
            # Create the file name
            image_path = os.path.join(image_dir, f"{model_name}_{timestamp}_{size}_{truncated_prompt}_{prompt_slug}.jpg")

            # Save the image
            with open(image_path, "wb") as image_file:
                image_file.write(image_data)

            return image_path
        except Exception as e:
            logger.error(f"Failed to generate image: {e}")
            return f"Failed to generate image: {e}"