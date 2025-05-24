from typing import Literal, Union, List, TypeAlias
from typing_extensions import TypedDict
from PIL import Image
import base64
from io import BytesIO
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ImageContent(TypedDict):
    type: Literal["image"]
    image: Union[str, Image.Image]


Content: TypeAlias = Union[str, ImageContent]


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: Content


Conversation: TypeAlias = List[Message]


def extract_images_from_conversation(conversation: Conversation) -> List[Image.Image]:
    images = []

    for message in conversation:
        content = message.get("content", [])
        if isinstance(content, list):
            for content_item in content:
                if (
                    isinstance(content_item, dict)
                    and content_item.get("type") == "image"
                    and "image" in content_item
                ):
                    image_data = content_item["image"]

                    if image_data.startswith("data:image/"):
                        base64_data = (
                            image_data.split(",", 1)[1]
                            if "," in image_data
                            else image_data
                        )
                        image_bytes = base64.b64decode(base64_data)
                        image = Image.open(BytesIO(image_bytes))
                        images.append(image)
                    else:
                        try:
                            image = Image.open(image_data)
                            images.append(image)
                        except Exception as e:
                            logger.warning(
                                f"Failed to load image: {image_data}, error: {e}"
                            )
        elif (
            isinstance(content, dict)
            and content.get("type") == "image"
            and "image" in content
        ):
            image_data = content["image"]
            if image_data.startswith("data:image/"):
                base64_data = (
                    image_data.split(",", 1)[1] if "," in image_data else image_data
                )
                image_bytes = base64.b64decode(base64_data)
                image = Image.open(BytesIO(image_bytes))
                images.append(image)
            else:
                try:
                    image = Image.open(image_data)
                    images.append(image)
                except Exception as e:
                    logger.warning(f"Failed to load image: {image_data}, error: {e}")

    return images


def encode_image_to_base64(image: Union[str, Image.Image]) -> str:
    if isinstance(image, str):
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    elif isinstance(image, Image.Image):
        buffered = BytesIO()
        image.save(buffered, format=image.format or "PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        raise ValueError(f"Invalid image type: {type(image)}")