from typing import Literal, Union, List, Optional
from PIL import Image
import base64
from io import BytesIO
import logging
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str = Field(..., description="Text content")


class ImageContent(BaseModel):
    type: Literal["image"] = "image"
    image: str = Field(
        ..., description="Base64 encoded image data (data:image/...;base64,...)"
    )

    @field_validator("image", mode="before")
    @classmethod
    def convert_to_base64(cls, v: Union[str, Image.Image]) -> str:
        if isinstance(v, Image.Image):
            return encode_image_to_base64(v)
        elif isinstance(v, str):
            if not v.startswith("data:image/") and len(v) < 100:
                logger.warning(f"Guessing input string as image path: {v}")
                return encode_image_to_base64(v)
            else:
                return v
        else:
            raise ValueError(f"Invalid image type: {type(v)}")

    @field_validator("image", mode="after")
    @classmethod
    def validate_image(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Invalid image type: {type(v)}")
        if not v.startswith("data:image/"):
            raise ValueError(
                "Image must be a base64 data URL starting with 'data:image/'"
            )
        return v

    def __repr__(self) -> str:
        if len(self.image) > 35:
            truncated = f"{self.image[:20]}...{self.image[-10:]}"
        else:
            truncated = self.image
        return f"ImageContent(type='{self.type}', image='{truncated}')"

    def __str__(self) -> str:
        return self.__repr__()


Content = Union[List[Union[TextContent, ImageContent]], str]


class Message(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(
        ..., description="The role of the message sender"
    )
    content: Content = Field(..., description="The content of the message")

    class Config:
        extra = "forbid"
        validate_assignment = True

    def get_images(self) -> List[Image.Image]:
        if isinstance(self.content, str):
            return []
        return [
            decode_base64_to_image(content.image)
            for content in self.content
            if isinstance(content, ImageContent)
        ]

    def has_images(self) -> bool:
        """Check if message contains any images"""
        if isinstance(self.content, str):
            return False
        return any(isinstance(content, ImageContent) for content in self.content)

    def has_text(self) -> bool:
        """Check if message contains any text"""
        if isinstance(self.content, str):
            return True
        return any(isinstance(content, TextContent) for content in self.content)


class Conversation(BaseModel):
    messages: List[Message] = Field(
        default_factory=list, description="List of messages in the conversation"
    )

    class Config:
        validate_assignment = True

    @field_validator("messages", mode="after")
    @classmethod
    def validate_messages(cls, value):
        """
        Validate that the first message is 'system', and then 'user' and 'assistant' alternate.
        """
        messages = (
            value.get("messages")
            if isinstance(value, dict)
            else getattr(value, "messages", None)
        )
        if messages is None:
            return value

        if not messages:
            return value

        # Check first message is 'system'
        first_role = (
            messages[0].role
            if hasattr(messages[0], "role")
            else messages[0].get("role")
        )
        if first_role != "system":
            raise ValueError(
                "The first message in a Conversation must have role 'system'."
            )

        # Check alternation: user, assistant, user, assistant, ...
        expected_roles = ["user", "assistant"]
        for idx, msg in enumerate(messages[1:], start=1):
            role = msg.role if hasattr(msg, "role") else msg.get("role")
            expected_role = expected_roles[(idx - 1) % 2]
            if role != expected_role:
                raise ValueError(
                    f"Message at position {idx} must have role '{expected_role}', got '{role}'."
                )
        return value

    def add_message(
        self,
        role: Literal["system", "user", "assistant"],
        content: Union[str, dict, Content],
    ) -> None:
        self.messages.append(Message(role=role, content=content))

    def get_last_role(self) -> Optional[Literal["system", "user", "assistant"]]:
        return self.messages[-1].role if self.messages else None

    def get_images(self) -> List[Image.Image]:
        return [image for message in self.messages for image in message.get_images()]

    def pprint(self):
        for idx, message in enumerate(self.messages):
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)
            if isinstance(content, dict) and "image" in content:
                image_data = content["image"]
                if isinstance(image_data, str) and image_data.startswith("data:image/"):
                    short_base64 = image_data[:50] + (
                        "..." if len(image_data) > 50 else ""
                    )
                    logger.info(f"[{idx}] {role}: <image: {short_base64}>")
                else:
                    logger.info(f"[{idx}] {role}: <image: {type(image_data)}>")
            elif isinstance(content, str):
                logger.info(f"[{idx}] {role}: {content}")
            else:
                logger.info(f"[{idx}] {role}: {content}")


def encode_image_to_base64(image_or_path: Union[str, Image.Image]) -> str:
    if isinstance(image_or_path, str):
        with open(image_or_path, "rb") as image_file:
            base64_data = base64.b64encode(image_file.read()).decode("utf-8")
            return f"data:image/png;base64,{base64_data}"
    elif isinstance(image_or_path, Image.Image):
        buffered = BytesIO()
        format_name = image_or_path.format or "PNG"
        image_or_path.save(buffered, format=format_name)
        base64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
        mime_type = f"image/{format_name.lower()}"
        return f"data:{mime_type};base64,{base64_data}"
    else:
        raise ValueError(f"Invalid image type: {type(image_or_path)}")


def decode_base64_to_image(base64_string: str) -> Image.Image:
    if base64_string.startswith("data:image/"):
        base64_data = (
            base64_string.split(",", 1)[1] if "," in base64_string else base64_string
        )
    else:
        base64_data = base64_string

    image_bytes = base64.b64decode(base64_data)
    return Image.open(BytesIO(image_bytes))
