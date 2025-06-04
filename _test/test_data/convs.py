import os
from PIL import Image

from data_service.typing.message import Conversation


TEST_CONVERSATIONS_PURE_TEXT = {
    "single_turn": Conversation(
        messages=[
            {
                "role": "user",
                "content": "Hello, could you introduce yourself?",
            },
            {
                "role": "assistant",
                "content": "Hello! I'm an AI assistant designed to help you with various tasks, including answering questions, providing suggestions, and engaging in conversations.",
            },
        ]
    ),
    "two_turn": Conversation(
        messages=[
            {
                "role": "user",
                "content": "What's the weather like today?",
            },
            {
                "role": "assistant",
                "content": "I apologize, but I don't have access to real-time weather information. I recommend checking a weather service or looking outside.",
            },
            {
                "role": "user",
                "content": "Could you tell me what time it is?",
            },
            {
                "role": "assistant",
                "content": "I apologize, but I don't have access to real-time clock information. Please check your device's time.",
            },
        ]
    ),
}

image_1_path = os.path.join(
    os.environ["MYGO_PROJECT_PATH"], "_test/test_data/images/demo_1.jpg"
)
image_2_path = os.path.join(
    os.environ["MYGO_PROJECT_PATH"], "_test/test_data/images/demo_2.jpg"
)
TEST_CONVERSATIONS_MULTIMODAL = {
    "image_text": Conversation(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.open(image_1_path),
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            },
            {
                "role": "assistant",
                "content": "The image depicts a woman and her dog sitting on a sandy beach at sunset.",
            },
        ]
    ),
    "image_only": Conversation(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.open(image_2_path),
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "There is an apple in the image.",
            },
        ]
    ),
    "text_only": Conversation(
        messages=[
            {
                "role": "user",
                "content": "Hello, how are you?",
            },
            {
                "role": "assistant",
                "content": "I'm doing well, thank you for asking!",
            },
        ]
    ),
    "multi_image": Conversation(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.open(image_1_path),
                    },
                    {
                        "type": "image",
                        "image": Image.open(image_2_path),
                    },
                    {
                        "type": "text",
                        "text": "Compare these two images and describe their differences.",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "The first image shows a woman and her dog on a beach at sunset, while the second image shows an apple. These are completely different subjects and settings.",
            },
        ]
    ),
    "multi_turn_image_text": Conversation(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.open(image_1_path),
                    },
                    {"type": "text", "text": "What's happening in this image?"},
                ],
            },
            {
                "role": "assistant",
                "content": "The image shows a woman and her dog enjoying a peaceful moment on a beach during sunset.",
            },
            {
                "role": "user",
                "content": "Can you describe the mood of this scene?",
            },
            {
                "role": "assistant",
                "content": "The scene appears very serene and peaceful. The warm colors of the sunset create a romantic atmosphere, and the presence of the dog suggests a moment of companionship and relaxation.",
            },
        ]
    ),
    "multi_turn_multi_image": Conversation(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.open(image_1_path),
                    },
                    {
                        "type": "text",
                        "text": "I'm planning a beach vacation. What should I bring based on this image?",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "Based on the beach scene, I recommend bringing: sunscreen, a beach towel, water, and perhaps a camera to capture beautiful sunset moments like this one.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.open(image_2_path),
                    },
                    {
                        "type": "text",
                        "text": "What about for a picnic? This is a different setting.",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "For a picnic setting like this, you should bring: a picnic blanket, fresh fruits (like the apple shown), a basket for food, and perhaps some refreshing drinks. The setting seems more casual and intimate.",
            },
        ]
    ),
}
