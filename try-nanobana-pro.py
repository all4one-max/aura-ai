import os

from google import genai
from PIL import Image


def main():
    project = os.getenv("GOOGLE_VERTEX_AI_PROJECT_ID")
    client = genai.Client(
        vertexai=True,
        project=project,
    )
    user_image = Image.open("/Users/sauravjha/Downloads/sj-wayanad.png")
    cowbody_shirt = Image.open("/Users/sauravjha/Downloads/cowbody-shirt.png")

    text_input = """You are a virtual try on assistant. You are given a user image (first image) and a product image (second image). You need to generate a realistic image of the user(first image) wearing the product(second image)."""

    # Generate an image from a text prompt
    response = client.models.generate_content(
        model="gemini-3-pro-image-preview",
        contents=[user_image, cowbody_shirt, text_input],
    )

    for part in response.parts:
        if part.text is not None:
            print(part.text)
        elif part.inline_data is not None:
            image = part.as_image()
            image.save("sj-wearing-cowboy-shirt.png")


if __name__ == "__main__":
    main()
