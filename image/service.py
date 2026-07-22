from image.base import ImageProvider


def generate_images(product_ideas: list[dict], provider: ImageProvider) -> list[dict]:
    """Return each idea augmented with an 'image_result' field."""
    results = []
    for idea in product_ideas:
        prompt = idea.get("image_prompt", idea.get("name", "product image"))
        results.append({**idea, "image_result": provider.generate(prompt)})
    return results
