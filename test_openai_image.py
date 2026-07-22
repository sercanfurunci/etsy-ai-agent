# ponytail: temporary test — remove once OpenAI provider is wired into main pipeline
from image.openai_provider import OpenAIImageProvider

PROMPT = "A handmade ceramic coffee mug with speckled glaze on a wooden table, natural light, product photography"

if __name__ == "__main__":
    print(f"Prompt: {PROMPT}\n")
    try:
        provider = OpenAIImageProvider()
        path = provider.generate(PROMPT)
        print(f"[OK] Image saved: {path}")
    except ValueError as e:
        print(f"[Config error] {e}")
    except Exception as e:
        print(f"[API error] {e}")
