from research.base import ResearchProvider
from research.models import Product


class MockResearchProvider(ResearchProvider):
    def search(self, query: str, limit: int = 20) -> list[Product]:
        return [
            Product(
                source="mock",
                title="Rainy Neon City Night – Cozy Lo-Fi Wall Art Print",
                description=(
                    "Single portrait poster depicting a rain-soaked urban street at night, "
                    "glowing vending machines and lanterns reflecting on wet pavement, "
                    "soft neon pinks and electric blues against deep indigo shadows, "
                    "hand-painted animation aesthetic with expressive brushwork, "
                    "nostalgic 1990s Saturday-morning animation atmosphere, "
                    "warm light spilling from a convenience store window, lone figure with umbrella. "
                    "Extremely popular with lo-fi music fans and cozy bedroom decorators."
                ),
                price=14.00,
                review_count=2187,
                rating=4.97,
                tags=["lo-fi art", "neon city", "rainy night", "cozy poster", "anime aesthetic",
                      "bedroom wall art", "1990s illustration", "printable", "night city"],
                shop_name="CozyCelStudio",
                product_url="https://mock.etsy.com/listing/1",
            ),
            Product(
                source="mock",
                title="Enchanted Bookstore with Sleeping Cats – Fantasy Illustration Print",
                description=(
                    "Cozy fantasy interior illustration: a candlelit independent bookshop with "
                    "floor-to-ceiling shelves, ladders, stacked grimoires, and three cats napping "
                    "among the books. Warm amber and sage green palette with soft dust-mote lighting. "
                    "Hand-drawn fantasy illustration style with fine ink linework and soft watercolour washes. "
                    "Highly detailed and collectible — sells strongly as a set of 2 with a companion "
                    "'Night Reading Nook' print. Best seller in cozy fantasy and bookish decor niche."
                ),
                price=18.00,
                review_count=3042,
                rating=4.98,
                tags=["cozy fantasy", "bookshop art", "cat poster", "fantasy illustration",
                      "bookish decor", "library print", "witchy aesthetic", "printable", "cottagecore"],
                shop_name="InkAndWonder",
                product_url="https://mock.etsy.com/listing/2",
            ),
            Product(
                source="mock",
                title="Frog Wizard in Mushroom Forest – Cute Fantasy Wall Art",
                description=(
                    "A small, round frog wearing a pointed wizard hat and mossy cloak stands "
                    "in a glowing forest of oversized bioluminescent mushrooms. Soft cel-shaded "
                    "illustration with rounded forms, earthy greens and purples, gentle ambient glow. "
                    "Expressive character design with kawaii proportions. "
                    "Sold as a set of 3: Frog Wizard, Snail Druid, and Hedgehog Alchemist — "
                    "consistent character scale, forest setting, and palette across all three prints. "
                    "Top performer in cute fantasy and cottagecore niches."
                ),
                price=16.00,
                review_count=1876,
                rating=4.96,
                tags=["frog art", "cute fantasy", "mushroom print", "cottagecore", "wizard poster",
                      "kawaii illustration", "forest art", "printable", "set of 3", "goblincore"],
                shop_name="MossCapStudios",
                product_url="https://mock.etsy.com/listing/3",
            ),
            Product(
                source="mock",
                title="Retro-Futuristic Ramen Shop at Night – Sci-Fi Diner Print",
                description=(
                    "A narrow neon-lit ramen counter set in a retro-futuristic cityscape — "
                    "chrome stools, holographic menu boards, a robot chef behind the counter, "
                    "steam rising from ceramic bowls. Colour palette: deep charcoal, neon magenta, "
                    "warm orange broth glow, chrome highlights. "
                    "Visual style: late-1980s retrofuturist illustration meets flat Japanese graphic design. "
                    "Fine linework, bold flat colour fills, no gradients. "
                    "Popular in sci-fi decor, ramen enthusiast, and Japanese street food niches. "
                    "Pairs naturally into a 5-print 'Cyber Street Food' series."
                ),
                price=15.00,
                review_count=1423,
                rating=4.94,
                tags=["ramen poster", "retro futuristic", "sci-fi art", "neon diner", "japanese food art",
                      "cyberpunk aesthetic", "kitchen poster", "printable", "flat design", "retrofuturism"],
                shop_name="NeonBowlPrints",
                product_url="https://mock.etsy.com/listing/4",
            ),
            Product(
                source="mock",
                title="Kawaii Capybara Café – Cute Animal Illustration Print",
                description=(
                    "A plump, cheerful capybara barista stands behind a pastel café counter, "
                    "wearing a small apron, surrounded by oversized pastries, steaming lattes, "
                    "and tiny flower arrangements. Soft pastel palette: cream, blush pink, "
                    "sage green, warm caramel. Illustration style: rounded kawaii character design "
                    "with clean digital linework and flat cel-shaded colour. "
                    "Extremely strong in cute animal, café aesthetic, and kids' room decor niches. "
                    "Natural series: each print features a different animal barista — "
                    "capybara, otter, red panda — same café setting and palette throughout."
                ),
                price=13.00,
                review_count=2654,
                rating=4.97,
                tags=["capybara art", "kawaii poster", "café illustration", "cute animal print",
                      "pastel wall art", "kids room decor", "animal barista", "printable", "cozy aesthetic"],
                shop_name="PastelCreaturesCo",
                product_url="https://mock.etsy.com/listing/5",
            ),
        ][:limit]
