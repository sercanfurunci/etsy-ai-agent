"""Builds data/ukiyoe_dataset.json from the curated data below.

Run: python3 scripts/build_dataset.py

Every entry in this file was authored individually (not synonym-expanded or
randomly padded). Category counts are documented in docs/ukiyoe-dataset.md
alongside the originally-requested targets and the reasoning for the gap
(quality over raw quantity, per project decision).
"""
import json
import re
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "ukiyoe_dataset.json"


def slugify(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


def entry(label: str, prompt: str, tags: list[str], weight: float = 1.0, **extra) -> dict:
    e = {
        "id": slugify(label),
        "label": label,
        "promptText": prompt,
        "tags": tags,
        "weight": weight,
        "rarity": extra.pop("rarity", round(1.0 / weight, 2) if weight else 1.0),
    }
    e.update(extra)
    return e


def build_list(rows: list[tuple]) -> list[dict]:
    out = []
    for row in rows:
        label, prompt, tags = row[0], row[1], row[2]
        weight = row[3] if len(row) > 3 else 1.0
        extra = row[4] if len(row) > 4 else {}
        out.append(entry(label, prompt, tags, weight, **extra))
    return out


def build_atmosphere_list(rows: list[tuple]) -> list[dict]:
    """For seasons/timesOfDay: every entry carries compatibleGenres/avoidWith/cooldownGroup.

    Row shape: (label, prompt, tags, weight, compatible_genres, avoid_with, cooldown_group_or_None)
    """
    out = []
    for label, prompt, tags, weight, compatible_genres, avoid_with, cooldown_group in rows:
        extra = {"compatibleGenres": compatible_genres, "avoidWith": avoid_with}
        if cooldown_group:
            extra["cooldownGroup"] = cooldown_group
        out.append(entry(label, prompt, tags, weight, **extra))
    return out


# ──────────────────────────────────────────────────────────────────────────
# ANIMALS
# ──────────────────────────────────────────────────────────────────────────

ANIMAL_BIRDS = build_list([
    ("red-crowned crane", "a red-crowned crane standing among wind-bent reeds", ["bird", "wetland", "elegant", "winter"]),
    ("white heron", "a white heron poised motionless at the water's edge", ["bird", "wetland", "quiet"]),
    ("grey heron", "a grey heron wading through shallow river water", ["bird", "wetland", "river"]),
    ("little egret", "a little egret preening among lotus stems", ["bird", "wetland", "pond"]),
    ("mandarin duck", "a pair of mandarin ducks drifting on a still pond", ["bird", "pond", "pair", "spring"]),
    ("wild goose", "a skein of wild geese crossing an autumn sky", ["bird", "migration", "autumn", "sky"]),
    ("cormorant", "a cormorant perched on a river stake, wings half spread to dry", ["bird", "river", "fishing"]),
    ("kingfisher", "a kingfisher darting low over a mountain stream", ["bird", "stream", "quick", "summer"]),
    ("Japanese bush warbler", "a bush warbler singing from a plum branch", ["bird", "spring", "song", "garden"]),
    ("brown-eared bulbul", "a bulbul perched among persimmon fruit", ["bird", "autumn", "orchard"]),
    ("mountain hawk-eagle", "a hawk-eagle circling above a forested ridge", ["bird", "raptor", "mountain"]),
    ("sparrowhawk", "a sparrowhawk perched watchfully on a bare branch", ["bird", "raptor", "winter"]),
    ("green pheasant", "a green pheasant stepping through spring grass", ["bird", "field", "spring"]),
    ("copper pheasant", "a copper pheasant with a long trailing tail among fallen leaves", ["bird", "forest", "autumn"]),
    ("Japanese quail", "a quail crouched low in a millet field", ["bird", "field", "small"]),
    ("golden plover", "a golden plover running along a tidal flat", ["bird", "coastal", "shore"]),
    ("common snipe", "a snipe probing the mud of a rice paddy at dusk", ["bird", "wetland", "dusk"]),
    ("barn swallow", "swallows skimming low over a village roofline", ["bird", "village", "summer", "flight"]),
    ("azure-winged magpie", "azure-winged magpies gathered in a bamboo grove", ["bird", "grove", "group"]),
    ("Eurasian jay", "a jay carrying an acorn through an oak wood", ["bird", "forest", "autumn"]),
    ("great spotted woodpecker", "a woodpecker at work on a cedar trunk", ["bird", "forest"]),
    ("lesser cuckoo", "the distant call of a cuckoo across a misty valley", ["bird", "sound", "summer"]),
    ("Eurasian skylark", "a skylark rising in song above a barley field", ["bird", "field", "sky", "song"]),
    ("white-naped crane", "a white-naped crane standing in a frost-covered field", ["bird", "winter", "field"]),
    ("black-tailed gull", "gulls wheeling above a fishing harbor", ["bird", "coastal", "harbor"]),
    ("grey wagtail", "a wagtail bobbing on a moss-covered stone in a stream", ["bird", "stream", "small"]),
    ("meadow bunting", "a bunting singing from a fence post at the field's edge", ["bird", "field", "song"]),
    ("Oriental greenfinch", "a greenfinch feeding among wild thistle", ["bird", "field", "small"]),
    ("jungle crow", "a lone crow perched atop a bare winter tree", ["bird", "winter", "solitary"]),
    ("great egret", "a great egret standing tall among rice seedlings", ["bird", "wetland", "field"]),
    ("Japanese stork", "a stork nesting atop an old pine", ["bird", "rare", "nesting"]),
    ("whooper swan", "whooper swans resting on a winter lake", ["bird", "winter", "lake"]),
    ("little grebe", "a grebe diving beneath the still surface of a pond", ["bird", "pond", "small"]),
    ("Ural owl", "an owl watching silently from a hollow tree at dusk", ["bird", "forest", "night"]),
    ("Japanese pheasant in flight", "a pheasant bursting upward from tall grass", ["bird", "motion", "field"]),
])

ANIMAL_MAMMALS = build_list([
    ("red fox", "a red fox slipping through frost-covered grass at dawn", ["mammal", "field", "cunning"], 0.4, {"cooldownGroup": "iconic-japanese-motifs"}),
    ("tanuki", "a round-bellied tanuki pausing beneath a temple gate at dusk", ["mammal", "folk", "playful"]),
    ("Japanese macaque", "a snow monkey soaking in a steaming mountain hot spring", ["mammal", "mountain", "winter"]),
    ("sika deer", "a sika deer grazing beneath autumn maples", ["mammal", "forest", "autumn"]),
    ("wild boar", "a wild boar crashing through undergrowth on a mountain slope", ["mammal", "forest", "powerful"]),
    ("Asian black bear", "a black bear foraging beside a mountain stream", ["mammal", "mountain", "forest"]),
    ("Japanese serow", "a serow standing motionless on a rocky mountain ledge", ["mammal", "mountain", "solitary"]),
    ("flying squirrel", "a flying squirrel gliding between cedar branches at night", ["mammal", "forest", "night"]),
    ("Japanese marten", "a marten darting along a fallen log in early snow", ["mammal", "forest", "winter"]),
    ("weasel", "a weasel pausing at the mouth of a stone wall burrow", ["mammal", "small", "field"]),
    ("Japanese hare", "a white winter hare crouched in a snowfield", ["mammal", "winter", "small"]),
    ("otter", "an otter surfacing among river reeds with a fish", ["mammal", "river", "playful"]),
    ("mole", "a mole's fresh mound rising along a garden path", ["mammal", "small", "subtle"]),
    ("bat", "bats wheeling above a temple roofline at dusk", ["mammal", "dusk", "flight"]),
    ("packhorse", "a heavily laden packhorse led along a mountain road", ["mammal", "labor", "travel"]),
    ("farm ox", "an ox pulling a plough through a flooded rice paddy", ["mammal", "labor", "field"]),
    ("water buffalo", "a water buffalo resting in a shallow irrigation channel", ["mammal", "field", "rest"]),
    ("village cat", "a cat curled asleep on a sun-warmed veranda", ["mammal", "domestic", "quiet"]),
    ("village dog", "a village dog trotting alongside a traveling merchant", ["mammal", "domestic", "companion"]),
    ("silkworm moth", "a silkworm moth resting on mulberry leaves in a farmhouse tray", ["mammal", "domestic", "craft"]),
])

ANIMAL_FISH = build_list([
    ("koi carp", "a koi carp gliding beneath drifting lily pads", ["fish", "pond", "elegant"], 0.4, {"cooldownGroup": "iconic-japanese-motifs"}),
    ("crucian carp", "crucian carp stirring the mud of a shallow paddy channel", ["fish", "pond", "rural"]),
    ("sweetfish", "sweetfish darting through a clear mountain stream", ["fish", "stream", "summer"]),
    ("catfish", "a giant catfish lurking beneath a river's muddy current", ["fish", "river", "folklore"]),
    ("eel", "an eel curling through reed shadows in shallow water", ["fish", "river", "night"]),
    ("sea bream", "a sea bream caught mid-leap above the harbor waves", ["fish", "coastal", "festive"]),
    ("bonito", "bonito breaking the surface as boats give chase offshore", ["fish", "sea", "fishing"]),
    ("loach", "loaches stirring the shallows of a flooded rice field", ["fish", "field", "small"]),
])

ANIMAL_MARINE = build_list([
    ("octopus", "an octopus coiled among rocks at low tide", ["marine", "coastal", "shore"]),
    ("squid", "squid drying on lines strung along a fishing dock", ["marine", "harbor", "craft"]),
    ("abalone", "a shell-diver surfacing with an abalone shell", ["marine", "coastal", "labor"]),
    ("sea turtle", "a sea turtle surfacing beyond a rocky point", ["marine", "sea", "rare"]),
    ("jellyfish", "a jellyfish drifting through pale green shallows", ["marine", "sea", "delicate"]),
    ("horseshoe crab", "a horseshoe crab crossing wet sand at dusk", ["marine", "shore", "ancient"]),
])

ANIMAL_INSECTS = build_list([
    ("firefly", "fireflies drifting above a humid summer paddy at dusk", ["insect", "summer", "night", "delicate"]),
    ("cicada", "a cicada clinging to a cedar trunk in the heat of summer", ["insect", "summer", "sound"]),
    ("dragonfly", "a red dragonfly resting on a reed tip", ["insect", "autumn", "pond"]),
    ("cricket", "a cricket singing beneath a paper lantern at night", ["insect", "autumn", "sound", "night"]),
    ("swallowtail butterfly", "a swallowtail butterfly settling on a citrus blossom", ["insect", "spring", "garden"]),
    ("praying mantis", "a praying mantis poised on a dew-heavy grass blade", ["insect", "field", "still"]),
    ("bee", "a bee moving among wisteria blossoms", ["insect", "spring", "garden"]),
])

ANIMAL_REPTILES = build_list([
    ("pond frog", "a frog perched on a broad lotus leaf after rain", ["amphibian", "pond", "rain"]),
    ("toad", "a toad crossing a moss-covered garden path at dusk", ["amphibian", "garden", "night"]),
    ("grass snake", "a slender snake gliding through tall summer grass", ["reptile", "field", "summer"]),
    ("salamander", "a salamander resting beneath a mossy streamside stone", ["amphibian", "stream", "shade"]),
    ("skink", "a skink basking on a sun-warmed garden wall", ["reptile", "garden", "small"]),
])

# ──────────────────────────────────────────────────────────────────────────
# PLANTS / FLORA
# ──────────────────────────────────────────────────────────────────────────

FLORA = build_list([
    ("cherry blossom", "a branch of blooming cherry blossoms against pale sky", ["spring", "iconic"], 0.35, {"cooldownGroup": "iconic-japanese-motifs"}),
    ("plum blossom", "early plum blossoms opening on a bare winter branch", ["late-winter", "delicate"]),
    ("wisteria", "cascading wisteria blossoms hanging from a garden trellis", ["spring", "garden"]),
    ("Japanese maple", "a maple tree ablaze with autumn red", ["autumn", "iconic-color"]),
    ("pine tree", "a wind-bent pine clinging to a rocky shoreline", ["evergreen", "coastal"]),
    ("bamboo", "tall bamboo stalks swaying in a dense grove", ["evergreen", "grove"]),
    ("weeping willow", "a willow trailing its branches over still water", ["riverside", "graceful"]),
    ("chrysanthemum", "chrysanthemums blooming in ordered rows in an autumn garden", ["autumn", "garden"]),
    ("peony", "a heavy-headed peony bloom bending its stem", ["late-spring", "garden"]),
    ("lotus", "lotus flowers rising above a still temple pond", ["summer", "pond", "sacred"]),
    ("iris", "purple iris blooming along a marsh's edge", ["early-summer", "wetland"]),
    ("morning glory", "morning glories climbing a bamboo trellis at dawn", ["summer", "garden"]),
    ("camellia", "a single red camellia fallen whole onto snow", ["winter", "symbolic"]),
    ("azalea", "a hillside covered in blooming azalea", ["spring", "hillside"]),
    ("hydrangea", "hydrangea blossoms heavy with rain", ["rainy-season", "garden"]),
    ("bush clover", "bush clover bending under the weight of autumn dew", ["autumn", "field"]),
    ("pampas grass", "silver pampas grass swaying beneath a harvest moon", ["autumn", "field"]),
    ("bellflower", "a bellflower nodding among roadside grasses", ["autumn", "field"]),
    ("cosmos", "a field of cosmos flowers swaying in an autumn breeze", ["autumn", "field"]),
    ("narcissus", "narcissus blooming beside a garden stone in early spring", ["early-spring", "garden"]),
    ("plum grove", "a grove of plum trees just beginning to flower", ["late-winter", "orchard"]),
    ("cedar", "an ancient cedar towering over a shrine path", ["evergreen", "sacred"]),
    ("cypress", "cypress trees lining a mountain pilgrim trail", ["evergreen", "mountain"]),
    ("persimmon tree", "a persimmon tree heavy with orange fruit against a bare sky", ["autumn", "orchard"]),
    ("mulberry tree", "mulberry leaves being gathered for silkworms", ["craft", "orchard"]),
    ("tea bush", "rows of tea bushes climbing a terraced hillside", ["agriculture", "hillside"]),
    ("rice plant", "young rice seedlings standing in flooded paddies", ["agriculture", "summer"]),
    ("reed", "reeds bending together along a riverbank in wind", ["wetland", "movement"]),
    ("water lily", "a water lily closed tight at the surface of a pond", ["pond", "quiet"]),
    ("wild violet", "wild violets scattered through spring undergrowth", ["spring", "small"]),
    ("dandelion", "dandelion seeds drifting across a village field", ["spring", "movement"]),
    ("peach blossom", "peach blossoms opening beside a farmhouse wall", ["spring", "orchard"]),
    ("magnolia", "a magnolia branch heavy with pale blossoms", ["early-spring", "garden"]),
    ("gardenia", "gardenia blossoms scenting a summer courtyard", ["summer", "garden"]),
    ("orchid", "a wild orchid growing from a mossy cliff crevice", ["mountain", "rare"]),
    ("moss", "deep green moss covering old stone steps", ["texture", "quiet"]),
    ("fern", "unfurling fern fronds in a shaded forest hollow", ["forest", "shade"]),
    ("gourd vine", "gourds ripening on a trellis beside a farmhouse", ["autumn", "agriculture"]),
    ("bitter melon vine", "bitter melon climbing a garden fence in summer", ["summer", "agriculture"]),
    ("daikon leaves", "daikon leaves drying in bundles beneath the eaves", ["autumn", "agriculture"]),
    ("chestnut tree", "chestnuts scattered beneath an autumn tree", ["autumn", "forest"]),
    ("ginkgo tree", "a ginkgo tree shedding golden leaves in a temple courtyard", ["autumn", "sacred"]),
    ("paulownia tree", "broad paulownia leaves shading a garden path", ["summer", "garden"]),
    ("camphor tree", "an ancient camphor tree marking a shrine boundary", ["sacred", "ancient"]),
    ("holly", "holly berries bright red against winter snow", ["winter", "small"]),
    ("rhododendron", "rhododendron blooming along a mountain trail", ["spring", "mountain"]),
    ("wild rose", "a wild rose climbing an old stone wall", ["summer", "field"]),
    ("lotus pond flora", "lotus leaves overlapping across a temple pond's surface", ["summer", "pond"]),
    ("seaweed", "kelp drying on racks along the shore", ["coastal", "craft"]),
    ("sedge grass", "sedge grass bending in a coastal wind", ["coastal", "wetland"]),
    ("thistle", "purple thistle standing tall in a fallow field", ["field", "texture"]),
    ("clover", "a patch of clover beside a well-worn village path", ["field", "small"]),
    ("sunflower", "sunflowers turning toward the late summer sun", ["summer", "field"]),
    ("safflower", "safflower harvested and drying for dye", ["craft", "agriculture"]),
    ("indigo plant", "indigo leaves piled beside a dye vat", ["craft", "agriculture"]),
    ("hemp plant", "hemp stalks drying in bundles in a farmyard", ["craft", "agriculture"]),
    ("millet", "millet heads bowing in an autumn field", ["autumn", "agriculture"]),
    ("barley field", "barley rippling gold under a summer wind", ["summer", "agriculture"]),
    ("sacred sakaki branch", "a sakaki branch left as an offering at a shrine altar", ["sacred", "ritual"]),
    ("wild grasses", "tall wild grasses catching the last light of dusk", ["field", "dusk"]),
    ("moss garden path", "a narrow path of stepping stones through deep moss", ["garden", "quiet"]),
])

# ──────────────────────────────────────────────────────────────────────────
# ENVIRONMENTS
# ──────────────────────────────────────────────────────────────────────────

ENVIRONMENTS = build_list([
    ("snow-covered mountain pass", "a narrow pass buried under fresh mountain snow", ["mountain", "winter"]),
    ("coastal cliffs", "wind-scoured cliffs rising above a restless sea", ["coastal", "dramatic"]),
    ("bamboo forest", "a dense bamboo forest filtering pale green light", ["forest", "quiet"]),
    ("terraced rice paddies", "flooded rice terraces stepping down a hillside", ["agriculture", "rural"]),
    ("willow-lined riverbank", "a riverbank shaded by trailing willow branches", ["river", "quiet"]),
    ("temple courtyard", "a swept gravel courtyard before a temple hall", ["sacred", "urban"]),
    ("shrine approach", "a long stone stairway rising toward a shrine gate", ["sacred", "mountain"]),
    ("harbor town at low tide", "fishing boats resting on exposed mudflats at low tide", ["coastal", "village"]),
    ("cloud-wreathed mountain pass", "a mountain pass disappearing into drifting cloud", ["mountain", "mysterious"]),
    ("koi garden pond", "a landscaped pond stocked with slow-moving koi", ["garden", "quiet"]),
    ("forest shrine clearing", "a small shrine standing in a clearing deep in the woods", ["forest", "sacred"]),
    ("volcanic hot spring", "steam rising from a rock-lined mountain hot spring", ["mountain", "winter"]),
    ("hilltop castle ruins", "crumbling stone walls overlooking a valley below", ["ruins", "dramatic"]),
    ("night market street", "lantern-lit stalls crowding a narrow market street", ["urban", "night"]),
    ("lakeside pavilion", "an open pavilion built out over a calm lake", ["lake", "quiet"]),
    ("cedar forest trail", "a trail winding beneath towering ancient cedars", ["forest", "sacred"]),
    ("cliffside pagoda", "a pagoda perched on a narrow cliff ledge", ["dramatic", "sacred"]),
    ("autumn maple valley", "a valley entirely aflame with autumn maple color", ["autumn", "dramatic"]),
    ("moonlit rooftop garden", "a small rooftop garden bathed in moonlight", ["urban", "night"]),
    ("remote fishing village", "a scattering of thatched houses along a quiet cove", ["coastal", "village"]),
    ("post-town crossroads", "travelers converging at a crossroads inn", ["urban", "travel"]),
    ("mountain hermitage", "a lone hut set among pines high on a mountainside", ["mountain", "solitary"]),
    ("tea plantation hillside", "neatly rowed tea bushes climbing a misty hillside", ["agriculture", "rural"]),
    ("salt flats", "workers raking salt across sunlit coastal flats", ["coastal", "labor"]),
    ("pine-covered island", "small pine-covered islets scattered across a calm bay", ["coastal", "iconic"]),
    ("river ferry crossing", "a ferry poling slowly across a wide river", ["river", "travel"]),
    ("stone bridge over gorge", "an arched stone bridge spanning a deep gorge", ["mountain", "dramatic"]),
    ("wisteria trellis garden", "a garden path arched over with blooming wisteria", ["garden", "spring"]),
    ("plum orchard", "rows of plum trees just beginning to bloom", ["orchard", "late-winter"]),
    ("snowfield with distant peak", "an open snowfield beneath a distant mountain peak", ["mountain", "winter"]),
    ("reed marsh", "a wide marsh of tall reeds under a grey sky", ["wetland", "quiet"]),
    ("tidal estuary", "mudflats and channels exposed at the mouth of a river", ["coastal", "wetland"]),
    ("castle-town street", "a bustling street lined with merchant shopfronts", ["urban", "lively"]),
    ("wooden-house back alley", "a quiet alley between weathered wooden houses", ["urban", "quiet"]),
    ("sake brewery courtyard", "barrels stacked in the courtyard of a sake brewery", ["urban", "craft"]),
    ("silk-weaving workshop", "looms clacking inside a dim weaving workshop", ["urban", "craft"]),
    ("papermaking riverside workshop", "sheets of washi drying on frames by a river", ["river", "craft"]),
    ("charcoal kiln clearing", "smoke rising from a charcoal kiln deep in the forest", ["forest", "labor"]),
    ("mountain waterfall basin", "mist rising from the base of a tall waterfall", ["mountain", "dramatic"]),
    ("cave shrine", "a small shrine set into the mouth of a mountain cave", ["sacred", "mountain"]),
    ("forest torii approach", "a torii gate framed by an avenue of tall trees", ["sacred", "forest"]),
    ("rooftop view over town", "tiled rooftops receding toward a hazy horizon", ["urban", "panoramic"]),
    ("floating teahouse", "a small teahouse built on stilts over a still lake", ["lake", "quiet"]),
    ("bamboo grove path at dusk", "a narrow path through bamboo losing light at dusk", ["forest", "dusk"]),
    ("rocky pine seashore", "wind-shaped pines rooted among coastal rocks", ["coastal", "iconic"]),
    ("rural farmhouse courtyard", "a swept dirt courtyard before a thatched farmhouse", ["rural", "quiet"]),
    ("thatched village square", "villagers gathered in a square of thatched houses", ["rural", "lively"]),
    ("inland sea strait", "fishing boats crossing a narrow strait between islands", ["coastal", "travel"]),
    ("mountain onsen village", "a small village built around steaming hot spring baths", ["mountain", "village"]),
    ("cherry blossom hill", "a hillside crowded with blossoming cherry trees", ["spring", "iconic"]),
    ("autumn foliage gorge", "a narrow gorge walled in blazing autumn color", ["autumn", "dramatic"]),
    ("wheat field at harvest", "workers cutting wheat beneath a wide harvest sky", ["agriculture", "summer"]),
    ("temple lotus pond", "a lotus pond enclosed within temple grounds", ["sacred", "summer"]),
    ("moon-viewing platform", "an open platform built for viewing the autumn moon", ["quiet", "autumn"]),
    ("snow monkey hot spring basin", "monkeys gathered in a snow-ringed hot spring", ["mountain", "winter"]),
    ("cormorant fishing river bend", "torch-lit boats fishing with cormorants at a river bend", ["river", "night"]),
    ("ancient cedar grove", "shafts of light falling through an old cedar grove", ["forest", "sacred"]),
    ("abandoned watchtower", "an old watchtower standing empty above a coastal plain", ["ruins", "coastal"]),
    ("lighthouse point", "a lighthouse standing alone on a rocky headland", ["coastal", "solitary"]),
    ("river delta rice fields", "a patchwork of rice fields spreading across a river delta", ["agriculture", "panoramic"]),
    ("mountain shrine steps", "a long flight of stone steps climbing to a mountain shrine", ["sacred", "mountain"]),
    ("covered bridge crossing", "a roofed wooden bridge spanning a mountain stream", ["mountain", "travel"]),
    ("fishing weir river bend", "a wooden weir channeling fish at a river bend", ["river", "labor"]),
    ("cliffside pilgrim trail", "a narrow trail hugging a cliff above the sea", ["coastal", "travel"]),
    ("moss garden", "a quiet garden carpeted entirely in soft moss", ["garden", "quiet"]),
    ("dry rock garden", "raked white gravel surrounding still stones", ["sacred", "minimal"]),
    ("floating lily pond", "water lilies spread across a still garden pond", ["garden", "summer"]),
    ("remote mountain waterfall", "a tall waterfall dropping through untouched forest", ["mountain", "dramatic"]),
    ("coastal pine windbreak", "a row of pines bent permanently by sea wind", ["coastal", "weathered"]),
    ("deep valley mist basin", "a valley filled with slow-moving morning mist", ["mountain", "mysterious"]),
])

# ──────────────────────────────────────────────────────────────────────────
# ARCHITECTURE
# ──────────────────────────────────────────────────────────────────────────

ARCHITECTURE = build_list([
    ("multi-tiered pagoda", "a five-tiered pagoda rising above the treeline", ["sacred", "iconic"], 0.45, {"cooldownGroup": "iconic-japanese-motifs"}),
    ("torii gate", "a weathered torii gate marking a shrine's threshold", ["sacred", "iconic"], 0.35, {"cooldownGroup": "iconic-japanese-motifs"}),
    ("thatched farmhouse", "a thatched farmhouse roof heavy with old moss", ["rural", "domestic"]),
    ("wooden watchtower", "a wooden watchtower rising above a village wall", ["rural", "defensive"]),
    ("castle keep", "a white-walled castle keep set against distant mountains", ["urban", "dramatic"]),
    ("stone bridge", "a low arched stone bridge over a quiet canal", ["urban", "quiet"]),
    ("covered wooden bridge", "a roofed bridge offering shelter mid-crossing", ["rural", "travel"]),
    ("teahouse pavilion", "an open-sided teahouse set among garden pines", ["garden", "quiet"]),
    ("shrine main hall", "the raised main hall of a Shinto shrine", ["sacred", "formal"]),
    ("temple bell tower", "a wooden tower housing a great bronze temple bell", ["sacred", "sound"]),
    ("merchant storehouse", "a white-plastered storehouse with heavy iron doors", ["urban", "commerce"]),
    ("post-town inn", "a two-story inn with lanterns lit along its facade", ["urban", "travel"]),
    ("stilted riverside teahouse", "a teahouse built out over the water on wooden stilts", ["river", "quiet"]),
    ("rice granary", "a raised wooden granary keeping grain off the damp ground", ["rural", "agriculture"]),
    ("village well house", "a small roofed structure sheltering the village well", ["rural", "domestic"]),
    ("lighthouse", "a simple stone lighthouse marking a rocky point", ["coastal", "solitary"]),
    ("fishing weir structure", "a woven wooden weir spanning a shallow river", ["river", "labor"]),
    ("mountain hermitage hut", "a single-room hut with a thin trail of hearth smoke", ["mountain", "solitary"]),
    ("floating shrine gate", "a torii gate standing directly in shallow tidal water", ["coastal", "sacred"]),
    ("moon-viewing pavilion", "an open pavilion oriented toward the rising moon", ["quiet", "formal"]),
    ("samurai residence gate", "the formal gate of a walled samurai residence", ["urban", "formal"]),
    ("market stall row", "a row of open-fronted stalls beneath striped awnings", ["urban", "commerce"]),
    ("sake brewery building", "a tall wooden brewery building with a cedar-ball sign", ["urban", "craft"]),
    ("paper workshop building", "a low workshop with racks of drying paper outside", ["craft", "riverside"]),
    ("silk mill building", "a long wooden mill building humming with looms", ["urban", "craft"]),
    ("public bathhouse", "a bathhouse with steam curling from its tall chimney", ["urban", "domestic"]),
    ("wooden aqueduct", "a raised wooden aqueduct carrying water across a gorge", ["mountain", "engineering"]),
    ("terraced stone retaining wall", "dry-stacked stone walls terracing a hillside", ["rural", "agriculture"]),
    ("mountain toll gate", "a simple wooden barrier gate on a mountain road", ["mountain", "travel"]),
    ("arched garden bridge", "a steeply arched bridge over an ornamental pond", ["garden", "formal"]),
    ("temple gate", "a heavy two-story gate guarding a temple's entrance", ["sacred", "formal"]),
    ("shrine torii avenue", "a long avenue of successive vermilion torii gates", ["sacred", "iconic"]),
    ("roped sacred rock", "a large rock bound with a thick sacred straw rope", ["sacred", "small"]),
    ("stone lantern row", "a row of moss-covered stone lanterns lining a path", ["garden", "sacred"]),
    ("drum tower", "a tower housing the great drum that marks the hours", ["urban", "sound"]),
    ("castle moat wall", "high stone walls rising directly from a still moat", ["urban", "defensive"]),
    ("river ferry dock", "a simple wooden dock where a ferry waits to cross", ["river", "travel"]),
    ("wooden pier", "a long wooden pier reaching out into a calm harbor", ["coastal", "travel"]),
    ("thatched roof barn", "a barn roof of thick golden thatch", ["rural", "agriculture"]),
    ("dovecote", "a small dovecote perched atop a garden wall", ["garden", "small"]),
    ("coastal watermill", "a watermill turning steadily beside a tidal channel", ["coastal", "engineering"]),
    ("mountain watermill", "a small watermill grinding grain beside a mountain stream", ["mountain", "engineering"]),
    ("covered market arcade", "a roofed arcade sheltering rows of market stalls", ["urban", "commerce"]),
    ("pilgrim rest house", "a simple rest house along a mountain pilgrim trail", ["mountain", "travel"]),
    ("cliffside shrine torii", "a torii gate standing at the very edge of a sea cliff", ["coastal", "dramatic"]),
    ("bamboo fence gate", "a simple gate woven from bamboo poles", ["rural", "small"]),
    ("garden teahouse gate", "a low rustic gate marking the entrance to a tea garden", ["garden", "formal"]),
    ("courtyard well", "a stone-rimmed well at the center of a quiet courtyard", ["domestic", "quiet"]),
    ("pagoda spire", "the ornate finial crowning a temple pagoda", ["sacred", "detail"]),
    ("shrine roof with finials", "a shrine roofline crossed with ornamental chigi finials", ["sacred", "formal"]),
    ("reroofing scaffolding", "wooden scaffolding raised for reroofing a shrine hall", ["sacred", "labor"]),
    ("rice-drying racks", "harvested rice hanging in long rows to dry", ["rural", "agriculture"]),
    ("fishing net drying racks", "nets strung out to dry along a harbor wall", ["coastal", "labor"]),
    ("outdoor bathhouse tub", "a large wooden tub steaming beside a mountain inn", ["mountain", "domestic"]),
])

# ──────────────────────────────────────────────────────────────────────────
# OBJECTS
# ──────────────────────────────────────────────────────────────────────────

OBJECTS = build_list([
    ("paper lantern", "a paper lantern swaying gently on its pole", ["light", "domestic"]),
    ("folding fan", "a painted folding fan held loosely in hand", ["craft", "detail"]),
    ("tea bowl", "a rough-glazed tea bowl set on a tatami mat", ["craft", "quiet"]),
    ("sake flask", "a ceramic sake flask warming beside a low table", ["craft", "domestic"]),
    ("wooden geta sandals", "a pair of wooden geta left beside a doorway", ["domestic", "detail"]),
    ("bamboo umbrella", "an oiled bamboo umbrella catching falling rain", ["rain", "craft"]),
    ("woven basket", "a woven bamboo basket heavy with harvested vegetables", ["craft", "rural"]),
    ("fishing net", "a fishing net drying in loose coils on a dock post", ["labor", "coastal"]),
    ("sword and scabbard", "a sword and lacquered scabbard resting across a stand", ["formal", "warrior"]),
    ("bow and arrows", "a bow and quiver of arrows leaning against a wall", ["warrior", "detail"]),
    ("taiko drum", "a large taiko drum awaiting a festival procession", ["festival", "sound"]),
    ("shamisen", "a shamisen resting across a performer's lap", ["music", "detail"]),
    ("biwa lute", "an old biwa lute held by a traveling musician", ["music", "travel"]),
    ("writing brush and inkstone", "a writing brush resting across an inkstone", ["craft", "quiet"]),
    ("scroll painting", "a hanging scroll painting displayed in an alcove", ["formal", "art"]),
    ("lacquered box", "a small lacquered box with a mother-of-pearl inlay", ["craft", "detail"]),
    ("wooden comb", "a carved wooden comb set beside a mirror stand", ["domestic", "detail"]),
    ("hair ornament", "an ornate hairpin catching the light", ["formal", "detail"]),
    ("straw raincoat", "a straw raincoat hung to dry beneath the eaves", ["rain", "rural"]),
    ("straw hat", "a wide straw hat shading a traveler's face", ["travel", "rural"]),
    ("handcart", "a wooden handcart loaded with market goods", ["labor", "urban"]),
    ("rickshaw", "a lone rickshaw waiting at a street corner", ["urban", "travel"]),
    ("fishing rod", "a bamboo fishing rod propped against a riverside stone", ["river", "quiet"]),
    ("rice-planting tools", "wooden tools laid ready at the edge of a paddy", ["rural", "labor"]),
    ("loom shuttle", "a wooden shuttle mid-pass across a loom's threads", ["craft", "detail"]),
    ("dyeing vat", "an indigo dyeing vat steaming in a workshop courtyard", ["craft", "labor"]),
    ("carving chisel", "a carver's chisel resting beside a half-finished woodblock", ["craft", "detail"]),
    ("printing block", "an inked woodblock ready for the next impression", ["craft", "detail"]),
    ("ceramic sake cup", "a small ceramic cup catching lamplight", ["domestic", "detail"]),
    ("teapot", "a cast-iron teapot resting over a low charcoal fire", ["domestic", "quiet"]),
    ("incense burner", "a bronze incense burner trailing a thin line of smoke", ["sacred", "quiet"]),
    ("prayer beads", "a string of prayer beads wound around a wrist", ["sacred", "detail"]),
    ("wind chime", "a glass wind chime stirring on a summer porch", ["summer", "sound"]),
    ("kite", "a large painted kite rising against a clear sky", ["festival", "sky"]),
    ("go board", "a go board mid-game beneath a garden pavilion", ["formal", "quiet"]),
    ("koto", "a koto resting on its stand, strings catching the light", ["music", "formal"]),
    ("palanquin", "an enclosed palanquin carried by four bearers", ["formal", "travel"]),
    ("lantern pole", "a tall lantern pole marking a shrine's entrance", ["sacred", "light"]),
    ("temple bell", "a great bronze temple bell hung in its wooden frame", ["sacred", "sound"]),
    ("offering box", "a wooden offering box worn smooth by countless hands", ["sacred", "detail"]),
])

# ──────────────────────────────────────────────────────────────────────────
# PEOPLE
# ──────────────────────────────────────────────────────────────────────────

PEOPLE_TRAVELERS = build_list([
    ("pilgrim", "a pilgrim in white robes climbing a stone shrine stairway", ["travel", "sacred"]),
    ("traveling medicine seller", "a medicine seller with a wooden case slung across his back", ["travel", "commerce"]),
    ("post-town courier", "a courier running a message between post towns at speed", ["travel", "motion"]),
    ("wandering poet", "a poet pausing beneath a pine to compose a verse", ["travel", "contemplative"]),
    ("itinerant monk", "a wandering monk walking the road with a single bowl", ["travel", "sacred"]),
    ("palanquin bearer", "bearers carrying a palanquin along a mountain road", ["travel", "labor"]),
    ("mountain guide", "a guide leading travelers along a steep pilgrim trail", ["travel", "mountain"]),
    ("traveling merchant", "a merchant with goods bundled high on his back", ["travel", "commerce"]),
    ("blind biwa player", "a blind musician led by a child along a village road", ["travel", "music"]),
    ("traveling puppeteer", "a puppeteer carrying folded stage props between towns", ["travel", "performance"]),
    ("packhorse driver", "a driver leading a train of loaded packhorses", ["travel", "labor"]),
    ("river porter", "a porter carrying a traveler across a shallow river ford", ["travel", "river"]),
])

PEOPLE_CRAFTSPEOPLE = build_list([
    ("woodblock carver", "a carver bent close over a half-finished woodblock", ["craft", "detail"]),
    ("indigo dyer", "a dyer lifting dripping cloth from a deep blue vat", ["craft", "labor"]),
    ("umbrella maker", "an umbrella maker stretching oiled paper over bamboo ribs", ["craft", "detail"]),
    ("paper lantern craftsman", "a craftsman painting characters onto a fresh lantern", ["craft", "detail"]),
    ("basket weaver", "a weaver's hands moving quickly through split bamboo", ["craft", "detail"]),
    ("potter", "a potter shaping clay at a slow-turning wheel", ["craft", "quiet"]),
    ("blacksmith", "a blacksmith striking glowing metal beside his forge", ["craft", "labor"]),
    ("carpenter", "a carpenter planing a beam beside a half-built house", ["craft", "labor"]),
    ("weaver at the loom", "a weaver's shuttle flashing through warp threads", ["craft", "detail"]),
    ("fan maker", "a fan maker gluing paper to a fresh bamboo frame", ["craft", "detail"]),
    ("lacquerware artisan", "an artisan applying fine gold lacquer to a bowl", ["craft", "detail"]),
    ("sword polisher", "a polisher drawing a blade slowly across a wet stone", ["craft", "detail"]),
    ("tatami maker", "a craftsman binding fresh reed into a tatami mat", ["craft", "labor"]),
    ("comb maker", "a comb maker carving fine teeth from boxwood", ["craft", "detail"]),
])

PEOPLE_PERFORMERS = build_list([
    ("puppet performer", "a puppeteer working three-man bunraku puppets on stage", ["performance", "urban"]),
    ("kabuki actor", "an actor mid-pose in bold kabuki costume", ["performance", "dramatic"]),
    ("street musician", "a musician drawing a small crowd at a crossroads", ["performance", "urban"]),
    ("dancer", "a dancer's sleeve caught mid-turn beneath festival lanterns", ["performance", "festival"]),
    ("storyteller", "a storyteller seated before a rapt circle of children", ["performance", "quiet"]),
    ("acrobat", "a street acrobat balanced on a bamboo pole", ["performance", "dramatic"]),
    ("shamisen player", "a shamisen player performing beneath a teahouse lantern", ["performance", "music"]),
    ("festival drum performer", "a drummer striking a great taiko at a night festival", ["performance", "festival"]),
])

PEOPLE_WARRIORS = build_list([
    ("samurai on horseback", "a samurai riding at a slow walk along a misty road", ["warrior", "iconic"], 0.4, {"cooldownGroup": "iconic-japanese-motifs"}),
    ("foot soldier", "a foot soldier resting briefly beside a mountain path", ["warrior", "quiet"]),
    ("archer", "an archer drawing his bow toward a distant target", ["warrior", "action"]),
    ("spear bearer", "a spear bearer standing guard beside a gate", ["warrior", "formal"]),
    ("mounted general", "a general surveying a valley from horseback", ["warrior", "dramatic"]),
    ("night watchman", "a watchman carrying a lantern along a sleeping street", ["warrior", "night"]),
    ("castle guard", "a guard standing motionless beside a castle gate", ["warrior", "formal"]),
    ("ronin", "a masterless ronin resting beneath a roadside tree", ["warrior", "solitary"]),
])

PEOPLE_RELIGIOUS = build_list([
    ("Buddhist monk", "a monk sweeping the stone path before a temple hall", ["sacred", "quiet"]),
    ("Shinto priest", "a priest performing a purification rite at a shrine altar", ["sacred", "formal"]),
    ("shrine maiden", "a shrine maiden performing a slow ceremonial dance", ["sacred", "formal"]),
    ("mountain ascetic", "an ascetic standing beneath a waterfall in meditation", ["sacred", "mountain"]),
    ("Buddhist nun", "a nun walking a quiet temple corridor at dawn", ["sacred", "quiet"]),
    ("pilgrim priest", "a traveling priest blessing villagers along the road", ["sacred", "travel"]),
    ("temple bell ringer", "a monk swinging the great wooden beam against the temple bell", ["sacred", "sound"]),
    ("fortune teller", "a fortune teller reading sticks beneath a shrine awning", ["sacred", "urban"]),
])

PEOPLE_VILLAGERS = build_list([
    ("tea field worker", "a worker picking new tea leaves at dawn", ["rural", "labor"]),
    ("salt farmer", "a farmer raking seawater across sunlit salt flats", ["coastal", "labor"]),
    ("charcoal burner", "a charcoal burner tending smoke rising from an earth kiln", ["forest", "labor"]),
    ("cormorant fisherman", "a fisherman directing his cormorants by torchlight", ["river", "night"]),
    ("ferryman", "a ferryman poling steadily across a wide river", ["river", "labor"]),
    ("shell gatherer", "a gatherer bent low across exposed tidal flats", ["coastal", "labor"]),
    ("rice farmer", "a farmer bent double, planting seedlings in a flooded field", ["rural", "labor"]),
    ("fishmonger", "a fishmonger calling out prices at a harbor stall", ["urban", "commerce"]),
    ("woodcutter", "a woodcutter resting his axe against a felled cedar", ["forest", "labor"]),
    ("net mender", "a fisherman's wife mending a torn net by lamplight", ["coastal", "domestic"]),
    ("farmwife at the well", "a farmwife drawing water at the village well", ["rural", "domestic"]),
    ("child at play", "a child chasing a paper kite across a village field", ["rural", "playful"]),
])

PEOPLE_COURTLY = build_list([
    ("court lady", "a court lady seated behind a painted folding screen", ["formal", "quiet"]),
    ("courtly poet", "a poet composing verse beneath a blossoming plum tree", ["formal", "contemplative"]),
    ("court musician", "a musician performing softly for an unseen audience", ["formal", "music"]),
    ("noblewoman with fan", "a noblewoman shading her face with a painted fan", ["formal", "detail"]),
    ("court attendant", "an attendant kneeling beside a low lacquered table", ["formal", "quiet"]),
    ("calligrapher", "a calligrapher's brush poised above fresh paper", ["formal", "craft"]),
    ("tea ceremony master", "a tea master whisking bright green tea with practiced calm", ["formal", "quiet"]),
    ("garden designer", "a designer directing the placement of a garden stone", ["formal", "craft"]),
])

# ──────────────────────────────────────────────────────────────────────────
# MYTHOLOGY / FOLKLORE
# ──────────────────────────────────────────────────────────────────────────

MYTH_YOKAI = build_list([
    ("kappa", "a kappa peering up from a shallow river, its head-dish full of water", ["yokai", "river", "trickster"]),
    ("tengu", "a tengu perched on a cedar branch, feathered and long-nosed", ["yokai", "mountain", "guardian"]),
    ("oni", "an oni glimpsed briefly beyond a mountain pass torii", ["yokai", "powerful"]),
    ("kitsune", "a fox spirit trailing blue foxfire through a moonlit field", ["yokai", "spirit", "iconic"], 0.5, {"cooldownGroup": "iconic-japanese-motifs"}),
    ("tanuki trickster", "a shape-shifting tanuki caught mid-transformation", ["yokai", "playful"]),
    ("yuki-onna", "a pale woman in white standing silently amid falling snow", ["yokai", "winter", "eerie"]),
    ("rokurokubi", "a long-necked spirit glimpsed at the edge of lantern light", ["yokai", "night", "eerie"]),
    ("kasa-obake", "a one-eyed umbrella spirit hopping down a rain-soaked alley", ["yokai", "playful", "night"]),
    ("noppera-bo", "a faceless figure turning slowly in an empty lane", ["yokai", "eerie"]),
    ("nurarihyon", "an elderly spirit slipping unnoticed through a lantern-lit doorway", ["yokai", "mysterious"]),
    ("azuki-arai", "a spirit heard washing beans beside a dark riverbank", ["yokai", "river", "sound"]),
    ("futakuchi-onna", "a woman whose second mouth is hidden beneath long hair", ["yokai", "eerie"]),
    ("konaki-jiji", "an infant's cry echoing from a bundle left on a mountain path", ["yokai", "mountain", "eerie"]),
    ("tsuchigumo", "a great earth spider glimpsed retreating into a cave", ["yokai", "mountain", "powerful"]),
])

MYTH_SPIRITS = build_list([
    ("mountain spirit", "a mountain spirit sensed but unseen among the peaks", ["spirit", "mountain", "sacred"]),
    ("river spirit", "an unseen presence stirring the current of a quiet river", ["spirit", "river", "sacred"]),
    ("tree spirit", "a kodama's faint glow deep within an old forest", ["spirit", "forest", "sacred"]),
    ("well spirit", "a stillness gathering around an old village well at dusk", ["spirit", "domestic", "eerie"]),
    ("wind spirit", "a sudden gust bending an entire field at once", ["spirit", "field", "powerful"]),
    ("rice-field spirit", "an offering left at the edge of a paddy for the harvest guardian", ["spirit", "agriculture", "sacred"]),
    ("household spirit", "a warm hearth glow said to be watched over by a household guardian", ["spirit", "domestic", "gentle"]),
    ("sea spirit", "a calm patch of sea where fishermen say a spirit rests", ["spirit", "sea", "sacred"]),
    ("storm spirit", "dark clouds gathering with unnatural speed over the bay", ["spirit", "weather", "powerful"]),
    ("ancestral spirit", "lanterns set adrift on a river to guide ancestral spirits home", ["spirit", "ritual", "gentle"]),
])

MYTH_LEGENDARY_ANIMALS = build_list([
    ("nine-tailed fox", "a nine-tailed fox glimpsed at the treeline in moonlight", ["legendary", "powerful", "iconic"], 0.4, {"cooldownGroup": "iconic-japanese-motifs"}),
    ("cloud dragon", "a dragon coiling through storm clouds above a mountain peak", ["legendary", "powerful", "iconic"], 0.4, {"cooldownGroup": "iconic-japanese-motifs"}),
    ("phoenix", "a phoenix-like ho-o bird descending toward a temple roof", ["legendary", "sacred"]),
    ("kirin", "a kirin standing motionless in a moonlit clearing", ["legendary", "gentle"]),
    ("shishi lion-dog", "a carved shishi guardian seeming to breathe in torchlight", ["legendary", "guardian"]),
    ("drum-bellied tanuki", "a tanuki with an impossibly large drum-taut belly", ["legendary", "playful"]),
    ("white serpent deity", "a white serpent coiled at the base of a shrine altar", ["legendary", "sacred"]),
    ("sea dragon king", "a vast presence sensed beneath deep offshore waters", ["legendary", "sea", "powerful"]),
    ("celestial crane", "a crane said to carry souls toward distant mountains", ["legendary", "sacred"]),
    ("thunder beast", "a shape glimpsed briefly within a lightning flash", ["legendary", "powerful"]),
])

MYTH_DEITIES = build_list([
    ("rice deity's fox messenger", "a stone fox messenger standing guard at a rice-harvest shrine", ["deity", "sacred", "agriculture"]),
    ("thunder deity's approach", "drums of thunder rolling ahead of a gathering storm", ["deity", "sacred", "powerful"]),
    ("wind deity's gust", "a sudden sacred wind scattering shrine offerings", ["deity", "sacred", "powerful"]),
    ("fisherman's luck deity", "a smiling figure honored at a small harbor shrine", ["deity", "sacred", "coastal"]),
    ("harvest deity's blessing", "a full granary honored with a small shrine offering", ["deity", "sacred", "agriculture"]),
    ("river and music deity", "a shrine to the river deity set where two streams meet", ["deity", "sacred", "river"]),
    ("guardian deity of the crossroads", "a small stone guardian watching over a rural crossroads", ["deity", "sacred", "travel"]),
    ("guardian of travelers", "a moss-covered stone figure dressed in a faded red bib", ["deity", "sacred", "travel"]),
    ("deity of mercy", "an old temple hall dedicated to boundless compassion", ["deity", "sacred"]),
    ("mountain guardian deity", "an unseen presence honored at a mountain summit shrine", ["deity", "sacred", "mountain"]),
    ("moon deity's light", "moonlight said to carry the deity's quiet blessing", ["deity", "sacred", "night"]),
    ("sun deity's dawn", "the first light of dawn breaking over a sacred eastern peak", ["deity", "sacred", "dawn"]),
])

MYTH_GHOSTLY = build_list([
    ("wandering yurei", "a pale figure drifting silently beneath a willow at night", ["ghost", "night", "eerie"]),
    ("vengeful spirit", "a cold presence gathering around a forgotten grave", ["ghost", "eerie"]),
    ("drowned spirit", "a faint shape rising briefly at the surface of a still river", ["ghost", "river", "eerie"]),
    ("warrior ghost", "an armored figure still standing watch on an old battlefield", ["ghost", "warrior", "eerie"]),
    ("lantern ghost", "a single lantern floating without a bearer down a dark road", ["ghost", "night", "eerie"]),
    ("bridge ghost", "a shadow that lingers at the midpoint of an old bridge", ["ghost", "eerie"]),
    ("well ghost", "a faint reflection that isn't quite the viewer's own", ["ghost", "domestic", "eerie"]),
    ("forest ghost", "a cold clearing where no birds ever seem to sing", ["ghost", "forest", "eerie"]),
    ("ghost ship crew", "a silent ship crossing the bay with no one at the oars", ["ghost", "sea", "eerie"]),
])

# ──────────────────────────────────────────────────────────────────────────
# SEASONS
#
# Independent axis (not just embedded in flora/weather tags). Every entry
# carries exactly one season-bucket tag from {spring, summer, rainy-season,
# autumn, winter} — the generator uses that bucket for hard weather
# compatibility (e.g. heavy snow excluded outside the winter bucket).
# ──────────────────────────────────────────────────────────────────────────

SEASONS = build_atmosphere_list([
    ("first signs of spring", "the first signs of spring stirring after winter", ["spring", "gentle", "transitional"], 1.0,
     ["kacho-e", "botanical-study", "fukei-ga"], ["deep-winter", "humid-midsummer"], None),
    ("early spring thaw", "the early spring thaw loosening the last frost", ["spring", "transitional"], 1.0,
     ["fukei-ga", "rural-life"], ["humid-midsummer", "peak-maple-season"], None),
    ("plum blossom season", "plum blossoms opening while the air is still cool", ["spring", "delicate"], 1.0,
     ["kacho-e", "botanical-study", "poetic-minimalism"], ["deep-winter", "humid-midsummer"], None),
    ("peak cherry blossom season", "cherry trees at the height of full bloom", ["spring", "iconic"], 0.4,
     ["kacho-e", "botanical-study", "bijin-ga-inspired"], ["deep-winter", "cicada-season"], "iconic-japanese-motifs"),
    ("late spring greenery", "the fresh, deepening green of late spring", ["spring"], 1.0,
     ["fukei-ga", "rural-life", "botanical-study"], ["deep-winter"], None),
    ("wisteria season", "wisteria cascading in its brief spring bloom", ["spring", "garden"], 0.9,
     ["kacho-e", "botanical-study"], ["deep-winter"], None),
    ("new spring growth", "tender new growth pushing up through last year's leaves", ["spring"], 1.0,
     ["botanical-study", "rural-life"], ["deep-winter"], None),
    ("snow-melt season", "the last mountain snow melting into spring streams", ["spring", "mountain", "transitional"], 0.9,
     ["fukei-ga", "sacred-landscape"], ["humid-midsummer"], None),
    ("early summer", "the freshness of early summer before the heat sets in", ["summer"], 1.0,
     ["fukei-ga", "rural-life"], ["deep-winter"], None),
    ("rainy season", "the long, grey rains of the wet season", ["rainy-season", "quiet"], 1.0,
     ["fukei-ga", "poetic-minimalism"], ["deep-winter"], None),
    ("plum rains", "the soft, persistent rains that arrive with early summer", ["rainy-season"], 0.9,
     ["botanical-study", "poetic-minimalism"], ["deep-winter"], None),
    ("iris season", "irises blooming along the marsh in early summer", ["summer", "wetland"], 0.9,
     ["kacho-e", "botanical-study"], ["deep-winter"], None),
    ("humid midsummer", "the heavy, humid air of full midsummer", ["summer"], 1.0,
     ["rural-life", "urban-edo", "maritime"], ["deep-winter", "peak-maple-season"], None),
    ("cicada season", "the droning chorus of cicadas filling the heat", ["summer", "sound"], 1.0,
     ["rural-life", "botanical-study"], ["deep-winter"], None),
    ("typhoon season", "the unsettled, storm-prone stretch of late summer", ["summer", "dramatic"], 0.8,
     ["dramatic-nature", "maritime"], ["deep-winter"], None),
    ("late summer heat", "the lingering, heavy heat of late summer", ["summer"], 1.0,
     ["rural-life", "maritime"], ["deep-winter"], None),
    ("early autumn", "the first cool edge entering the air of early autumn", ["autumn", "transitional"], 1.0,
     ["fukei-ga", "travel-scene"], ["humid-midsummer"], None),
    ("harvest season", "fields being brought in during the harvest season", ["autumn", "agriculture"], 1.0,
     ["rural-life", "fukei-ga"], ["humid-midsummer"], None),
    ("chrysanthemum season", "chrysanthemums blooming in the cool autumn air", ["autumn", "garden"], 0.9,
     ["kacho-e", "botanical-study"], ["humid-midsummer"], None),
    ("persimmon season", "persimmons ripening bright against bare branches", ["autumn", "orchard"], 0.9,
     ["rural-life", "botanical-study"], ["humid-midsummer"], None),
    ("peak maple season", "maple foliage at its most vivid autumn red", ["autumn", "iconic"], 0.45,
     ["fukei-ga", "dramatic-nature", "botanical-study"], ["humid-midsummer", "peak-cherry-blossom-season"], "iconic-japanese-motifs"),
    ("equinox transition", "the balance of light and dark at the autumn equinox", ["autumn", "transitional"], 0.9,
     ["poetic-minimalism", "sacred-landscape"], ["humid-midsummer"], None),
    ("late autumn", "the thinning light and bare branches of late autumn", ["autumn"], 1.0,
     ["fukei-ga", "travel-scene"], ["humid-midsummer"], None),
    ("first frost", "the first frost silvering the grass overnight", ["autumn", "transitional"], 0.9,
     ["poetic-minimalism", "winter-scene"], ["humid-midsummer"], None),
    ("first snowfall", "the season's first snow settling quietly over everything", ["winter", "transitional"], 0.9,
     ["winter-scene", "poetic-minimalism"], ["humid-midsummer", "cicada-season"], None),
    ("deep winter", "the stillness of deep winter at its coldest", ["winter"], 1.0,
     ["winter-scene", "sacred-landscape"], ["humid-midsummer", "cicada-season"], None),
    ("midwinter stillness", "an unmoving quiet settled over a midwinter landscape", ["winter", "quiet"], 0.9,
     ["winter-scene", "poetic-minimalism"], ["humid-midsummer"], None),
    ("hard freeze", "a hard freeze locking the rivers and paddies in ice", ["winter", "dramatic"], 0.8,
     ["winter-scene", "dramatic-nature"], ["humid-midsummer"], None),
    ("winter thaw", "the first give in the ice as winter begins to loosen", ["winter", "transitional"], 0.9,
     ["winter-scene", "fukei-ga"], ["cicada-season"], None),
])

# ──────────────────────────────────────────────────────────────────────────
# TIMES OF DAY
#
# Independent axis. Every entry carries exactly one macro daypart tag,
# "day" or "night", used by the generator for hard lighting compatibility
# (e.g. full-moon lighting excluded during a "day" time-of-day pick).
# ──────────────────────────────────────────────────────────────────────────

TIMES_OF_DAY = build_atmosphere_list([
    ("before sunrise", "the last darkness before the sky begins to lighten", ["night", "dawn", "quiet"], 1.0,
     ["fukei-ga", "sacred-landscape"], ["high-noon", "midday-glare"], None),
    ("pre-dawn darkness", "a near-total darkness just before the first light", ["night", "dawn"], 0.9,
     ["nocturne", "sacred-landscape"], ["high-noon", "midday-glare"], None),
    ("blue hour before dawn", "a cool blue light gathering ahead of sunrise", ["night", "dawn"], 0.9,
     ["fukei-ga", "poetic-minimalism"], ["high-noon"], None),
    ("dawn chorus", "the first birdsong rising in the dim light before sunrise", ["night", "dawn", "sound"], 0.9,
     ["kacho-e", "fukei-ga"], ["high-noon"], None),
    ("sunrise", "the sun breaking the horizon in a wash of colour", ["day", "dawn", "iconic"], 0.9,
     ["fukei-ga", "sacred-landscape", "travel-scene"], ["midnight", "starlit-night"], None),
    ("early morning mist", "soft mist still clinging to the ground shortly after sunrise", ["day", "dawn"], 1.0,
     ["fukei-ga", "poetic-minimalism"], ["midnight"], None),
    ("mid-morning", "the settled, even light of mid-morning", ["day"], 1.0,
     ["urban-edo", "rural-life"], ["midnight", "starlit-night"], None),
    ("high noon", "the sun directly overhead, shadows pulled short", ["day", "bright"], 1.0,
     ["urban-edo", "rural-life", "maritime"], ["midnight", "moonrise", "starlit-night"], None),
    ("midday glare", "a flat, bright glare typical of the day's peak heat", ["day", "bright"], 0.9,
     ["maritime", "rural-life"], ["midnight", "starlit-night"], None),
    ("clear morning light", "clean, bright light early in the working day", ["day"], 1.0,
     ["rural-life", "urban-edo"], ["midnight"], None),
    ("bright forenoon", "the strengthening light of the forenoon", ["day"], 0.9,
     ["urban-edo", "travel-scene"], ["midnight"], None),
    ("early afternoon", "the mellowing light of early afternoon", ["day"], 1.0,
     ["rural-life", "travel-scene"], ["midnight"], None),
    ("late afternoon", "long shadows stretching out in late afternoon", ["day"], 1.0,
     ["fukei-ga", "travel-scene"], ["midnight", "starlit-night"], None),
    ("warm mid-afternoon light", "warm, slightly slanted light of the mid-afternoon", ["day"], 0.9,
     ["botanical-study", "rural-life"], ["midnight"], None),
    ("hazy noon warmth", "a soft haze softening the brightness of noon", ["day"], 0.9,
     ["poetic-minimalism", "fukei-ga"], ["midnight"], None),
    ("golden hour", "the warm, low-angled light of the hour before sunset", ["day", "warm"], 1.0,
     ["fukei-ga", "botanical-study", "travel-scene"], ["starlit-night"], None),
    ("overcast daylight", "flat, even daylight beneath a uniform grey sky", ["day", "quiet"], 1.0,
     ["poetic-minimalism", "rural-life"], ["midnight", "starlit-night"], None),
    ("sunset", "the sun sinking below the horizon in a wash of colour", ["night", "dusk", "iconic"], 0.8,
     ["fukei-ga", "maritime", "travel-scene"], ["high-noon"], None),
    ("afterglow", "the last warm colour lingering after the sun has set", ["night", "dusk"], 0.9,
     ["fukei-ga", "poetic-minimalism"], ["high-noon"], None),
    ("twilight", "the deepening blue of early twilight", ["night", "dusk"], 1.0,
     ["nocturne", "yokai-folklore"], ["high-noon"], None),
    ("blue hour", "the short, deep-blue window just after sunset", ["night", "dusk"], 0.9,
     ["nocturne", "early-shin-hanga-inspired"], ["high-noon"], None),
    ("early evening", "the first lanterns being lit as evening settles in", ["night", "dusk"], 1.0,
     ["urban-edo", "nocturne"], ["high-noon"], None),
    ("dusk market hour", "the bustle of a market winding down at dusk", ["night", "dusk", "urban"], 0.8,
     ["urban-edo", "nocturne"], ["high-noon"], None),
    ("lantern-lit night", "a night lit only by scattered paper lanterns", ["night"], 0.9,
     ["nocturne", "urban-edo", "yokai-folklore"], ["high-noon", "midday-glare"], None),
    ("moonrise", "the moon just clearing the horizon", ["night", "iconic"], 0.7,
     ["nocturne", "yokai-folklore"], ["high-noon"], "iconic-japanese-motifs"),
    ("deep dusk", "the last grey light fading into full night", ["night", "dusk"], 0.9,
     ["nocturne", "dramatic-nature"], ["high-noon"], None),
    ("midnight", "the stillness of true midnight", ["night"], 0.9,
     ["nocturne", "yokai-folklore"], ["sunrise", "high-noon", "midday-glare"], None),
    ("hour of the ox", "the deep, still hour of night said to belong to spirits", ["night", "folklore"], 0.5,
     ["yokai-folklore", "nocturne"], ["high-noon", "midday-glare"], "supernatural-hours"),
    ("starlit night", "a moonless night lit only by a dense field of stars", ["night"], 0.8,
     ["nocturne", "sacred-landscape"], ["high-noon", "midday-glare"], None),
    ("eclipse light", "an eerie half-light as the sun is briefly obscured", ["night", "rare"], 0.3,
     ["yokai-folklore", "dramatic-nature"], ["midnight"], "rare-celestial-events"),
])

# ──────────────────────────────────────────────────────────────────────────
# WEATHER
# ──────────────────────────────────────────────────────────────────────────

WEATHER = build_list([
    ("light snowfall", "light snow drifting down without wind", ["winter", "gentle"]),
    ("heavy snowstorm", "a heavy snowstorm blurring the edges of the scene", ["winter", "dramatic"]),
    ("gentle spring rain", "a soft spring rain falling on new leaves", ["spring", "gentle"]),
    ("sudden summer downpour", "a sudden downpour driving travelers to shelter", ["summer", "dramatic"]),
    ("autumn drizzle", "a fine autumn drizzle greying the distant hills", ["autumn", "quiet"]),
    ("dense morning mist", "dense mist erasing the far side of the valley", ["mysterious", "quiet"]),
    ("rolling fog", "fog rolling steadily in from the sea", ["coastal", "mysterious"]),
    ("clear blue sky", "a clear, cloudless sky stretching to the horizon", ["bright", "calm"]),
    ("gathering storm clouds", "dark clouds building rapidly over the mountains", ["dramatic", "powerful"]),
    ("distant thunder", "the low rumble of thunder from beyond the ridge", ["dramatic", "sound"]),
    ("lightning flash", "a single flash of lightning splitting the dusk sky", ["dramatic", "powerful"]),
    ("strong coastal wind", "a strong onshore wind bending the coastal pines", ["coastal", "powerful"]),
    ("gentle breeze", "a light breeze stirring hanging lanterns", ["calm", "gentle"]),
    ("approaching typhoon", "a typhoon's outer bands lashing the harbor", ["dramatic", "powerful"]),
    ("hail shower", "a brief hail shower rattling on the roof tiles", ["dramatic", "brief"]),
    ("morning frost", "frost silvering the grass at first light", ["winter", "quiet"]),
    ("dew-laden morning", "heavy dew clinging to every blade of grass", ["quiet", "gentle"]),
    ("shimmering heat haze", "heat haze rising visibly from a summer road", ["summer", "bright"]),
    ("humid summer air", "thick humid air hanging heavy over the paddies", ["summer", "close"]),
    ("crisp autumn air", "sharp, clear autumn air carrying the scent of fallen leaves", ["autumn", "bright"]),
    ("biting winter wind", "a biting wind driving snow sideways across the road", ["winter", "powerful"]),
    ("still windless day", "an unusually still day with not a leaf stirring", ["calm", "quiet"]),
    ("rainbow after rain", "a faint rainbow arching over a rain-freshened valley", ["gentle", "hopeful"]),
    ("low storm clouds", "low, fast-moving clouds pressing close to the hilltops", ["dramatic", "close"]),
    ("sea spray mist", "salt spray drifting over the harbor wall", ["coastal", "bright"]),
    ("drifting cherry petals", "cherry petals blown loose in a sudden gust", ["spring", "gentle"]),
    ("steady autumn rain", "a steady, unhurried autumn rain soaking the fields", ["autumn", "quiet"]),
    ("cloud shadows over hills", "moving cloud shadows chasing across distant hills", ["calm", "dramatic"]),
    ("hazy spring afternoon", "a soft haze softening the edges of a spring afternoon", ["spring", "gentle"]),
    ("sudden gust scattering leaves", "a sudden gust scattering fallen leaves across a path", ["autumn", "brief"]),
    ("still humid night air", "warm, still air holding the sounds of insects close", ["summer", "night"]),
    ("cold mountain wind", "a cold wind sweeping down from a snow-capped ridge", ["mountain", "winter"]),
    ("warm southern breeze", "an early warm breeze carrying the first scent of spring", ["spring", "gentle"]),
    ("overcast winter sky", "a flat, pale grey sky holding back its snow", ["winter", "quiet"]),
    ("sun breaking through clouds", "a shaft of sun breaking briefly through storm clouds", ["dramatic", "hopeful"]),
    ("veil of falling snow", "a soft veil of falling snow blurring the far shore", ["winter", "quiet"]),
    ("driving rain on rooftops", "rain drumming hard against tiled rooftops", ["dramatic", "sound"]),
    ("quiet drizzle at dusk", "a light drizzle settling in as the light fades", ["quiet", "gentle"]),
    ("sea fog rolling into harbor", "thick fog swallowing the harbor's fishing boats", ["coastal", "mysterious"]),
    ("dry autumn wind", "a dry wind rattling dead leaves along a stone path", ["autumn", "brief"]),
    ("gathering monsoon clouds", "heavy monsoon clouds massing over the coast", ["summer", "dramatic"]),
    ("clear frosty night", "a clear night sky sharp with cold and stars", ["winter", "night"]),
    ("warm updraft over fields", "a warm updraft shimmering above sun-baked rice fields", ["summer", "bright"]),
    ("gentle snow flurries", "light snow flurries swirling without settling", ["winter", "gentle"]),
])

# ──────────────────────────────────────────────────────────────────────────
# LIGHTING
# ──────────────────────────────────────────────────────────────────────────

LIGHTING = build_list([
    ("soft dawn glow", "the soft pink glow of first light along the horizon", ["dawn", "gentle", "day"]),
    ("harsh midday sun", "flat, bright midday light with hard-edged shadows", ["day", "bright"]),
    ("golden late-afternoon light", "warm golden light raking low across the scene", ["afternoon", "warm", "day"]),
    ("deep dusk shadow", "deep blue shadow pooling as the light fades", ["dusk", "quiet", "night"]),
    ("cool blue twilight", "a cool, even blue light settling over the land", ["dusk", "calm", "night"]),
    ("warm lantern glow", "the warm amber glow of a nearby paper lantern", ["night", "warm"]),
    ("flickering candlelight", "candlelight trembling faintly against a dim wall", ["night", "intimate"]),
    ("moonlight through clouds", "moonlight breaking intermittently through drifting cloud", ["night", "mysterious"]),
    ("full moon brilliance", "a bright full moon casting sharp, silver shadows", ["night", "iconic"], 0.5, {"cooldownGroup": "iconic-japanese-motifs"}),
    ("crescent moon glow", "a thin crescent moon low over the treeline", ["night", "quiet"]),
    ("starlit darkness", "faint starlight the only illumination across a dark field", ["night", "quiet"]),
    ("hearth firelight", "the warm glow of a hearth fire lighting a room from within", ["night", "warm"]),
    ("torch procession light", "a line of torches winding down a dark mountainside", ["night", "dramatic"]),
    ("mist-filtered light", "sunlight diffused to a soft grey glow through mist", ["mysterious", "gentle"]),
    ("dappled leaf-light", "light scattered into small shifting coins beneath a canopy", ["forest", "gentle", "day"]),
    ("silhouette backlight", "a figure rendered as pure silhouette against bright sky", ["dramatic", "graphic"]),
    ("rim light on ridge", "a thin bright line outlining a distant mountain ridge", ["dramatic", "quiet"]),
    ("lantern-lined street glow", "a street glowing warmly under a line of hanging lanterns", ["night", "urban"]),
    ("reflected water light", "sunlight scattered and doubled across rippling water", ["bright", "gentle", "day"]),
    ("snow-reflected brightness", "unusually bright, even light bounced up from fresh snow", ["winter", "bright", "day"]),
    ("dramatic stormlight", "a low, ominous light beneath heavy storm clouds", ["dramatic", "powerful"]),
    ("rain-diffused grey light", "flat grey light softened by steady falling rain", ["quiet", "gentle"]),
    ("shrine bonfire glow", "the leaping orange glow of a ceremonial bonfire", ["night", "sacred"]),
    ("amber lantern procession", "a slow procession of amber lanterns along a dark path", ["night", "ritual"]),
    ("deep shadow interior", "a dim interior lit only by a single high window", ["quiet", "intimate"]),
    ("lattice sunbeams", "sunbeams falling in narrow bands through a wooden lattice", ["sacred", "detail", "day"]),
    ("glowing embers", "the last embers of a fire glowing faintly in darkness", ["night", "quiet"]),
    ("fading afterglow", "the last warm color fading from the sky after sunset", ["dusk", "quiet", "night"]),
    ("cold winter light", "pale, thin light typical of a short winter day", ["winter", "quiet", "day"]),
    ("hazy overcast light", "even, shadowless light beneath a thin overcast", ["quiet", "gentle"]),
    ("lightning-flash illumination", "a brief, stark flash of white illuminating the scene", ["dramatic", "brief"]),
    ("underwater blue-green light", "cool blue-green light filtering down through shallow water", ["mysterious", "gentle"]),
    ("early spring pale light", "the pale, still-cool light of an early spring morning", ["spring", "gentle", "day"]),
    ("hazy harvest-moon light", "a warm amber haze surrounding a rising harvest moon", ["autumn", "warm", "night"]),
    ("dim workshop lamplight", "a single oil lamp lighting a craftsman's late work", ["night", "intimate"]),
])

# ──────────────────────────────────────────────────────────────────────────
# MOODS
# ──────────────────────────────────────────────────────────────────────────

MOODS = build_list([
    (m, f"a {m} atmosphere", ["mood"]) for m in [
        "tranquil", "melancholic", "mysterious", "powerful", "sacred", "lonely", "nostalgic",
        "joyful", "dramatic", "contemplative", "serene", "restless", "wistful", "reverent",
        "playful", "ominous", "hopeful", "weary", "triumphant", "tender", "austere", "wondrous",
        "brooding", "festive", "solemn", "yearning", "defiant", "peaceful", "eerie", "jubilant",
        "resolute", "forlorn", "awe-struck", "quiet", "intimate", "stormy", "meditative",
        "vigilant", "elegiac", "exuberant", "subdued", "expectant", "haunted", "gentle",
        "fierce", "humble", "transcendent", "curious", "resigned", "exultant",
    ]
])

# ──────────────────────────────────────────────────────────────────────────
# SYMBOLISM
# ──────────────────────────────────────────────────────────────────────────

SYMBOLISM = build_list([
    (s, f"symbolising {s}", ["symbolism"]) for s in [
        "impermanence", "renewal", "courage", "wisdom", "harmony", "solitude", "journey",
        "balance", "resilience", "hope", "transience", "endurance", "purity", "loyalty",
        "sacrifice", "longing", "transformation", "humility", "perseverance", "gratitude",
        "remembrance", "devotion", "freedom", "patience", "protection", "prosperity",
        "filial piety", "honor", "redemption", "awakening", "simplicity", "unity", "fate",
        "duality", "reverence for nature", "cyclical time", "homecoming", "parting",
        "vigilance", "quiet defiance", "inner peace", "ambition", "mortality", "continuity",
        "kinship", "discipline", "wonder", "humility before nature", "guardianship", "letting go",
    ]
])

# ──────────────────────────────────────────────────────────────────────────
# COMPOSITIONS
# ──────────────────────────────────────────────────────────────────────────

COMPOSITIONS = build_list([
    ("panoramic horizontal landscape", "a panoramic horizontal composition sweeping across the landscape", ["landscape", "wide"]),
    ("tall vertical pillar composition", "a tall vertical composition drawing the eye upward", ["vertical", "dramatic"]),
    ("diagonal branch crossing frame", "a branch cutting diagonally across the frame", ["dynamic", "botanical"]),
    ("foreground obscuring element", "a foreground element partially obscuring the scene beyond", ["layered", "depth"]),
    ("tiny figure beneath monumental nature", "a tiny figure dwarfed beneath monumental natural forms", ["scale", "dramatic"]),
    ("compressed urban street", "a tightly compressed street scene layered with detail", ["urban", "dense"]),
    ("layered mist bands", "the scene divided into layered horizontal bands of mist", ["atmospheric", "landscape"]),
    ("circular current composition", "the composition turning around a circular current or eddy", ["dynamic", "water"]),
    ("off-center subject, negative space", "an off-center subject balanced by broad negative space", ["minimal", "balanced"]),
    ("close botanical crop", "a close, cropped study of a single botanical detail", ["intimate", "botanical"]),
    ("aerial river geometry", "a high aerial view revealing a river's winding geometry", ["aerial", "landscape"]),
    ("low viewpoint through reeds", "a low viewpoint looking out through a screen of reeds", ["intimate", "wetland"]),
    ("view through a doorway", "the scene framed and softened by a doorway's edge", ["framed", "intimate"]),
    ("framed by bridge rails", "the view framed between the rails of a wooden bridge", ["framed", "structural"]),
    ("distant silhouette", "a distant figure or form rendered as pure silhouette", ["minimal", "graphic"]),
    ("repeated rhythmic figures", "a rhythmic repetition of similar figures across the frame", ["pattern", "narrative"]),
    ("procession crossing image", "a procession of figures crossing steadily through the frame", ["narrative", "dynamic"]),
    ("triptych-like staging", "the scene staged as though spanning three connected panels", ["narrative", "wide"]),
    ("split foreground/background action", "distinct action unfolding in both foreground and background", ["layered", "narrative"]),
    ("seen through rain curtains", "the subject seen through streaking curtains of falling rain", ["atmospheric", "dramatic"]),
    ("steep cliff descent", "a steep diagonal composition following a cliffside descent", ["dramatic", "vertical"]),
    ("shoreline from a boat", "the shoreline composed as seen from a boat offshore", ["framed", "coastal"]),
    ("vertical waterfall axis", "a strong vertical axis following a falling waterfall", ["vertical", "dramatic"]),
    ("overlapping rooftops", "layers of overlapping rooftops receding into the distance", ["urban", "layered"]),
    ("moon as compositional anchor", "the moon placed as a fixed anchor point in the composition", ["minimal", "night"]),
    ("cropped animal entering frame", "an animal cropped at the frame's edge, caught mid-entrance", ["dynamic", "cropped"]),
    ("asymmetrical counterweight", "an asymmetrical arrangement balanced by a small counterweight element", ["balanced", "minimal"]),
    ("serpentine path composition", "a winding, serpentine path leading the eye through the scene", ["dynamic", "landscape"]),
    ("zigzag movement composition", "a zigzagging line of movement cutting through the frame", ["dynamic", "dramatic"]),
    ("quiet centered icon", "a single quiet subject placed calmly at the frame's center", ["minimal", "calm"]),
    ("dense decorative field", "a densely patterned field of repeating decorative detail", ["pattern", "dense"]),
    ("sparse poetic study", "a sparse, restrained study leaving most of the frame empty", ["minimal", "poetic"]),
    ("radial composition around focal point", "elements radiating outward from a single focal point", ["dynamic", "balanced"]),
    ("receding diagonal rooflines", "rooflines receding in strong diagonal perspective", ["urban", "dynamic"]),
    ("framed by torii gate", "the distant scene framed neatly within a torii gate's posts", ["framed", "sacred"]),
    ("nested layers of depth", "the scene built from several nested layers of depth", ["layered", "landscape"]),
    ("cropped at the waist", "a figure cropped at the waist in intimate portrait framing", ["intimate", "portrait"]),
    ("scattered petals as rhythm", "scattered falling petals setting a rhythm across the frame", ["pattern", "gentle"]),
    ("converging riverbanks", "two riverbanks converging toward a distant vanishing point", ["landscape", "dynamic"]),
    ("terraced horizontal bands", "the hillside composed as a series of terraced horizontal bands", ["landscape", "pattern"]),
    ("single branch against empty sky", "a single bare branch set against an otherwise empty sky", ["minimal", "poetic"]),
    ("cascading composition down slope", "the composition cascading downward following a hill's slope", ["dynamic", "landscape"]),
    ("tight cluster of figures", "a tight, intimate cluster of figures gathered together", ["intimate", "narrative"]),
    ("wide foreground, distant action", "a wide empty foreground with small distant action beyond", ["minimal", "landscape"]),
    ("silhouette against glowing horizon", "a silhouette set against a brightly glowing horizon line", ["dramatic", "graphic"]),
    ("framed by willow branches", "the scene softened and framed by hanging willow branches", ["framed", "gentle"]),
    ("spiral cloud formation", "clouds arranged in a slow spiraling formation overhead", ["dynamic", "sky"]),
    ("mirrored reflection composition", "the scene doubled by a still, mirror-like reflection", ["symmetrical", "water"]),
    ("stepped rooflines into mist", "rooflines stepping back in stages until lost in mist", ["urban", "atmospheric"]),
    ("single lantern as focal anchor", "a single glowing lantern anchoring an otherwise dark frame", ["minimal", "night"]),
    ("scattered birds in flight pattern", "birds scattered across the sky in a loose flight pattern", ["dynamic", "sky"]),
    ("bridge as central diagonal", "a bridge crossing the frame as a strong central diagonal", ["structural", "dynamic"]),
    ("path curving into distance", "a path curving gently away toward a distant vanishing point", ["landscape", "dynamic"]),
    ("rock formation as frame", "a large rock formation used to frame the scene beyond", ["framed", "dramatic"]),
    ("layered horizon bands at sea", "the sea composed in calm layered horizontal bands", ["landscape", "coastal"]),
])

# ──────────────────────────────────────────────────────────────────────────
# PERSPECTIVES
# ──────────────────────────────────────────────────────────────────────────

PERSPECTIVES = build_list([
    (p, p, ["perspective"]) for p in [
        "an eye-level view", "a worm's-eye view from below", "an over-the-shoulder view",
        "a distant vantage point", "a view framed through foreground branches",
        "a view glimpsed through a torii gate", "a view reflected in still water",
        "a receding vanishing-point view", "an elevated oblique view", "a bird's-eye aerial view",
        "a ground-level close view", "a view from a passing boat", "a view from a mountain ledge",
        "a view through a window", "a view from within a palanquin", "a view looking up a stairway",
        "a view down a winding path", "a view across a courtyard", "a view from a bridge",
        "a view from behind a curtain of rain", "a view through drifting mist",
        "a view from a rooftop", "a view along a riverbank", "a view from beneath a leaf canopy",
        "a view from a doorway threshold", "a view from horseback", "a view from a moving cart",
        "a view through a bamboo lattice", "a view along a shoreline", "a view from a hilltop shrine",
        "a view through rising smoke", "a view from within a boat cabin",
        "a view from a snow-covered slope", "a view through hanging lanterns",
        "a view from a garden veranda",
    ]
])

# ──────────────────────────────────────────────────────────────────────────
# MOVEMENT
# ──────────────────────────────────────────────────────────────────────────

MOVEMENT = build_list([
    (m, m, ["movement"]) for m in [
        "drifting snowflakes", "falling cherry petals", "rippling water", "rising mist",
        "swirling wind", "cresting waves", "fluttering banners", "swaying reeds",
        "circling birds in flight", "a galloping horse", "a running figure",
        "flowing river current", "billowing smoke", "a wavering candle flame",
        "rustling bamboo leaves", "a cascading waterfall", "scattering leaves in wind",
        "drifting clouds", "a gliding boat", "a leaping fish", "a soaring crane",
        "darting fireflies", "a spinning kite", "a swaying lantern", "a rolling fog bank",
        "a surging tide", "trembling branches", "an unfurling banner", "a striding traveler",
        "a dancing figure mid-motion", "whirling dust", "flickering firelight",
        "leaves drifting on water", "steam rising from a hot spring",
        "a spider web quivering in the wind",
    ]
])

# ──────────────────────────────────────────────────────────────────────────
# PALETTES
# ──────────────────────────────────────────────────────────────────────────

PALETTES = build_list([
    ("aizuri-e blue monochrome", "rendered entirely in graded shades of indigo blue", ["monochrome", "blue"], 0.5, {"cooldownGroup": "indigo-palettes"}),
    ("benizuri-e rose and green", "soft rose pink and muted green with pale paper ground", ["polychrome", "soft"]),
    ("muted mineral spring", "muted mineral green, soft pink, and warm grey", ["spring", "soft"]),
    ("persimmon and charcoal", "warm persimmon orange against deep charcoal", ["autumn", "warm"]),
    ("tea brown and moss", "earthy tea brown paired with quiet moss green", ["earthy", "quiet"]),
    ("pale celadon and warm grey", "pale celadon green balanced by warm grey", ["cool", "calm"]),
    ("vermilion shrine palette", "bold vermilion red with black and warm wood tones", ["sacred", "bold"]),
    ("winter blue-grey", "cool blue-grey tones suited to snow and stillness", ["winter", "cool"]),
    ("plum, soot, and parchment", "deep plum, soft soot black, and warm parchment", ["night", "warm"]),
    ("golden ochre and deep green", "golden ochre grounded by deep forest green", ["autumn", "rich"]),
    ("faded rose and river blue", "faded dusty rose paired with soft river blue", ["gentle", "cool"]),
    ("storm violet and iron grey", "storm violet against heavy iron grey", ["dramatic", "cool"]),
    ("moonlit silver-blue", "cool silver-blue tones suited to night scenes", ["night", "cool"]),
    ("autumn maple red and tan", "bright maple red balanced with warm tan", ["autumn", "warm"]),
    ("pale dawn peach", "pale peach and soft gold suited to early morning", ["dawn", "soft"]),
    ("summer turquoise and reed green", "bright turquoise water tones with reed green", ["summer", "bright"]),
    ("black ink monochrome", "rendered in ink black and its diluted greys alone", ["monochrome", "restrained"]),
    ("sepia and smoke", "warm sepia brown softened with pale smoke grey", ["muted", "warm"]),
    ("restrained polychrome", "a restrained palette of three or four quiet colors", ["balanced", "quiet"]),
    ("dark nocturne palette", "deep indigo-black and muted amber for night scenes", ["night", "dramatic"], 0.6, {"cooldownGroup": "indigo-palettes"}),
    ("cedar green and rain grey", "deep cedar green paired with soft rain grey", ["forest", "quiet"]),
    ("saffron with indigo accent", "warm saffron yellow with indigo used only as a small accent", ["warm", "balanced"]),
    ("dusty rose and slate", "dusty rose softened against cool slate grey", ["gentle", "cool"]),
    ("amber lantern glow", "warm amber and deep brown suited to lantern light", ["night", "warm"]),
    ("frost white and pine green", "crisp frost white with deep pine green", ["winter", "cool"]),
    ("copper sunset and charcoal", "warm copper orange fading into charcoal shadow", ["dusk", "warm"]),
    ("jade and warm ivory", "cool jade green balanced by warm ivory ground", ["calm", "balanced"]),
    ("plum blossom pink and snow white", "soft plum-blossom pink against clean snow white", ["late-winter", "soft"]),
    ("autumn gold and burnt sienna", "rich autumn gold paired with burnt sienna", ["autumn", "warm"]),
    ("deep forest green and mist grey", "deep forest green softened by pale mist grey", ["forest", "quiet"]),
    ("coral dawn and pale gold", "soft coral pink and pale gold at first light", ["dawn", "soft"]),
    ("iron blue and rust", "cool iron blue paired with warm rust orange", ["contrast", "bold"]),
    ("warm sand and driftwood grey", "warm sand tones with weathered driftwood grey", ["coastal", "warm"]),
    ("lotus pink and pond green", "soft lotus pink against still pond green", ["summer", "soft"]),
    ("midnight indigo with warm ochre", "midnight indigo balanced by a warm ochre accent", ["night", "balanced"], 0.6, {"cooldownGroup": "indigo-palettes"}),
    ("straw yellow and charcoal", "pale straw yellow grounded by deep charcoal", ["rural", "warm"]),
    ("dusty lavender and grey", "soft dusty lavender paired with quiet grey", ["gentle", "cool"]),
    ("harvest gold and umber", "rich harvest gold deepening into umber shadow", ["autumn", "warm"]),
    ("cool jade and pale lilac", "cool jade green softened with pale lilac", ["calm", "cool"]),
    ("crimson maple and dusk blue", "crimson maple red set against deepening dusk blue", ["autumn", "dramatic"]),
    ("weathered bronze and moss", "weathered bronze tones with quiet moss green", ["earthy", "quiet"]),
    ("pale peach and river silver", "pale peach warmth against cool river silver", ["gentle", "balanced"]),
    ("ash grey and ember orange", "cool ash grey punctuated by warm ember orange", ["dramatic", "contrast"]),
    ("deep plum and gold leaf", "deep plum shadow accented with gold leaf detail", ["formal", "rich"]),
    ("seafoam and driftwood tan", "pale seafoam green with warm driftwood tan", ["coastal", "soft"]),
])

# ──────────────────────────────────────────────────────────────────────────
# PRINT TECHNIQUES
# ──────────────────────────────────────────────────────────────────────────

PRINT_TECHNIQUES = build_list([
    (t, t, ["technique"]) for t in [
        "multi-block polychrome printing", "blue monochrome printing", "hand-applied gradation (bokashi)",
        "bold black keyline carving", "fine hairline linework", "embossed blind-printed texture",
        "mica-dusted shimmer", "layered flat colour blocks", "a woodgrain-textured background pass",
        "delicate botanical linework", "dynamic diagonal carving strokes", "a soft brushed ink wash effect",
        "restrained two-colour printing", "bold narrative-scene carving", "atmospheric sky gradation",
        "crisp architectural linework", "a textured rain-line overlay", "hand-rubbed pigment texture",
        "layered mist gradation", "fine crosshatched shading", "flowing calligraphic linework",
        "sharp-edged silhouette carving", "dappled light stippling", "subtle overprinting texture",
        "bold outline with flat infill", "tonal woodgrain shading", "delicate feather-fine linework",
        "rough folk-print carving", "layered translucent colour washes", "precise architectural crosshatching",
        "a soft-edge bokashi horizon", "dense decorative pattern carving", "minimal single-colour linework",
        "rich saturated block printing", "early shin-hanga atmospheric shading",
    ]
])

# ──────────────────────────────────────────────────────────────────────────
# SURFACE TEXTURES (paper / print surface)
# ──────────────────────────────────────────────────────────────────────────

SURFACE_TEXTURES = build_list([
    (t, t, ["texture"]) for t in [
        "warm ivory washi grain", "cool grey mulberry-paper texture", "soft cream fibrous texture",
        "an aged parchment tone", "fine-grained hosho-paper texture", "a subtle vertical fiber texture",
        "warm oatmeal paper tone", "a pale blue-grey paper tone", "a lightly mica-flecked surface",
        "a smooth pale grey surface", "warm sepia-toned aged paper", "faint horizontal laid lines",
        "a soft deckled edge texture", "a pale golden parchment tone", "cool silver-grey fiber texture",
        "warm biscuit-toned paper", "a subtle speckled fiber texture", "a muted straw-toned surface",
        "a pale jade-tinted paper tone", "a soft dove-grey surface", "warm terracotta-toned paper",
        "a faint woodgrain impression", "a delicate crepe-textured surface", "a pale moss-toned paper",
        "warm parchment with light foxing", "a cool bone-white surface", "a subtle linen-weave texture",
        "a pale smoke-grey tone", "warm honey-toned paper", "a soft ash-toned surface",
        "a fine ridged fiber texture", "a pale lavender-grey tone", "warm dust-toned parchment",
        "a cool celadon paper tint", "a muted charcoal-flecked surface",
    ]
])

# ──────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ──────────────────────────────────────────────────────────────────────────

GLOBAL_STYLES = [
    {
        "id": "late-edo-landscape-print",
        "label": "Late Edo landscape print",
        "promptText": "Rendered as a late Edo-period landscape woodblock print, with confident black outlines and flat layered colour.",
        "compatibleGenres": ["fukei-ga", "sacred-landscape", "travel-scene", "maritime"],
        "lineCharacter": "confident, medium-weight outlines",
        "colorBehavior": "flat layered fields with graded skies",
        "textureBehavior": "visible woodgrain in open areas",
        "weight": 1.0,
    },
    {
        "id": "restrained-kacho-e",
        "label": "Restrained bird-and-flower print",
        "promptText": "Rendered as a restrained kachō-e bird-and-flower print, with close observation and quiet negative space.",
        "compatibleGenres": ["kacho-e", "botanical-study", "poetic-minimalism"],
        "lineCharacter": "fine, precise linework",
        "colorBehavior": "limited, naturalistic palette",
        "textureBehavior": "minimal, paper left visible",
        "weight": 1.0,
    },
    {
        "id": "bold-narrative-energy",
        "label": "Bold narrative energy",
        "promptText": "Rendered with bold narrative energy in the manner of dramatic Edo-period warrior prints, without copying any specific composition.",
        "compatibleGenres": ["musha-e", "dramatic-nature", "narrative-triptych-inspired"],
        "lineCharacter": "thick, expressive outlines",
        "colorBehavior": "saturated, high-contrast blocks",
        "textureBehavior": "dynamic carved texture in motion areas",
        "weight": 0.8,
    },
    {
        "id": "lyrical-atmospheric-landscape",
        "label": "Lyrical atmospheric landscape",
        "promptText": "Rendered as a lyrical, atmospheric landscape print with soft gradation and quiet depth, inspired broadly by Edo travel-print traditions.",
        "compatibleGenres": ["fukei-ga", "travel-scene", "sacred-landscape"],
        "lineCharacter": "soft, understated outlines",
        "colorBehavior": "gentle bokashi gradation",
        "textureBehavior": "atmospheric mist layering",
        "weight": 1.0,
    },
    {
        "id": "delicate-botanical-woodblock",
        "label": "Delicate botanical woodblock treatment",
        "promptText": "Rendered as a delicate botanical woodblock study, precise and quietly observant.",
        "compatibleGenres": ["botanical-study", "kacho-e", "poetic-minimalism"],
        "lineCharacter": "fine hairline detail",
        "colorBehavior": "soft, naturalistic tones",
        "textureBehavior": "minimal, precise",
        "weight": 0.9,
    },
    {
        "id": "aizuri-e-monochrome",
        "label": "Aizuri-e monochrome print",
        "promptText": "Rendered entirely in graded blue monochrome, in the aizuri-e printing tradition.",
        "compatibleGenres": ["nocturne", "maritime", "sacred-landscape"],
        "lineCharacter": "clean, graphic outlines",
        "colorBehavior": "single-hue graded blue",
        "textureBehavior": "smooth gradation, minimal grain",
        "weight": 0.6,
    },
    {
        "id": "early-meiji-dramatic-light",
        "label": "Early Meiji dramatic light",
        "promptText": "Rendered with the dramatic light and deeper shadow of early Meiji-period woodblock prints.",
        "compatibleGenres": ["dramatic-nature", "urban-edo", "nocturne"],
        "lineCharacter": "firm outlines with tonal shading",
        "colorBehavior": "deeper contrast, richer shadow",
        "textureBehavior": "subtle tonal gradation",
        "weight": 0.8,
    },
    {
        "id": "early-shin-hanga-depth",
        "label": "Early shin-hanga atmospheric depth",
        "promptText": "Rendered with the atmospheric depth and soft realism of early shin-hanga prints.",
        "compatibleGenres": ["fukei-ga", "nocturne", "sacred-landscape"],
        "lineCharacter": "soft, refined outlines",
        "colorBehavior": "subtle tonal gradation",
        "textureBehavior": "atmospheric, layered depth",
        "weight": 0.9,
    },
    {
        "id": "hand-carved-folk-character",
        "label": "Hand-carved folk woodblock character",
        "promptText": "Rendered with the rougher, hand-carved character of rural folk woodblock printing.",
        "compatibleGenres": ["rural-life", "yokai-folklore", "travel-scene"],
        "lineCharacter": "irregular, hand-carved outlines",
        "colorBehavior": "earthy, limited palette",
        "textureBehavior": "visible carving texture",
        "weight": 0.7,
    },
]

# ──────────────────────────────────────────────────────────────────────────
# GENRE PROFILES
# ──────────────────────────────────────────────────────────────────────────

GENRE_PROFILES = {
    "kacho-e": {
        "label": "Bird-and-flower print",
        "weight": 1.0,
        "preferredSubjectTags": ["bird", "insect", "spring", "garden"],
        "preferredCompositions": ["close-botanical-crop", "diagonal-branch-crossing-frame"],
        "preferredMoods": ["quiet", "contemplative", "tranquil"],
        "preferredPalettes": ["muted-mineral-spring", "restrained-polychrome"],
        "preferredSeasonTags": ["spring", "summer"],
        "preferredTimeTags": ["day", "dawn"],
        "disallowedTags": ["battle", "warrior"],
    },
    "fukei-ga": {
        "label": "Landscape print",
        "weight": 1.0,
        "preferredSubjectTags": ["mountain", "river", "coastal", "landscape"],
        "preferredCompositions": ["panoramic-horizontal-landscape", "layered-mist-bands"],
        "preferredMoods": ["tranquil", "contemplative", "awe-struck"],
        "preferredPalettes": ["winter-blue-grey", "cedar-green-and-rain-grey"],
        "disallowedTags": [],
    },
    "musha-e": {
        "label": "Warrior print",
        "weight": 0.6,
        "preferredSubjectTags": ["warrior", "dramatic"],
        "preferredCompositions": ["procession-crossing-image", "tight-cluster-of-figures"],
        "preferredMoods": ["dramatic", "triumphant", "fierce"],
        "preferredPalettes": ["vermilion-shrine-palette", "iron-blue-and-rust"],
        "disallowedTags": ["botanical", "minimal"],
    },
    "bijin-ga-inspired": {
        "label": "Courtly figure study",
        "weight": 0.7,
        "preferredSubjectTags": ["formal", "quiet", "detail"],
        "preferredCompositions": ["cropped-at-the-waist", "off-center-subject-negative-space"],
        "preferredMoods": ["elegant", "quiet", "reverent"],
        "preferredPalettes": ["dusty-rose-and-slate", "pale-dawn-peach"],
        "disallowedTags": ["battle"],
    },
    "yokai-folklore": {
        "label": "Yokai and folklore scene",
        "weight": 0.7,
        "preferredSubjectTags": ["yokai", "spirit", "ghost", "legendary"],
        "preferredCompositions": ["silhouette-against-glowing-horizon", "single-lantern-as-focal-anchor"],
        "preferredMoods": ["eerie", "mysterious", "haunted"],
        "preferredPalettes": ["dark-nocturne-palette", "plum-soot-and-parchment"],
        "disallowedTags": [],
        "supernaturalOverride": True,
    },
    "maritime": {
        "label": "Maritime and coastal scene",
        "weight": 0.9,
        "preferredSubjectTags": ["coastal", "sea", "marine", "harbor"],
        "preferredCompositions": ["shoreline-from-a-boat", "layered-horizon-bands-at-sea"],
        "preferredMoods": ["powerful", "dramatic", "quiet"],
        "preferredPalettes": ["seafoam-and-driftwood-tan", "warm-sand-and-driftwood-grey"],
        "disallowedTags": [],
    },
    "rural-life": {
        "label": "Rural and agricultural life",
        "weight": 1.0,
        "preferredSubjectTags": ["rural", "agriculture", "labor", "village"],
        "preferredCompositions": ["terraced-horizontal-bands", "tight-cluster-of-figures"],
        "preferredMoods": ["nostalgic", "peaceful", "weary"],
        "preferredPalettes": ["straw-yellow-and-charcoal", "harvest-gold-and-umber"],
        "disallowedTags": [],
    },
    "urban-edo": {
        "label": "Urban Edo street scene",
        "weight": 0.9,
        "preferredSubjectTags": ["urban", "commerce", "travel"],
        "preferredCompositions": ["compressed-urban-street", "overlapping-rooftops"],
        "preferredMoods": ["lively", "festive", "curious"],
        "preferredPalettes": ["amber-lantern-glow", "persimmon-and-charcoal"],
        "disallowedTags": [],
    },
    "sacred-landscape": {
        "label": "Sacred landscape and shrine scene",
        "weight": 1.0,
        "preferredSubjectTags": ["sacred", "shrine", "temple"],
        "preferredCompositions": ["framed-by-torii-gate", "mountain-shrine-steps"],
        "preferredMoods": ["reverent", "sacred", "quiet"],
        "preferredPalettes": ["vermilion-shrine-palette", "jade-and-warm-ivory"],
        "disallowedTags": [],
    },
    "botanical-study": {
        "label": "Intimate botanical study",
        "weight": 0.8,
        "preferredSubjectTags": ["flora", "garden", "botanical"],
        "preferredCompositions": ["close-botanical-crop", "single-branch-against-empty-sky"],
        "preferredMoods": ["quiet", "intimate", "contemplative"],
        "preferredPalettes": ["pale-celadon-and-warm-grey", "cool-jade-and-pale-lilac"],
        "disallowedTags": ["battle", "monumental"],
    },
    "nocturne": {
        "label": "Night scene",
        "weight": 0.8,
        "preferredSubjectTags": ["night", "dusk"],
        "preferredCompositions": ["single-lantern-as-focal-anchor", "moon-as-compositional-anchor"],
        "preferredMoods": ["mysterious", "quiet", "haunted"],
        "preferredPalettes": ["dark-nocturne-palette", "moonlit-silver-blue"],
        "preferredTimeTags": ["night", "dusk"],
        "disallowedTags": [],
    },
    "winter-scene": {
        "label": "Winter scene",
        "weight": 0.8,
        "preferredSubjectTags": ["winter", "snow"],
        "preferredCompositions": ["sparse-poetic-study", "single-branch-against-empty-sky"],
        "preferredMoods": ["quiet", "austere", "melancholic"],
        "preferredPalettes": ["frost-white-and-pine-green", "winter-blue-grey"],
        "preferredSeasonTags": ["winter"],
        "disallowedTags": [],
    },
    "travel-scene": {
        "label": "Travel and road scene",
        "weight": 0.9,
        "preferredSubjectTags": ["travel", "road", "post-town"],
        "preferredCompositions": ["serpentine-path-composition", "path-curving-into-distance"],
        "preferredMoods": ["nostalgic", "weary", "hopeful"],
        "preferredPalettes": ["tea-brown-and-moss", "warm-sand-and-driftwood-grey"],
        "disallowedTags": [],
    },
    "dramatic-nature": {
        "label": "Dramatic natural forces",
        "weight": 0.7,
        "preferredSubjectTags": ["powerful", "storm", "dramatic"],
        "preferredCompositions": ["vertical-waterfall-axis", "zigzag-movement-composition"],
        "preferredMoods": ["powerful", "dramatic", "awe-struck"],
        "preferredPalettes": ["storm-violet-and-iron-grey", "ash-grey-and-ember-orange"],
        "disallowedTags": ["minimal"],
    },
    "poetic-minimalism": {
        "label": "Poetic minimalism",
        "weight": 0.8,
        "preferredSubjectTags": ["quiet", "small", "solitary"],
        "preferredCompositions": ["sparse-poetic-study", "quiet-centered-icon"],
        "preferredMoods": ["contemplative", "quiet", "serene"],
        "preferredPalettes": ["black-ink-monochrome", "restrained-polychrome"],
        "disallowedTags": ["dense", "dramatic"],
    },
    "narrative-triptych-inspired": {
        "label": "Narrative triptych-inspired scene",
        "weight": 0.6,
        "preferredSubjectTags": ["narrative", "group", "action"],
        "preferredCompositions": ["triptych-like-staging", "split-foreground-background-action"],
        "preferredMoods": ["dramatic", "dynamic", "epic"],
        "preferredPalettes": ["deep-plum-and-gold-leaf", "crimson-maple-and-dusk-blue"],
        "disallowedTags": [],
    },
    "early-shin-hanga-inspired": {
        "label": "Early shin-hanga inspired atmosphere",
        "weight": 0.7,
        "preferredSubjectTags": ["atmospheric", "quiet", "landscape"],
        "preferredCompositions": ["layered-mist-bands", "stepped-rooflines-into-mist"],
        "preferredMoods": ["contemplative", "serene", "wistful"],
        "preferredPalettes": ["moonlit-silver-blue", "coral-dawn-and-pale-gold"],
        "disallowedTags": [],
    },
}

# ──────────────────────────────────────────────────────────────────────────
# COMPATIBILITY / EXCLUSION RULES (tag-based)
# ──────────────────────────────────────────────────────────────────────────

COMPATIBILITY_RULES = [
    {"id": "marine-favors-water", "ifTag": "marine", "preferEnvironmentTags": ["coastal", "river", "wetland"]},
    {"id": "fish-favors-water", "ifTag": "fish", "preferEnvironmentTags": ["river", "pond", "coastal"]},
    {"id": "snow-monkey-favors-winter", "ifTag": "winter", "preferSeasonTags": ["winter"], "preferEnvironmentTags": ["mountain"]},
    {"id": "lotus-favors-summer", "ifId": "lotus", "preferSeasonTags": ["summer"], "preferEnvironmentTags": ["pond", "wetland"]},
    {"id": "firefly-favors-summer-night", "ifId": "firefly", "preferSeasonTags": ["summer"], "preferTimeTags": ["night", "dusk"]},
    {"id": "cherry-blossom-favors-spring", "ifId": "cherry-blossom", "preferSeasonTags": ["spring"]},
    {"id": "maple-favors-autumn", "ifId": "japanese-maple", "preferSeasonTags": ["autumn"]},
    {"id": "urban-merchants-favor-streets", "ifTag": "commerce", "preferEnvironmentTags": ["urban", "village"]},
    {"id": "warrior-avoids-botanical-composition", "ifTag": "warrior", "avoidCompositionTags": ["botanical", "minimal"]},
    {"id": "insect-avoids-monumental-composition", "ifTag": "insect", "avoidCompositionTags": ["dramatic", "wide"]},
    {"id": "marine-avoids-dry-mountain", "ifTag": "marine", "avoidEnvironmentTags": ["mountain"]},
    {"id": "yokai-can-override-season", "ifTag": "yokai", "supernaturalOverride": True},
    {"id": "coastal-fisherfolk-favor-coastal-env", "ifTag": "coastal", "preferEnvironmentTags": ["coastal", "harbor"]},
    {"id": "sacred-figures-favor-sacred-env", "ifTag": "sacred", "preferEnvironmentTags": ["sacred"]},
    {"id": "night-lighting-favors-night-time", "ifTag": "night", "preferTimeTags": ["night", "dusk"]},
    {"id": "cicada-favors-midsummer-day-or-evening", "ifId": "cicada", "preferSeasonTags": ["summer"], "preferTimeTags": ["day", "dusk"]},
    {"id": "maple-favors-autumn-season-too", "ifTag": "autumn", "preferSeasonTags": ["autumn"]},
    {"id": "cherry-blossom-tag-favors-spring-season", "ifTag": "spring", "preferSeasonTags": ["spring"]},
]

EXCLUSION_RULES = [
    {"id": "no-heavy-snow-in-midsummer", "conflict": [{"tag": "winter"}, {"tag": "summer"}], "unless": "supernaturalOverride"},
    {"id": "no-battle-with-botanical-study-composition", "conflict": [{"tag": "warrior"}, {"compositionTag": "botanical"}]},
    {"id": "no-insect-with-panoramic-battlefield", "conflict": [{"tag": "insect"}, {"compositionTag": "wide", "subjectTag": "warrior"}]},
    {"id": "no-underwater-subject-in-dry-mountain-env", "conflict": [{"tag": "marine"}, {"environmentTag": "mountain"}]},
]

# ──────────────────────────────────────────────────────────────────────────
# GENERATION SETTINGS
# ──────────────────────────────────────────────────────────────────────────

GENERATION_SETTINGS = {
    "batchSize": 10,
    "historyWindow": 50,
    "titleHistoryWindow": 20000,
    "pairHistoryWindow": 100,
    "maxGenrePerBatch": 2,
    "maxPaletteFamilyPerBatch": 2,
    "maxCompositionFamilyPerBatch": 2,
    "maxNocturnalPerBatch": 2,
    "maxMinimalPerBatch": 2,
    "iconicMotifPenalty": 0.3,
    "recentUsePenalty": 0.15,
    "minimumDistinctCompositionFamilies": 5,
    "minimumDistinctPaletteFamilies": 5,
    "minimumSeasonsOrAtmospheresPerBatch": 3,
    "requireAtLeastOneHumanScenePerBatch": True,
    "requireAtLeastOneAnimalScenePerBatch": True,
    "requireAtLeastOneBotanicalScenePerBatch": True,
    "requireAtLeastOneArchitecturalScenePerBatch": True,
    "requireAtLeastOneDramaticScenePerBatch": True,
    "requireAtLeastOneQuietScenePerBatch": True,
    "wordCountTarget": {"min": 70, "max": 130},
    "promptWordCountTarget": [70, 130],
}

# ──────────────────────────────────────────────────────────────────────────
# ICONIC MOTIFS (documentation of which entries carry a repetition penalty)
# ──────────────────────────────────────────────────────────────────────────

ICONIC_MOTIFS = [
    {"category": "subjects.animals.mammals", "id": "red-fox", "reason": "overused fox motif"},
    {"category": "subjects.animals.fish", "id": "koi-carp", "reason": "overused koi motif"},
    {"category": "subjects.animals.birds", "id": "red-crowned-crane", "reason": "overused crane motif"},
    {"category": "flora", "id": "cherry-blossom", "reason": "overused sakura motif"},
    {"category": "subjects.architecture", "id": "torii-gate", "reason": "overused torii motif"},
    {"category": "subjects.architecture", "id": "multi-tiered-pagoda", "reason": "overused pagoda motif"},
    {"category": "subjects.people.warriors", "id": "samurai-on-horseback", "reason": "overused samurai motif"},
    {"category": "subjects.mythology.legendaryAnimals", "id": "cloud-dragon", "reason": "overused dragon motif"},
    {"category": "subjects.mythology.legendaryAnimals", "id": "nine-tailed-fox", "reason": "overused fox-spirit motif"},
    {"category": "subjects.mythology.yokai", "id": "kitsune", "reason": "overused fox-spirit motif"},
    {"category": "lighting", "id": "full-moon-brilliance", "reason": "overused full-moon motif"},
    {"category": "palettes", "id": "aizuri-e-blue-monochrome", "reason": "indigo overuse — capped via cooldownGroup indigo-palettes"},
    {"category": "palettes", "id": "dark-nocturne-palette", "reason": "indigo overuse — capped via cooldownGroup indigo-palettes"},
    {"category": "palettes", "id": "midnight-indigo-with-warm-ochre", "reason": "indigo overuse — capped via cooldownGroup indigo-palettes"},
    {"category": "seasons", "id": "peak-cherry-blossom-season", "reason": "overused sakura-season motif"},
    {"category": "seasons", "id": "peak-maple-season", "reason": "overused maple-season motif"},
    {"category": "timesOfDay", "id": "moonrise", "reason": "overused full-moon-adjacent motif"},
]

# ──────────────────────────────────────────────────────────────────────────
# METADATA
# ──────────────────────────────────────────────────────────────────────────

METADATA = {
    "name": "Ukiyo-e Generative Art Dataset",
    "version": "1.0.0",
    "language": "en",
    "description": (
        "A curated, tag-driven dataset for procedurally assembling historically-informed "
        "traditional Japanese ukiyo-e-style artwork concepts. Built for combinatorial prompt "
        "generation rather than direct historical reproduction."
    ),
    "createdFor": "AI artwork concept and prompt generation for a printable wall-art pipeline",
    "historicalScope": [
        "Edo-period ukiyo-e", "Meiji-period woodblock aesthetics", "kachō-e", "fūkei-ga",
        "musha-e", "yōkai and folklore imagery", "early shin-hanga influence",
    ],
    "disclaimer": (
        "This dataset is historically informed and visually inspired by Japanese woodblock "
        "print traditions. It does not claim that generated concepts are historically authentic "
        "reproductions, and folklore entries are presented as folklore, not as confirmed fact."
    ),
}


def build_dataset() -> dict:
    dataset = {
        "metadata": METADATA,
        "globalStyles": GLOBAL_STYLES,
        "genreProfiles": GENRE_PROFILES,
        "subjects": {
            "animals": {
                "birds": ANIMAL_BIRDS,
                "mammals": ANIMAL_MAMMALS,
                "fish": ANIMAL_FISH,
                "marineLife": ANIMAL_MARINE,
                "insects": ANIMAL_INSECTS,
                "reptilesAndAmphibians": ANIMAL_REPTILES,
            },
            "people": {
                "travelers": PEOPLE_TRAVELERS,
                "craftspeople": PEOPLE_CRAFTSPEOPLE,
                "performers": PEOPLE_PERFORMERS,
                "warriors": PEOPLE_WARRIORS,
                "religiousFigures": PEOPLE_RELIGIOUS,
                "villagers": PEOPLE_VILLAGERS,
                "courtlyFigures": PEOPLE_COURTLY,
            },
            "mythology": {
                "yokai": MYTH_YOKAI,
                "spirits": MYTH_SPIRITS,
                "legendaryAnimals": MYTH_LEGENDARY_ANIMALS,
                "deities": MYTH_DEITIES,
                "ghostlyFigures": MYTH_GHOSTLY,
            },
            "architecture": ARCHITECTURE,
            "objects": OBJECTS,
        },
        "flora": FLORA,
        "environments": ENVIRONMENTS,
        "seasons": SEASONS,
        "timesOfDay": TIMES_OF_DAY,
        "weather": WEATHER,
        "lighting": LIGHTING,
        "moods": MOODS,
        "symbolism": SYMBOLISM,
        "compositions": COMPOSITIONS,
        "perspectives": PERSPECTIVES,
        "movement": MOVEMENT,
        "palettes": PALETTES,
        "printTechniques": PRINT_TECHNIQUES,
        "surfaceTextures": SURFACE_TEXTURES,
        "iconicMotifs": ICONIC_MOTIFS,
        "compatibilityRules": COMPATIBILITY_RULES,
        "exclusionRules": EXCLUSION_RULES,
        "generationSettings": GENERATION_SETTINGS,
    }
    return dataset


def category_counts(dataset: dict) -> dict:
    counts = {}

    def walk(node, path):
        if isinstance(node, list):
            if node and isinstance(node[0], dict) and "id" in node[0]:
                counts[path] = len(node)
            return
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else key)

    walk(dataset["subjects"], "subjects")
    for key in ("flora", "environments", "seasons", "timesOfDay", "weather", "lighting", "moods",
                "symbolism", "compositions", "perspectives", "movement", "palettes",
                "printTechniques", "surfaceTextures", "globalStyles"):
        counts[key] = len(dataset[key])
    return counts


def main() -> None:
    dataset = build_dataset()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")

    counts = category_counts(dataset)
    total = sum(counts.values())
    print(f"Wrote {OUTPUT_PATH} ({total} entries across {len(counts)} leaf categories)\n")
    for key, count in sorted(counts.items()):
        print(f"  {key:<45} {count}")


if __name__ == "__main__":
    main()
