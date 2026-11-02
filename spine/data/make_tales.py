"""Generate the spine corpus `tales.txt`: short, simple stories, one per line.

The corpus is synthetic so that it is licence-free, small enough for a laptop, and
*instrumented*: properties that later units must detect are planted on purpose.

  * Names recur within a story ("Mia ... Mia"), and half of the stories give a pet a made-up
    name ("a cat called Bivo") that recurs later; the made-up names are random, so the only way
    to predict their second occurrence is to copy from context (Unit 04 lookup, Unit 05 recall,
    Unit 11 induction heads).
  * About 6% of stories are exact or near duplicates of another story (Unit 07 dedup).
  * Some arithmetic sentences ("Now Tom has 7 apples.") are wrong on purpose:
    a fraction ARITH_NOISE of answers is off, imitating noisy web text (Unit 09
    post-training with a verifiable reward fixes exactly this).
  * The last HELD_OUT stories are the held-out split; a few of them are also planted
    in the training part (Unit 07/10 contamination check must find them).

The planted indices are written to `tales_manifest.json` so tests can check recall.
Run:  python make_tales.py            (deterministic; seed 2027)
Replace with a real corpus (e.g. a TinyStories subset) via --data on any stage.
"""
import json
import pathlib
import random

SEED = 2027
N_STORIES = 1400
HELD_OUT = 140
ARITH_NOISE = 0.3
DUP_FRACTION = 0.06
N_CONTAMINATED = 6

NAMES = ["Mia", "Tom", "Lily", "Ben", "Sam", "Anna", "Max", "Zoe", "Leo", "Ivy", "Noah", "Ella", "Kai", "Ruby", "Finn", "Lena"]
ANIMALS = ["cat", "dog", "bird", "fox", "duck", "frog", "rabbit", "bear", "mouse", "goat"]
THINGS = ["kite", "ball", "boat", "book", "drum", "hat", "box", "cup", "shell", "lamp"]
FRUIT = ["apples", "pears", "plums", "nuts", "eggs", "cakes", "stars", "stones"]
PLACES = ["the park", "the hill", "the river", "the garden", "the beach", "the farm", "the woods", "the town"]
COLORS = ["red", "blue", "green", "yellow", "small", "big", "old", "shiny"]
FEEL_BAD = ["sad", "upset", "worried", "cross"]
FEEL_GOOD = ["happy", "glad", "proud", "calm"]
WEATHER = ["sunny", "windy", "rainy", "cold", "warm"]
PROBLEMS = [
    "the {thing} fell into the water",
    "the {thing} was stuck in a tree",
    "the {animal} ran away with the {thing}",
    "the {thing} broke into two pieces",
    "the wind took the {thing} far away",
]
FIXES = [
    "{friend} found a long stick and helped get it back",
    "{friend} had a good idea and they fixed it together",
    "they asked {friend}, and {friend} knew what to do",
    "{name} tried again and again until it worked",
]
PET_FIRST = "BKLMNRSTZ"          # capitals that already occur in the corpus, so V does not change
CONSONANTS = "bdfgklmnprstv"
VOWELS = "aeiou"


def pet_name(rng):
    name = rng.choice(PET_FIRST) + rng.choice(VOWELS) + rng.choice(CONSONANTS) + rng.choice(VOWELS)
    return name + (rng.choice(CONSONANTS) if rng.random() < 0.5 else "")


MORALS = [
    "Friends help each other.",
    "It is good to keep trying.",
    "Sharing makes play more fun.",
    "Being kind is the best thing.",
    "",
]


def story(rng):
    name, friend = rng.sample(NAMES, 2)
    thing, animal, place = rng.choice(THINGS), rng.choice(ANIMALS), rng.choice(PLACES)
    color = rng.choice(COLORS)
    article = "an" if color[0] in "aeiou" else "a"
    parts = [f"{name} had {article} {color} {thing}."]
    pet = None
    if rng.random() < 0.5:
        pet = pet_name(rng)
        parts.append(f"{name} also had a {rng.choice(ANIMALS)} called {pet}.")
    parts.append(f"One {rng.choice(WEATHER)} day {name} went to {place} with {friend}.")
    if rng.random() < 0.5:
        parts.append(f"A {animal} came to play with {name} and {friend}.")
    problem = rng.choice(PROBLEMS).format(thing=thing, animal=animal)
    parts.append(f"Then {problem}.")
    parts.append(f"{name} felt {rng.choice(FEEL_BAD)}.")
    if pet:
        parts.append(f"{pet} ran to {name}.")
    fix = rng.choice(FIXES).format(friend=friend, name=name)
    parts.append(fix[0].upper() + fix[1:] + ".")
    arithmetic = None
    if rng.random() < 0.6:
        a, b = rng.randint(1, 5), rng.randint(1, 4)
        fruit = rng.choice(FRUIT)
        c = a + b
        if rng.random() < ARITH_NOISE:
            c = rng.choice([x for x in range(2, 10) if x != a + b])
        prompt = f"{name} had {a} {fruit}. Then {name} found {b} more. Now {name} has "
        parts.append(prompt + f"{c} {fruit}.")
        arithmetic = {"prompt_suffix": prompt, "a": a, "b": b, "written": c}
    parts.append(f"{name} and {friend} were {rng.choice(FEEL_GOOD)}.")
    if pet:
        parts.append(f"{name} gave {pet} a hug.")
    moral = rng.choice(MORALS)
    if moral:
        parts.append(moral)
    return " ".join(parts), arithmetic


def near_duplicate(text, rng):
    """Change one word: what a scraper sees when the same story is posted twice with an edit."""
    words = text.split(" ")
    i = rng.randrange(len(words))
    if words[i].rstrip(".,") in COLORS:
        words[i] = rng.choice(COLORS) + ("." if words[i].endswith(".") else "")
    else:
        words.insert(i, "very" if rng.random() < 0.5 else "then")
    return " ".join(words)


def build(seed=SEED):
    rng = random.Random(seed)
    stories, arith = [], []
    for _ in range(N_STORIES):
        s, a = story(rng)
        stories.append(s)
        arith.append(a)
    train_n = N_STORIES - HELD_OUT
    # plant duplicates inside the training part
    dups = []
    for _ in range(int(DUP_FRACTION * train_n)):
        src = rng.randrange(train_n)
        dst = rng.randrange(train_n)
        touched = {d for _, d, _ in dups} | {s for s, _, _ in dups}
        if src == dst or dst in touched or src in {d for _, d, _ in dups}:
            continue  # never overwrite a story that is already a source or a copy
        exact = rng.random() < 0.5
        stories[dst] = stories[src] if exact else near_duplicate(stories[src], rng)
        arith[dst] = arith[src]
        dups.append((src, dst, "exact" if exact else "near"))
    # plant held-out stories into the training part (contamination)
    contaminated = []
    used = {d for _, d, _ in dups} | {s for s, _, _ in dups}
    for h in rng.sample(range(train_n, N_STORIES), N_CONTAMINATED):
        dst = rng.choice([i for i in range(train_n) if i not in used])
        used.add(dst)
        stories[dst] = stories[h]
        arith[dst] = arith[h]
        contaminated.append((h, dst))
    manifest = {
        "seed": seed,
        "n_stories": N_STORIES,
        "held_out_start": train_n,
        "arith_noise": ARITH_NOISE,
        "duplicates": [{"source": s, "copy": d, "kind": k} for s, d, k in dups],
        "contaminated": [{"held_out": h, "train_copy": d} for h, d in contaminated],
    }
    return stories, manifest


def main():
    here = pathlib.Path(__file__).parent
    stories, manifest = build()
    (here / "tales.txt").write_text("\n".join(stories) + "\n", encoding="utf-8")
    (here / "tales_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    text = "\n".join(stories)
    print(f"{len(stories)} stories, {len(text):,} characters, {len(set(text))} distinct characters")
    print(f"{len(manifest['duplicates'])} planted duplicates, {len(manifest['contaminated'])} contaminated held-out stories")


if __name__ == "__main__":
    main()
