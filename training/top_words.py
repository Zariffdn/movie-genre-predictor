"""What the model learned: each genre's strongest words, and where some odd ones come from.

    python training/top_words.py

Part 1 reads app/assets/model.json and lists, for each genre, the 10 words
with the largest positive weight. This is the table in STUDY.md section 13.

Part 2 needs the corpus (run download_data.py first). For a few surprising
words it counts the plots that contain the word and how many of those have
the genre, compared with how common the genre is overall. This is the
evidence for the dataset quirks discussed in STUDY.md.
"""

import json
from pathlib import Path

from data import load_movies
from tokenizer import tokenize

MODEL_PATH = Path(__file__).resolve().parent.parent / "app" / "assets" / "model.json"
TOP = 10

# (word, genre) pairs discussed in STUDY.md.
SUSPECTS = [
    ("woody", "Family"), ("bugs", "Family"), ("daffy", "Family"), ("porky", "Family"),
    ("stooges", "Comedy"), ("mumbai", "Musical"), ("she", "Romance"), ("comedy", "Comedy"),
]


def main() -> None:
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    vocabulary = model["vocabulary"]
    rank_of = {}
    print(f"Top {TOP} words per genre (largest positive weight in model.json)")
    for genre in model["genres"]:
        weights = genre["weights"]
        order = sorted(range(len(vocabulary)), key=lambda i: -weights[i])
        rank_of[genre["name"]] = {vocabulary[i]: rank for rank, i in enumerate(order, start=1)}
        print(f"  {genre['name']:16} {', '.join(vocabulary[i] for i in order[:TOP])}")

    movies = load_movies()
    token_sets = [set(tokenize(movie.plot)) for movie in movies]
    print(f"\nWhere some of those words come from (all {len(movies):,} movies)")
    for word, genre in SUSPECTS:
        having = [movie for movie, tokens in zip(movies, token_sets) if word in tokens]
        share = sum(genre in movie.genres for movie in having) / len(having)
        overall = sum(genre in movie.genres for movie in movies) / len(movies)
        print(f"  {word:8} #{rank_of[genre][word]:<3} for {genre:8}  appears in {len(having):>6,} plots; "
              f"{share:4.0%} of those are {genre} (vs {overall:.0%} of all movies)")


if __name__ == "__main__":
    main()
