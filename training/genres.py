"""Which genres the model predicts, and how raw Freebase labels map onto them.

The corpus tags each movie with Freebase genre labels, and there are 363 of
them. Many describe how or where a film was made rather than what happens
in it ("Black-and-white", "World cinema", "Short Film", "Bollywood"), so
they aren't predicted. We keep 15 story genres, each with at least 1,000
movies, and fold obvious sub-genres into them. For example,
"Slasher" counts as Horror, and "Romantic comedy" counts as both Romance
and Comedy.

A movie whose labels don't map to any of the 15 genres is dropped, because
"no label" in Freebase usually means "not tagged", not "none of these".
"""

GENRE_SOURCES: dict[str, list[str]] = {
    "Action": [
        "Action", "Action/Adventure", "Action Thrillers", "Action Comedy",
        "Martial Arts Film",
    ],
    "Adventure": [
        "Adventure", "Action/Adventure", "Family-Oriented Adventure",
        "Fantasy Adventure", "Adventure Comedy", "Swashbuckler films",
        "Costume Adventure",
    ],
    "Comedy": [
        "Comedy", "Comedy film", "Romantic comedy", "Comedy-drama",
        "Black comedy", "Parody", "Satire", "Slapstick", "Sex comedy",
        "Screwball comedy", "Crime Comedy", "Comedy of manners",
        "Horror Comedy", "Fantasy Comedy", "Musical comedy", "Domestic Comedy",
        "Comedy of Errors", "Action Comedy", "Adventure Comedy",
        "Workplace Comedy", "Political satire", "Comedy Thriller",
        "Media Satire",
    ],
    "Crime": [
        "Crime Fiction", "Crime Thriller", "Crime Drama", "Crime Comedy",
        "Gangster Film", "Film noir", "Heist", "Caper story", "Detective",
        "Detective fiction",
    ],
    "Drama": [
        "Drama", "Romantic drama", "Comedy-drama", "Family Drama",
        "Crime Drama", "Political drama", "Melodrama", "Costume drama",
        "Historical drama", "Marriage Drama", "Courtroom Drama",
        "Childhood Drama", "Musical Drama",
    ],
    "Family": [
        "Family Film", "Children's/Family", "Children's",
        "Children's Fantasy", "Family-Oriented Adventure",
    ],
    "Fantasy": [
        "Fantasy", "Children's Fantasy", "Fantasy Comedy", "Fantasy Adventure",
    ],
    "Horror": [
        "Horror", "Slasher", "Zombie Film", "Horror Comedy",
        "Natural horror films", "Sci-Fi Horror", "Haunted House Film",
    ],
    "Musical": ["Musical", "Musical comedy", "Musical Drama"],
    "Mystery": ["Mystery", "Detective", "Detective fiction"],
    "Romance": ["Romance Film", "Romantic drama", "Romantic comedy"],
    "Science Fiction": [
        "Science Fiction", "Sci-Fi Horror", "Alien Film", "Time travel",
        "Dystopia",
    ],
    "Thriller": [
        "Thriller", "Crime Thriller", "Psychological thriller", "Suspense",
        "Action Thrillers", "Political thriller", "Erotic thriller",
        "Comedy Thriller",
    ],
    "War": ["War film", "Combat Films"],
    "Western": ["Western"],
}

GENRES: list[str] = sorted(GENRE_SOURCES)


def map_labels(freebase_labels: list[str]) -> list[str]:
    """Turn a movie's raw Freebase labels into our genres (sorted, no duplicates)."""
    labels = set(freebase_labels)
    return [genre for genre in GENRES if labels & set(GENRE_SOURCES[genre])]
