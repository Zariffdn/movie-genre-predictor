from genres import GENRE_SOURCES, GENRES, map_labels


def test_fifteen_genres_in_alphabetical_order():
    assert len(GENRES) == 15
    assert GENRES == sorted(GENRES)


def test_sub_genre_maps_to_every_parent():
    assert map_labels(["Romantic comedy"]) == ["Comedy", "Romance"]


def test_format_and_origin_labels_are_ignored():
    assert map_labels(["Black-and-white", "World cinema", "Short Film"]) == []


def test_no_duplicates_when_several_labels_map_to_one_genre():
    assert map_labels(["Horror", "Slasher", "Zombie Film"]) == ["Horror"]


def test_every_genre_has_source_labels():
    assert all(sources for sources in GENRE_SOURCES.values())
