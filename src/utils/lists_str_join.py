def lists_str_join(l1: list[str], l2: list[str]) -> str:
    """
    Combines the two lists of strings into a single, space-separated, string with no duplicate entries
    :param l1: first list
    :param l2: second list
    :return: combined string
    """
    return " ".join(list(set(l1 + l2)))
