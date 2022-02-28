def index_to_loc(idx, size):
    x, y = divmod(idx, size-2)
    y += 1
    x += 1
    return x, y
