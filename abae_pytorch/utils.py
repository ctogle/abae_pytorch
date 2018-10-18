

def linecount(path, chunk=(8192 * 1024)):
    lc = 0
    with open(path, 'rb') as f:
        buf = f.read(chunk)
        while buf:
            lc += buf.count(b'\n')
            buf = f.read(chunk)
    return lc


