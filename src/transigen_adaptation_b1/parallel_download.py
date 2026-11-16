"""Resumable parallel HTTP range download (stdlib only).

Writes to <dest>.part with a sidecar <dest>.part.state listing finished chunks, verifies the final
size against Content-Length and renames atomically. Never overwrites an existing <dest>.
"""
import argparse, json, os, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = "curl/8"


def content_length(url):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("dest")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--chunk-mib", type=int, default=64)
    a = ap.parse_args()
    if os.path.exists(a.dest):
        print("present, skipping:", a.dest)
        return
    size = content_length(a.url)
    chunk = a.chunk_mib * 1024 * 1024
    n = (size + chunk - 1) // chunk
    part, state = a.dest + ".part", a.dest + ".part.state"
    done = set()
    if os.path.exists(part) and os.path.exists(state):
        st = json.load(open(state))
        if st["size"] == size and st["chunk"] == chunk and st["url"] == a.url:
            done = set(st["done"])
    else:
        for p in (part, state):
            if os.path.exists(p):
                os.remove(p)
    fd = os.open(part, os.O_RDWR | os.O_CREAT, 0o644)
    os.ftruncate(fd, size)
    lock = threading.Lock()
    t0 = time.time()
    got = [0]

    def save_state():
        tmp = state + ".tmp"
        json.dump(dict(url=a.url, size=size, chunk=chunk, done=sorted(done)), open(tmp, "w"))
        os.replace(tmp, state)

    def fetch(i):
        lo, hi = i * chunk, min(size, (i + 1) * chunk) - 1
        for attempt in range(1000):
            try:
                req = urllib.request.Request(a.url, headers={"User-Agent": UA, "Range": f"bytes={lo}-{hi}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    if r.status != 206:
                        raise IOError(f"status {r.status}")
                    off = lo
                    while True:
                        buf = r.read(4 * 1024 * 1024)
                        if not buf:
                            break
                        os.pwrite(fd, buf, off)
                        off += len(buf)
                if off != hi + 1:
                    raise IOError(f"short read {off - lo} of {hi + 1 - lo}")
                with lock:
                    done.add(i)
                    got[0] += hi + 1 - lo
                    if len(done) % 16 == 0 or len(done) == n:
                        save_state()
                        print(f"{len(done)}/{n} chunks, {got[0] / 1e6 / (time.time() - t0):.1f} MB/s this session", flush=True)
                return
            except Exception as e:  # network errors are retried; the chunk is rewritten from its start
                time.sleep(min(60, 5 + attempt * 5))
                last = e
        raise RuntimeError(f"chunk {i} failed: {last}")

    todo = [i for i in range(n) if i not in done]
    print(f"{a.url}: {size} bytes, {n} chunks, {len(todo)} to fetch", flush=True)
    with ThreadPoolExecutor(a.threads) as ex:
        list(ex.map(fetch, todo))
    save_state()
    os.fsync(fd)
    os.close(fd)
    assert len(done) == n and os.path.getsize(part) == size
    os.replace(part, a.dest)
    os.remove(state)
    print("complete:", a.dest, size, flush=True)


if __name__ == "__main__":
    sys.exit(main())
