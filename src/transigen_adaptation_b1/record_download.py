"""Record size, SHA-256 and the S3 multipart ETag check for each downloaded file (stdlib only)."""
import argparse, hashlib, json, os, subprocess, datetime


def head(url):
    out = subprocess.run(["curl", "-sI", "-A", "curl/8", url], capture_output=True, text=True).stdout
    h = {}
    for line in out.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            h[k.strip().lower()] = v.strip()
    return h


def digests(path, part_size):
    sha = hashlib.sha256()
    part_md5 = []
    whole = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            buf = f.read(part_size)
            if not buf:
                break
            sha.update(buf)
            whole.update(buf)
            part_md5.append(hashlib.md5(buf).digest())
    multi = hashlib.md5(b"".join(part_md5)).hexdigest() + f"-{len(part_md5)}"
    return sha.hexdigest(), whole.hexdigest(), multi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    rows = []
    for rel in a.files:
        path = os.path.join(a.raw, os.path.basename(rel))
        url = f"{a.base}/{rel}"
        h = head(url)
        etag = h.get("etag", "").strip('"')
        size = os.path.getsize(path)
        ok = None
        note = ""
        sha = md5 = multi = None
        # S3 multipart ETag = md5(concat(md5(part)))-nparts; try the usual part sizes.
        if "-" in etag:
            nparts = int(etag.split("-")[1])
            for mib in (8, 16, 32, 64, 5, 15, 100):
                ps = mib * 1024 * 1024
                if (size + ps - 1) // ps != nparts:
                    continue
                sha, md5, multi = digests(path, ps)
                ok = multi == etag
                note = f"multipart etag check with {mib} MiB parts"
                if ok:
                    break
        if sha is None:
            sha, md5, multi = digests(path, 64 * 1024 * 1024)
            if "-" not in etag:
                ok = md5 == etag
                note = "single-part etag is the md5"
            else:
                note = "no candidate part size matched the part count; etag not verified"
        rows.append(dict(file=os.path.basename(rel), url=url, bytes=size,
                         content_length=int(h.get("content-length", -1)),
                         last_modified=h.get("last-modified"), etag=etag, etag_verified=ok,
                         etag_note=note, sha256=sha,
                         retrieved=datetime.datetime.now().astimezone().isoformat(timespec="seconds")))
        print(rows[-1], flush=True)
        assert size == rows[-1]["content_length"], "size differs from Content-Length"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(source="CMap LINCS 2020 (clue.io/data/CMap2020), public S3 bucket", files=rows), f, indent=1)


if __name__ == "__main__":
    main()
