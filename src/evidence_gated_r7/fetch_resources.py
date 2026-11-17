"""Phase 0 download of the external knowledge resources (no expression data is touched).

ChEMBL: release SQLite dump from the EBI FTP (release pinned by file name), checked against the release checksums.
OmniPath: interactions web service (datasets omnipath + collectri, human, license=academic), stored with the query URL,
          the download time and sha256. Every request uses a generic User-Agent (R7_UA) and no personal data.
"""
import argparse, json, os, subprocess, time, urllib.request

from .common import ext, dump, sha256, pman

CHEMBL_BASE = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_{r}/"
OMNI = ("https://omnipathdb.org/interactions?datasets=omnipath,collectri&organisms=9606&genesymbols=yes"
        "&fields=sources,references,curation_effort,type&license=academic&format=tsv")


def get(url, path, timeout=3600):
    req = urllib.request.Request(url, headers={"User-Agent": os.environ.get("R7_UA", "Mozilla/5.0")})
    tmp = path + ".part"
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            f.write(b)
    os.replace(tmp, path)
    return dict(url=url, path=path, bytes=os.path.getsize(path), sha256=sha256(path), fetched=time.strftime("%Y-%m-%dT%H:%M:%S%z"))


def chembl(release):
    d = ext(f"chembl_{release}"); os.makedirs(d, exist_ok=True)
    base = CHEMBL_BASE.format(r=release)
    rec = {}
    for f in ["checksums.txt", "LICENSE", "REQUIRED.ATTRIBUTION", f"chembl_{release}_release_notes.txt", f"chembl_{release}_sqlite.tar.gz"]:
        p = os.path.join(d, f)
        rec[f] = get(base + f, p) if not os.path.exists(p) else dict(path=p, bytes=os.path.getsize(p), sha256=sha256(p), note="already present")
        print(f, rec[f]["bytes"], flush=True)
    sums = open(os.path.join(d, "checksums.txt")).read()
    tgz = f"chembl_{release}_sqlite.tar.gz"
    ok = rec[tgz]["sha256"] in sums
    assert ok, "sqlite archive sha256 not in the release checksums"
    db = [l for l in subprocess.run(["tar", "tzf", os.path.join(d, tgz)], capture_output=True, text=True).stdout.split() if l.endswith(".db")]
    assert len(db) == 1, db
    if not os.path.exists(os.path.join(d, db[0])):
        subprocess.run(["tar", "xzf", tgz, db[0]], cwd=d, check=True)
    rec["sqlite_db"] = dict(path=os.path.join(d, db[0]), bytes=os.path.getsize(os.path.join(d, db[0])))
    rec["checksum_verified"] = ok
    return rec


def omnipath():
    d = ext("omnipath"); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "interactions_omnipath_collectri.tsv")
    rec = get(OMNI, p, timeout=7200) if not os.path.exists(p) else dict(url=OMNI, path=p, bytes=os.path.getsize(p), sha256=sha256(p), note="already present")
    with open(p) as f:
        rec["header"] = f.readline().rstrip("\n").split("\t")
        rec["rows"] = sum(1 for _ in f)
    for q, name in [("https://omnipathdb.org/about", "about.txt"), ("https://omnipathdb.org/resources", "resources.json")]:
        try:
            rec[name] = get(q, os.path.join(d, name), timeout=600)
        except Exception as e:    # informational only
            rec[name] = dict(url=q, error=repr(e))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chembl_release", default="37")
    ap.add_argument("--what", default="chembl,omnipath")
    a = ap.parse_args()
    out = {}
    path = pman("resource_versions.json")
    if os.path.exists(path):
        out = json.load(open(path))
    if "omnipath" in a.what:
        try:
            out["omnipath"] = omnipath()
        except Exception as e:
            out["omnipath"] = dict(url=OMNI, status="unavailable", error=repr(e))
        dump(out, path)
    if "chembl" in a.what:
        out["chembl"] = dict(release=f"ChEMBL {a.chembl_release}", files=chembl(a.chembl_release),
                             license="CC BY-SA 3.0 (release LICENSE file)", access="EBI FTP release directory")
        dump(out, path)
    out["user_agent"] = os.environ.get("R7_UA")
    dump(out, path)
    print(json.dumps(out, indent=1)[:4000])


if __name__ == "__main__":
    main()
