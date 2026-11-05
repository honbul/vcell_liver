"""Extract compact execution evidence from complete paper-track Slurm runs."""
import csv
import re

from artifact_io import digest

CHECKPOINT = re.compile(r'^EPOCH_CHECKPOINT (\d+) (\S+)$',re.M)


def submissions(path):
    with path.open() as f:
        rows = dict((stage, job) for stage,job,*_ in csv.reader(f,delimiter='\t'))
    for stage in ('R1-seed1','R2','R1-seeds2-5'):
        if not rows[stage].split(';',1)[0].isdigit():
            raise ValueError(f'Invalid Slurm handle for {stage}')
    return {stage: job.split(';',1)[0] for stage,job in rows.items()}


def audit(log, end_marker, expected_epochs):
    raw = log.read_text(errors='replace').splitlines()
    # Keep only checkpoint and completion evidence from our adapter, not noisy
    # third-party download paths or tqdm output.
    content = '\n'.join(raw)
    points = [(int(e),float(v)) for e,v in CHECKPOINT.findall(content)]
    n = len(points)
    if n != expected_epochs or [e for e,v in points] != list(range(1,expected_epochs+1)):
        raise ValueError(f'Unexpected epoch checkpoint history: {log}')
    if sum(line.startswith(end_marker+' ') for line in raw) != 1:
        raise ValueError(f'Final export marker missing or repeated: {log}')
    return dict(log_name=log.name, log_sha256=digest(log), epochs=n,
                checkpoints=[dict(epoch=e,selection_value=v) for e,v in points])
