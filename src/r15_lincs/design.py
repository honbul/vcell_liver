"""A checked R15 data access layer with independent DMSO roles and parent splits."""

from pathlib import Path
from zlib import crc32

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from src.r11_chemical_probe.common import read_json, sha256
from src.r11_chemical_probe.dataset import Block, CONTROL, ENTITY, GROUP, KEY, LINE

from .data import STUDIES, table


def fold_id(parent, salt, folds):
    return crc32(f"{salt}|{parent}".encode()) % folds


def chemical_features(smiles):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    if any(m is None for m in mols):
        raise ValueError("Frozen structure axis contains an unparseable molecule")
    return np.stack([generator.GetFingerprintAsNumPy(m) for m in mols]).astype(np.float32)


class Bank:
    def __init__(self, root, protocol, scenario):
        self.root, self.scenario = Path(root), scenario
        main, origins, prior, blocked = [], {}, [], set()
        self.controls, self.y, self.receipts = {}, {}, {}
        for study in STUDIES:
            folder = self.root/study
            receipt = read_json(folder/"receipt.json")
            if receipt["protocol_sha256"] != sha256(protocol):
                raise ValueError(f"R15 cache protocol mismatch for {study}")
            self.receipts[study] = sha256(folder/"receipt.json")
            expected = ["target_rows.tsv","target_wells.npy","vehicle.npy","landmarks.tsv","chemicals.tsv","tahoe_identity.tsv"]
            if scenario == "B":
                expected += ["transfer_rows.tsv","transfer.npy"]
            for name in expected:
                if sha256(folder/name) != receipt["files"][name]:
                    raise ValueError(f"R15 cache artifact changed: {folder/name}")
            rows = pd.read_csv(folder/"target_rows.tsv",sep="\t",keep_default_na=False)
            rows["study"] = study
            rows[GROUP], rows[ENTITY], rows[LINE] = rows.parent, rows.parent, rows.cell_id
            rows[KEY[0]] = study + ":" + rows.inst_id
            rows["plate_id"] = study + ":" + rows.rna_plate
            rows[CONTROL] = rows.plate_id
            rows["time_h"], rows["dose_uM"] = rows.hours, rows.dose
            main.append(rows)
            self.controls[study] = np.load(folder/"vehicle.npy",mmap_mode="r",allow_pickle=False)
            self.y[study] = folder/"target_wells.npy"
            axis = pd.read_csv(folder/"landmarks.tsv",sep="\t")
            if origins and not axis.pr_gene_id.equals(self.landmarks.pr_gene_id):
                raise ValueError("L1000 landmark axes disagree by Entrez ID")
            self.landmarks = axis
            origins[study]=table(folder/"chemicals.tsv")
            blocked.update(table(folder/"tahoe_identity.tsv").parent)
            if scenario == "B":
                meta = pd.read_csv(folder/"transfer_rows.tsv",sep="\t",keep_default_na=False)
                matrix = np.load(folder/"transfer.npy",mmap_mode="r",allow_pickle=False)
                prior += [(r.parent,r.hours,r.tahoe_group,matrix[i]) for i,r in enumerate(meta.itertuples(index=False))]
        self.rows = pd.concat(main,ignore_index=True)
        self.blocked = blocked | set(self.rows.loc[self.rows.tahoe_exclude,GROUP])
        self.specs = pd.concat(origins.values(),ignore_index=True).sort_values("pert_id",kind="stable").drop_duplicates("parent")
        self.specs = self.specs.loc[self.specs.structure.ne("")].set_index("parent",verify_integrity=True)
        self.fps = chemical_features(self.specs.index)
        self.inputs = {}
        self.external_inputs = {}
        if scenario == "B":
            ix = pd.DataFrame([(x[0],x[1],str(x[2])) for x in prior],columns=["parent","hours","tahoe_group"])
            for columns, store in ((["parent","hours"],self.inputs),(["tahoe_group","hours"],self.external_inputs)):
                keep = ix.loc[ix.tahoe_group.ne("")] if columns[0]=="tahoe_group" else ix
                for key, indices in keep.groupby(columns).groups.items():
                    store[key]=np.stack([prior[i][3] for i in indices]).mean(axis=0)

    def target(self):
        use = self.rows.loc[self.rows[LINE].eq("HEPG2") & self.rows.study.eq("GSE92742") & self.rows.hours.eq(6)].copy()
        use = use.loc[use.quality & ~use.prior_exact]
        if self.scenario == "B":
            use = use.loc[use.parent.map(lambda p:(p,6) in self.inputs)]
        return use.reset_index(drop=True)

    def auxiliary(self):
        use = self.rows.loc[~(self.rows[LINE].eq("HEPG2") & self.rows.study.eq("GSE92742") & self.rows.hours.eq(6))].copy()
        use = use.loc[use.quality]
        low = self.rows.loc[self.rows[LINE].eq("HEPG2") & self.rows.study.eq("GSE92742") & self.rows.hours.eq(6)
                            & ~self.rows.quality & ~self.rows.prior_exact]
        use = pd.concat((use,low),ignore_index=True)
        if self.scenario == "B":
            use = use.loc[[key in self.inputs for key in zip(use.parent,use.hours)]]
        return use.reset_index(drop=True)

    def response(self, rows, role):
        y = np.empty((len(rows),len(self.landmarks)),dtype=np.float32)
        for study, indices in rows.groupby("study").indices.items():
            array = np.load(self.y[study],mmap_mode="r",allow_pickle=False)
            y[indices] = array[rows.iloc[indices]["index"].to_numpy()]
        return y-self.control(rows,role)

    def control(self, rows, role):
        output = np.empty((len(rows),len(self.landmarks)),dtype=np.float32)
        for study, indices in rows.groupby("study").indices.items():
            output[indices]=self.controls[study][rows.iloc[indices].ctrl_index.to_numpy(),"AB".index(role)]
        return output

    def chemistry(self, rows):
        pos = self.specs.index.get_indexer(rows[GROUP])
        if (pos<0).any():
            raise ValueError("Query structure is outside its registered feature axis")
        return self.fps[pos]

    def transfer(self, rows):
        if self.scenario == "A":
            return np.zeros((len(rows),len(self.landmarks)),dtype=np.float32)
        return np.stack([self.inputs[p,h] for p,h in zip(rows[GROUP],rows.hours)])


class FitSource:
    """Expose the T2 fit interface while keeping the requested holdout read-only."""

    def __init__(self, bank, allowed, cfg, assets, role, features=None, train_delta=None):
        self.bank,self.allowed,self.cfg,self.assets,self.role = bank,set(allowed.parent),cfg,assets,role
        self.r11=assets
        self.dose=10.
        self.features=features
        self.train_delta=train_delta

    def block(self, fit, query, panel, observe_query):
        if observe_query or set(fit.parent) & set(query.parent) or not set(fit.parent) <= self.allowed:
            raise ValueError("Unexpected response access in a LINCS fit")
        self.data=pd.concat((fit,query),ignore_index=True)

        def assemble(rows, treated):
            rows=rows.reset_index(drop=True)
            control=self.bank.control(rows,self.role)
            chem={"E":self.bank.chemistry(rows)}
            if self.features is not None:
                chem["C"] = self.features.pos(rows[GROUP])
            return Block(rows,control,chem,None if treated is None else treated+control)

        observed=self.bank.response(fit,self.role) if self.train_delta is None else self.train_delta
        return assemble(fit,observed),assemble(query,None),panel

    def control_block(self, panel):
        rows=self.data.drop_duplicates([CONTROL,LINE]).reset_index(drop=True)
        return rows,self.bank.control(rows,self.role)
