"""Fetch only the official single-dose dataset using validated ZIP byte ranges, never the full archive."""
import io,struct,urllib.request,zipfile,zlib
from research_core import *

url='https://drive.usercontent.google.com/download?id=11fzDbp0B19Dy47MtD742Jl4Hz2bdSiiq&export=download&confirm=t'
root=ASSETS/'official/scVIDR';root.mkdir(parents=True,exist_ok=True);name='nault2021_singleDose.h5ad';out=root/name;assert not out.exists(),'Immutable dataset exists'
def get_range(start,end=None):
    header=f'bytes={start}-{end}' if end is not None else 'bytes=-65536'
    req=urllib.request.Request(url,headers={'Range':header})
    response=urllib.request.urlopen(req,timeout=90)
    assert response.status==206,'Server must honor ranges; refusing a full archive download'
    content=response.headers.get('Content-Range','');assert content.startswith('bytes ')
    if end is not None:assert content.split('/')[0]==f'bytes {start}-{end}',content
    return response
with get_range(None) as response:
    archive_bytes=int(response.headers['Content-Range'].split('/')[-1]);tail=response.read()
end=tail.rfind(b'PK\x05\x06');assert end>=0;record=struct.unpack_from('<4s4H2LH',tail,end);central_size,central_offset=record[5:7]
assert central_size<65536 and end>=central_size
fake=bytearray(tail[end-central_size:]);struct.pack_into('<L',fake,central_size+16,0);archive=zipfile.ZipFile(io.BytesIO(fake));entry=archive.getinfo(name);assert entry.compress_type==8
with get_range(entry.header_offset,entry.header_offset+1023) as response:header=response.read()
local=struct.unpack_from('<4s5H3L2H',header);assert local[0]==b'PK\x03\x04';start=entry.header_offset+30+local[-2]+local[-1];last=start+entry.compress_size-1
part=out.with_suffix('.h5ad.part');decoder=zlib.decompressobj(-15);crc=0;total=0
try:
    with get_range(start,last) as response,part.open('wb') as output:
        while chunk:=response.read(8*1024*1024):
            data=decoder.decompress(chunk);output.write(data);crc=zlib.crc32(data,crc);total+=len(data)
        data=decoder.flush();output.write(data);crc=zlib.crc32(data,crc);total+=len(data)
    assert decoder.eof and total==entry.file_size and crc==entry.CRC,(total,entry.file_size,crc,entry.CRC)
    part.rename(out)
except Exception:
    part.unlink(missing_ok=True);raise
write_json(ROOT/'manifests/research_phase2/scvidr_single_dose.json',dict(status='downloaded_verified',source_repository='https://github.com/BhattacharyaLab/scVIDR',upstream_commit='bcb6a96c85cddc4eef052be739c122efb737fe0e',source_url=url,archive_total_bytes=archive_bytes,archive_entries=len(archive.infolist()),selected_entry=name,compressed_bytes=entry.compress_size,size_bytes=out.stat().st_size,crc32=entry.CRC,sha256=sha(out),path=str(out),method='Validated HTTP206 central-directory and selected-entry ranges; full1.74GB archive was not downloaded',task='conditional_mouse_liver_chemical_response; data contract still requires inspection'))
print('SCVIDR_SINGLE_DOSE_DOWNLOADED',total,sha(out),flush=True)
