# SignalRegionStudyV2

## KNU Tier2/Tier3 Storage and HTCondor

### Storage Element (SE) Access

**SE_UserHome path:**
```
/pnfs/knu.ac.kr/data/cms/store/user/{CERN_ID}
```
- `SE_UserHome` symlink in home directory points to this path
- Note: Uses CERN ID, not KNU ID

**Access protocols:**
| Protocol | Path Format | Notes |
|----------|-------------|-------|
| xrootd | `root://cluster142.knu.ac.kr//store/user/{userid}/...` | Recommended for HTCondor jobs |
| dcap | `dcap://cluster142.knu.ac.kr//pnfs/knu.ac.kr/data/cms/store/user/{userid}/...` | Read-only, no auth required, KNU internal only |
| gsidcap | Same as dcap with grid auth | Read/write, works outside KNU |
| NFS | Direct path `/pnfs/knu.ac.kr/...` | Available in HTCondor (since 2023.06.23), no overwrite/append |

### HTCondor at KNU

**UI servers:**
- Tier-2: `kcms-t2.knu.ac.kr` (or `cms.knu.ac.kr`, `cms01.knu.ac.kr`)
- Tier-3: `kcms-t3.knu.ac.kr` (or `cms02.knu.ac.kr`, `cms03.knu.ac.kr`)

**Reading SE data in HTCondor jobs:**
- HTCondor worker nodes can directly access `/pnfs/` via NFS mount
- For better parallel I/O, use xrootd protocol: `root://cluster142.knu.ac.kr//store/user/...`
- dCache NFS does NOT support overwriting or appending files

**Singularity containers (optional):**
```
+SingularityImage = "/cvmfs/singularity.opensciencegrid.org/opensciencegrid/osgvo-el9:latest"
```

**Job monitoring:**
- `condor_q` - check job status
- `condor_tail -f <job_id>` - real-time stdout/stderr
- `condor_ssh_to_job <job_id>` - SSH into running job for debugging

**Reference:** [KNU T2/T3 Wiki](http://t2-cms.knu.ac.kr/wiki/index.php/HTCondor)
