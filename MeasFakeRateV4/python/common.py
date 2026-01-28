RUN2_ERAS = ["2016preVFP", "2016postVFP", "2017", "2018"]
RUN3_ERAS = ["2022", "2022EE", "2023", "2023BPix"]

def get_run_period(era):
    if era in RUN2_ERAS:
        return "Run2"
    elif era in RUN3_ERAS:
        return "Run3"
    else:
        raise ValueError(f"Unknown era: {era}")

def findbin(ptcorr, abseta, ptcorr_bins, abseta_bins):
    if ptcorr > 200.:
        ptcorr = 199.
        
    prefix = ""
    # find bin index for ptcorr
    for i, _ in enumerate(ptcorr_bins[:-1]):
        if ptcorr_bins[i] < ptcorr+0.1 < ptcorr_bins[i+1]:
            prefix += f"ptcorr_{int(ptcorr_bins[i])}to{int(ptcorr_bins[i+1])}"
            break
            
    # find bin index for abseta
    abseta_idx = -1
    for i, _ in enumerate(abseta_bins[:-1]):
        if abseta_bins[i] < abseta+0.001 < abseta_bins[i+1]:
            abseta_idx = i
            break
            
    if abseta_idx == 0:   prefix += "_EB1"
    elif abseta_idx == 1: prefix += "_EB2"
    elif abseta_idx == 2: prefix += "_EE"
    else:        raise ValueError(f"Wrong abseta {abseta}")
    
    return prefix