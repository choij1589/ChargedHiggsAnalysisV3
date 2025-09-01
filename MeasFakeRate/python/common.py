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