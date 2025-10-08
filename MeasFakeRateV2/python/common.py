def findbin(ptcorr, eta_value, ptcorr_bins, eta_bins, is_run3=False):
    if ptcorr > 200.:
        ptcorr = 199.

    prefix = ""
    # find bin index for ptcorr
    for i, _ in enumerate(ptcorr_bins[:-1]):
        if ptcorr_bins[i] < ptcorr+0.1 < ptcorr_bins[i+1]:
            prefix += f"ptcorr_{int(ptcorr_bins[i])}to{int(ptcorr_bins[i+1])}"
            break

    if is_run3:
        # For Run3, use signed eta binning
        eta_idx = -1
        for i, _ in enumerate(eta_bins[:-1]):
            if eta_bins[i] <= eta_value + 0.001 and eta_value - 0.001 < eta_bins[i+1]:
                eta_idx = i
                break

        # Run3 eta bin naming (matches analyzer logic)
        if eta_idx == 0:   prefix += "_EEm"      # negative endcap
        elif eta_idx == 1: prefix += "_EB2m"     # negative barrel outer
        elif eta_idx == 2: prefix += "_EB1m"     # negative barrel inner
        elif eta_idx == 3: prefix += "_EB1p"     # positive barrel inner
        elif eta_idx == 4: prefix += "_EB2p"     # positive barrel outer
        elif eta_idx == 5: prefix += "_EEp"      # positive endcap
        else: raise ValueError(f"Wrong eta {eta_value} for Run3 binning")
    else:
        # For Run2, use absolute eta binning (backward compatibility)
        abseta = abs(eta_value)
        abseta_idx = -1
        for i, _ in enumerate(eta_bins[:-1]):  # eta_bins is actually abseta_bins for Run2
            if eta_bins[i] < abseta + 0.001 and abseta - 0.001 < eta_bins[i+1]:
                abseta_idx = i
                break

        if abseta_idx == 0:   prefix += "_EB1"
        elif abseta_idx == 1: prefix += "_EB2"
        elif abseta_idx == 2: prefix += "_EE"
        else: raise ValueError(f"Wrong abseta {abseta}")

    return prefix