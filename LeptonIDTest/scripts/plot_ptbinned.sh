#!/bin/bash

# Script to generate all pt-binned plots for electron/muon ID variables and efficiency plots using GNU parallel
# Usage: ./scripts/plot_ptbinned.sh [era] [object]

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <era> [object]"
    echo "Example: $0 2017 electron"
    echo "Example: $0 2017 muon"
    echo "If object is not specified, both electron and muon will be processed"
    exit 1
fi

ERA=$1
OBJECT=${2:-"both"}
REGIONS=("InnerBarrel" "OuterBarrel" "Endcap")

# Define variables based on object type
if [ "$OBJECT" = "electron" ]; then
    OBJECTS=("electron")
    ELECTRON_HISTKEYS=("mvaNoIso" "sip3d" "miniIso")
    ELECTRON_PLOTVARS=("mvaNoIso" "sip3d" "miniIso")
    ELECTRON_PTBINS=("pt10to15" "pt15to20" "pt20to25" "pt25to35" "pt35to50" "pt50to70" "pt70toInf")
elif [ "$OBJECT" = "muon" ]; then
    OBJECTS=("muon")
    MUON_HISTKEYS=("sip3d" "miniIso")
    MUON_PLOTVARS=("sip3d" "miniIso")
    MUON_PTBINS=("pt10to15" "pt15to20" "pt20to30" "pt30to50" "pt50to70" "pt70toInf")
else
    OBJECTS=("electron" "muon")
    ELECTRON_HISTKEYS=("mvaNoIso" "sip3d" "miniIso")
    ELECTRON_PLOTVARS=("mvaNoIso" "sip3d" "miniIso")
    ELECTRON_PTBINS=("pt10to15" "pt15to20" "pt20to25" "pt25to35" "pt35to50" "pt50to70" "pt70toInf")
    MUON_HISTKEYS=("sip3d" "miniIso")
    MUON_PLOTVARS=("sip3d" "miniIso")
    MUON_PTBINS=("pt10to15" "pt15to20" "pt20to30" "pt30to50" "pt50to70" "pt70toInf")
fi

export PATH=$PWD/python:$PATH

# Create a function to run the ID variable plotting command
plot_idvar_command() {
    local region=$1
    local histkey=$2
    local ptbin=$3
    local era=$4
    local object=$5
    
    if [ "$ptbin" = "inclusive" ]; then
        echo "Plotting ID var ${object} ${region}/${histkey} (inclusive)..."
        plot_idvar.py --era $era --object $object --region $region --histkey $histkey
    else
        echo "Plotting ID var ${object} ${region}/${histkey}/${ptbin}..."
        plot_idvar.py --era $era --object $object --region $region --histkey $histkey --ptbin $ptbin
    fi
}

# Create a function to run the efficiency plotting command
plot_efficiency_command() {
    local region=$1
    local plotvar=$2
    local ptbin=$3
    local era=$4
    local object=$5
    
    if [ "$ptbin" = "inclusive" ]; then
        echo "Plotting efficiency ${object} ${region}/${plotvar} (inclusive)..."
        plot_efficiency.py --era $era --object $object --region $region --plotvar $plotvar
    else
        echo "Plotting efficiency ${object} ${region}/${plotvar}/${ptbin}..."
        plot_efficiency.py --era $era --object $object --region $region --plotvar $plotvar --ptbin $ptbin
    fi
}

# Export the functions so parallel can use them
export -f plot_idvar_command
export -f plot_efficiency_command

# Process each object type
for current_object in "${OBJECTS[@]}"; do
    echo "Generating pt-binned plots for $current_object era $ERA using GNU parallel..."
    
    # Set variables based on current object
    if [ "$current_object" = "electron" ]; then
        HISTKEYS=("${ELECTRON_HISTKEYS[@]}")
        PLOTVARS=("${ELECTRON_PLOTVARS[@]}")
        PTBINS=("${ELECTRON_PTBINS[@]}")
    else
        HISTKEYS=("${MUON_HISTKEYS[@]}")
        PLOTVARS=("${MUON_PLOTVARS[@]}")
        PTBINS=("${MUON_PTBINS[@]}")
    fi
    
    echo "Generating pt-binned ID variable plots for $current_object era $ERA..."
    
    # Generate ID variable plots
    {
        # Pt-binned ID variable plots
        for region in "${REGIONS[@]}"; do
            for histkey in "${HISTKEYS[@]}"; do
                for ptbin in "${PTBINS[@]}"; do
                    echo "idvar $region $histkey $ptbin $ERA $current_object"
                done
                # Add inclusive plot
                echo "idvar $region $histkey inclusive $ERA $current_object"
            done
        done
    } | parallel -j 4 --colsep ' ' 'case {1} in idvar) plot_idvar_command {2} {3} {4} {5} {6} ;; esac'
    
    echo "Generating pt-binned efficiency plots for $current_object era $ERA..."
    
    # Generate efficiency plots
    {
        # Pt-binned efficiency plots
        for region in "${REGIONS[@]}"; do
            for plotvar in "${PLOTVARS[@]}"; do
                for ptbin in "${PTBINS[@]}"; do
                    echo "efficiency $region $plotvar $ptbin $ERA $current_object"
                done
                # Add inclusive plot
                echo "efficiency $region $plotvar inclusive $ERA $current_object"
            done
        done
    } | parallel -j 4 --colsep ' ' 'case {1} in efficiency) plot_efficiency_command {2} {3} {4} {5} {6} ;; esac'
done

echo "All pt-binned plots generated for era $ERA!"
for current_object in "${OBJECTS[@]}"; do
    echo "$current_object ID variable plots saved in: $WORKDIR/LeptonIDTest/plots/$ERA/$current_object/idvar/"
    echo "$current_object efficiency plots saved in: $WORKDIR/LeptonIDTest/plots/$ERA/$current_object/efficiency/"
done
