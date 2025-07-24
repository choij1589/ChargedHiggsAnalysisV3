#!/usr/bin/env python
import os
import sys
import argparse
import ROOT
import cmsstyle as CMS

# Add Common/Tools to path
sys.path.append(os.path.join(os.environ['WORKDIR'], 'Common', 'Tools'))
from plotter import LumiInfo

parser = argparse.ArgumentParser()
parser.add_argument("--era", type=str, default="2023", help="Era")
parser.add_argument("--leg", type=str, choices=["mu8", "mu17"], required=True, help="Muon leg (mu8 or mu17)")
args = parser.parse_args()

# Determine ROOT file based on leg selection
if args.leg == "mu8":
    root_file = f"results/{args.era}/ROOT/efficiency_DLT_Mu8Leg.root"
elif args.leg == "mu17":
    root_file = f"results/{args.era}/ROOT/efficiency_DLT_Mu17Leg.root"

## Get efficiency histograms
f = ROOT.TFile.Open(root_file)
h_data = f.Get("data"); h_data.SetDirectory(0)
h_sim = f.Get("sim"); h_sim.SetDirectory(0)
h_sf_2d = f.Get("sf"); h_sf_2d.SetDirectory(0)
f.Close()

# Set up ROOT for batch mode and CMS style
ROOT.gROOT.SetBatch(True)

# Set era-dependent energy
if args.era in ["2016preVFP", "2016postVFP", "2017", "2018"]:
    energy = 13
else:  # 2022, 2022EE, 2023, 2023BPix
    energy = 13.6

CMS.SetEnergy(energy)
CMS.SetLumi(LumiInfo[args.era], run=args.era)
CMS.SetExtraText("Preliminary")

# Get histogram structure info
n_eta_bins = h_data.GetNbinsX()
n_pt_bins = h_data.GetNbinsY()
print(f"2D histogram structure: {n_eta_bins} eta bins x {n_pt_bins} pt bins")

# Create output directory
output_dir = f"plots/{args.era}/DblMu_{args.leg.upper()}Leg"
os.makedirs(output_dir, exist_ok=True)

def create_eta_projections():
    """Create 1D eta projections for each pt bin with scale factor panel"""
    print("Creating eta projections for each pt bin...")
    
    for pt_bin in range(1, n_pt_bins + 1):
        # Get pt bin range
        pt_low = h_data.GetYaxis().GetBinLowEdge(pt_bin)
        pt_high = h_data.GetYaxis().GetBinUpEdge(pt_bin)
        
        # Project to eta axis for this pt bin
        h_data_proj = h_data.ProjectionX(f"data_eta_ptbin{pt_bin}", pt_bin, pt_bin)
        h_sim_proj = h_sim.ProjectionX(f"sim_eta_ptbin{pt_bin}", pt_bin, pt_bin)
        h_sf = h_sf_2d.ProjectionX(f"sf_eta_ptbin{pt_bin}", pt_bin, pt_bin)
        
        # Set histogram properties
        h_data_proj.SetDirectory(0)
        h_sim_proj.SetDirectory(0)
        h_sf.SetDirectory(0)
        
        # Set titles
        h_data_proj.SetTitle("Data")
        h_sim_proj.SetTitle("Simulation")
        
        # Get axis range
        xmin, xmax = h_data_proj.GetXaxis().GetXmin(), h_data_proj.GetXaxis().GetXmax()
        
        # Create CMS-style canvas with two pads
        canvas = CMS.cmsDiCanvas(f"c_eta_ptbin{pt_bin}", xmin, xmax, 0.0, 1.1, 0.5, 1.5, 
                                "#eta", "Efficiency", "Data/MC", square=True, iPos=0, extraSpace=0)
        
        # Upper pad - efficiency comparison
        canvas.cd(1)
        
        # Create legend
        leg = CMS.cmsLeg(0.7, 0.6, 0.85, 0.75, textSize=0.04, columns=1)
        leg.AddEntry(h_data_proj, "Data", "PE")
        leg.AddEntry(h_sim_proj, "Simulation", "PE")
        
        # Draw histograms with CMS style
        CMS.cmsDraw(h_data_proj, "PE", mcolor=ROOT.kBlack, msize=1.0, lwidth=2)
        CMS.cmsDraw(h_sim_proj, "PE", mcolor=ROOT.kRed, msize=1.0, lwidth=2)
        leg.Draw()
        
        # Add title
        title_label = ROOT.TLatex()
        title_label.SetNDC()
        title_label.SetTextFont(42)
        title_label.SetTextSize(0.04)
        trigger_label = f"{args.leg.upper()} Leg" if args.leg in ["mu8", "mu17"] else args.leg
        title_label.DrawLatex(0.7, 0.55, f"{trigger_label}")
        title_label.DrawLatex(0.7, 0.50, f"p_{{T}} = {pt_low:.0f}-{pt_high:.0f} GeV")
        
        canvas.cd(1).RedrawAxis()
        
        # Lower pad - scale factor
        canvas.cd(2)
        
        # Draw reference line at 1.0
        ref_line = ROOT.TLine()
        ref_line.SetLineStyle(ROOT.kDotted)
        ref_line.SetLineColor(ROOT.kBlack)
        ref_line.SetLineWidth(2)
        ref_line.DrawLine(xmin, 1.0, xmax, 1.0)
        
        # Draw scale factor
        CMS.cmsDraw(h_sf, "PE", mcolor=ROOT.kBlack, msize=1.0, lwidth=2)
        
        canvas.cd(2).RedrawAxis()
        
        # Save canvas
        canvas.SaveAs(f"{output_dir}/eff_vs_eta_ptbin{pt_bin}_{pt_low:.0f}to{pt_high:.0f}GeV.png")

def create_pt_projections():
    """Create 1D pt projections for each eta bin with scale factor panel"""
    print("Creating pt projections for each eta bin...")
    
    for eta_bin in range(1, n_eta_bins + 1):
        # Get eta bin range
        eta_low = h_data.GetXaxis().GetBinLowEdge(eta_bin)
        eta_high = h_data.GetXaxis().GetBinUpEdge(eta_bin)
        
        # Project to pt axis for this eta bin
        h_data_proj = h_data.ProjectionY(f"data_pt_etabin{eta_bin}", eta_bin, eta_bin)
        h_sim_proj = h_sim.ProjectionY(f"sim_pt_etabin{eta_bin}", eta_bin, eta_bin)
        h_sf = h_sf_2d.ProjectionY(f"sf_pt_etabin{eta_bin}", eta_bin, eta_bin)
        
        # Set histogram properties
        h_data_proj.SetDirectory(0)
        h_sim_proj.SetDirectory(0)
        h_sf.SetDirectory(0)
        
        # Set titles
        h_data_proj.SetTitle("Data")
        h_sim_proj.SetTitle("Simulation")
        
        # Get axis range
        xmin, xmax = h_data_proj.GetXaxis().GetXmin(), h_data_proj.GetXaxis().GetXmax()
        
        # Create CMS-style canvas with two pads
        canvas = CMS.cmsDiCanvas(f"c_pt_etabin{eta_bin}", xmin, xmax, 0.0, 1.1, 0.5, 1.5, 
                                "p_{T} [GeV]", "Efficiency", "Data/MC", square=True, iPos=0, extraSpace=0)
        
        # Upper pad - efficiency comparison
        canvas.cd(1)
        
        # Create legend
        leg = CMS.cmsLeg(0.7, 0.6, 0.85, 0.75, textSize=0.04, columns=1)
        leg.AddEntry(h_data_proj, "Data", "PE")
        leg.AddEntry(h_sim_proj, "Simulation", "PE")
        
        # Draw histograms with CMS style
        CMS.cmsDraw(h_data_proj, "PE", mcolor=ROOT.kBlack, msize=1.0, lwidth=2)
        CMS.cmsDraw(h_sim_proj, "PE", mcolor=ROOT.kRed, msize=1.0, lwidth=2)
        leg.Draw()
        
        # Add title
        title_label = ROOT.TLatex()
        title_label.SetNDC()
        title_label.SetTextFont(42)
        title_label.SetTextSize(0.04)
        trigger_label = f"{args.leg.upper()} Leg" if args.leg in ["mu8", "mu17"] else args.leg
        title_label.DrawLatex(0.7, 0.55, f"{trigger_label}")
        title_label.DrawLatex(0.7, 0.50, f"#eta = {eta_low:.1f}-{eta_high:.1f}")
        
        canvas.cd(1).RedrawAxis()
        
        # Lower pad - scale factor
        canvas.cd(2)
        
        # Draw reference line at 1.0
        ref_line = ROOT.TLine()
        ref_line.SetLineStyle(ROOT.kDotted)
        ref_line.SetLineColor(ROOT.kBlack)
        ref_line.SetLineWidth(2)
        ref_line.DrawLine(xmin, 1.0, xmax, 1.0)
        
        # Draw scale factor
        CMS.cmsDraw(h_sf, "PE", mcolor=ROOT.kBlack, msize=1.0, lwidth=2)
        
        canvas.cd(2).RedrawAxis()
        
        # Save canvas
        canvas.SaveAs(f"{output_dir}/eff_vs_pt_etabin{eta_bin}_{format(eta_low, '.3f').replace('.', 'p')}{format(eta_high, '.3f').replace('.', 'p')}.png")

def create_eta_summary():
    """Create eta summary plot with pt-averaged efficiencies"""
    print("Creating eta summary plot...")
    
    # Create 1D histograms for eta summary
    h_data_eta = h_data.ProjectionX("data_eta_summary", 1, n_eta_bins)
    h_sim_eta = h_sim.ProjectionX("sim_eta_summary", 1, n_eta_bins)
    h_sf_eta = h_sf_2d.ProjectionX("sf_eta_summary", 1, n_eta_bins)
    
    # For each eta bin, calculate pt-averaged efficiency
    for eta_bin in range(1, n_eta_bins + 1):
        # Get efficiency values for all pt bins in this eta bin
        data_effs = []
        sim_effs = []
        sf_effs = []
        data_errs = []
        sim_errs = []
        sf_errs = []
        
        for pt_bin in range(1, n_pt_bins + 1):
            data_eff, data_err = h_data.GetBinContent(eta_bin, pt_bin), h_data.GetBinError(eta_bin, pt_bin)
            sim_eff, sim_err = h_sim.GetBinContent(eta_bin, pt_bin), h_sim.GetBinError(eta_bin, pt_bin)
            sf_eff, sf_err = h_sf_2d.GetBinContent(eta_bin, pt_bin), h_sf_2d.GetBinError(eta_bin, pt_bin)
            
            if data_eff > 0 and sim_eff > 0:
                data_effs.append(data_eff)
                data_errs.append(data_err)
                sim_effs.append(sim_eff)
                sim_errs.append(sim_err)
                sf_effs.append(sf_eff)
                sf_errs.append(sf_err)
        
        # Calculate simple average
        if len(data_effs) > 0:
            avg_data_eff = sum(data_effs) / len(data_effs)
            avg_sim_eff = sum(sim_effs) / len(sim_effs)
            avg_sf_eff = sum(sf_effs) / len(sf_effs)
            avg_data_err = sum(data_errs) / len(data_errs)
            avg_sim_err = sum(sim_errs) / len(sim_errs)
            avg_sf_err = sum(sf_errs) / len(sf_errs)
        else:
            avg_data_eff = avg_sim_eff = avg_sf_eff = 0.0
            avg_data_err = avg_sim_err = avg_sf_err = 0.0
        
        h_data_eta.SetBinContent(eta_bin, avg_data_eff)
        h_data_eta.SetBinError(eta_bin, avg_data_err)
        h_sim_eta.SetBinContent(eta_bin, avg_sim_eff)
        h_sim_eta.SetBinError(eta_bin, avg_sim_err)
        h_sf_eta.SetBinContent(eta_bin, avg_sf_eff)
        h_sf_eta.SetBinError(eta_bin, avg_sf_err)
    
    # Set histogram properties
    h_data_eta.SetDirectory(0)
    h_sim_eta.SetDirectory(0)
    h_sf_eta.SetDirectory(0)
    
    # Set titles
    h_data_eta.SetTitle("Data")
    h_sim_eta.SetTitle("Simulation")
    
    # Get axis range
    xmin, xmax = h_data_eta.GetXaxis().GetXmin(), h_data_eta.GetXaxis().GetXmax()
    
    # Create CMS-style canvas with two pads
    canvas = CMS.cmsDiCanvas("c_eta_summary", xmin, xmax, 0.0, 1.1, 0.5, 1.5, 
                            "#eta", "Efficiency", "Data/MC", square=True, iPos=0, extraSpace=0)
    
    # Upper pad - efficiency comparison
    canvas.cd(1)
    
    # Create legend
    leg = CMS.cmsLeg(0.7, 0.6, 0.85, 0.75, textSize=0.04, columns=1)
    leg.AddEntry(h_data_eta, "Data", "PE")
    leg.AddEntry(h_sim_eta, "Simulation", "PE")
    
    # Draw histograms with CMS style
    CMS.cmsDraw(h_data_eta, "PE", mcolor=ROOT.kBlack, msize=1.0, lwidth=2)
    CMS.cmsDraw(h_sim_eta, "PE", mcolor=ROOT.kRed, msize=1.0, lwidth=2)
    leg.Draw()
    
    # Add title
    title_label = ROOT.TLatex()
    title_label.SetNDC()
    title_label.SetTextFont(42)
    title_label.SetTextSize(0.04)
    trigger_label = f"{args.leg.upper()} Leg" if args.leg in ["mu8", "mu17"] else args.leg
    title_label.DrawLatex(0.7, 0.55, f"{trigger_label}")
    title_label.DrawLatex(0.7, 0.50, "p_{T}-averaged")
    
    canvas.cd(1).RedrawAxis()
    
    # Lower pad - scale factor
    canvas.cd(2)
    
    # Draw reference line at 1.0
    ref_line = ROOT.TLine()
    ref_line.SetLineStyle(ROOT.kDotted)
    ref_line.SetLineColor(ROOT.kBlack)
    ref_line.SetLineWidth(2)
    ref_line.DrawLine(xmin, 1.0, xmax, 1.0)
    
    # Draw scale factor
    CMS.cmsDraw(h_sf_eta, "PE", mcolor=ROOT.kBlack, msize=1.0, lwidth=2)
    
    canvas.cd(2).RedrawAxis()
    
    # Save canvas
    canvas.SaveAs(f"{output_dir}/eff_vs_eta_summary.png")

def create_pt_summary():
    """Create pt summary plot with eta-averaged efficiencies"""
    print("Creating pt summary plot...")
    
    # Create 1D histograms for pt summary
    h_data_pt = h_data.ProjectionY("data_pt_summary", 1, n_pt_bins)
    h_sim_pt = h_sim.ProjectionY("sim_pt_summary", 1, n_pt_bins)
    h_sf_pt = h_sf_2d.ProjectionY("sf_pt_summary", 1, n_pt_bins)
    
    # For each pt bin, calculate eta-averaged efficiency
    for pt_bin in range(1, n_pt_bins + 1):
        # Get efficiency values for all eta bins in this pt bin
        data_effs = []
        sim_effs = []
        sf_effs = []
        data_errs = []
        sim_errs = []
        sf_errs = []
        
        for eta_bin in range(1, n_eta_bins + 1):
            data_eff, data_err = h_data.GetBinContent(eta_bin, pt_bin), h_data.GetBinError(eta_bin, pt_bin)
            sim_eff, sim_err = h_sim.GetBinContent(eta_bin, pt_bin), h_sim.GetBinError(eta_bin, pt_bin)
            sf_eff, sf_err = h_sf_2d.GetBinContent(eta_bin, pt_bin), h_sf_2d.GetBinError(eta_bin, pt_bin)
            
            if data_eff > 0 and sim_eff > 0:
                data_effs.append(data_eff)
                data_errs.append(data_err)
                sim_effs.append(sim_eff)
                sim_errs.append(sim_err)
                sf_effs.append(sf_eff)
                sf_errs.append(sf_err)
        
        # Calculate simple average
        if len(data_effs) > 0:
            avg_data_eff = sum(data_effs) / len(data_effs)
            avg_sim_eff = sum(sim_effs) / len(sim_effs)
            avg_sf_eff = sum(sf_effs) / len(sf_effs)
            avg_data_err = sum(data_errs) / len(data_effs)
            avg_sim_err = sum(sim_errs) / len(sim_effs)
            avg_sf_err = sum(sf_errs) / len(sf_effs)
        else:
            avg_data_eff = avg_sim_eff = avg_sf_eff = 0.0
            avg_data_err = avg_sim_err = avg_sf_err = 0.0
        
        h_data_pt.SetBinContent(pt_bin, avg_data_eff)
        h_data_pt.SetBinError(pt_bin, avg_data_err)
        h_sim_pt.SetBinContent(pt_bin, avg_sim_eff)
        h_sim_pt.SetBinError(pt_bin, avg_sim_err)
        h_sf_pt.SetBinContent(pt_bin, avg_sf_eff)
        h_sf_pt.SetBinError(pt_bin, avg_sf_err)
    
    # Set histogram properties
    h_data_pt.SetDirectory(0)
    h_sim_pt.SetDirectory(0)
    h_sf_pt.SetDirectory(0)
    
    # Set titles
    h_data_pt.SetTitle("Data")
    h_sim_pt.SetTitle("Simulation")
    
    # Get axis range
    xmin, xmax = h_data_pt.GetXaxis().GetXmin(), h_data_pt.GetXaxis().GetXmax()
    
    # Create CMS-style canvas with two pads (using linear scale for pt)
    canvas = CMS.cmsDiCanvas("c_pt_summary", xmin, xmax, 0.0, 1.1, 0.5, 1.5, 
                            "p_{T} [GeV]", "Efficiency", "Data/MC", square=True, iPos=0, extraSpace=0)
    
    # Set linear scale for both pads
    canvas.cd(1).SetLogx(0)
    canvas.cd(2).SetLogx(0)
    
    # Upper pad - efficiency comparison
    canvas.cd(1)
    
    # Create legend
    leg = CMS.cmsLeg(0.7, 0.5, 0.85, 0.7, textSize=0.04, columns=1)
    leg.AddEntry(h_data_pt, "Data", "PE")
    leg.AddEntry(h_sim_pt, "Simulation", "PE")
    
    # Draw histograms with CMS style
    CMS.cmsDraw(h_data_pt, "PE", mcolor=ROOT.kBlack, msize=1.0, lwidth=2)
    CMS.cmsDraw(h_sim_pt, "PE", mcolor=ROOT.kRed, msize=1.0, lwidth=2)
    leg.Draw()
    
    # Add title
    title_label = ROOT.TLatex()
    title_label.SetNDC()
    title_label.SetTextFont(42)
    title_label.SetTextSize(0.04)
    trigger_label = f"{args.leg.upper()} Leg" if args.leg in ["mu8", "mu17"] else args.leg
    title_label.DrawLatex(0.7, 0.45, f"{trigger_label}")
    title_label.DrawLatex(0.7, 0.40, "#eta-averaged")
    
    canvas.cd(1).RedrawAxis()
    
    # Lower pad - scale factor
    canvas.cd(2)
    
    # Draw reference line at 1.0
    ref_line = ROOT.TLine()
    ref_line.SetLineStyle(ROOT.kDotted)
    ref_line.SetLineColor(ROOT.kBlack)
    ref_line.SetLineWidth(2)
    ref_line.DrawLine(xmin, 1.0, xmax, 1.0)
    
    # Draw scale factor
    CMS.cmsDraw(h_sf_pt, "PE", mcolor=ROOT.kBlack, msize=1.0, lwidth=2)
    
    canvas.cd(2).RedrawAxis()
    
    # Save canvas
    canvas.SaveAs(f"{output_dir}/eff_vs_pt_summary.png")

# Create the projections and summary plots
create_eta_projections()
create_pt_projections()
create_eta_summary()
create_pt_summary()

print(f"All plots saved to {output_dir}")