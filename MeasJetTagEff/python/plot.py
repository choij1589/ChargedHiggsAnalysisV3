#!/usr/bin/env python
import os
import sys
import argparse
import ROOT
import cmsstyle as CMS
import math

from plotter import LumiInfo, PALETTE

parser = argparse.ArgumentParser(description="Plot jet tag efficiency")
parser.add_argument("--era", type=str, required=True, help="Era")
args = parser.parse_args()
WORKDIR = os.getenv("WORKDIR")

## Get efficiency histograms
f = ROOT.TFile.Open(f"{WORKDIR}/SKNanoOutput/MeasJetTagEff/{args.era}/TTLL_powheg.root")
h_l_den = f.Get(f"tagging#b##era#{args.era}##flavor#0##systematic#central##den"); h_l_den.SetDirectory(0)
h_c_den = f.Get(f"tagging#b##era#{args.era}##flavor#4##systematic#central##den"); h_c_den.SetDirectory(0)
h_b_den = f.Get(f"tagging#b##era#{args.era}##flavor#5##systematic#central##den"); h_b_den.SetDirectory(0)

h_l_num = f.Get(f"tagging#b##era#{args.era}##tagger#deepJet##working_point#M##flavor#0##systematic#central##num"); h_l_num.SetDirectory(0)
h_c_num = f.Get(f"tagging#b##era#{args.era}##tagger#deepJet##working_point#M##flavor#4##systematic#central##num"); h_c_num.SetDirectory(0)
h_b_num = f.Get(f"tagging#b##era#{args.era}##tagger#deepJet##working_point#M##flavor#5##systematic#central##num"); h_b_num.SetDirectory(0)

h_l_eff = h_l_num.Clone(f"light_eff_{args.era}")
h_l_eff.Divide(h_l_den)

h_c_eff = h_c_num.Clone(f"c_eff_{args.era}")
h_c_eff.Divide(h_c_den)

h_b_eff = h_b_num.Clone(f"b_eff_{args.era}")
h_b_eff.Divide(h_b_den)

# Set era-dependent energy
if args.era in ["2016preVFP", "2016postVFP", "2017", "2018"]:
    energy = 13
else:  # 2022, 2022EE, 2023, 2023BPix
    energy = 13.6
CMS.SetEnergy(energy)
CMS.SetLumi(LumiInfo[args.era], run=args.era)
CMS.SetExtraText("Preliminary")

# Get histogram structure info
n_eta_bins = h_l_den.GetNbinsX()
n_pt_bins = h_l_den.GetNbinsY()

output_dir = f"plots/{args.era}"
os.makedirs(output_dir, exist_ok=True)

def create_eta_projections():
    """Create efficiency vs eta plots for different pt bins"""
    for pt_bin in range(1, n_pt_bins + 1):
        pt_low = h_l_den.GetYaxis().GetBinLowEdge(pt_bin)
        pt_high = h_l_den.GetYaxis().GetBinUpEdge(pt_bin)

        # Create eta projections for each flavor
        h_l_eff_proj = h_l_eff.ProjectionX(f"light_eff_{args.era}_pt_{pt_bin}", pt_bin, pt_bin); h_l_eff_proj.SetDirectory(0)
        h_c_eff_proj = h_c_eff.ProjectionX(f"c_eff_{args.era}_pt_{pt_bin}", pt_bin, pt_bin); h_c_eff_proj.SetDirectory(0)
        h_b_eff_proj = h_b_eff.ProjectionX(f"b_eff_{args.era}_pt_{pt_bin}", pt_bin, pt_bin); h_b_eff_proj.SetDirectory(0)

        # Set axis ranges
        xmin, xmax = h_l_eff_proj.GetXaxis().GetXmin(), h_l_eff_proj.GetXaxis().GetXmax()
        ymin, ymax = 0.0, 1.1

        # Create canvas for eta projection
        canvas = CMS.cmsCanvas("", xmin, xmax, ymin, ymax, "#eta", "Efficiency")

        # Set histogram styles
        h_l_eff_proj.SetLineColor(PALETTE[2])
        h_c_eff_proj.SetLineColor(PALETTE[1])
        h_b_eff_proj.SetLineColor(PALETTE[0])

        h_l_eff_proj.SetLineWidth(2)
        h_c_eff_proj.SetLineWidth(2)
        h_b_eff_proj.SetLineWidth(2)

        h_l_eff_proj.SetMarkerColor(PALETTE[2])
        h_c_eff_proj.SetMarkerColor(PALETTE[1])
        h_b_eff_proj.SetMarkerColor(PALETTE[0])

        h_l_eff_proj.SetMarkerStyle(20)
        h_c_eff_proj.SetMarkerStyle(21)
        h_b_eff_proj.SetMarkerStyle(22)

        # Draw histograms
        CMS.cmsObjectDraw(h_b_eff_proj, "PE", MarkerSize=0.8)
        CMS.cmsObjectDraw(h_c_eff_proj, "PE same", MarkerSize=0.8)
        CMS.cmsObjectDraw(h_l_eff_proj, "PE same", MarkerSize=0.8)

        # Create legend
        leg = CMS.cmsLeg(0.7, 0.75, 0.9, 0.9, textSize=0.04)
        CMS.addToLegend(leg, (h_b_eff_proj, "b-jets", "PE"))
        CMS.addToLegend(leg, (h_c_eff_proj, "c-jets", "PE"))
        CMS.addToLegend(leg, (h_l_eff_proj, "light-jets", "PE"))
        leg.Draw()

        # Add pt range text
        CMS.drawText(f"{pt_low:.0f} < p_{{T}} < {pt_high:.0f} GeV", posX=0.2, posY=0.85, font=42, align=0, size=0.04)

        # Save canvas
        canvas.SaveAs(f"{output_dir}/eta_projection_pt_{pt_bin}.png")

def create_pt_projections():
    """Create efficiency vs pt plots for different eta bins"""
    for eta_bin in range(1, n_eta_bins + 1):
        eta_low = h_l_den.GetXaxis().GetBinLowEdge(eta_bin)
        eta_high = h_l_den.GetXaxis().GetBinUpEdge(eta_bin)

        # Create pt projections for each flavor
        h_l_eff_proj = h_l_eff.ProjectionY(f"light_eff_{args.era}_eta_{eta_bin}", eta_bin, eta_bin); h_l_eff_proj.SetDirectory(0)
        h_c_eff_proj = h_c_eff.ProjectionY(f"c_eff_{args.era}_eta_{eta_bin}", eta_bin, eta_bin); h_c_eff_proj.SetDirectory(0)
        h_b_eff_proj = h_b_eff.ProjectionY(f"b_eff_{args.era}_eta_{eta_bin}", eta_bin, eta_bin); h_b_eff_proj.SetDirectory(0)

        # Set axis ranges
        xmin, xmax = h_l_eff_proj.GetXaxis().GetXmin(), h_l_eff_proj.GetXaxis().GetXmax()
        ymin, ymax = 0.0, 1.2

        # Create canvas for pt projection
        canvas = CMS.cmsCanvas("", xmin, xmax, ymin, ymax, "p_{T} [GeV]", "Efficiency", iPos=0, extraSpace=0)

        # Set histogram styles
        h_l_eff_proj.SetLineColor(PALETTE[2])
        h_c_eff_proj.SetLineColor(PALETTE[1])
        h_b_eff_proj.SetLineColor(PALETTE[0])

        h_l_eff_proj.SetLineWidth(2)
        h_c_eff_proj.SetLineWidth(2)
        h_b_eff_proj.SetLineWidth(2)

        h_l_eff_proj.SetMarkerColor(PALETTE[2])
        h_c_eff_proj.SetMarkerColor(PALETTE[1])
        h_b_eff_proj.SetMarkerColor(PALETTE[0])

        h_l_eff_proj.SetMarkerStyle(ROOT.kFullCircle)
        h_c_eff_proj.SetMarkerStyle(ROOT.kFullCircle)
        h_b_eff_proj.SetMarkerStyle(ROOT.kFullCircle)

        # Draw histograms
        CMS.cmsObjectDraw(h_b_eff_proj, "PE", MarkerSize=0.8)
        CMS.cmsObjectDraw(h_c_eff_proj, "PE same", MarkerSize=0.8)
        CMS.cmsObjectDraw(h_l_eff_proj, "PE same", MarkerSize=0.8)

        # Create legend
        leg = CMS.cmsLeg(0.7, 0.7, 0.9, 0.88, textSize=0.04)
        CMS.addToLegend(leg, (h_b_eff_proj, "b-jets", "PE"))
        CMS.addToLegend(leg, (h_c_eff_proj, "c-jets", "PE"))
        CMS.addToLegend(leg, (h_l_eff_proj, "light-jets", "PE"))
        leg.Draw()

        # Add eta range text
        CMS.drawText(f"{eta_low:.1f} < |#eta| < {eta_high:.1f}", posX=0.2, posY=0.85, font=42, align=0, size=0.04)

        # Save canvas
        canvas.SaveAs(f"{output_dir}/pt_projection_eta_{eta_bin}.png")

def create_eta_summary():
    """Create eta summary plot with pt-averaged efficiencies"""
    print("Creating eta summary plot...")
    
    # Create 1D histograms for eta summary by projecting all pt bins
    h_l_eff_eta = h_l_eff.ProjectionX("light_eff_eta_summary", 1, n_pt_bins); h_l_eff_eta.SetDirectory(0)
    h_c_eff_eta = h_c_eff.ProjectionX("c_eff_eta_summary", 1, n_pt_bins); h_c_eff_eta.SetDirectory(0)
    h_b_eff_eta = h_b_eff.ProjectionX("b_eff_eta_summary", 1, n_pt_bins); h_b_eff_eta.SetDirectory(0)
    
    # For each eta bin, calculate pt-averaged efficiency
    for eta_bin in range(1, n_eta_bins + 1):
        # Get efficiency values for all pt bins in this eta bin
        l_effs, c_effs, b_effs = [], [], []
        l_errs, c_errs, b_errs = [], [], []
        
        for pt_bin in range(1, n_pt_bins + 1):
            l_eff, l_err = h_l_eff.GetBinContent(eta_bin, pt_bin), h_l_eff.GetBinError(eta_bin, pt_bin)
            c_eff, c_err = h_c_eff.GetBinContent(eta_bin, pt_bin), h_c_eff.GetBinError(eta_bin, pt_bin)
            b_eff, b_err = h_b_eff.GetBinContent(eta_bin, pt_bin), h_b_eff.GetBinError(eta_bin, pt_bin)
            
            if l_eff > 0:
                l_effs.append(l_eff)
                l_errs.append(l_err)
            if c_eff > 0:
                c_effs.append(c_eff)
                c_errs.append(c_err)
            if b_eff > 0:
                b_effs.append(b_eff)
                b_errs.append(b_err)
        
        # Calculate simple average
        avg_l_eff = sum(l_effs) / len(l_effs) if len(l_effs) > 0 else 0.0
        avg_c_eff = sum(c_effs) / len(c_effs) if len(c_effs) > 0 else 0.0
        avg_b_eff = sum(b_effs) / len(b_effs) if len(b_effs) > 0 else 0.0
        
        avg_l_err = sum(l_errs) / len(l_errs) if len(l_errs) > 0 else 0.0
        avg_c_err = sum(c_errs) / len(c_errs) if len(c_errs) > 0 else 0.0
        avg_b_err = sum(b_errs) / len(b_errs) if len(b_errs) > 0 else 0.0
        
        h_l_eff_eta.SetBinContent(eta_bin, avg_l_eff)
        h_l_eff_eta.SetBinError(eta_bin, avg_l_err)
        h_c_eff_eta.SetBinContent(eta_bin, avg_c_eff)
        h_c_eff_eta.SetBinError(eta_bin, avg_c_err)
        h_b_eff_eta.SetBinContent(eta_bin, avg_b_eff)
        h_b_eff_eta.SetBinError(eta_bin, avg_b_err)
    
    # Set axis ranges
    xmin, xmax = h_l_eff_eta.GetXaxis().GetXmin(), h_l_eff_eta.GetXaxis().GetXmax()
    ymin, ymax = 0.0, 1.1
    
    # Create canvas for eta summary
    canvas = CMS.cmsCanvas("", xmin, xmax, ymin, ymax, "|#eta|", "Efficiency", iPos=0, extraSpace=0)
    
    # Set histogram styles
    h_l_eff_eta.SetLineColor(PALETTE[2])
    h_c_eff_eta.SetLineColor(PALETTE[1])
    h_b_eff_eta.SetLineColor(PALETTE[0])
    
    h_l_eff_eta.SetLineWidth(2)
    h_c_eff_eta.SetLineWidth(2)
    h_b_eff_eta.SetLineWidth(2)
    
    h_l_eff_eta.SetMarkerColor(PALETTE[2])
    h_c_eff_eta.SetMarkerColor(PALETTE[1])
    h_b_eff_eta.SetMarkerColor(PALETTE[0])
    
    h_l_eff_eta.SetMarkerStyle(ROOT.kFullCircle)
    h_c_eff_eta.SetMarkerStyle(ROOT.kFullCircle)
    h_b_eff_eta.SetMarkerStyle(ROOT.kFullCircle)
    
    # Draw histograms
    CMS.cmsObjectDraw(h_b_eff_eta, "PE", MarkerSize=0.8)
    CMS.cmsObjectDraw(h_c_eff_eta, "PE same", MarkerSize=0.8)
    CMS.cmsObjectDraw(h_l_eff_eta, "PE same", MarkerSize=0.8)
    
    # Create legend
    leg = CMS.cmsLeg(0.7, 0.7, 0.9, 0.88, textSize=0.04)
    CMS.addToLegend(leg, (h_b_eff_eta, "b-jets", "PE"))
    CMS.addToLegend(leg, (h_c_eff_eta, "c-jets", "PE"))
    CMS.addToLegend(leg, (h_l_eff_eta, "light-jets", "PE"))
    leg.Draw()
    
    # Add averaged label
    CMS.drawText("p_{T}-averaged", posX=0.2, posY=0.85, font=42, align=0, size=0.04)
    
    # Save canvas
    canvas.SaveAs(f"{output_dir}/eff_vs_eta_summary.png")

def create_pt_summary():
    """Create pt summary plot with eta-averaged efficiencies"""
    print("Creating pt summary plot...")
    
    # Create 1D histograms for pt summary by projecting all eta bins
    h_l_eff_pt = h_l_eff.ProjectionY("light_eff_pt_summary", 1, n_eta_bins); h_l_eff_pt.SetDirectory(0)
    h_c_eff_pt = h_c_eff.ProjectionY("c_eff_pt_summary", 1, n_eta_bins); h_c_eff_pt.SetDirectory(0)
    h_b_eff_pt = h_b_eff.ProjectionY("b_eff_pt_summary", 1, n_eta_bins); h_b_eff_pt.SetDirectory(0)
    
    # For each pt bin, calculate eta-averaged efficiency
    for pt_bin in range(1, n_pt_bins + 1):
        # Get efficiency values for all eta bins in this pt bin
        l_effs, c_effs, b_effs = [], [], []
        l_errs, c_errs, b_errs = [], [], []
        
        for eta_bin in range(1, n_eta_bins + 1):
            l_eff, l_err = h_l_eff.GetBinContent(eta_bin, pt_bin), h_l_eff.GetBinError(eta_bin, pt_bin)
            c_eff, c_err = h_c_eff.GetBinContent(eta_bin, pt_bin), h_c_eff.GetBinError(eta_bin, pt_bin)
            b_eff, b_err = h_b_eff.GetBinContent(eta_bin, pt_bin), h_b_eff.GetBinError(eta_bin, pt_bin)
            
            if l_eff > 0:
                l_effs.append(l_eff)
                l_errs.append(l_err)
            if c_eff > 0:
                c_effs.append(c_eff)
                c_errs.append(c_err)
            if b_eff > 0:
                b_effs.append(b_eff)
                b_errs.append(b_err)
        
        # Calculate simple average
        avg_l_eff = sum(l_effs) / len(l_effs) if len(l_effs) > 0 else 0.0
        avg_c_eff = sum(c_effs) / len(c_effs) if len(c_effs) > 0 else 0.0
        avg_b_eff = sum(b_effs) / len(b_effs) if len(b_effs) > 0 else 0.0
        
        avg_l_err = sum(l_errs) / len(l_errs) if len(l_errs) > 0 else 0.0
        avg_c_err = sum(c_errs) / len(c_errs) if len(c_errs) > 0 else 0.0
        avg_b_err = sum(b_errs) / len(b_errs) if len(b_errs) > 0 else 0.0
        
        h_l_eff_pt.SetBinContent(pt_bin, avg_l_eff)
        h_l_eff_pt.SetBinError(pt_bin, avg_l_err)
        h_c_eff_pt.SetBinContent(pt_bin, avg_c_eff)
        h_c_eff_pt.SetBinError(pt_bin, avg_c_err)
        h_b_eff_pt.SetBinContent(pt_bin, avg_b_eff)
        h_b_eff_pt.SetBinError(pt_bin, avg_b_err)
    
    # Set axis ranges
    xmin, xmax = h_l_eff_pt.GetXaxis().GetXmin(), h_l_eff_pt.GetXaxis().GetXmax()
    ymin, ymax = 0.0, 1.3
    
    # Create canvas for pt summary
    canvas = CMS.cmsCanvas("", xmin, xmax, ymin, ymax, "p_{T} [GeV]", "Efficiency", iPos=0, extraSpace=0)
    
    # Set histogram styles
    h_l_eff_pt.SetLineColor(PALETTE[2])
    h_c_eff_pt.SetLineColor(PALETTE[1])
    h_b_eff_pt.SetLineColor(PALETTE[0])
    
    h_l_eff_pt.SetLineWidth(2)
    h_c_eff_pt.SetLineWidth(2)
    h_b_eff_pt.SetLineWidth(2)
    
    h_l_eff_pt.SetMarkerColor(PALETTE[2])
    h_c_eff_pt.SetMarkerColor(PALETTE[1])
    h_b_eff_pt.SetMarkerColor(PALETTE[0])
    
    h_l_eff_pt.SetMarkerStyle(ROOT.kFullCircle)
    h_c_eff_pt.SetMarkerStyle(ROOT.kFullCircle)
    h_b_eff_pt.SetMarkerStyle(ROOT.kFullCircle)
    
    # Draw histograms
    CMS.cmsObjectDraw(h_b_eff_pt, "PE", MarkerSize=0.8)
    CMS.cmsObjectDraw(h_c_eff_pt, "PE same", MarkerSize=0.8)
    CMS.cmsObjectDraw(h_l_eff_pt, "PE same", MarkerSize=0.8)
    
    # Create legend
    leg = CMS.cmsLeg(0.7, 0.7, 0.9, 0.88, textSize=0.04)
    CMS.addToLegend(leg, (h_b_eff_pt, "b-jets", "PE"))
    CMS.addToLegend(leg, (h_c_eff_pt, "c-jets", "PE"))
    CMS.addToLegend(leg, (h_l_eff_pt, "light-jets", "PE"))
    leg.Draw()
    
    # Add averaged label
    CMS.drawText("|#eta|-averaged", posX=0.2, posY=0.85, font=42, align=0, size=0.04)
    
    # Save canvas
    canvas.SaveAs(f"{output_dir}/eff_vs_pt_summary.png")

# Create the projections and summary plots
create_eta_projections()
create_pt_projections()
create_eta_summary()
create_pt_summary()

print(f"All plots saved to {output_dir}")