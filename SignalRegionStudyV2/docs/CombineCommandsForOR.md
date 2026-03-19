## Combine commands commonly used during the Stat/Combine review

-   [Combine commands commonly used during the Stat/Combine review](https://twiki.cern.ch/twiki/bin/view/CMS/CombineCommandsForOR#Combine_commands_commonly_used_d)
    -   [Git repository](https://twiki.cern.ch/twiki/bin/view/CMS/CombineCommandsForOR#Git_repository)
        -   [Upload the datacard, root input to repo assigned by the CAT group](https://twiki.cern.ch/twiki/bin/view/CMS/CombineCommandsForOR#Upload_the_datacard_root_input_t)
        -   [Rename the systematics](https://twiki.cern.ch/twiki/bin/view/CMS/CombineCommandsForOR#Rename_the_systematics)
        -   [Push root files](https://twiki.cern.ch/twiki/bin/view/CMS/CombineCommandsForOR#Push_root_files)
        -   [Upload which signal's data-card?](https://twiki.cern.ch/twiki/bin/view/CMS/CombineCommandsForOR#Upload_which_signal_s_data_card)
        -   [Update the CI file](https://twiki.cern.ch/twiki/bin/view/CMS/CombineCommandsForOR#Update_the_CI_file)
        -   [My repo doesn't have a systematics directory](https://twiki.cern.ch/twiki/bin/view/CMS/CombineCommandsForOR#My_repo_doesn_t_have_a_systemati)
    -   [Combine tests/plots for the Stat/Combine review](https://twiki.cern.ch/twiki/bin/view/CMS/CombineCommandsForOR#Combine_tests_plots_for_the_Stat)
        -   [Goodness-of-fit Test](https://twiki.cern.ch/twiki/bin/view/CMS/CombineCommandsForOR#Goodness_of_fit_Test)
        -   [Making pull-plots (blinded)](https://twiki.cern.ch/twiki/bin/view/CMS/CombineCommandsForOR#Making_pull_plots_blinded)
        -   [Bias-test (signal-injection test)](https://twiki.cern.ch/twiki/bin/view/CMS/CombineCommandsForOR#Bias_test_signal_injection_test)
        -   [Bias-test (signal-injection test) with post-fit toys](https://twiki.cern.ch/twiki/bin/view/CMS/CombineCommandsForOR#Bias_test_signal_injection_t_AN1)
        -   [Impact](https://twiki.cern.ch/twiki/bin/view/CMS/CombineCommandsForOR#Impact)
    -   [Appendix](https://twiki.cern.ch/twiki/bin/view/CMS/CombineCommandsForOR#Appendix)
        -   [FAQ](https://twiki.cern.ch/twiki/bin/view/CMS/CombineCommandsForOR#FAQ)
        -   [Interesting facts about Combine](https://twiki.cern.ch/twiki/bin/view/CMS/CombineCommandsForOR#Interesting_facts_about_Combine)
        -   [Plotting script for the injection test (modified from source and with help from GaneshParida)](https://twiki.cern.ch/twiki/bin/view/CMS/CombineCommandsForOR#Plotting_script_for_the_injectio)
        -   [How to hack the script test/diffNuisances.py to visualise B-only fit](https://twiki.cern.ch/twiki/bin/view/CMS/CombineCommandsForOR#How_to_hack_the_script_test_diff)
        -   [Some smoothing and symmetrising example codes (credit: Aliya Nigamova)](https://twiki.cern.ch/twiki/bin/view/CMS/CombineCommandsForOR#Some_smoothing_and_symmetrising)
        -   [Efficient ways to generate toys and do MultiDimFit in Combine](https://twiki.cern.ch/twiki/bin/view/CMS/CombineCommandsForOR#Efficient_ways_to_generate_toys)

## Git repository

### Upload the datacard, root input to repo assigned by the CAT group

We first clone the initial content to your local machine:

```
git clone --recurse-submodules https://gitlab.cern.ch/cms-analysis/b2g/[your_repo]/datacards.git
```

And then update the `systematics` submodule because the recommended list is constantly updated:

```
cd systematics
git checkout master
git pull
```

Open `.gitlab-ci.yml` in your repo. It references the git-CI stored at the CAT [repo![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://gitlab.cern.ch/cms-analysis/general/datacard-ci/-/blob/v2.4/.gitlab-ci.yml?ref_type=tags). And search for following lines:

```
include:
  - project: 'cms-analysis/general/datacard-ci'
    ref: [branch/tag name]
```

The branch/tag after `ref` may an old version from the datacard CI [repo![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://gitlab.cern.ch/cms-analysis/general/datacard-ci/-/blob/v2.4/.gitlab-ci.yml?ref_type=tags), if your repo had been created early. If so, please change it to `v2.4` or newer tags to run the latest CI scripts.

Then please put your datacard and associated combine input to the directory called `input`. Note that Git CI will do `combineCards.py` automatically over all of the uploaded cards. So if you have multiple regions' separate cards, this is not a problem. But cards for different signal masses will not work. Please read the later topic "Upload which signal's data-card" for more details.

Before uploading everything to your repo, please check the systematics naming locally first. One of the CI [job![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://gitlab.cern.ch/cms-analysis/general/datacard-ci/-/blob/master/yaml/check_systematics.yml?ref_type=heads) that needs to pass is to check if your datacard follows the [CMS convention![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://gitlab.cern.ch/cms-analysis/general/systematics). The is necessary because CMS will make your cards public and the uniform naming will give the outsiders an easier life to understand your cards. It also facilitates the combination of analyses.

**You need to install both Combine and [CombineHarvester![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/#citation) locally to be able to proceed!** Now we will create a `yml` file listing the systematics

```
python3 systematics/check_names.py --input input/[datacard name].txt  --mass 200 --make-template input/syst_template.yml
```

The file `input/syst_template.yml` is made of entries copied from `systemtics/systematics_master.yml`, if the systematics listed in the latter were found in your card. For example, if you have the luminosity NP called `lumi_2018`, the yml file will include:

```
class: luminosity
  description: portion of luminosity uncertainty on 13TeV pp data in a given year
    which is uncorrelated with other Run 2 years.
```

And note that `--mass 200` is to replace the `$MASS` variable inside the datacard. If it is not used in your card, feel free to ignore it here and in latter commands.

Now rerun the script to check the card against the template:

```
python3 systematics/check_names.py --input input/[datacard_name].txt  --mass 200 --systematics-dict input/syst_template.yml
```

The print out will show all systematics unfound in the official list `systematics/systematics_master.yml`. The print out would look like:

```
 ######################################## 

Entry for systematics CMS_eff_b does not exist in input dictionary 

...

Run on
  datacard: input/[datacard_name].txt
  systematics file: input/syst_template.yml
  master_file: systematics/systematics_master.yml
Summary: ... nuisances checked, nuisance check found issues related to ... nuisance parameter names

 ######################################## 


```

In this example, `CMS_eff_b` is used in the datacard but not in the recommended list. If you know an NP commonly used in CMS but not included in [systematics\_master.yml![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://gitlab.cern.ch/cms-analysis/general/systematics/-/blob/master/systematics_master.yml), please contact the CAT convenors to include them. You could then update the `systematics` submodules as we did in the beginning. Then that systematics will pass the check.

It is common to have systematics not included in the official list. This CMS policy is new and most analysers had already named their cards before this policy was announced or weren't aware that it was announced. In the next section, we will go through the method provided by the CAT group to rename the systematics and associated histograms.

### Rename the systematics

Please create a yml file called `rename_dict.yml`. It should contain a map between your name and the recommended name. For example:

```
lumi_13TeV: lumi_2016
CMS_scale_t_1prong0pi0_13TeV: CMS_scale_t_DM0_2016
CMS_scale_t_1prong1pi0_13TeV: CMS_scale_t_DM1_2016
CMS_scale_t_3prong0pi0_13TeV: CMS_scale_t_DM10_2016
CMS_eff_t_highpt: CMS_eff_t_high_pt_2016
CMS_eff_t: CMS_eff_t_2016
CMS_eff_b: CMS_eff_b
top_pt_ttbar_shape: top_pt_reweighting
```

The name before the colon is the name to be replaced; the name after is the corresponding systematic name in `systematics/systematics_master.yml`. The latter will replace the former.

If you have custom NP specific to your analysis, then please also include entries like the following in `rename_dict.yml`:

```
xsec_diboson: CMS_HIG17020_xsec_diboson
norm_jetFakes: CMS_HIG17020_norm_jetFakes
acceptance_bbH: CMS_HIG17020_acceptance_bbH
```

The name after the colon should include the your CADI (HIG17020 in the example) at the front of the new names. Now perform renaming of the systematics in the card and the root file simultaneously:

```
python3 systematics/check_names.py --input input/[datacard_name].txt  --mass 200 --rename-dict rename_dict.yml --systematics-dict input/syst_template.yml
```

Don't be worried about the messages about NP `does not exist` here. The script still used the older template for name matching. The new datacard with renamed NPs and root file containing the renamed input histograms are stored in a new directory called `renamed_cards`. Now rerun the script with the renamed cards:

```
python3 systematics/check_names.py --input renamed_cards/datacard.txt  --mass 200 --make-template renamed_cards/syst_template.yml
```

Unfortunately `make-template` won't capture the custom names. So you have to put them in by hand into `renamed_cards/syst_template.yml`. For example:

```
CMS_HIG17020_xsec_diboson:
    class: custom
    description: Uncertainty on the diboson production cross section.
CMS_HIG17020_norm_jetFakes:
    class: custom
    description: Uncertainty on the normalization of events with jets faking taus.
CMS_HIG17020_acceptance_bbH:
    class: custom
    description: Uncertainty on the acceptance of the signal model for bbHtautau
```

Note the description is compulsory. And finally redo the check:

```
python3 systematics/check_names.py --input renamed_cards/datacard.txt  --mass 200 --systematics-dict renamed_cards/syst_template.yml
```

If you see `no issues related to nuisance parameter names found.` in the print out, you are good to go!! Please replace the old cards and input root file in the `input` directory with the new cards and root file in the `renamed_cards` directory (e.g. `cp renamed_cards/*.{txt,root,yml} input`). And then `git add` and `git commit` the new cards, root files, and `input/syst_template.yml`. The last one is needed because the [CI job![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://gitlab.cern.ch/cms-analysis/general/datacard-ci/-/blob/master/yaml/check_systematics.yml?ref_type=heads) skips the `make-template` step. And the name `syst_template.yml` can be adjusted with the variable `NPs_dictionary` in your `.gitlab-ci.yml` file. Beware that the CI job for the systematic checking won't start if the line `RUN_checkSyst: 'true'` is not included in your `.gitlab-ci.yml` file.

Last note: currently the script can only deal with one card at the time. But you can use the path specification option `--output` to give a specific name to the output datacard and root file from different regions and years. Otherwise, the renamed cards and root files will always have the same default name in the `renamed_cards` diretcory and will overwrite each other. Therefore, if you have many cards, it might be easier to combine the cards with `combineCards.py` first, and then start with the output combined card.

This introduction to the renaming is my personal take from the [CMS convention![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://gitlab.cern.ch/cms-analysis/general/systematics) page and the [tutorial![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://gitlab.cern.ch/cms-analysis/analysisexamples/systematics-demo/-/blob/master/tutorial.md?ref_type=heads) from the CAT group. The CAT website also has some [information![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://cms-analysis.docs.cern.ch/code/datacard_validation_ci/?h=ci) about all the CI jobs running on the datacard repo.

![Warning, important](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/warning.gif "Warning, important") **KNOWN BUG**: The script won't copy lines of `nuisance edit` from your original card to the renamed card! Please manually add it back.

### Push root files

The datacard repo supported [LFS![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://github.com/git-lfs/git-lfs#installing), which makes uploading/downloading large files more efficient. (The installation guide is in the link, but it might exist already in your computing cluster. So try it first and see if you need to install it) Let's start to track some root files via LFS with the following commands:

```
git lfs track "/path/to/*.root"
git add .gitattributes
git commit -m "track /path/to/*.root files using Git LFS"
```

Note that `.gitattributes` needed to be committed after one runs `git lfs track` or `git lfs untrack`.

Now LFS knows the which file to track, we still need add the particular files to be uploaded to the git repo:

```
git add /path/to/*.root
```

Then do `git commit` either with `-m` to include short messages or enter the text editor for longer messages. Finally check that all the large input files have been tracked:

```
git lfs ls-files
```

Then you are good to =git pus=\` both the cards and the input root files.

P.S. The link mentioned that existing files tracked by git has to use an alternative method:

```
git lfs migrate import --include="*.root" --everything
```

I am not sure in which situation this is required. I have not seen it myself.

### Upload which signal's data-card?

It is strongly encouraged to use the mass insertion function in Combine. The instructions are on the [Combine page![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/model_building_tutorial2024/model_building_exercise/#keywords). This will reduce the number of cards to be stored and corrected for systematic naming. One just specify one or multiple mass points for CI jobs using the variable `MH` in the file [.gitlab-ci.yml![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://gitlab.cern.ch/cms-analysis/general/datacard-ci/-/blob/master/.gitlab-ci.yml?ref_type=heads) of your own directory. (So you will get Hessian-approximation of impacts for multiple masses from the CI [jobs![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://gitlab.cern.ch/cms-analysis/general/datacard-ci/-/blob/master/yaml/run_t2w_mdf_impacts.yml?ref_type=heads))

But it is also fine to upload separate data-cards for each signals. Simply order the directories properly in the `input` directory, then copy one particular signal's cards for checks and paste them directory under `input`. This is because the CI [job![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://gitlab.cern.ch/cms-analysis/general/datacard-ci/-/blob/master/yaml/combinecards.yml?ref_type=heads) only check cards there.

### Update the CI file

If your repo was created a while ago, the content of `.gitlab-ci.yml` might miss some newer customisation variables added to the central [version![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://gitlab.cern.ch/cms-analysis/general/datacard-ci/-/blob/master/.gitlab-ci.yml?ref_type=heads). Your own `.gitlabe-ci.yml` file sources the central version via the `include command` and allows users to change of some default parameters For example, `CARD_rgx` can be given a [regular expression![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://en.wikipedia.org/wiki/Regular_expression#POSIX_basic_and_extended) to specify particular datacards for CI. And `INPUT_DIR` gives users freedom to place the cards, so that they don't have to place duplicate cards directly under `input` for the CI. Feel free to use these customisation and `git commit` your own `.gitlab-ci.yml`.

### My repo doesn't have a systematics directory

This could be because you directory was created in an early version. Please add the submodule via

```
git submodule init
git submodule add https://gitlab.cern.ch/cms-analysis/general/systematics systematics
```

They will be uploaded to your repo when you do `git push origin`.

## Combine tests/plots for the Stat/Combine review

### Goodness-of-fit Test

During the Stat Review, we would like to check the compatibility between background and data without unblinding the Signal Region. Therefore one need to find a Control Region or a Validation Region sharing the same dominant background as the Signal Region. [Channel masks![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/part5/longexercise/?h=signal+mask#f-use-of-channel-masking) are useful to prevent Combine from seeing data in the Signal Region.

We will first compute the Goodness-of-fit metric using the [saturated model![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/what_combine_does/statistical_tests/?h=satur#goodness-of-fit-tests-using-the-likelihood-ratio).

```
combine -M GoodnessOfFit -d [work_space_file_name].root --algo=saturated -n _[custom_name] --freezeParameters r --setParameters r=0,[mask_signal_region_XX=1,mask_signal_region_YY=1,...]
```

The signal strength `r` is fixed to zero via `--freezeParameters r --setParameters r=0` and `signal_region_XX` is the name of the signal regions written in the data-card. Now we will make toys to emulate the distribution of the Goodness-of-fit metric, assuming the background-only hypothesis.

```
combine -M GoodnessOfFit -d [work_space_file_name].root --algo=saturated -n _[custom_name] --freezeParameters r --setParameters r=0,[mask_signal_region_XX=1,mask_signal_region_YY=1,...] -t 500 --toysFrequentist
```

The option `--toysFrequentist` has two functions: (1) data is first fit to the background-only model (r was frozen) to adjust the Nuisance Parameters (NP) best describing data (2) randomise the nominal NP values according to the post-fit values and the post-fit uncertainty. The second function is critical because the nominal values of the NPs are from calibration, extrapolation, and ansatz. So they are subject to change when the calibration/extrapolation was performed with different data. (For theoretical uncertainty this is debatable. But if on constrain the theoretical NP in the fit, then these parameters are data-dependent.) The number of toys of 500 is recommended. But when your data has larger deviation from backgrounds and you have many histograms bins times control regions (which increases the degree-of-freedom), then 1000 or even 2000 toys are needed to have a better estimate of the p-value. Now we evaluate the p-value using Combine script.

```
combineTool.py -M CollectGoodnessOfFit --input higgsCombine_[custom_name].GoodnessOfFit.mH120.root higgsCombine_[custom_ name].GoodnessOfFit.mH120.123456.root -o gof.json
plotGof.py gof.json --statistic saturated --mass 120.0 -o [output_name] --title-right="[some title]"
```

If one uses the `-m` option to adjust the signal mass, then the `mH` value in the first line won't be 120 but is the user input. The mass value in the second line is required because it is used to query the goodness-of-fit metric recorded in the json file.

### Making pull-plots (blinded)

It is useful just to make the pull plots without wasting time on evaluating the impact. This is possible by plotting the result from `FitDiagnostics`. The instructions are on this [section![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/part5/longexercise/#c-using-fitdiagnostics) of Combine. Furthermore, if we want to know the pulls without unblinding data in the signal regions, we can add the option `--skipSBFit` to the command for `FitDiganostics`. This is particular useful when you see a bad Goodness-of-fit (low p-values) and want to understand which NP shifts significantly from the nominal.

Note: if you use `--skipSBFit`, the plotting script `test/diffNuisances.py` might break. Please use the latest version [here![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit/blob/main/test/diffNuisances.py)

### Bias-test (signal-injection test)

The test is performed by generating toy-data with certain signal strength, fitting toy-data with background model, and checking the mean of the post-fit r from all toy-data. This test is used widely in Higgs analysis with parametric fit as the true background distribution could deviate from the chosen one, such that the disagreement might leave rooms for an excess. This kind of tests are well-documented such as in [Combine![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/tutorial2023/parametric_exercise/?h=bias#bias-studies) webpage, so it won't be explained here. Here, we focus on the analysis without parametric background, and will use the same background models for generating toys and fit. This seems odd, because by [law of large numbers![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://en.wikipedia.org/wiki/Law_of_large_numbers), the agreement is perfect as long as we have infinite number of toys. But it often happens that we can detect systematic uncertainty that ill-behaved when some bins in the toy-data downward fluctuates a lot. So this test can still serve as a test of fit stability. (In fact, the law of large numbers don't hold when the variables are near some hard-boundary of the underlying distribution.)

First, we will generate toys with a fixed strength of r via the option `--expectSignal`:

```
combine -M GenerateOnly -d [work_space_file_name].root --expectSignal [0 or other injected value] --saveToys --toysFrequentist --bypassFrequentistFit -t [number of toys, 500 should be enough] -n _[custom name] --rMax [some value larger than the injected value]
```

We need the option `--toysFrequentist` to randomise the Nuisance Parameters, as mentioned in the section of [Goodness-of-fit Test](https://twiki.cern.ch/twiki/bin/view/CMS/CombineCommandsForOR#MyAnchor0). But since we don't want to unblind data at this stage, we will add an additional option `--bypassFrequentistFit`. **_Important!_** `rMax` must be set by hand if the injected value is larger than 20, the default value of `rMax`. Otherwise, Combine will just generate toys with this value. Next, we will fit the toys using `MultiDimFit`:

```
combine -M MultiDimFit [work_space_file_name].root -t [number of toys] -n _[custom name] --toysFile higgsCombine_[custom name].GenerateOnly.mH[120 or custom value].123456.root --algo singles --toysFrequentist --rMin [large negative] --rMax [large number]
```

**_Important!_** The values of `--rMin` and `--rMax` have to be wide enough to include ![$[r_{fit}-5\sigma_{fit},r_{fit}+5\sigma_{fit}]$](https://twiki.cern.ch/twiki/pub/CMS/CombineCommandsForOR/latex3298c7f1af2338f327b61f4a74470ac3.png). Otherwise there might be fit-failures or incorrectly reported post-fit r and uncertainty due to limited fitting range. To better assess the uncertainty of r (![$\sigma_{fit}$](https://twiki.cern.ch/twiki/pub/CMS/CombineCommandsForOR/latexaaddc51c855c5f6ed61a670863641475.png)), we can use the post-fit uncertainty reported by the impact plots. Of course, if the bound of r is too narrow even for the 1 ![$[\sigma]$](https://twiki.cern.ch/twiki/pub/CMS/CombineCommandsForOR/latex7f2309506e3aee319664d1da81089351.png) range, then the uncertainty of r on the impact plot would also be wrong.

**_Last step_**: we will use a [script![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit/blob/v10.1.0/data/tutorials/parametric_exercise/plot_bias_pull.py) from Combine to plot the distribution of post-fit r w.r.t to the injected value and relative to the uncertainty of r. Note that this script is designed for parametric backgrounds, so the descriptions in the script might look strange to histogram-based analysis. The script also didn't catch suspicious fit results, where the ![$r_{fit}-\sigma_{fit}$](https://twiki.cern.ch/twiki/pub/CMS/CombineCommandsForOR/latex9d9d93a9c70435bdb54a980ff2979e65.png) = `rMin` or ![$r_{fit}+\sigma_{fit}$](https://twiki.cern.ch/twiki/pub/CMS/CombineCommandsForOR/latexac9ab8906bf7d8058201297e2bc21e3d.png) = `rMax`. So a modified version is described in the latter part of this TWiki page.

**_Important!_**: analysers should repeat the test with both **_r = 0 and r = expected limit_**, because the shape of the systematics can change with r. One should also test **_at least one low mass and one high mass_** signal.

### Bias-test (signal-injection test) with post-fit toys

It is encouraged but not required to use post-fit NPs obtained from a background-only fit to a Control/Validation region. The purpose of the injection test during the Object Review to detect ill-behaved NPs/templates. But if one finds large pulls in the background-only fit in the Control/Validation region, it might be interesting to see what will happen if such post-fit value is assumed by data in the SR. To do so we need to save the post-fit values in a snapshot. This is done by

```
combine -M MultiDimFit -d [work_space_file_name].root --algo fixed --fixedPointPOIs r=0 -n _[custom name] --saveWorkspace --setParameters [mask_signal_region_XX=1,mask_signal_region_YY=1,...]
```

Here we force `r` to be 0 via a special option `--fixedPointPOIs`. And we have to mask the signal regions to only fit to data in the Control/Validation Regions. Now we can grab the post-fit values from the snapshot and generate toys.

```
combine -M GenerateOnly -d higgsCombine_[custom_name].MultiDimFit.mH120.root --snapshotName MultiDimFit --expectSignal [0 or other injected value] --saveToys --toysFrequentist --bypassFrequentistFit -t [number of toys, 500 should be enough] -n _[custom name] --rMax [some value larger than the injected value]
```

The reason for including `--toysFrequentist` is similar to before. And `--bypassFrequentistFit` is necessary as we now are now exposed to the signal regions. `--rMax` is needed as the original value of \`\`rMax\`\` will prevent us from moldifying r beyond it. Finally we can do the fit in the signal regions, which was blinded in the first step. Thus we will blind the control regions in the fitting step so that we won't repeatedly fitting the parameter to the same region twice:

```
combine -M MultiDimFit [work_space_file_name].root -t [number of toys] -n _[custom name] --toysFile higgsCombine_[custom name].GenerateOnly.mH[120 or custom value if used].123456.root --algo singles --toysFrequentist --rMin [large negative] --rMax [large number] --setParameters [mask_control_region_XX=1,mask_control_region_YY=1,...]
```

The rest is the same as in the previous section about using blinded signal regions to perform the injection test.

### Impact

Impact plots are made to assess the relative importance of nuisance parameters. During the Stat Review, it also has the purpose of detecting strange systematic modelling. The commands for impact plot making is written in the [Combine website![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/part5/longexercise/?h=impact#b-nuisance-parameter-impacts). But there are still important points to take notice.

-   During the Stat Review, we still blind the Signal Region. So one should use Asimov data by adding an option of `-t -1` in both the initial fit command and the fit step.
-   The impact plots have a corner to show the post-fit Nuisance Parameters. So analysers may want to use this to check the post-fit NPs from the Control/Validation Regions. Just note that this is not a background-only fit and the usually tiny signal in these regions might be shifted largely away from 0 to achieve best-fit. So if the analysers have non-zero signal in these regions and just want to check post-fit NPs, [another script![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit/blob/v10.1.0/test/diffNuisances.py) in Combine is more useful than making impacts, because it also make a background-only fit.
-   It is very important to check if the uncertainty in r on the top right of the impact plots. If they plus the ![$\hat{r}$](https://twiki.cern.ch/twiki/pub/CMS/CombineCommandsForOR/latex26eb15f4c86688a4dba321c8507af95e.png) happens to be some integers, like 2, -1, 0, or even 20, then it is likely that the fit fails to find an uncertainty. One should first try to broaden the range via `--rMin` and `--rMax`. If it doesn't help, then one need to increase the verbose-level (say =--verbose=2) to extract warnings to debug.
-   **_Important!_**: as for the injection-test, the analyser should make impact plots for **_both r = 0 and r = expected limit_** (or larger values like the theory prediction) because the systematic shapes could change. For similar argument, **_at least one low and one high signal mass_** should be used for making impact plots.
-   **_Important!_**: we generally look for one-sided impact on the impact plots. This is because normally, if the systematic variation increases the background+signal model, r should decreases to match the (toy) data; and vice versa. And one possibility of having both impact going in the same direction is that the systematic histograms are fluctuating strongly due to too few un-weighted MC events and/or large MC weights. If this is true, one should smooth/symmetrise the systematic histograms. But there are many exception where smooth variations lead to one-sided impact. Please look at examples in the [FAQ](https://twiki.cern.ch/twiki/bin/view/CMS/FAQ).

## Appendix

### FAQ

**_Q_** How to save B-only fit snapshot?

**_A_** We will use an algorithm called `fixed` of `MultiDimFit` to do this. The snapshot will be saved in the output root file.

```
combine -M MultiDimFit -d [work_space_file_name].root --algo fixed --fixedPointPOIs r=0 -n _[custom name] --saveWorkspace --setParameters [mask_signal_region_XX=1,mask_signal_region_YY=1,...] 
```

**_Q_** My impact plots have unnatural cut-off.

**_A_** This means that a lot of impacts are asymmetric in a similar fashion. Usually this is because the fit range of r is too narrow. One should toggle with the options `--rMin` and `--rMax` until they are wide enough to include the ![$1\sigma$](https://twiki.cern.ch/twiki/pub/CMS/CombineCommandsForOR/latex0a358324b2e692c0b8a80f6dc72efab3.png) range of r. This symptom is particularly common for high mass signals.

**_Q_** Some of my impact is one-sided

**_A_** One possibility of having both impact going in the same direction is that the systematic histograms are fluctuating strongly due to too few un-weighted MC events and/or large MC weights. If this is true, one should [smooth/symmetrise](https://twiki.cern.ch/twiki/bin/view/CMS/CombineCommandsForOR#MyAnchor4) the systematic histograms. If not true, one first check if the post-fit error is missing (black bars are one-sided) and if the NP is in one of the following.

1.  Shape variations that lead to always low or high normalisation than the nominal background around the signal peak. For example, horizontal shape variation for the scale variation of the resonant Higgs backgrounds.
2.  One analysis has reported one-sided impact for a rate-parameter that constrained a signal-region via a control region. The detail is unclear but it is possibly due to strong correlation of the rate-parameter with other constrained NPs.
3.  MC stats uncertainty (shown as "propbin\_xxx" in the impact). Since they are single-bin Nuisance Parameters, to they don't always have linear correlation with r.
4.  Numeric instability for impact much smaller than the uncertainty of r.

**_Q_** Some of Nuisance Parameters are highly constrained (post-fit uncertainty significantly less than the pre-fit value, usually scaled to 1 on plots).

**_A_** According to Gaussian approximation, the post-fit uncertainty of a particular NP is equal to ![$(1/\sigma^2_{data}+1/\sigma^2_{pre-fit})^{-1/2}$](https://twiki.cern.ch/twiki/pub/CMS/CombineCommandsForOR/latexb3da51a98a3e2f4ee13c068af0f057dc.png). Therefore, a constrained uncertainty is due to the fact that the systematic variation w.r.t the background model (![$\sigma_{pre-fit}$](https://twiki.cern.ch/twiki/pub/CMS/CombineCommandsForOR/latex913cc8b5a3fe0b057161aa849bfa9af2.png)) is close or even larger than the statistical uncertainty of the analysed region (![$\sigma_{data}$](https://twiki.cern.ch/twiki/pub/CMS/CombineCommandsForOR/latexe3763bcaf0b990858c71bebdbf9899ef.png)). This implies that data in your regions have similar or higher sensitivity to the systematic effect than data in the calibrated regions. This can be normal due to over-conservative (i.e. intentional overestimation) pre-fit uncertainty. E.g. enveloping individual systematics into one super big uncertainty, or modelling uncertainty unrelated to any calibration. But too large uncertainty can also be artificially created by large-weight events in MC shifting toward low-stat bins, such as JEC and JER. Therefore, if the constrained NP is from calibration, one should also see if there are bins with large variations relative to the nominal and with high statistical errors. In this case, [smoothing](https://twiki.cern.ch/twiki/bin/view/CMS/CombineCommandsForOR#MyAnchor4) could be adopted.

**_Q_** I saw large biases (>0.2) in signal-injection test

**_A_** Suggested checks:

1.  Check if you have set the range of r wide enough. Please refer to the instructions [here](https://twiki.cern.ch/twiki/bin/view/CMS/CombineCommandsForOR#MyAnchor1). It matters for not only `MultiDimFit`, but also `GenerateOnly`.
2.  Check if you have too many events failing, then please refer to the later questions regarding high failure rates.
3.  Check if you have used asymmetric errors and remove fishy post-fit results. These are done in the appended [plotting script](https://twiki.cern.ch/twiki/bin/view/CMS/CombineCommandsForOR#MyAnchor2).

**_Q_** I have fit failures when I do signal-injection test with `MultiDimFit`

**_A_** Some possible causes have been seen in analyses in the following.

1.  A clear error message of "WARNING: MultiDimFit failed". One possibility is that the fit convergence is too slow. You could increase the verbose level to see complaints from [RooFit](https://twiki.cern.ch/twiki/bin/view/CMS/RooFit) that it had reached the maximum function calls while the EDM (estimated distance to minimum) is still larger than tolerance. One brute force solution is to increase the function call, such as "--X-rtd MINIMIZER\_MaxCalls=9999999"
2.  A long message with some mention of nan value. This could be due to too large systematic variation such as having the shape uncertainty of -100%. When a toy data have much lower counts than the nominal background, that NP might be un-intentionally pulled to more than 1 ![$\sigma$](https://twiki.cern.ch/twiki/pub/CMS/CombineCommandsForOR/latex9d5d1c5d77960a5fc995b5c3ccb11d74.png), causing problems.
3.  You used `--robustFit 1` and you saw "Error: closed range without finding crossing." `robustFit` used an adaptive stepping to find the uncertainty of r according to the [Cramér–Rao bound![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Rao_bound) in `src/FitterAlgoBase.cc`. It stopped the search once the shrinking steps are below a tuneable tolerance. But then afterwards, it would declare failure if the Negative-Log-Likelihood of the searched r is more than 1% away from the theoretical NLL value. One could dive into the source code to tune the tolerance or hack Combine by scaling the signal histogram to a smaller height. This would shrink the effective stepping of r and just remember to scale the best-fit r and its uncertainty back using the inverse of the original scaling.
4.  You saw many messages like `Warning - No valid low-error found, will report difference to minimum of range for`. This error is commonly due to Hessian errors, when the routine `HESSE` identifies negative eigenvalues in the Hessian matrix, indicating possibility that the minimised NLL is not at its minimum. A common cause is due to strong correlation of NPs, which makes the Hessian matrix inaccurate. One solution is to use the option `--cminDefaultMinimizerStrategy 0` to turn off eigenvalues evaluation. The `--robustHesse 1` can re-scale the NLL to improve the accuracy of the Hessian matrix, but it is not fully integrated into \`\`MultiDimFit\`\` yet so that the existing errors from the original model couldn't be skipped.

**_Q_** What are `--toyFrequentist` and `--bypassFrequentistFit`?

**_A_** The option `--toysFrequentist` makes a fit to data, and then randomise the post-fit NP in the toys based on the post-fit Nuisance Parameters. So if you are generated toys in the SR but still can't unblind data in the SR, you have to add `--bypassFrequentistFit` to skip the fit but keep the randomisation. The latter is important in toy generation because the nominal values of the NPs are from calibration, extrapolation, and ansatz.

**_Q_** My Goodness-of-fit is super large (super low p-value)

**_A_** you have to check the post-fit histograms and the post-fit NPs. [FitDiagnostics![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/part5/longexercise/?h=savewithunc#c-post-fit-distributions) offers a nice tool for this, and only the B-only fit result is relevant to [GoF](https://twiki.cern.ch/twiki/bin/view/CMS/GoF) test. There are two distinctive scenarios, and the real situation could a mix of both.

1.  When the systematic variations are small and unable to make-up for the discrepancy between the background model and data. In this case, one can ask if existing systematic correction is incorrectly applied or unaccounted effects are present.
2.  Or, data agree with background (but some data bins should still be beyond the post-fit uncertainty), while some post-fit NPs deviate hugely from the nominal value (usually 0). Then the user should ask if the shifted NP is expected to deviate by such a amount larger than the expectation, or there are some missing corrections/bugs covered by these extraordinary shifts.

**_Q_** How can I do just B-only fit using `FitDiagnostics` in a Control/Validation Region with little or no signals?

**_A_** There is an option in Combine called `--skipSBFit` to enable this. For `MultiDimFit` it is impossible. But you can hack Combine by referring to the Q: "How to save B-only fit snapshot?"

**_Q_** How can I visualise the B-only fit without S+B fit because my CR/VR has little or no signal?

**_A_** `FitDiagnostics` always perform the B-only fit (`r` fixed to 0) and the S+B fit (`r` determined by fit). By adding the option `--saveShapes --saveWithUncertainties`, one can visualise the post-fit results in the output root file. The former one will be called "fit\_b" and the latter "fit\_s". Furthermore, one can even fix `r` to some non-zero value during the B-only fit by adding the option `--customStartingPoint` and `--setParameters r=[some value]`.

**_Q_** I need to calculate [significance![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/part3/commonstatsmethods/?h=signifi#asymptotic-significances), but the default definition doesn't allow the best-fit r to go negative

**_A_** Firstly, include `--uncapped`. Secondly, set the minimum of r to a low enough negative number via either `--rMin [very negative number]` or `--setParameterRanges r=[very negative number],[very large positive number]`.

Other [FAQs from Combine![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/part4/usefullinks/#cms-statistics-committee)

### Interesting facts about Combine

**_Q_** How does Combine do lnN uncertainty?

**_A_** Let's say you put a value of 1.1 in your data-card, then the histograms (functions for unbinned analysis) will be scaled by ![$1.1^\theta$](https://twiki.cern.ch/twiki/pub/CMS/CombineCommandsForOR/latex8623bacb0032ac62f5f676c37d4c8b27.png) according to the normal Gaussian probability distribution of ![$\theta$](https://twiki.cern.ch/twiki/pub/CMS/CombineCommandsForOR/latex0cbf49e3d2aab45a2b2afc1a480be989.png) (constraint). [source![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/part2/settinguptheanalysis/#a-simple-counting-experiment)

**_Q_** How does Combine do shape uncertainty?

**_A_** The normalisation component follows the lnN scaling, while the shape component follows an non-linear interpolation for ![$-1<\theta<1$](https://twiki.cern.ch/twiki/pub/CMS/CombineCommandsForOR/latexcfccf9da361b0f8b240ba774d2411a82.png) and linear extrapolation otherwise. [source![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/part2/settinguptheanalysis/#template-shape-uncertainties)

**_Q_** How does Combine do MC-stats (bin-by-bin) uncertainty?

**_A_** It first calculates the effective un-weighted events in each bin = ![$(n_{bin}/err_{bin})^2$](https://twiki.cern.ch/twiki/pub/CMS/CombineCommandsForOR/latexef2a2956ad5aab9ab46edf460aaaefaf.png). If it is higher than the user-set threshold, the bin content will add-or-subtract the bin-error according to a normal distribution (![$+1\sigma\rightarrow n_{bin}+err_{bin}$](https://twiki.cern.ch/twiki/pub/CMS/CombineCommandsForOR/latexc17b771c7d2759322ba99d1234893227.png)). If the effective events are less than the threshold, each process has its own uncertainty. They follow a Poisson distribution according to the number of un-weighted events. [source![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/part2/bin-wise-stats/?h=automcstats#usage-instructions)

**_Q_** Can I profile the ABCD (matrix method) data-driven uncertainty directly using Combine? So that I don't have to do annoying error-propagation that is only an approximation?

**_A_** Yes you can. More info [here![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/abcd_rooparametrichist_tutorial/rooparametrichist_exercise/?h=abcd#generate-input-data)

**_Q_** What is the meaning of pulls (blue-crosses) in the impact plot

**_A_** The default one shown is the [diffPullAsym![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/part3/nonstandard/?h=pull#pulls). It can be thought of as the number of standard-deviation that the NP to fit both the pre-fit nominal, data, and other NP correlated via the model (in histograms). One can derive the formula by assuming a two-bin thought-experiment, where the pre-fit uncertainty is defined in one of the bin and takes Gaussian approximation.

**_Q_** How does the saturated model calculate the Goodness-of-Fit?

**_A_** It takes the difference between the minimised 2\*Negative-Log-Likelihood of (toy-)data and the NLL of saturated model. It would follow asymptotically a ![$\chi^2$](https://twiki.cern.ch/twiki/pub/CMS/CombineCommandsForOR/latexa824a16ee3d3fdd79aa99228fe77ff0f.png) distribution of NB-NR degree-of-freedom. NB is the total number of bins among all regions/categories in the fit and NR is the number of rate-parameter (flat parameters, unconstrained nuisance parameters). And the metric is equal to the following formula.

![GoF.png](https://twiki.cern.ch/twiki/pub/CMS/CombineCommandsForOR/GoF.png "The goodness-of-fit metric when all nuisance parameters follow Gaussian constraints and without rate-parameters. The term with a natural-log needs to be removed for bins with 0 data.")

### Plotting script for the injection test (modified from [source![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit/blob/v10.1.0/data/tutorials/parametric_exercise/plot_bias_pull.py) and with help from [GaneshParida](https://twiki.cern.ch/twiki/bin/edit/CMS/GaneshParida?topicparent=CMS.CombineCommandsForOR;nowysiwyg=1 "this topic does not yet exist; you can create it."))

```
import ROOT
import argparse

ROOT.gROOT.SetBatch(True)

# Set up argument parser
parser = argparse.ArgumentParser(description="Process ROOT fit files with configurable parameters.")
parser.add_argument("--r_truth", type=float, required=True, help="Truth value of r")
parser.add_argument("--r_max", type=float, required=True, help="Maximum value of r")
parser.add_argument("--r_min", type=float, required=True, help="Minimum value of r")
parser.add_argument("--file", type=str, required=True, help="ROOT file containing the fit")

args = parser.parse_args()

# Assign parsed arguments to variables
r_truth = args.r_truth
r_max = args.r_max
r_min = args.r_min
root_file = args.file

# Open file with fits
c = ROOT.TChain("limit")
c.Add(root_file)

hist_pull = ROOT.TH1F("pull", "Pull distribution: truth=%lf" % (r_truth), 100, -5, 5)
hist_pull.GetXaxis().SetTitle("Pull = (r_{truth}-r_{fit})/#sigma_{fit}")
hist_pull.GetYaxis().SetTitle("Entries")

N_toys = int(c.GetEntries()/3) #every toy has 3 r-values: r_hi = r_fit-err_low , r_fit , r_lo = r_fit+err_high

for i_toy in range(N_toys):
    # Best-fit value
    c.GetEntry(i_toy * 3)
    r_fit = getattr(c, "r")

    # -1 sigma value
    c.GetEntry(i_toy * 3 + 1)
    r_lo = getattr(c, "r")

    # +1 sigma value
    c.GetEntry(i_toy * 3 + 2)
    r_hi = getattr(c, "r")

    diff = r_truth - r_fit
    # Use uncertainty depending on where mu_truth is relative to mu_fit
    if diff > 0:
        sigma = abs(r_hi - r_fit) #when r_fit < r_truth, r_fit needs to go upward to approach r_truth, thus the up-error is used
    else:
        sigma = abs(r_lo - r_fit) #when r_fit > r_truth, r_fit needs to go downward to approach r_truth, thus the low-error is used

    if abs(r_hi-r_max)>1e-3: #Minos didn't return the rMax properly
        if abs(r_lo-r_min)>1e-3: #Minos didn't return the rMin properly
            if sigma>1e-3: #Errors returned by Minos too small, indicating again fit issues. If your r range is wider, you might need to level up this cut
                hist_pull.Fill(diff / sigma)
            else:
                print("r_fit: %f, sigma: %f is too small" % (r_fit, sigma))
        else:
            print("r: %f, r_lo: %f touches rMin" % (r_fit, r_lo))
    else:
        print("r: %f, r_hi: %f touches rMax" % (r_fit, r_hi))

canv = ROOT.TCanvas()
hist_pull.Draw()
print("Filled toys: %d" % hist_pull.GetEntries())

# Fit Gaussian to pull distribution
ROOT.gStyle.SetOptFit(111)
hist_pull.Fit("gaus")

canv.SaveAs("pull_r%.2f.png" % (r_truth))
```

### How to hack the script test/diffNuisances.py to visualise B-only fit

  ```
  233gr_fit_s.SetTitle("fit_b_s")
  234
  235#error_poi = fpf_s.find(options.poi).getError() #comment this line
  236
  237# loop over all fitted parameters
  238
```  ```
  407# end of loop over s and b
  408
  409    #row += ["%+4.2f" % fit_s.correlation(name, options.poi)] #comment this line
  410    #row += ["%+4.3f" % (nuis_x.getError() * fit_s.correlation(name, options.poi) * error_poi)] #comment this line
  411    if flag or options.show_all_parameters:
```  ```
  542names = list(table.keys())
  543""" comment the following lines
  544names.sort()
  545if options.sortBy == "correlation":
  546    names = [[abs(float(table[t][-2])), t] for t in table.keys()]
  547    names.sort()
  548    names.reverse()
  549    names = [n[1] for n in names]
  550elif options.sortBy == "impact":
  551    names = [[abs(float(table[t][-1])), t] for t in table.keys()]
  552    names.sort()
  553    names.reverse()
  554    names = [n[1] for n in names]
  555"""
```

And these lines have bugs to be fixed:

  ```
  555highlighters = {1: highlight, 2: morelight}
  556for n in names:
  557    v = table[n]
  558    if pmsub != None:
  559        v = [re.sub(pmsub[0], pmsub[1], i) for i in v]
  560    if sigsub != None:
  561        v = [re.sub(sigsub[0], sigsub[1], i) for i in v]
  562    if (n, "b") in isFlagged:
  563        #v[-3] = highlighters[isFlagged[(n, "b")]] % v[-3] #bugs
  564        v[-2] = highlighters[isFlagged[(n, "b")]] % v[-2] #fixed
  565    if (n, "s") in isFlagged:
  566        #v[-2] = highlighters[isFlagged[(n, "s")]] % v[-2] #bugs
  567        v[-1] = highlighters[isFlagged[(n, "s")]] % v[-1] #fixed
  568    if options.format == "latex":
  569        n = n.replace(r"_", r"\_")
  570    if options.absolute_values:
  571        #print(fmtstring % (n, v[0], v[1], v[2], v[3], v[4])) #bugs
  572        print(fmtstring % (n, v[0], v[1], v[2], 0., 0.)) #fixed
  573    else:
  574        #print(fmtstring % (n, v[0], v[1], v[2], v[3])) #bugs
  575        print(fmtstring % (n, v[0], v[1], 0., 0.)) #fixed
  576
```

### Some smoothing and symmetrising example codes (credit: Aliya Nigamova)

```
import ROOT
import ctypes
import math

def symm_down_as_neg_up(hist_u,hist_d,nominal):
  #Replace the down variations with the negative of up-nominal, good for small uncertainty
  hist_d = nominal.Clone()
  hist_d.Scale(2)
  hist_d.Add(hist_u,-1)
  return hist_d

def symm_down_as_inverse_up(hist_u,hist_d,nominal):
  #Replace the down variations with the the ratio nominal/up, good for larger uncertainty
  ratio = hist_u.Divide(nominal)
  hist_d = nominal.Clone()
  hist_d.Divide(ratio)
  return hist_d

def symm_up_down_diff(hist_u,hist_d,nominal):
  #Use the difference between the up and down as new symmetric uncertainy, useful for both variations on one-side
  diff = hist_u.Clone()
  diff.Add(hist_d,-1)
  diff.Scale(0.5)
  hist_u = nominal.Add(diff)
  hist_d = nominal.Add(diff,-1)
  return hist_u,hist_d

def smooth_TH1(h):
  #Classic 353QH smoothing in TH1 it aims to remove local bumps and dips. Applied to single histogram
  #WARNING!! Bin errors won't get smoothed by TH1::Smooth() so only recommended to use for up/down histograms, not nominal!
  h.Smooth(1) #Apply smooth once
  return h

'''
The algorithms below also apply only to one histogram. And sometimes smoothing the ratio of up to nominal or down to nominal will achieve 
a better result. Then please adapt the functions by using the ratio as input. The error bar was treated carefully below in case the user wanted
 to use it for the nominal histogram, whose bin errors are used for MC-stat uncertainty. For up or down histograms, the bin errors are irrelevant. 
More proper smoothing needs to be stronger for bins with larger MC-stat errors and vice versa. Unfortunately, the algorithms below didn't take 
this into account.
'''
def smooth_markov(h):
  #Use the Markov method to suppress spiky statistical fluctuation of single histogram
  dat,err = [],[]
  dat = [h.GetBinContent(i+1) for i in range(h.GetNBinsX())] #WARNING!!! Overflow is ignored here and below!
  n_eff = [(h.GetBinContent(i+1)/h.GetBinError(i+1))**2 for i in range(h.GetNbinsX())] #Effective evt num = (n_bin/err_bin)^2
  s = ROOT.TSpectrum()
  smoothed = h.Clone("smoothed")
  smoothed.Reset() #Just keep the bin-edges
  s.SmoothMarkov(dat,h.GetNBinsX(),3)
  for i in range(smoothed.GetNBinsX()):
    smoothed.SetBinContent(i+1,dat[i])
    smoothed.SetBinError(i+1,dat[i]/math.sqrt(n_eff[i])) #To keep (n_bin_smoothed/err_bin)^2 == n_eff
  return smoothed

def smooth_lowess_tgraph(h):
  #Use the Lowess algorithm, a kind of local regression to suppress spiky statistical fluctuation of single histogram
  g=ROOT.TGraph()
  for i in range(h.GetNbinsX()):
    g.SetPoint(i,h.GetBinCenter(i+1), h.GetBinContent(i+1)) #WARNING!!! Overflow is ignored here & below!
  smooth = ROOT.TGraphSmooth("normal")
  gout = smooth.SmoothLowess(g,"",0.5) #The smoothing span = 0.5 and other parameters are tunable
  hist_out = h.Clone("smoothed")
  hist_out.Reset() #Just keep the bin-edges
  #Please make sure you don't have empty bins otherwise the following lines will get nan issues
  n_eff = [(h.GetBinContent(i+1)/h.GetBinError(i+1))**2 for i in range(h.GetNbinsX())] #Effective evt num = (n_bin/err_bin)^2
  for i in range(h.GetNbinsX()):
    y = gout.GetPointY(i)
    hist_out.SetBinContent(i+1,y)
    hist_out.SetBinError(i+1,y/math.sqrt(n_eff[i])) #To keep (n_bin_smoothed/err_bin)^2 == n_eff
  return hist_out

def smooth_tspline(h):
  #Use splines to create smooth interpolation so proper-points have to be hand-picked. Applied to single histogram
  g=ROOT.TGraph()
  for i in range(h.GetNbinsX()): #All bins used as an example. You have to choose the good bins to interpolate the bad bins
    g.SetPoint(i,h.GetBinCenter(i+1), h.GetBinContent(i+1)) #WARNING!!! Overflow is ignored here and below!
  tspline = ROOT.TSpline3('sp',g)
  n_eff = [(h.GetBinContent(i+1)/h.GetBinError(i+1))**2 for i in range(h.GetNbinsX())] #Effective evt num = (n_bin/err_bin)^2
  for i in range(h.GetNbinsX()):
    h.SetBinContent(i+1,tspline.Eval(h.GetBinCenter(i+1)))
    h.SetBinError(i+1,tspline.Eval(h.GetBinCenter(i+1))/math.sqrt(n_eff[i])) #To keep (n_bin_smoothed/err_bin)^2 == n_eff
  return h
```

### Efficient ways to generate toys and do `MultiDimFit` in Combine

The following scripts were modified the grid submission scripts generated by `combineTool.py`. See [source![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/part3/runningthetool/?h=grid#splitting-jobs-for-a-multi-dimensional-likelihood-scan).

1\. Condor submission script (`condor_submit [this file]`)

```
executable = generate.sh #Used in step 2. change to fit.sh in step 3. This comment will incur errors in submission. Please remove it.
arguments = $(ProcId)
output                = condor.$(ClusterId).$(ProcId).out
error                 = condor.$(ClusterId).$(ProcId).err
log                   = condor.$(ClusterId).log
Request_Memory        = 4000
Request_Runtime       = 86400

# Send the job to Held state on failure.
# on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)

# Periodically retry the jobs every 10 minutes, up to a maximum of 5 retries.
# periodic_release =  (NumJobStarts < 3) && ((CurrentTime - EnteredCurrentStatus) > 600)


queue 10 #Depends on how many splitting you would like. I will split 1000 toys into 10 jobs with 100 toys each. This comment will incur errors in submission. Please remove it.

```

2\. 1000 toys' generation (`generate.sh`)

```
#!/bin/sh
ulimit -s unlimited
set -e
cd /path/to/Combine/CMSSW_[version_numbers]/src
export SCRAM_ARCH=el9_amd64_gcc12 #Depends on your Combine compiler
source /cvmfs/cms.cern.ch/cmsset_default.sh #Seems to be unnecessary on lxplus?
eval `scramv1 runtime -sh`
cd /path/to/your/workspace/

# $1 controls the seed values. We split 1000 toys into 10 jobs so the seed will go from 0 to 9
combine -M GenerateOnly -d [your-workspace].root -t 100 -n _[custom-name] --toysFrequentist --bypassFrequentistFit --expectSignal [0 or other values] -s $1 --saveToys 
``` 

3\. Do fit (`fit.sh`)

```
#!/bin/sh
ulimit -s unlimited
set -e
cd /path/to/Combine/CMSSW_[version_numbers]/src
export SCRAM_ARCH=el9_amd64_gcc12 #Depends on your Combine compiler
source /cvmfs/cms.cern.ch/cmsset_default.sh #Seems to be unnecessary on lxplus?
eval `scramv1 runtime -sh`
cd /path/to/your/workspace/

# $1 controls the seed values. We split 1000 toys into 10 jobs so the seed will go from 0 to 9
combine -M MultiDimFit -d [your-workspace].root -t 100 -n _[custom_name]_s$1 --toysFrequentist --toysFile higgsCombine_[custom-name].GenerateOnly.mH120.$1.root --algo singles --rMin [many-sigma-below-expected-r] --rMax [many-sigma-above-expected-r]
```

\-- [KuanYuLin](https://twiki.cern.ch/twiki/bin/edit/Main/KuanYuLin?topicparent=CMS.CombineCommandsForOR;nowysiwyg=1 "this topic does not yet exist; you can create it.") - 2025-02-10