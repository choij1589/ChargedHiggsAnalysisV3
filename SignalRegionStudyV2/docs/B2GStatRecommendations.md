[B2G](https://twiki.cern.ch/twiki/bin/view/CMS/B2G) PAG recommendations and links on statistical treatment in an analysis.

-   [Statistics committee documentation](https://twiki.cern.ch/twiki/bin/view/CMS/B2GStatisticsRecommendations#Statistics_committee_documentati)
-   [B2G recommendations](https://twiki.cern.ch/twiki/bin/view/CMS/B2GStatisticsRecommendations#B2G_recommendations)
    -   [optimisation](https://twiki.cern.ch/twiki/bin/view/CMS/B2GStatisticsRecommendations#optimisation)
    -   [plot style](https://twiki.cern.ch/twiki/bin/view/CMS/B2GStatisticsRecommendations#plot_style)
    -   [PDF/scale uncertainties](https://twiki.cern.ch/twiki/bin/view/CMS/B2GStatisticsRecommendations#PDF_scale_uncertainties)
-   [B2G blinding policy](https://twiki.cern.ch/twiki/bin/view/CMS/B2GStatisticsRecommendations#B2G_blinding_policy)
-   [Further recommendations](https://twiki.cern.ch/twiki/bin/view/CMS/B2GStatisticsRecommendations#Further_recommendations)
    -   [goodness-of-fit test recommendations](https://twiki.cern.ch/twiki/bin/view/CMS/B2GStatisticsRecommendations#goodness_of_fit_test_recommendat)
    -   [route to approval](https://twiki.cern.ch/twiki/bin/view/CMS/B2GStatisticsRecommendations#route_to_approval)
    -   [statistical health checks](https://twiki.cern.ch/twiki/bin/view/CMS/B2GStatisticsRecommendations#statistical_health_checks)
    -   [Further recommended contents of analysis documentation](https://twiki.cern.ch/twiki/bin/view/CMS/B2GStatisticsRecommendations#Further_recommended_contents_of)
-   [B2G Combine Documentation Supplement](https://twiki.cern.ch/twiki/bin/view/CMS/B2GStatisticsRecommendations#B2G_Combine_Documentation_Supple)
    -   [B2G Requested Statistical Tests](https://twiki.cern.ch/twiki/bin/view/CMS/B2GStatisticsRecommendations#B2G_Requested_Statistical_Tests)
    -   [Generating Pseudoexperiments (Toys) with Combine](https://twiki.cern.ch/twiki/bin/view/CMS/B2GStatisticsRecommendations#Generating_Pseudoexperiments_Toy)

## Statistics committee documentation

[Statistics committee](https://twiki.cern.ch/twiki/bin/view/CMS/StatisticsCommittee) mission: help CMS to obtain statistically optimal results

available [recommendations](https://twiki.cern.ch/twiki/bin/view/CMS/StatisticsCommittee#Recommendations_from_the_Committ), e.g.:

-   goodness-of-fit tests
-   look-elsewhere effect
-   choosing the bin size for a histogram
-   quoting significant digits

recent developments presented at [Mini-Workshop![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://indico.cern.ch/event/577649/) in November 2016

-   new ROOT features
-   combination using BLUE multi-channel look-elsewhere effect

## B2G recommendations

Before object review, it is necessary to complete a number of statistical tests. An overview of these tests and some general combine help are available [below](https://twiki.cern.ch/twiki/bin/view/CMS/B2GStatisticsRecommendations#B2G_Combine_Documentation_Supple).

## optimisation

Remember, you are trying to find a new physics signal! Your analysis should therefore be optimal to **discover** what you’re looking for. Good measures for optimisation when the tails of your distributions for background are not well described:

Perform a search **before** you set a limit: provide p-values in your analysis note when unblinding (in [combine](https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideHiggsAnalysisCombinedLimit) they come out for free, also possible in [theta![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](http://www-ekp.physik.uni-karlsruhe.de/~ott/theta/theta-auto/)).

Also, try to take systematic uncertainties into account. A tighter cut that could improve your signal-to-backgroudn ratio might mean that the systematic uncertainty increases. Mind that this could also mean a re-optimisation is needed once the systematic uncertainties are available.

-   consider look-elsewhere effect if significance ![](https://twiki.cern.ch/twiki/pub/CMS/B2GStatisticsRecommendations/_MathModePlugin_eca73c2f4a030ee788d9a6e730e4decd.png)

## plot style

-   background plot style: bin your histograms (in particular those of your new physics signal) according to expected resolution
-   try to avoid empty bins
    -   background: should in principle show Poisson error bars (grass) when 0 expected data events ([PoissonErrorBars](https://twiki.cern.ch/twiki/bin/view/CMS/PoissonErrorBars)) which is **not** recommended by B2G

## PDF/scale uncertainties

-   account for acceptance effects in limit calculation
-   provide normalisation either as uncertainty band in the limit plot and/or quote it in the text

## B2G blinding policy

You are to remain blind to the data in the (or close to the) full selection of the signal region where you are performing your search. However, to check the modelling, you are advised to rely on control regions such as sidebands. The overall analysis development needs to be done blinded to avoid biases.

AN and paper/PAS should be written with the signal region blinded. The full object review will be performed blinded as well. After iterations with the POG contacts, sub-conveners and conveners, and after the pre-approval is called, you will **be asked to unblind** your signal region and present the unblinded results at the end of the pre-approval. You can update your documentation with the new results but in a private version only: the frozen version should remain frozen until the pre-approval presentation.

## Further recommendations

## goodness-of-fit test recommendations

Maintained by statistics committee: [Recommendation](https://twiki.cern.ch/twiki/bin/view/CMS/StatComGOF).

It is recommended when comparing Poisson data to a model to use a likelihood ratio test with respect to the saturated model. Instructions for combine can be found [here](https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideHiggsAnalysisCombinedLimit#Goodness_of_fit_tests). In general, the goodness-of-fit should be checked (and stated) for any fit, e.g. when you parametrise your background processes (cf. alpha-ratio method) and also for the final signal extraction/limit.

## route to approval

Analysts should submit statistics questionnaires when the analysis strategy is settled and a CADI line exists, but at least two weeks before pre-approval. A member of the statistics committee checks answers and sends comments within 1-2 weeks to the analysis hypernews. Mind that filling the questionnaire is one thing - having statistical treatment properly documented in the AN of equal, if not higher importance. Please make sure you describe what you (= the tool of your choice) are doing, e.g.:

-   how do you account for limited MC statistics (Barlow-Beeston light? Did you check that the statistical uncertainty in each bin is < 20-30%)?
-   template morphing (horizontal/vertical)?
-   profiling and marginalisation?

## statistical health checks

It is very important to check the general health checks of your data cards. The limits/p-values/fits are most likely THE result of your analysis and it is therefore important to check their output. Examples can be found at [Higgs combine diagnostics tutorial![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://indico.cern.ch/event/577649/contributions/2339439/attachments/1380195/2097804/diagnostics_tut.pdf) and [combine twiki](https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideHiggsAnalysisCombinedLimit#Maximum_likelihood_fits_and_diag).

## Further recommended contents of analysis documentation

-   ±1 sigma plots of all uncertainties (at least you should make them for yourself)
-   post-fit nuisance parameter pulls and correlations
-   nuisance parameter impacts on signal strength (e.g. for two different mass points)
-   bias studies (if relevant)

Further details can e.g. be found in [this talk![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://indico.cern.ch/event/578320/contributions/2343028/attachments/1382461/2102515/20161205_B2GWorkshop_Statistics.pdf), given at the [B2G Winter Workshop 2016![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://indico.cern.ch/event/578320/).

## [B2G](https://twiki.cern.ch/twiki/bin/view/CMS/B2G) Combine Documentation Supplement

The official documentation for the Higgs Combine Tool (commonly called "Combine") can be found [here![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/). If this is your first time learning about Combine, we highly recommend that you start with the official documentation.

While any new tool can be time-consuming to learn, documented tools with a large user base are typically more approachable because of the availability of simple examples and expertise that others can share. This is one of the major advantages of Combine and to your benefit!

These pages are meant to compile these resources and supplement them for the reference of [B2G](https://twiki.cern.ch/twiki/bin/view/CMS/B2G) analyses. It explains the standard statistical tests by [B2G](https://twiki.cern.ch/twiki/bin/view/CMS/B2G), when and when not to use certain Combine options, and common pitfalls in parameterizing models.

## Resources

Official Combine documentation

Combine Harvester (Combine wrapper)

HyperNews

[Practical Statistics for Particle Physicists![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://arxiv.org/pdf/1609.04150.pdf)

## Introduction from the [B2G](https://twiki.cern.ch/twiki/bin/view/CMS/B2G) Combine Contact

The greatest disadvantage of Combine is that users are prone to treat it like a black box that performs statistical magic to produce the final analysis result.

The most common errors come from one analysis borrowing the data card and/or commands from another analysis, modifying the card and commands without reading the documentation, and running Combine regardless.

Using Combine is not a simple matter nor should it be. The statistical analysis of the data you've selected and model you've built _is_ your result - it should not be an afterthought.

We urge you to take caution whenever you're sent Combine commands recommended for your analysis but which were constructed for another analysis. You should understand each portion of the command and agree that it is suitable for your analysis before you start wondering why the results are not what you expected.

An understanding of practical statistics and common statistical methods used in particle physics will help with constructing input models and developing an expectation for how Combine will manipulate and fit the model.

If this worries you, it's probably best to first learn or review some practical statistics with a trusted resource such as the paper linked above.

Below, we provide a supplement to such a resource to help build an intuition on how Combine and likelihood fits in general behave and how the conceptual goals of Combine manifest as input and output for the tool so that we can demystify the black box!

## Likelihoods as physical systems

The first two paragraphs of the ["Likelihood" entry on Wikipedia![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://en.wikipedia.org/wiki/Likelihood_function) are a more succinct and sufficient description than I could hope to come up with,

> In statistics, the likelihood function (often simply called the likelihood) measures the goodness of fit of a statistical model to a sample of data for given values of the unknown parameters. It is formed from the joint probability distribution of the sample, but viewed and used as a function of the parameters only, thus treating the random variables as fixed at the observed values.

> The likelihood function describes a hypersurface whose peak, if it exists, represents the combination of model parameter values that maximize the probability of drawing the sample obtained. The procedure for obtaining these arguments of the maximum of the likelihood function is known as maximum likelihood estimation, which for computational convenience is usually done using the natural logarithm of the likelihood, known as the log-likelihood function. Additionally, the shape and curvature of the likelihood surface represent information about the stability of the estimates, which is why the likelihood function is often plotted as part of a statistical analysis.

We include [here![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://docs.google.com/presentation/d/1-a-BdsdvwMGYuqYZfHHA5hl7fp6ns5WTGiWjNdWYv2g/edit?usp=sharing) a set of slides written by Lucas Corcodilos for the [B2G](https://twiki.cern.ch/twiki/bin/view/CMS/B2G) long exercise held during the Fermilab CMS [DAS](https://twiki.cern.ch/twiki/bin/view/CMS/DAS) (2021). This gives a crash course on the big picture of likelihood models and why we minimize (fit) them.

As a summary though, the likelihood (or really, the negative log-likelihood) is simply a function in many dimensions that we want to minimize and not so different from a complex mechanical system of blocks, ramps, springs, etc. The parameter values are like the position of each block in space, constraints on parameters are like springs, correlating parameters is like connecting blocks with incompressible and massless rods, the parameter of interest (POI) is the position of a specific object that you want to measure, and fitting is like releasing the system from rest (comparing the model to data) until it reaches it's equilibrium (minimized energy in the system).

While the likelihood may be a very complex function, the parts that make it up are simple and together provide the model that attempts to describe the real world physics - just like in your most feared mechanics exam problem!

The difference lies in how we determine the minimum. In a classical mechanics example, we can solve the system analytically because the function being evaluated does not take data/measurements as input. The likelihood is a way to measure the agreement of the model to the data and maximizing the likelihood (minimizing the negative log-likelihood) is the process of solving for the optimal model numerically. This is done by scanning the likelihood in steps until the minimization algorithm is confident it has found a minimum.

Finally, the final uncertainty on each parameter is found by calculating the effect on the likelihood by varying each parameter individually. If a small change in the parameter creates a large change in the value of the likelihood, then the uncertainty on the parameter is small and vice versa.

Likelihoods are complex functions but they are made of simple pieces and the more we understand the pieces of your model and how they are connected, the easier it is to understand the result of a maximum likelihood fit to data.

## Combine as a likelihood builder - data cards, workspaces, and models

Possibly related to the greatest disadvantage of Combine that I described above is the greatest advantage: Combine handles all of the model and likelihood constructing for you. This is really its main purpose and its namesake is from "combining" the likelihoods of statistical independent channels for simultaneous fits to data.

Providing a data card is how you communicate your desired model to Combine. It then converts the text into a workspace (`RooWorkspace`) which contains the actual model in ROOT. <sup> <span><a href="https://twiki.cern.ch/twiki/bin/edit/CMS/Fn-1-6361?topicparent=CMS.B2GStatisticsRecommendations;nowysiwyg=1" rel="nofollow" title="this topic does not yet exist; you can create it.">1</a></span></sup> This model is used with the data to create a likelihood which is then turned into the negative log-likelihood and then minimized with MINUIT.

If performing a maximum likelihood fit (via `FitDiagnostics` or `MultiDimFit`), the output is a fit result (`RooFitResult`) with the final values of all parameters in the fit. If using another mode of Combine, the same steps follow but the output will vary based on what is being done (limit, goodness of fit calculation, etc).

These steps provide you with quite a bit of freedom for manipulating data cards, workspaces, and models.

For example, you could do `python text2workspace.py -b card.txt -o workspace.root` and then open `workspace.root` and manipulate it manually (you could for instance, change all parameters and save it as a snapshot which could be loaded in the Combine call) before passing it to Combine. <sup> <span><a href="https://twiki.cern.ch/twiki/bin/edit/CMS/Fn-2-6361?topicparent=CMS.B2GStatisticsRecommendations;nowysiwyg=1" rel="nofollow" title="this topic does not yet exist; you can create it.">2</a></span></sup> You could also use the output parameters of a fit as the input to generate toy data (via `GenerateOnly`).

However, Combine clearly automates a lot of this for you by default and this can lead to unintended mistakes. So when constructing commands for Combine, always consider what model you are providing to it and in what ways it might be using it given the basic steps Combine follows, as I've outlined above.

For example, if you've asked Combine to perform fits to 50 toy data with `combine -M [FitDiagnostics](https://twiki.cern.ch/twiki/bin/edit/CMS/FitDiagnostics?topicparent=CMS.B2GStatisticsRecommendations;nowysiwyg=1 "this topic does not yet exist; you can create it.") -d card.txt -t 50`, ask yourself what model Combine is using to generate those toys! <sup> <span><a href="https://twiki.cern.ch/twiki/bin/edit/CMS/Fn-3-6361?topicparent=CMS.B2GStatisticsRecommendations;nowysiwyg=1" rel="nofollow" title="this topic does not yet exist; you can create it.">3</a></span></sup> (_Hint: You've only provided one model and not asked that the toys be frequentist._)

1.  Note that you can use `text2workspace.py` to isolate this functionality. [↩](https://twiki.cern.ch/twiki/bin/edit/CMS/Fnref-1-6361?topicparent=CMS.B2GStatisticsRecommendations;nowysiwyg=1 "this topic does not yet exist; you can create it.")
    
2.  Combine can take either a `.txt` or `.root` file in the `-d` option. The .txt file will be interpreted as a data card and the `.root` file will be interpreted as a workspace with a model already built. [↩](https://twiki.cern.ch/twiki/bin/edit/CMS/Fnref-2-6361?topicparent=CMS.B2GStatisticsRecommendations;nowysiwyg=1 "this topic does not yet exist; you can create it.")
    
3.  In this command, Combine is using the pre-fit model created from card.txt to generate toys. The values of the parameters of each toy are generated from their pre-fit uncertainty constraints. If the values of the parameters are not meaningful before fitting to data, this command would not produce useful results! [↩](https://twiki.cern.ch/twiki/bin/edit/CMS/Fnref-3-6361?topicparent=CMS.B2GStatisticsRecommendations;nowysiwyg=1 "this topic does not yet exist; you can create it.")
    

## [B2G](https://twiki.cern.ch/twiki/bin/view/CMS/B2G) Requested Statistical Tests

By default, [B2G](https://twiki.cern.ch/twiki/bin/view/CMS/B2G) requests several statistical tests and indicators to verify the health of your fits.

-   Nuisance parameter pulls
-   Nuisance parameter impacts
-   Goodness of Fit test
-   Signal injection tests
-   F-tests (if applicable)

While other tests might also be appropriate, these are the minimum that the Combine contact will look/ask for during the analysis object review. Below is an explanation of each test and how to use Combine to execute the test. Note that there are not always standard commands that work for every analysis so in those cases, some suggestions are presented with possible variations given the goals of the analysis (ex. to remain blinded, to derive a background from a control region, etc) but that does not mean they should by copy-pasted!

## Nuisance parameter pulls

Perhaps poorly named, the nuisance parameter pulls are the comparison of the nuisance parameter values and their errors pre- and post-fit. The plot produced by Combine looks like the following figure:

[![](https://twiki.cern.ch/twiki/pub/CMS/B2GStatisticsRecommendations/nuisance_pulls.png)](https://twiki.cern.ch/twiki/pub/CMS/B2GStatisticsRecommendations/nuisance_pulls.png)

Example of a nuisance pull plot

This can be produced via the diffNuisances.py script that comes with every Combine installation. More information about nuisance parameter pulls can be found [here![](https://twiki.cern.ch/twiki/pub/TWiki/TWikiDocGraphics/external-link.gif)](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/part3/nonstandard/#pre-and-post-fit-nuisance-parameters-and-pulls) in the Combine documentation.

## Nuisance parameter impacts

The impact of a nuisance parameter is how much the value of the POI (default is signal strength, `r`) changes when the parameter is varied up and down by the uncertainty on the parameter. In most cases, the uncertainty on the parameter should be post-fit.

Calculating the impacts of the nuisance parameters is described in the official Combine documentation [here](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/part3/nonstandard/#nuisance-parameter-impacts). The provided commands use the `combineTool.py` from Combine Harvester to perform an initial fit to the data and then loop over the nuisance parameters and vary them according their post-fit values and uncertainties and re-fit the data with nuisance parameters frozen one at a time.

Please consider the following notes when constructing the commands to use for your analysis.

-   The commands assume that you need to construct a workspace from your data card via `text2workspace.py`. This will produce a pre-fit workspace to perform the initial fit and the following fits with frozen nuisances. Consider whether this is the model which you'd like to use.
-   The standard Combine command line options are available for use and will be passed by `combineTool.py` to Combine if they are not `combineTool.py` specific.
-   If using `r` as your POI, consider removing "physics" bounds on the parameter. For example, many analysis require `r >` 0= to enforce a positive excess. While this is applicable to the physical interpretation of the fit results, it is harmful in a test of variations in `r` as a fit parameter. In this case, the fit will "bounce off" the boundary on `r` if the post-fit `r` value is already close to 0 and the calculated impact of each nuisance will be biased (and possibly one-sided). The boundaries on `r` can be changed with the `--rMin` and `--rMax` options.

## Goodness of Fit test

The Goodness of Fit ([GoF](https://twiki.cern.ch/twiki/bin/view/CMS/GoF)) test compares the value of a [test statistic](https://en.wikipedia.org/wiki/Test_statistic) calculated in data against the value of the test statistic calculated for fits to toy data generated from a model. The goal of the [GoF](https://twiki.cern.ch/twiki/bin/view/CMS/GoF) test is to measure the health of the background estimation by using the background model to fit toys created from the generative model in combine.

!!! warning If your analysis is blinded, it is important that this test stays blind to collision data in the signal region.

This is where dedicated control regions come into play. Analyses often use measurement or control regions to perform the statistical tests when blinded. For instructions on how to generate toys while blinded, see the page [Generating Pseudoexperiments with Combine](https://twiki.cern.ch/twiki/bin/view/CMS/B2GStatisticsRecommendations#Generating_Pseudoexperiments_Toy).

!!! example Measuring the Goodness of Fit in data

```
 =```bash combine -M GoodnessOfFit -d mycard.txt --algo=saturated <other options> ``` This will output a file with a single value called "limit" which is actually your saturated test statistic value for the fit to data.= 
```

!!! example Measuring the Goodness of Fit in toys Please first read the page on [Generating Pseudoexperiments with Combine](https://twiki.cern.ch/twiki/bin/view/CMS/B2GStatisticsRecommendations#Generating_Pseudoexperiments_Toy). From this, you should be able to develop a plan for an input toys file. The Goodness of Fit of these toys can be evaluated using the following command: `bash combine -M [GoodnessOfFit](https://twiki.cern.ch/twiki/bin/edit/CMS/GoodnessOfFit?topicparent=CMS.B2GStatisticsRecommendations;nowysiwyg=1 "this topic does not yet exist; you can create it.") --algo=saturated -d <card or workspace> --toysFile higgsCombineTest.GenerateOnly.mH120.123456.root <possibly --toysFrequentist>` Note: If `--toysFrequentist` is "on", combine will load in the toy model stored in the file provided by `--toysFile` as the pre-fit model to fit the toy data set. If `--toysFrequentist` is "off", combine will use whatever model is specified via the `-d` option (either a card or [RooWorkspace](https://twiki.cern.ch/twiki/bin/edit/CMS/RooWorkspace?topicparent=CMS.B2GStatisticsRecommendations;nowysiwyg=1 "this topic does not yet exist; you can create it.")) to fit the toy data sets. Since toys are not generated at this step, the `--bypassFrequentistFit` option does nothing.

The default test statistic of Combine is the [saturated test statistic](http://www.physics.ucla.edu/~cousins/stats/cousins_saturated.pdf). Details about additional [GoF](https://twiki.cern.ch/twiki/bin/view/CMS/GoF) tests are available in [the combine docs](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/part3/commonstatsmethods/#goodness-of-fit-tests).

!!! warning Warning for 2D Methods The Kolmorgorov-Smirnov (KS) test and Anderson-Darling (AD) test are meant for 1D only. Even mapping the 2D fit to 1D can lead to uninterperatable results using these tests because cumulative distributions are not well defined in more than one dimension. We recommend using the default saturated test statistic.

The final comparison should be the distribution of the test statistic for the toys (histogram typically) with the value of the test statistic in data (an arrow or line on top of the toy histogram). It's also recommended to fit the toy distribution with a Gaussian curve and to calculate the p-value of the data relative to the Gaussian fit.

## Signal injection tests

The Signal Injection test (sometimes called Bias test) analyzes the bias of a model to over or under model the true signal contribution in data. Since we don't know how much signal is in the data, we generate toys with a known, fixed amount of signal, ask the fit to find the amount of signal, and compare how well it does over many toys where the signal is fixed but the background is varied.

**NOTE:** This is a non-trivial point - "where the signal is fixed but the background is varied." That means one needs a background-only model that is already data-like before the signal can be fixed to a value _on top_ of the background.

For example, the following command will not work correctly.

```
 =# THIS DOES NOT WORK CORRECTLY - DO NOT COPY IT combine -M GenerateOnly -d <card or workspace> -t 500 --toysFrequentist --expectSignal 1= 
```

This command is asking a fit to data to be performed first (via `--toysFrequentist`) _with the signal frozen to 1_. Even if you are unblinded (and so `--toysFrequentist` is appropriate), you do not want to assume the data has signal in it. Doing so will bias your background model and thus bias your signal injection test.

If the model in your input workspace is already data-like, you can just add `--bypassFrequentist` fit. If not, you'll need to create a data-like model (instructions are provided in [Generating Pseudoexperiments with Combine](https://twiki.cern.ch/twiki/bin/view/CMS/B2GStatisticsRecommendations#Generating_Pseudoexperiments_Toy))

!!! example Performing a Signal Injection Test with Combine Here is an example command for running this test after you have created a data-like workspace and have produced toys with signal already injected. `bash combine -M [FitDiagnostics](https://twiki.cern.ch/twiki/bin/edit/CMS/FitDiagnostics?topicparent=CMS.B2GStatisticsRecommendations;nowysiwyg=1 "this topic does not yet exist; you can create it.") -d <morphedWorkSpace> --toysFile <toyFile.root> -t <numberOfToys> --rMin <minimumPOIvalue> --rmax <maximumPOIvalue>` Note: If you expect r=0, then `minimumPOIvalue` should be negative. If `minimumPOIvalue` is fixed at 0, then the fit will hit a "wall" at 0 biasing the fit towards a positive value of r.

!!! warning Low event yield searches There have been several analyses that have seen biases, even when running the test correctly. If an analysis has this issue and has very low per-bin yields, we ask that they scale their backgrounds artificially by a factor of 10 (signal remains the same) and re-perform the test. Because there is a natural barrier at zero for event yields and no equivalent upper bound, this test can have a natural bias. By scaling the backgrounds up, the model can still be tested for bias while staying away from this natural barrier at 0.

Before plotting the distribution of `r`, make a cut on the `fit_status` to require that it be equal to 0 (fit finished with no errors). If this kills a good majority of your toy fits (> 10%), there may be a problem and the behavior should be investigated. For those fits which concluded without error, the final distribution of `r` should be presented in two ways. The first is to plot (as a histogram) the distribution of `r` minus the amount injected (so that an unbiased test is always a distribution centered at 0). This should be fit with a Gaussian and the mean and width reported. The second is to plot (as a histogram) the following equation:

`(r-r_inj)/r_Err`

where `r_inj` is the amount injected and `r_Err` is `rHiErr*(r-r_inj<0)+rLoErr*(r-r_inj>0)` and where `rHiErr` is the upper error on `r` and `rLoErr` is the lower error on `r` The plotted quantity is referred to as the pull and it should be fitted with a Gaussian as well. In a perfect test, the mean of this Gaussian is 0 and the width is 1.

B2G requests that you perform several of these tests. For three sufficiently different signals (ex. mass points), you should test the cases of injecting `r = 0` and `r` values corresponding to the lower, median, and upper expected limits for that signal sample (ie. the expected limit and the edges of the 68% CL band). Thus, for three different signal samples, you should have 12 total signal injection tests.

## F-tests (if applicable)

If there is a need for an F-test, please ask the Combine contacts for instruction.

## Generating Pseudoexperiments (Toys) with Combine

Several requested tests require generating toy data ("toys") from a model. Before discussing these tests, toy generation should be considered on its own.

Two questions should be considered:

-   From what model do I want to generate my toys?
-   With what model do I want to fit my toys?

The answers to these questions differ between tests and between analyses and so you should consider the statistical interpretation of your choices before running anything. The information below should clarify the behavior of the various Combine options so that you can correctly construct your test around the Combine infrastructure.

## Commands to generate toys

Toys can be generated in Combine in two different ways:

1.  with the dedicated `GenerateOnly` mode or
2.  with the `-t <ntoys>` option passed to any other mode which will generate the toys simultaneously to evaluating them.

The toys generated from the first option can be passed to another Combine command with `--toysFile higgCombine<name>.GenerateOnly.<seed>.root -t <ntoys>`.

The second option is provided to simplify workflows but if used incorrectly, will generate toys from a model that was not intended or fit with a model that was not intended. It should only be used if you understand what model is being used when and you've confirmed this is the behavior you've intended.

## Options to choose models

The following options (applicable to these docs) modify the models used in generating toys or performing fits (or both).

-   `-d`: The input card or workspace (ROOT file) that will be considered your "pre-fit" model.
-   `--snapshotName`: If the option provided to `-d` is a ROOT file with a workspace `w`, this option will load a `snapshot` (set of parameter values) stored inside the workspace to use as your pre-fit "starting point."
-   `--toysFrequentist`:

-   For the toy generation, this option... 0. fits the data if `--bypassFrequentistFit` is not provided;
    1.  samples a model from the current model parameter values+/-errors -- if bypassing the frequentist fit, this is just the parameter values you feed in via the -d option (card or [RooWorkspace](https://twiki.cern.ch/twiki/bin/edit/CMS/RooWorkspace?topicparent=CMS.B2GStatisticsRecommendations;nowysiwyg=1 "this topic does not yet exist; you can create it."));
    2.  assigns the parameters in this toy model their pre-fit errors;
    3.  generates a toy data set from this toy model.
-   For fits,
    -   If `--toysFrequentist` is "on", combine will load in the toy model stored in the file provided by `--toysFile` as the pre-fit model to fit the toy data set.
    -   If `--toysFrequentist` is "off", combine will use whatever model is specified via the `-d` option (either a card or [RooWorkspace](https://twiki.cern.ch/twiki/bin/edit/CMS/RooWorkspace?topicparent=CMS.B2GStatisticsRecommendations;nowysiwyg=1 "this topic does not yet exist; you can create it.")) to fit the toy data sets.
-   In summary, a fit to data is performed first, and then for each toy it is the external constraints that are

-   `--expectSignal`: Freezes `r` to the provided value.
    -   If using with `--toysFrequentist` during toy generation, this will freeze `r` during the frequentist fit to data. If the value provided to `--expectSignal` is non-zero, this is not what you want most of the time! Your frequentist fit to data should usually be background-only.

## Generating toys when blinded

Before describing the process in detail lets define a new term Measurement Region (MR). In this case, the MR is the region where the nuisance parameters will be measured in data so that the distribution of toys is more data-like. This region is orthogonal to the SR to ensure that the analysis is blinded. Some groups using data driven approaches define their own MR outside of the SR and CR, while others use the CR.

## 1\. Measure the Nuisance Parameters in the MR:

If you provide to Combine a card or workspace with the MR and the SR, it will automatically try to perform the fit simultaneously for MR+SR. If you are blinded, it is imperative that the SR be masked. You can explicitly drop it from the card/workspace but we'd recommend instead using the channel masks feature of Combine so that the total likelihood can be built and used in later steps.

!!! example Masking the Signal Region in Combine Full documentation for masking channels can be found in the [combine docs](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/part3/nonstandard/#channel-masking) In order to mask the SR, the channel masks must be enabled when creating the workspace: `bash text2workspace.py datacard.txt --channel-masks` This creates a masking parameter for each channel labeled `mask_CHANNELNAME`. Assuming the signal channel is called `signal`, then masking the SR is done by running combine with the option `--setParameters mask_signal=1`. This can be done in-line with your main Combine command with something like the following: `bash combine -M [FitDiagnostics](https://twiki.cern.ch/twiki/bin/edit/CMS/FitDiagnostics?topicparent=CMS.B2GStatisticsRecommendations;nowysiwyg=1 "this topic does not yet exist; you can create it.") -d mycard.txt --text2workspace "--channel-masks" --setParameters mask_signal=1`

## 2\. Generate Toys:

The toys should be generated for the MR+SR using the nuisance parameter values from step 1. To do this the signal region must now be unmasked. This **does not** mean that the test is unblinded. Rather, it means that toys will be generated for the SR using the nuisance parameter values from the MR-only fit. 500 toys is a large value without being overkill and is our recommended starting point.

!!! danger Keeping the Signal Region Blinded There are two important options to know about when generating toys: `--toysFrequentist` and `--bypassFrequentistFit`. If you are blinded and have turned off the SR mask, never use `--toysFrequentist` by itself (ie. without `--bypassFrequentistFit`) in the generation step or this will unblind the SR (it will ignore the post-MR-fit parameter values that were set and load in whatever it finds via the frequentist fit).

!!! tip Providing a Model The model you provide via `-d` is not necessarily the model from which toys are generated. If you use `--toysFrequentist` without `--bypassFrequentistFit`, the model provided via `-d` will be fit to data and that fit result will be used to randomly generate a new model and your toy will be sampled from this new model. If you use `--toysFrequentist --bypassFrequentistFit`, the same will occur except the initial fit to data will not occur. If you use neither option, the toy will be generated directly from `-d`. Note that this is not considered "frequentist" and we recommend that `--toysFrequentist` be used. If you are blinded, you should also use `--bypassFrequentistFit`. However, keep in mind that your input model from `-d` should not need to be fit to data (ie. it was already in a control/measurement region).

!!! example Toy Generation using the Measurement Region Post-fit values The original model passed to combine in step 1 now needs to be manipulated so that the parameters of the model are at their post-MR-fit values. To do this, we've provided a tool called [importPars.py](https://github.com/lcorcodilos/2DAlphabet/blob/bstar/importPars.py) that can be used as standalone or inside of a python script. One can use this tool standalone like so: `bash python importPars.py /path/to/SRfit/card.txt /path/to/CRfit/fitDiagnostics.root` This will output `morphedWorkspace.root` which contains a workspace with parameters set to the values in the fit result from fitDiagnostics.root.

```
 =This workspace should be used as input to the GenerateOnly Combine step. ```bash combine -M GenerateOnly -d morphedWorkspace.root --toysFrequentist --bypassFrequentistFit --expectSignal <r> -t <N> --saveToys ```= 
```

## 3\. Input the toys:

In your next step, whether it be [FitDiagnostics](https://twiki.cern.ch/twiki/bin/edit/CMS/FitDiagnostics?topicparent=CMS.B2GStatisticsRecommendations;nowysiwyg=1 "this topic does not yet exist; you can create it."), [GoodnessOfFit](https://twiki.cern.ch/twiki/bin/edit/CMS/GoodnessOfFit?topicparent=CMS.B2GStatisticsRecommendations;nowysiwyg=1 "this topic does not yet exist; you can create it."), or something else, you can pass the newly generated toys via the following option `--toysFile higgsCombineTest.GenerateOnly.123456.root` (or similar). You should also be sure to include the number of toys you'd like to use `-t <N>` and `--toysFrequentist` again.

!!! tip Why do I need `--toysFrequentist` again? When fitting already generated toys, the `--toysFrequentist` option is still important - though not always wanted! When frequentist toys are generated, they are saved along with the model that generated the toy (remember, this is not your pre-fit model). When you specify `--toysFrequentist` during the fitting of the toys, the generative model will be used as the model to fit the new toy _with the constraints moved to nominal values of the generative model_. For example, pretend you have a model with one constrained nuisance parameter with a pre-fit value of 0 +/- 1. You perform the frequentist fit to data which says that the likelihood is minimized at 0.4 +/- 0.9 for this parameter. Combine will sample the parameter and let's say it picks 0.6. A new set of constraints will be created _centered on 0.6_ with a width of 1.0 (from your pre-fit model). Now a toy will be generated from the Gaussian PDF with mean 0.6 and width 1.0. Once fitting, this model (0.6 +/- 1.0) will be loaded to fit the toy that it generated.

\-- [ClemensLange](https://twiki.cern.ch/twiki/bin/view/Main/ClemensLange) - 2017-01-19