# Boundaries and Clusters

## Requirements
Requirements:
  + Java 17
  + Maven 3.8.7
  + Python 3.9.7
    + Python requirements are in __requirements.py__
  + On MacOS
    + gtar
  + texlive-full (To generate tikz plots)

## Troubleshooting
Troubleshooting:

+ no libjvm, libjli:

  + Probably bug in MacOS javabridge. Need to use CP_JAVA_HOME instead of JAVA_HOME. Missing __/__ a the end of path. Please do:

  ```
  source ./cp_javahome_export.sh
  ```
+ To build python-javabridge on freebsd first do:

```
source ./freebsd_pack_build.sh
```

## Running experiments

Creating environment:
```
make
```
Running bagging experiment:
```
make  run_bagging
```

Running boosting experiment:
```
make  run_boosting
```

## Results

The results are placed inside __results__ directory. There are two subdirectories __results/bagging__ and __results/boosting__ that stores results for bagging and boosting committees respectively. Each of the directories contain the following files and directories:

  + <setname>.npy -- raw dataset-related results stored as numpy object
  + <setname>.pdf -- dataset-related plots for different classifiers and quality criteria. The results are ploted as a function of committee size. 
  + <setname> directory -- separate plots for each committee size. Boxplots and mean and variance plots.
  + <setname>_tex directory -- the same plots as in <setname>.pdf but in tikz format
  + Ranks:
    + Ranks.pdf -- plot of average ranks for each method and criterion. The ranks are averaged over all base classifiers.
    + Ranks directory -- the same plots but in tikz format
    + Ranks_std directory -- ranked standard deviation plots in tikz format
    + Rank.tex -- results in tabular form
    + Ranks_std.pdf -- average ranks plots for standard deviation.
    + Rank_std.tex -- standard deviation results in tabular form
    + Ranks_wra.pdf -- also rank plots but using weighted ranking approach.
    + Ranks_wra directory -- the same plots but in tikz format
  + Ranks_all:
    + Ranks_all.pdf -- separate rank plots for each base classifier
    + Ranks_all directory -- plots in tikz format
    + Ranks_all_wra.pdf  -- plots for weighted ranks approach
    + Ranks_all_std.pdf -- plots of ranked standard deviations
    + Ranks_all_std directory -- the same plots but in tikz format
  + Trends:
    + Trend_means_all.tex(md) -- trend related (base classifiers separately) results in tabular form
    + Trend_stds_all.tex(md) -- trend related (base classifiers separately) results for standard deviation in tabular form
    + Trend_means.tex(md) -- trend related results in tabular form
    + Trend_stds.tex(md) -- trend related results for standard deviation in tabular form
