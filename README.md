# Math 6911

This is the final project folder of York master 6911

## Workflow
You would use the Jupyter Notebook to prepare data. After that, you need to
set up local Zipline environment and ingest data into the system. In the very end,
after you execute the algorithmic backtest, pyfolio would be used to generate report. 


## Dependencies
In general, Zipline is a strong but relatively awkward framework. I would suggest
you to set up two independent virtual environments. One to prepare data (Python > 3.8)
and the other for Zipline (Python == 3.6). In the current stage, I heavily rely on
a few packages. We can bypass later in the project.
1. [Optimization](https://pypi.org/project/pyportfolioopt/)
2. [Visualization](https://quantopian.github.io/pyfolio/)
3. [Backtest](https://www.zipline.io/bundles.html#ingesting-data-from-csv-files)
