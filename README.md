# LLM benchmark performance structure and alignment prediction 

## Question
Is there any latent structure to LLM capabilities that explains performance across different benchmarks, and can this be used to predict alignment (honesty)?

## Approach
- Small but quite dense dataset (54 models across 12 benchmarks, from [Artificial Analysis](https://artificialanalysis.ai/).
- PCA with parallel analysis to reveal 'capability structure', inspired by [Epoch AI's approach](https://epoch.ai/gradient-updates/benchmark-scores-general-capability-claudiness)
- Hierarchical regression approach to identify components that predict honesty beyond creator/general capability

## Main takeaways
- One dominant component captured ~80% of variance in data, suggesting that different benchmarks may be largely measuring the same thing. This PC was roughly just an average across all benchmarks, thus representing general intelligence/ability. 
- Honesty was predicatable - but mostly by creator. General capability (PC1) added some predictive power.
- Projection of models onto a higher-order component (PC3, representing something like an agentic-vs-math dimension) may also give additional predictive power - but hard to tell if this is real from my limited data.

See [analysis notebook](https://github.com/moriohamada/capability_structure/blob/main/analysis_notebook.ipynb) for further discussion and details. 

