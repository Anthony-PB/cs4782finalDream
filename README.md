### Introduction
- Purpose of this Git repo (mention how this is a project that attempts to re-implement your paper of choice)
- Introduce the paper chosen and its main contribution.

This github repo recreates DreamBooth (from the paper "DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation" by Nataniel Ruiz et al.), which fine tunes diffusion models to be able to generate specific subjects in novel environments from very few examples (3-5 to be specific).

### Chosen Result
- Identify the specific result you aimed to reproduce and its significance in the context of the paper’s
main contribution(s).
- Include the relevant figure, table, or equation reference from the original paper.

DreamBooth is very effective at preserving unique features of a specific subject while placing them in unique environments unlike those in any of the training examples. How well this is done can be measured by subject and prompt fidelity metrics that the original paper invented.
How well the original implementation performed at these is shown in the following tables.

| Method | DINO↑ | CLIP-I↑ | CLIP-T↑ |
|---|---:|---:|---:|
| Real Images | 0.774 | 0.885 | N/A |
| DreamBooth (Imagen) | **0.696** | **0.812** | **0.306** |
| DreamBooth (Stable Diffusion) | 0.668 | 0.803 | 0.305 |
| Textual Inversion (Stable Diffusion) | 0.569 | 0.780 | 0.255 |

**Table 1.** Subject fidelity (DINO, CLIP-I) and prompt fidelity (CLIP-T, CLIP-T-L) quantitative metric comparison.

| Method | Subject Fidelity↑ | Prompt Fidelity↑ |
|---|---:|---:|
| DreamBooth (Stable Diffusion) | **68%** | **81%** |
| Textual Inversion (Stable Diffusion) | 22% | 12% |
| Undecided | 10% | 7% |

**Table 2.** Subject fidelity and prompt fidelity user preference.

### GitHub Contents
• Make a brief note about the content structure of your project.

`code/` contains all the code of the re-implementation
`data/` contains the dataset from the original paper in `data/dreambooth-dataset/` and an extra dataset that we added in `data/killian/`.
`poster/` contains a poster summarizing this project
`report/` contains a report explaing the project in more depth than is done here.
`results/` contains results of our testing (TODO)

### Re-implementation Details
- Describe your approach to re-implementation or experimentation.
- Include key details about models, datasets, tools, and evaluation metrics.
- Mention any challenges or modifications made to the original approach.

### Reproduction Steps
As meta as this section is, it essentially documents steps someone would need to follow to implement your GitHub repo in a local environment.
- Describe ”how someone using your GitHub can re-implement your re-implementation?”
- Provide instructions for running your code, including any dependencies, required libraries, and command line arguments.
- Specify the computational resources (e.g., GPU) needed to reproduce your results.

### Results/Insights
- Present your re-implementation results as a comparison to the original paper’s findings. Describes ”what can someone expect as the end-result of using your GitHub repo?”

### Conclusion
- Summarize the key takeaways from your re-implementation effort and the lessons learned.

### References
- Include a list of references, including the original paper and any additional resources used in your re-implementation.

### Acknowledgements
- Recognition goes a long way in setting up the context of your work. Your acknowledgements also act as an indirect validation about the quality of the work. For e.g., having done this project as part of coursework is a sign that the work was potentially peer-reviewed or graded- i.e. added authenticity.

Ruiz, Nataniel, et al. ‘DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation’. arXiv [Cs.CV], 2023, arxiv.org/abs/2208.12242. arXiv.

https://www.cs.cornell.edu/courses/cs4782/2026sp/docs/final_deliverables.pdf
