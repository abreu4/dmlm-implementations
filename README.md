## Deploying machine learning models

My Python machine learning pipeline implementations from [this](https://www.udemy.com/course/deployment-of-machine-learning-models/) course.

Students are required to write code for 3 types of machine learning pipelines:

* Research pipeline
* Procedural pipeline
* Third-party based pipeline (ex.: scikit-learn)

I skipped the procedural pipeline given this approach is sub-optimal for production environments.

* `research/` includes a Jupyter notebook for quick model development and research, including code for data treatment, feature extraction and feature selection.
* `production/` includes a complete scikit-learn-based pipeline, optimal for containerizing and deploying models in CI/CD context.