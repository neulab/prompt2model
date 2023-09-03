# Contributing to prompt2model

Thanks for your interest in contributing to prompt2model!
We appreciate your support and welcome your
 contributions. Here's a guide to help you get started:

## Developer Installation

Installing dependencies:
Run the following command to install all the dependencies after you have forked the repository.

```bash
pip install .[dev]
```

If you're a developer, it's recommended to install
pre-commit hooks before starting your development
work. These hooks ensure code formatting, linting,
and type-checking. To install the pre-commit hooks,
run the following command:

```bash
pre-commit install
```

Additionally, it's essential to run tests to verify the
functionality of your code. Execute the following
command to run the tests:

```bash
pytest
```

## Contribution Guide

To contribute to prompt2model, or if you have any questions,
please reach out to us!

- open an [issue](https://github.com/neulab/prompt2model/issues) or submit a PR
- join us on [discord](https://discord.gg/UCy9csEmFc)
- or reach out to [@vijaytarian](https://twitter.com/vijaytarian)
  and [@Chenan3_Zhao](https://twitter.com/Chenan3_Zhao) on Twitter.

## Making a Release

If you have admin privileges for the repository,
you can create a new release of the prompt2model
library. We utilize the
[hatchling](https://github.com/pypa/hatch) build
system, which simplifies the process of making
new releases.

To create a new release, follow these steps:

1. Create a new version tag on GitHub, adhering to
the [semantic versioning](https://semver.org/) guidelines.
2. Once the tag is created, the continuous integration
(CI) system will automatically build and publish the
new version to PyPI.

By following these steps, you can effectively make
new releases of the library and contribute to its
ongoing development.
