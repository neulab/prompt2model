# Contributing to prompt2model

If you're reading this, you're probably interested in contributing to prompt2model. Thank you for your interest!

## Developer Installation

If you're a developer, you should also install pre-commit hooks before developing.

```bash
pre-commit install
```

These will do things like ensuring code formatting, linting, and type checking.

You'll also want to run tests to make sure your code is working by running the following:

```bash
pytest
```

## Contribution Guide

If you want to make a contribution you can:
1. Browse existing issues and take one to handle
2. Create a new issue to discuss a feature that you might want to contribute
3. Send a PR directly

We'd recommend the first two to increase the chances of your PR being accepted, but if you're confident in your contribution, you can go ahead and send a PR directly.

## Making a Release

If you are an admin of the repository, you can make a new release of the library.

We are using the [hatchling](https://github.com/pypa/hatch) build system, which makes it easy to make new library releases.
In order to do so, just create a new version tag on github (it has to be a valid [semantic version](https://semver.org/)) and the CI will automatically build and publish the new version to PyPI.
