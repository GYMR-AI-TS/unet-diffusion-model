# unet-diffusion-model

Diffusion models are generative models that iteratively refine noisy data to approximate a target data distribution. They are widely used in tasks like image generation, where the goal is to generate high-fidelity samples from random noise. The U-Net architecture is a popular choice for diffusion models due to its unique design and capabilities.

## Installation

```bash
pip install -e .
```

## CI/CD Pipeline

This project includes an automated CI pipeline that runs on every push and pull request.

### What the CI Does

The CI pipeline automatically:

- **Lints code** with [Ruff](https://github.com/astral-sh/ruff) to catch style and logical errors
- **Checks formatting** with [Black](https://github.com/psf/black) to ensure consistent code style
- **Runs tests** with [pytest](https://pytest.org/) to verify functionality
- **Measures coverage** with pytest-cov to track test coverage

### When CI Runs

The pipeline automatically triggers on:
- Push to `main`
- Pull requests targeting `main`

### CI Status

You can check the CI status for a push or PR by looking at the **Checks** section in GitHub.

### Before Pushing Code

Before pushing, make sure your code passes local checks:

```bash
# Format code with Black
black .

# Lint with Ruff
ruff check . --fix

# Run tests
pytest tests/ -v
```

This ensures your PR will pass CI checks.
