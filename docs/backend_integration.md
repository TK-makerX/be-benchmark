# Custom backends for the interface demonstration

1. Copy `configs/backend_registry.template.json` to a location outside the repository.
2. Add a backend identifier, supported editor families, supported tasks and a connector import path.
3. Point `BE_SHELL_REGISTRY` to that private registry.
4. Implement authentication, model loading, numerical inference and output validation in the private connector.

For the study adapters, model input windows and checkpoints, see [model deployment](../reproducibility/deployment/README.md).
