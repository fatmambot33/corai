# Release Process

## Creating a New Release

To create a new release of this package:

1. **Update the version in `pyproject.toml`**
   - Increment the version number following semantic versioning
   - Commit and push the change to the main/develop branch

2. **Create a GitHub release**
   - Go to the GitHub repository releases page
   - Click "Create a new release"
   - Create a new tag matching the version in `pyproject.toml` (e.g., `0.1.4`)
   - Add release notes describing the changes
   - Publish the release

3. **Automated workflow**
   - The `Upload Python Package` workflow will automatically trigger
   - It will verify the version matches the release tag
   - It will build and publish to PyPI

## Version Validation

The release workflow includes an automatic version validation step that:
- Extracts the version from `pyproject.toml`
- Compares it with the GitHub release tag
- Fails the build if there's a mismatch

This prevents the following error:
```
HTTPError: 400 Bad Request from https://upload.pypi.org/legacy/
File already exists
```

## Important Notes

- **PyPI does not allow re-uploading the same version**, even if deleted
- Always ensure `pyproject.toml` version matches the release tag **before** creating the release
- If a release fails due to version mismatch, you must:
  1. Delete the failed release tag
  2. Update the version in `pyproject.toml`
  3. Create a new release with the corrected version

## Troubleshooting

### Release workflow failed with "File already exists"

This occurs when:
- The version in `pyproject.toml` doesn't match the release tag
- The workflow tried to upload a version that already exists on PyPI

**Solution**: Create a new release with an incremented version number.

### Version mismatch error

If the workflow fails with "Version mismatch!" error:
1. Delete the release and tag from GitHub
2. Update the version in `pyproject.toml` to match your intended release version
3. Commit and push the change
4. Create a new release with the matching tag
