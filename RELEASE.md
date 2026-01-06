# Release Process

## Branch Strategy

- **`develop`**: Main development branch where all new features and fixes are merged
- **`main`**: Stable release branch - only contains production-ready code
- **Releases**: Must be created from the `main` branch only

## Creating a New Release

To create a new release of this package:

1. **Merge changes to main branch**
   - Ensure all changes are merged and tested in `develop`
   - Create a PR to merge `develop` into `main`
   - Get approval and merge the PR

2. **Update the version in `pyproject.toml`**
   - Increment the version number following semantic versioning
   - This should be done on `main` branch before creating the release
   - Commit and push the change to `main`

3. **Create a GitHub release**
   - Go to the GitHub repository releases page
   - Click "Create a new release"
   - **Important**: Set target branch to `main` (not `develop`)
   - Create a new tag matching the version in `pyproject.toml` (e.g., `0.1.4`)
   - Add release notes describing the changes
   - Publish the release

4. **Automated workflow**
   - The `Upload Python Package` workflow will automatically trigger
   - It will build and publish to PyPI
   - **Important**: Ensure version is updated and release is from main before publishing

## Manual Validation Checklist

Before creating a release, verify:
- [ ] All changes are merged from `develop` to `main`
- [ ] Version in `pyproject.toml` matches the intended release tag
- [ ] Release will be created targeting the `main` branch
- [ ] Version doesn't already exist on PyPI

This prevents the following error:
```
HTTPError: 400 Bad Request from https://upload.pypi.org/legacy/
File already exists
```

## Important Notes

- **Releases must be created from the `main` branch**, not `develop`
- **PyPI does not allow re-uploading the same version**, even if deleted
- Always ensure `pyproject.toml` version matches the release tag **before** creating the release
- If a release fails due to version mismatch, you must:
  1. Delete the failed release tag
  2. Update the version in `pyproject.toml` on the `main` branch
  3. Create a new release with the corrected version from `main`

## Troubleshooting

### Release workflow failed with "File already exists"

This occurs when:
- The release was created from `develop` branch instead of `main`
- The version in `pyproject.toml` doesn't match the release tag
- The workflow tried to upload a version that already exists on PyPI

**Solution**: 
1. Delete the failed release and tag from GitHub
2. Verify changes are merged to `main` branch
3. Update version in `pyproject.toml` on `main` to a new version number
4. Commit and push changes to `main`
5. Create a new release from `main` with the matching new version number

### Wrong branch used for release

If you accidentally created a release from `develop`:
1. Delete the release and tag from GitHub
2. Merge your changes from `develop` to `main`
3. Verify version in `pyproject.toml` on `main` is correct
4. Create a new release targeting the `main` branch

### Version already exists on PyPI

If the version number already exists on PyPI:
1. Delete the release and tag from GitHub
2. Increment the version in `pyproject.toml` on `main` branch
3. Commit and push the change to `main`
4. Create a new release from `main` with the new version tag
