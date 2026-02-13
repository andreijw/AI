# Copilot Deployment Workflow Setup

This document describes how to set up the Copilot deployment workflow with signed commits.

## Overview

The Copilot deployment workflow enables automated commits from GitHub Actions with GPG signature verification. This ensures that all automated changes are properly signed and verified.

## Prerequisites

- GitHub repository with Actions enabled
- GPG key pair for signing commits

## Setup Instructions

### 1. Generate GPG Key (if you don't have one)

```bash
# Generate a new GPG key
gpg --full-generate-key

# Follow the prompts:
# - Key type: RSA and RSA
# - Key size: 4096
# - Expiration: your choice (e.g., 1 year)
# - Name: github-actions[bot]
# - Email: github-actions[bot]@users.noreply.github.com
```

### 2. Export GPG Key

```bash
# List your GPG keys to find the key ID
gpg --list-secret-keys --keyid-format=long

# Export the private key (replace KEY_ID with your actual key ID)
gpg --armor --export-secret-keys KEY_ID

# The output will be your GPG_PRIVATE_KEY
```

### 3. Add Secrets to GitHub Repository

Add the following secrets to your GitHub repository:

1. Go to: Settings → Secrets and variables → Actions → New repository secret

2. Add `GPG_PRIVATE_KEY`:
   - Name: `GPG_PRIVATE_KEY`
   - Value: The output from the export command above (including `-----BEGIN PGP PRIVATE KEY BLOCK-----` and `-----END PGP PRIVATE KEY BLOCK-----`)

3. Add `GPG_PASSPHRASE`:
   - Name: `GPG_PASSPHRASE`
   - Value: The passphrase you used when creating the GPG key (leave empty if no passphrase)

### 4. Add GPG Public Key to GitHub

1. Export your public key:
   ```bash
   gpg --armor --export KEY_ID
   ```

2. Add the public key to the appropriate place in GitHub:
   - **For a personal user account (signing your own commits)**:
     - Go to your user profile: Settings → SSH and GPG keys → New GPG key
     - Paste the public key and save
   - **For automated/bot usage (for example, keys named `github-actions[bot]`)**:
     - Prefer a dedicated machine user (a separate GitHub account used only by automation):
       - Sign in as the machine user, then go to: Settings → SSH and GPG keys → New GPG key
       - Paste the public key and save
     - Do **not** add a bot GPG key to your personal account

### 5. Enable Vigilant Mode (Optional)

To show "Verified" badges on all signed commits:

1. Go to: Settings → SSH and GPG keys
2. Enable "Flag unsigned commits as unverified"

## Workflow Features

The Copilot deployment workflow includes:

- **Automated Tasks**: Runs predefined automated tasks (customize as needed)
- **GPG Signing**: All commits are signed with GPG
- **Signature Verification**: Verifies commit signatures before pushing changes to the repository
- **Change Detection**: Only commits when there are actual changes
- **Manual Trigger**: Currently configured for manual dispatch only until deployment tasks are implemented

## Workflow Triggers

The workflow runs on:
- Manual trigger via workflow dispatch

**Note**: The workflow is currently configured to run only on manual dispatch. Once you've implemented your deployment tasks, you can add push triggers for the `main` or `develop` branches by updating the workflow's `on:` section.

## Customization

### Adding Automated Tasks

Edit the "Run automated tasks" step in `.github/workflows/copilot-deploy.yml`:

```yaml
- name: Run automated tasks
  run: |
    echo "Running Copilot automated tasks..."
    # Add your custom tasks here
    # Examples:
    # - Dependency updates: python -m pip install --upgrade -r requirements.txt
    # - Code formatting: ruff format .
    # - Security scanning: pip-audit
    echo "Deployment tasks completed successfully"
```

### Enabling Automatic Triggers

Once you've implemented your deployment tasks, you can enable the workflow to run automatically on push events. Edit the `on:` section in `.github/workflows/copilot-deploy.yml`:

```yaml
on:
  push:
    branches: [main, develop]
  workflow_dispatch:
```

This will trigger the workflow on every push to `main` or `develop` branches, in addition to manual dispatch.

### Changing Commit Message

Modify the commit message in the "Commit and push changes" step:

```yaml
git commit -S -m "chore: automated deployment updates by Copilot [bot]"
```

## Troubleshooting

### Issue: "gpg: signing failed: Inappropriate ioctl for device"

**Solution**: The ghaction-import-gpg action handles GPG configuration automatically. If you still encounter this:
1. Check that your GPG_PRIVATE_KEY is correctly formatted
2. Verify that your GPG_PASSPHRASE matches the key

### Issue: "Commit signature verification failed"

**Solution**: 
1. Ensure the GPG public key is added to your GitHub account
2. Verify the private key and passphrase are correct in secrets
3. Check that the email in the GPG key matches the git committer email

### Issue: "Permission denied" when pushing

**Solution**:
1. Ensure the workflow has proper permissions (contents: write)
2. Check that branch protection rules allow Actions to push

## Security Considerations

- **Never commit GPG keys to the repository**
- Store keys only in GitHub Secrets
- Use a dedicated GPG key for automation (don't reuse personal keys)
- Regularly rotate GPG keys (set expiration dates)
- Review and audit automated commits regularly

## Testing the Workflow

To test the workflow:

1. Trigger it manually:
   - Go to Actions tab
   - Select "Copilot Deployment" workflow
   - Click "Run workflow"

2. Check the workflow run:
   - Verify all steps complete successfully
   - Check that commits (if any) are signed and verified
   - Review the commit signature in the git log

## Additional Resources

- [GitHub GPG Commit Signature Verification](https://docs.github.com/en/authentication/managing-commit-signature-verification)
- [GitHub Actions Security](https://docs.github.com/en/actions/security-guides)
- [GPG Documentation](https://gnupg.org/documentation/)
