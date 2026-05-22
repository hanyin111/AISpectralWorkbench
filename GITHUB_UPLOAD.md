# GitHub Upload Guide

This repository has already been initialized locally with Git.

## Option 1: Upload with Git commands

1. Create an empty repository on GitHub.

   Recommended repository name:

   ```text
   ai-spectral-analysis-workbench
   ```

   Do not initialize it with README, `.gitignore`, or LICENSE, because this local repository already contains them.

2. Copy the repository URL from GitHub.

   Example:

   ```text
   https://github.com/your-username/ai-spectral-analysis-workbench.git
   ```

3. Run these commands in this project folder:

   ```bat
   git remote add origin https://github.com/your-username/ai-spectral-analysis-workbench.git
   git push -u origin main
   ```

4. If Git asks you to log in, use your GitHub username and a personal access token.

## Option 2: Upload with GitHub Desktop

1. Install GitHub Desktop from:

   ```text
   https://desktop.github.com/
   ```

2. Sign in to GitHub Desktop.
3. Choose `File` -> `Add local repository`.
4. Select this project folder.
5. Click `Publish repository`.

## Option 3: Upload from the GitHub website

1. Create a new repository on GitHub.
2. Open the repository page.
3. Click `Add file` -> `Upload files`.
4. Upload the contents of `github_source_package.zip`.

## What should not be uploaded

The following files and folders are intentionally ignored:

- `build/`
- `dist/`
- `release/`
- `AISpectralWorkbench.spec`
- generated exe files
- local experiment images and TXT/CSV data
- thesis documents

If you want to distribute the packaged exe, upload it to GitHub Releases instead of committing it to the repository.

